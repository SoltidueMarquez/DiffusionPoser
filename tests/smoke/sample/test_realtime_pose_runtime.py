from __future__ import annotations

import numpy as np
import torch
from ema_pytorch import EMA

from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM
from data_loaders.realtime_pose_config import TaIDConfig
from model.realtime_pose_target_dit import RealtimePoseTargetDiT
from sample.realtime_pose_runtime import RealtimePoseRuntime, step_realtime_pose_batch
from sample.utils import load_checkpoint_model
from tests.smoke.realtime_pose_fixtures import IDENTITY_6D, build_toy_realtime_source


class _RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.valid_counts: list[int] = []
        self.batch_sizes: list[int] = []
        self.prepare_grad_enabled: list[bool] = []

    def forward(self, value, *_args, **_kwargs):
        return value + self.anchor

    def prepare_conditioning(
        self,
        pose_history,
        tracker_history,
        current_tracker,
        trajectory_history,
        current_trajectory,
        valid_frame_mask,
    ):
        batch_size = int(pose_history.shape[0])
        assert tuple(pose_history.shape) == (batch_size, 60, 144)
        assert tuple(tracker_history.shape) == (batch_size, 60, 6, 13)
        assert tuple(current_tracker.shape) == (batch_size, 6, 13)
        assert tuple(trajectory_history.shape) == (batch_size, 60, 5)
        assert tuple(current_trajectory.shape) == (batch_size, 1, 5)
        assert tuple(valid_frame_mask.shape) == (batch_size, 60)
        self.valid_counts.extend(
            int(value) for value in valid_frame_mask.sum(dim=1).detach().cpu().tolist()
        )
        self.batch_sizes.append(batch_size)
        self.prepare_grad_enabled.append(torch.is_grad_enabled())
        return {"frame": len(self.valid_counts)}


class _OneStepProjectedDiffusion:
    def projected_ddim_sample_loop(
        self,
        model,
        shape,
        projection_fn,
        model_kwargs,
        device,
        **kwargs,
    ):
        del model, model_kwargs, kwargs
        raw = torch.as_tensor(
            np.tile(IDENTITY_6D, 24), device=device, dtype=torch.float32
        ).reshape(1, REALTIME_POSE_TARGET_DIM).expand(shape[0], -1).clone()
        deployed = projection_fn(raw)
        return {
            "sample": deployed,
            "raw_pred_xstart": raw,
            "deployed_pred_xstart": deployed,
        }


def _step(runtime: RealtimePoseRuntime, source: dict[str, np.ndarray], frame: int, valid: np.ndarray):
    configured = np.ones(6, dtype=bool)
    return runtime.step(
        source["tracker_pos_world"][frame],
        source["tracker_rot_world_6d"][frame],
        configured,
        valid,
        float(source["root_pos_world"][frame, 1]),
    )


def test_runtime_cold_start_cache_and_hard_recovery_gate() -> None:
    source = build_toy_realtime_source(frame_count=20)
    model = _RecordingModel()
    runtime = RealtimePoseRuntime(
        model,
        _OneStepProjectedDiffusion(),
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
    valid = np.ones(6, dtype=bool)
    stable_results = [_step(runtime, source, frame, valid) for frame in range(16)]

    assert model.valid_counts[:3] == [0, 1, 2]
    assert model.valid_counts[-1] == 15
    assert not any(model.prepare_grad_enabled)
    assert stable_results[0].hard_rotation_state.tolist() == [True, False, False, False, False, False]
    assert stable_results[14].hard_rotation_state.all()
    assert len(runtime.pose_history) == 16
    assert len(runtime.tracker_history) == 16
    assert len(runtime.trajectory_history) == 16

    dropout = valid.copy()
    dropout[[1, 3]] = False
    dropped = _step(runtime, source, 16, dropout)
    assert not dropped.hard_rotation_state[[1, 3]].any()
    assert runtime.previous_d_off[[1, 3]].tolist() == [1, 1]
    assert runtime.previous_d_on[[1, 3]].tolist() == [0, 0]

    reconnected = _step(runtime, source, 17, valid)
    assert not reconnected.hard_rotation_state[[1, 3]].any()
    assert runtime.previous_d_off[[1, 3]].tolist() == [0, 0]
    assert runtime.previous_d_on[[1, 3]].tolist() == [1, 1]
    assert reconnected.resolved_pose.hard_rotation_max_error < 1e-5
    assert reconnected.raw_pred_xstart.shape == (REALTIME_POSE_TARGET_DIM,)
    assert reconnected.deployed_pred_xstart.shape == (REALTIME_POSE_TARGET_DIM,)
    assert reconnected.current_tracker_raw.shape == (6, 13)
    assert reconnected.taid_prior_root_head is None
    assert reconnected.taid_prior_joints_head is None


def test_ema_checkpoint_returns_inner_model_and_runs_runtime(tmp_path) -> None:
    model_path = tmp_path / "model000000001.pt"
    ema_path = tmp_path / "ema000000001.pt"
    online_model = _RecordingModel()
    ema = EMA(online_model, include_online_model=False)
    with torch.no_grad():
        ema.ema_model.anchor.fill_(2.0)
    torch.save(online_model.state_dict(), model_path)
    torch.save(ema.state_dict(), ema_path)

    loaded_model, weight_source = load_checkpoint_model(
        _RecordingModel(),
        model_path,
        device=torch.device("cpu"),
        use_ema=True,
    )
    assert weight_source == "ema"
    assert isinstance(loaded_model, _RecordingModel)
    assert not isinstance(loaded_model, EMA)
    assert not loaded_model.training
    torch.testing.assert_close(loaded_model.anchor, torch.tensor(2.0))

    source = build_toy_realtime_source(frame_count=1)
    runtime = RealtimePoseRuntime(
        loaded_model,
        _OneStepProjectedDiffusion(),
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
    result = _step(runtime, source, 0, np.ones(6, dtype=bool))
    assert result.deployed_pred_xstart.shape == (REALTIME_POSE_TARGET_DIM,)
    assert loaded_model.valid_counts == [0]


def test_taid_single_and_batch_runtime_prepare_full_geometry_contract() -> None:
    source = build_toy_realtime_source(frame_count=2)
    model = RealtimePoseTargetDiT(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        motion_layers=1,
        dropout=0.0,
        taid_config=TaIDConfig(ablation="B6", innovation_dim=16),
    ).eval()
    diffusion = _OneStepProjectedDiffusion()
    first = RealtimePoseRuntime(
        model,
        diffusion,
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
    single = _step(first, source, 0, np.ones(6, dtype=bool))
    assert single.deployed_pred_xstart.shape == (REALTIME_POSE_TARGET_DIM,)
    assert single.taid_prior_root_head is not None
    assert single.taid_prior_root_head.shape == (4,)
    assert single.taid_prior_joints_head is not None
    assert single.taid_prior_joints_head.shape == (24, 3)

    second = RealtimePoseRuntime(
        model,
        diffusion,
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
    results = step_realtime_pose_batch(
        [first, second],
        np.stack([source["tracker_pos_world"][1], source["tracker_pos_world"][0]]),
        np.stack([source["tracker_rot_world_6d"][1], source["tracker_rot_world_6d"][0]]),
        np.ones((2, 6), dtype=bool),
        np.ones((2, 6), dtype=bool),
        np.asarray([source["root_pos_world"][1, 1], source["root_pos_world"][0, 1]]),
    )
    assert len(results) == 2
    assert all(result.deployed_pred_xstart.shape == (REALTIME_POSE_TARGET_DIM,) for result in results)
    np.testing.assert_allclose(
        results[1].taid_prior_root_head,
        single.taid_prior_root_head,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        results[1].taid_prior_joints_head,
        single.taid_prior_joints_head,
        atol=1e-6,
    )
