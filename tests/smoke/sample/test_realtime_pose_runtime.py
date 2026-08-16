from __future__ import annotations

import numpy as np
import pytest
import torch
from ema_pytorch import EMA

from data_loaders.generate_realtime_pose_tasks import compute_source_joint_rotations_world
from data_loaders.realtime_pose_geometry import (
    extract_rotation_heading_np,
)
from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_LENGTH
from model.realtime_pose_spatiotemporal_dit import RealtimePoseSpatioTemporalDiT
from sample.realtime_pose_runtime import (
    RealtimePoseRuntime,
    WorldPoseState,
    step_realtime_pose_batch,
)
from sample.utils import load_checkpoint_model
from tests.smoke.realtime_pose_fixtures import (
    IDENTITY_6D,
    build_toy_realtime_source,
)


class _RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.history_valid_counts: list[int] = []
        self.batch_sizes: list[int] = []
        self.prepare_grad_enabled: list[bool] = []
        self.head_paths: list[torch.Tensor] = []

    def forward(self, value, *_args, **_kwargs):
        return value + self.anchor

    def prepare_conditioning(
        self,
        history_pose_observation,
        head_path_window,
        history_region_confidence,
        window_valid_mask,
        frame_offsets,
    ):
        batch_size = int(history_pose_observation.shape[0])
        assert history_pose_observation.shape == (batch_size, 10, 144)
        assert head_path_window.shape == (batch_size, 11, 5)
        assert history_region_confidence.shape == (batch_size, 10, 5)
        assert window_valid_mask.shape == (batch_size, 11)
        assert frame_offsets.shape == (batch_size, 21)
        self.history_valid_counts.extend(
            int(value) for value in window_valid_mask[:, :-1].sum(dim=1).cpu().tolist()
        )
        self.batch_sizes.append(batch_size)
        self.prepare_grad_enabled.append(torch.is_grad_enabled())
        self.head_paths.append(head_path_window.detach().cpu().clone())
        return {"batch_size": batch_size}


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
        assert shape[1:] == (REALTIME_POSE_TARGET_LENGTH, REALTIME_POSE_TARGET_DIM)
        frame = torch.as_tensor(
            np.tile(IDENTITY_6D, 24), device=device, dtype=torch.float32
        )
        raw = frame.reshape(1, 1, REALTIME_POSE_TARGET_DIM).expand(*shape).clone()
        deployed = projection_fn(raw)
        return {
            "sample": deployed,
            "raw_pred_xstart": raw,
            "deployed_pred_xstart": deployed,
        }


def _step(runtime, source, frame, valid):
    return runtime.step(
        source["tracker_pos_world"][frame],
        source["tracker_rot_world_6d"][frame],
        np.ones(6, dtype=bool),
        valid,
        float(source["root_pos_world"][frame, 1]),
    )


def _runtime(model, diffusion, source, **kwargs) -> RealtimePoseRuntime:
    kwargs.setdefault("ik_direction_only_quality", 0.4)
    kwargs.setdefault("ik_residual_scale", 0.1)
    return RealtimePoseRuntime(
        model,
        diffusion,
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
        **kwargs,
    )


def test_runtime_uses_60_dense_frames_and_synchronized_anchors():
    source = build_toy_realtime_source(frame_count=65)
    model = _RecordingModel()
    runtime = _runtime(model, _OneStepProjectedDiffusion(), source)
    valid = np.ones(6, dtype=bool)
    results = [_step(runtime, source, frame, valid) for frame in range(61)]
    assert model.history_valid_counts[0] == 0
    assert model.history_valid_counts[1] == 1
    assert model.history_valid_counts[59] == 9
    assert model.history_valid_counts[60] == 10
    assert len(runtime.pose_history) == 60
    assert len(runtime.tracker_history) == 60
    assert not (results[0].inpaint_confidence[1:] > 0.0).any()
    assert not (results[1].inpaint_confidence[1:] > 0.0).any()
    current_head_path = model.head_paths[-1][0, -1]
    torch.testing.assert_close(current_head_path[:2], torch.zeros(2))
    torch.testing.assert_close(current_head_path[3:], torch.tensor([0.0, 1.0]))
    assert results[-1].raw_pred_pose_horizon.shape == (11, 144)
    assert results[-1].deployed_pred_pose_horizon.shape == (11, 144)


def test_runtime_can_seed_sixty_ground_truth_frames_without_sampling():
    source = build_toy_realtime_source(frame_count=61)
    rotations = compute_source_joint_rotations_world(source)
    model = _RecordingModel()
    runtime = _runtime(model, _OneStepProjectedDiffusion(), source)
    valid = np.ones(6, dtype=bool)
    for frame_index in range(60):
        runtime.append_ground_truth_frame(
            WorldPoseState(
                joint_rotations_world=rotations[frame_index],
                root_yaw_world=float(
                    extract_rotation_heading_np(rotations[frame_index, 0])
                ),
                hip_height=float(source["pelvis_height"][frame_index, 0]),
                root_position_world=source["root_pos_world"][frame_index],
            ),
            source["tracker_pos_world"][frame_index],
            source["tracker_rot_world_6d"][frame_index],
            valid,
            valid,
            float(source["root_pos_world"][frame_index, 1]),
        )

    assert model.batch_sizes == []
    assert len(runtime.pose_history) == 60
    np.testing.assert_allclose(
        runtime.pose_history[-1].joint_rotations_world,
        rotations[59],
        atol=1e-6,
    )
    _step(runtime, source, 60, valid)
    assert model.history_valid_counts == [10]
    assert len(runtime.pose_history) == 60


def test_runtime_dropout_and_reconnect_preserve_duration_semantics():
    source = build_toy_realtime_source(frame_count=18)
    runtime = _runtime(_RecordingModel(), _OneStepProjectedDiffusion(), source)
    valid = np.ones(6, dtype=bool)
    stable = [_step(runtime, source, frame, valid) for frame in range(16)]
    assert not stable[0].hard_rotation_state[1:].any()
    assert stable[14].hard_rotation_state.all()
    dropout = valid.copy()
    dropout[[1, 3]] = False
    dropped = _step(runtime, source, 16, dropout)
    assert not dropped.hard_rotation_state[[1, 3]].any()
    assert runtime.previous_d_off[[1, 3]].tolist() == [1, 1]
    reconnected = _step(runtime, source, 17, valid)
    assert not reconnected.hard_rotation_state[[1, 3]].any()
    assert runtime.previous_d_on[[1, 3]].tolist() == [1, 1]


def test_runtime_rejects_future_rolling_prior_in_first_round():
    source = build_toy_realtime_source(frame_count=1)
    with pytest.raises(ValueError, match="显式禁用"):
        _runtime(
            _RecordingModel(),
            _OneStepProjectedDiffusion(),
            source,
            use_future_rolling_prior=True,
        )


def test_runtime_uses_linear_tracker_warmup_for_current_joint_confidence():
    source = build_toy_realtime_source(frame_count=4)
    runtime = _runtime(
        _RecordingModel(),
        _OneStepProjectedDiffusion(),
        source,
        tracker_confidence_warmup=2,
    )
    valid = np.ones(6, dtype=bool)
    results = [_step(runtime, source, frame, valid) for frame in range(4)]

    np.testing.assert_array_equal(runtime.previous_d_on, np.full(6, 4))
    assert results[0].inpaint_confidence[0].max() == pytest.approx(0.5)
    assert results[1].inpaint_confidence[0].max() == pytest.approx(1.0)


def test_batched_runtime_samples_one_window_per_sequence():
    source = build_toy_realtime_source(frame_count=1)
    model = _RecordingModel()
    diffusion = _OneStepProjectedDiffusion()
    runtimes = [
        _runtime(model, diffusion, source)
        for _ in range(2)
    ]
    results = step_realtime_pose_batch(
        runtimes,
        np.repeat(source["tracker_pos_world"][:1], 2, axis=0),
        np.repeat(source["tracker_rot_world_6d"][:1], 2, axis=0),
        np.ones((2, 6), dtype=bool),
        np.ones((2, 6), dtype=bool),
        np.zeros(2, dtype=np.float32),
    )
    assert len(results) == 2
    assert model.batch_sizes == [2]
    assert all(result.deployed_pred_pose_horizon.shape == (11, 144) for result in results)


def test_batched_runtime_forwards_and_validates_known_noise():
    class NoiseRecordingDiffusion(_OneStepProjectedDiffusion):
        def __init__(self) -> None:
            self.known_noise: torch.Tensor | None = None

        def projected_ddim_sample_loop(self, *args, **kwargs):
            self.known_noise = kwargs["known_noise"].detach().cpu().clone()
            return super().projected_ddim_sample_loop(*args, **kwargs)

    source = build_toy_realtime_source(frame_count=1)
    model = _RecordingModel()
    diffusion = NoiseRecordingDiffusion()
    runtime = _runtime(model, diffusion, source)
    known_noise = torch.randn(1, REALTIME_POSE_TARGET_LENGTH, REALTIME_POSE_TARGET_DIM)
    runtime.step(
        source["tracker_pos_world"][0],
        source["tracker_rot_world_6d"][0],
        np.ones(6, dtype=bool),
        np.ones(6, dtype=bool),
        float(source["root_pos_world"][0, 1]),
        known_noise=known_noise,
    )
    torch.testing.assert_close(diffusion.known_noise, known_noise)

    with pytest.raises(ValueError, match="known_noise"):
        step_realtime_pose_batch(
            [runtime],
            source["tracker_pos_world"][:1],
            source["tracker_rot_world_6d"][:1],
            np.ones((1, 6), dtype=bool),
            np.ones((1, 6), dtype=bool),
            np.zeros(1, dtype=np.float32),
            known_noise=torch.zeros(1, 144),
        )


def test_runtime_pushes_only_horizon_zero_into_pose_history():
    class DistinctFutureDiffusion(_OneStepProjectedDiffusion):
        def projected_ddim_sample_loop(self, *args, **kwargs):
            result = super().projected_ddim_sample_loop(*args, **kwargs)
            result["raw_pred_xstart"][:, 1:, :6] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            )
            result["deployed_pred_xstart"] = kwargs["projection_fn"](
                result["raw_pred_xstart"]
            )
            result["sample"] = result["deployed_pred_xstart"]
            return result

    source = build_toy_realtime_source(frame_count=1)
    runtime = _runtime(_RecordingModel(), DistinctFutureDiffusion(), source)
    result = _step(runtime, source, 0, np.ones(6, dtype=bool))
    assert not np.allclose(
        result.deployed_pred_pose_horizon[0], result.deployed_pred_pose_horizon[1]
    )
    np.testing.assert_allclose(
        runtime.pose_history[-1].joint_rotations_world,
        result.resolved_pose.joint_rotations_world,
        atol=1e-6,
    )


def test_ema_checkpoint_returns_inner_model_and_runs_runtime(tmp_path):
    model_path = tmp_path / "model000000001.pt"
    ema_path = tmp_path / "ema000000001.pt"
    online_model = _RecordingModel()
    ema = EMA(online_model, include_online_model=False)
    with torch.no_grad():
        ema.ema_model.anchor.fill_(2.0)
    torch.save(online_model.state_dict(), model_path)
    torch.save(ema.state_dict(), ema_path)
    loaded_model, weight_source = load_checkpoint_model(
        _RecordingModel(), model_path, device=torch.device("cpu"), use_ema=True
    )
    assert weight_source == "ema"
    assert not loaded_model.training
    torch.testing.assert_close(loaded_model.anchor, torch.tensor(2.0))

    source = build_toy_realtime_source(frame_count=1)
    runtime = _runtime(loaded_model, _OneStepProjectedDiffusion(), source)
    result = _step(runtime, source, 0, np.ones(6, dtype=bool))
    assert result.deployed_pred_pose_horizon.shape == (11, 144)
    assert loaded_model.history_valid_counts == [0]


def test_legacy_single_frame_checkpoint_is_rejected_explicitly(tmp_path):
    model = RealtimePoseSpatioTemporalDiT(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        max_seq_len=21,
    )
    state = model.state_dict()
    state.pop("joint_diffusion_horizon_length")
    state["future_leg_head.weight"] = torch.zeros(1)
    path = tmp_path / "model000000001.pt"
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="单帧 checkpoint 与联合 11 帧模型不兼容"):
        load_checkpoint_model(model, path, device=torch.device("cpu"), use_ema=False)
