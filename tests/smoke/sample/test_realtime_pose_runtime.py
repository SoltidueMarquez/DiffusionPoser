from __future__ import annotations

import numpy as np
import pytest
import torch
from ema_pytorch import EMA

from data_loaders.realtime_pose_config import TrackerReliabilityConfig
from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_LENGTH
from sample.realtime_pose_runtime import RealtimePoseRuntime, step_realtime_pose_batch
from sample.utils import load_checkpoint_model
from model.realtime_pose_spatiotemporal_dit import RealtimePoseSpatioTemporalDiT
from tests.smoke.realtime_pose_fixtures import IDENTITY_6D, build_toy_realtime_source


class _RecordingModel(torch.nn.Module):
    def __init__(
        self,
        reliability_config: TrackerReliabilityConfig | None = None,
    ) -> None:
        super().__init__()
        self.reliability_config = (
            reliability_config or TrackerReliabilityConfig()
        ).validate()
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
        tracker_window,
        head_path_window,
        history_region_confidence,
        window_valid_mask,
        frame_offsets,
    ):
        batch_size = int(history_pose_observation.shape[0])
        assert history_pose_observation.shape == (batch_size, 10, 144)
        assert tracker_window.shape == (batch_size, 11, 6, 13)
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


def test_runtime_uses_60_dense_frames_and_synchronized_anchors():
    source = build_toy_realtime_source(frame_count=65)
    model = _RecordingModel()
    runtime = RealtimePoseRuntime(
        model,
        _OneStepProjectedDiffusion(),
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
    valid = np.ones(6, dtype=bool)
    results = [_step(runtime, source, frame, valid) for frame in range(61)]
    assert model.history_valid_counts[0] == 0
    assert model.history_valid_counts[1] == 1
    assert model.history_valid_counts[59] == 9
    assert model.history_valid_counts[60] == 10
    assert len(runtime.pose_history) == 60
    assert len(runtime.tracker_history) == 60
    current_head_path = model.head_paths[-1][0, -1]
    torch.testing.assert_close(current_head_path[:2], torch.zeros(2))
    torch.testing.assert_close(current_head_path[3:], torch.tensor([0.0, 1.0]))
    assert results[-1].raw_pred_pose_horizon.shape == (11, 144)
    assert results[-1].deployed_pred_pose_horizon.shape == (11, 144)


def test_runtime_dropout_and_reconnect_preserve_duration_semantics():
    source = build_toy_realtime_source(frame_count=18)
    runtime = RealtimePoseRuntime(
        _RecordingModel(),
        _OneStepProjectedDiffusion(),
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
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
    assert reconnected.resolved_pose.hard_rotation_max_error < 1e-5


def test_runtime_uses_the_model_reliability_config():
    source = build_toy_realtime_source(frame_count=4)
    model_config = TrackerReliabilityConfig(
        d_warm_pos=2,
        d_warm_rot=3,
        d_hard=2,
        duration_cap=2,
    )
    runtime = RealtimePoseRuntime(
        _RecordingModel(model_config),
        _OneStepProjectedDiffusion(),
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
    valid = np.ones(6, dtype=bool)
    results = [_step(runtime, source, frame, valid) for frame in range(4)]

    assert runtime.reliability_config is model_config
    np.testing.assert_array_equal(runtime.previous_d_on, np.full(6, 2))
    np.testing.assert_allclose(results[0].kappa_position, np.full(6, 0.5))
    np.testing.assert_allclose(results[1].kappa_position, np.ones(6))
    assert results[1].hard_rotation_state.all()


def test_batched_runtime_samples_one_window_per_sequence():
    source = build_toy_realtime_source(frame_count=1)
    model = _RecordingModel()
    diffusion = _OneStepProjectedDiffusion()
    runtimes = [
        RealtimePoseRuntime(
            model,
            diffusion,
            torch.device("cpu"),
            source["joint_offsets_parent"],
            source["joint_rest_local_rotations_6d"],
        )
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
    runtime = RealtimePoseRuntime(
        _RecordingModel(),
        DistinctFutureDiffusion(),
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
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
    runtime = RealtimePoseRuntime(
        loaded_model,
        _OneStepProjectedDiffusion(),
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
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
