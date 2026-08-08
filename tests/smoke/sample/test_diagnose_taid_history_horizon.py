from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import sample.diagnose_taid_history_horizon as diagnostic
from data_loaders.realtime_pose_geometry import build_pose_target_np
from data_loaders.tracker_timeline import build_isolated_condition_timeline
from eval.evaluate_realtime_pose import evaluate_file
from sample.realtime_pose_runtime import RealtimePoseRuntime
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source
from tests.smoke.sample.test_realtime_pose_runtime import (
    _OneStepProjectedDiffusion,
    _RecordingModel,
)


class _HistoryRecordingModel(_RecordingModel):
    def __init__(self) -> None:
        super().__init__()
        self.pose_histories: list[np.ndarray] = []

    def prepare_conditioning(
        self,
        pose_history,
        tracker_history,
        current_tracker,
        trajectory_history,
        current_trajectory,
        valid_frame_mask,
    ):
        self.pose_histories.append(pose_history.detach().cpu().numpy().copy())
        return super().prepare_conditioning(
            pose_history,
            tracker_history,
            current_tracker,
            trajectory_history,
            current_trajectory,
            valid_frame_mask,
        )


class _NoiseRecordingDiffusion(_OneStepProjectedDiffusion):
    def __init__(self) -> None:
        self.noises: list[np.ndarray] = []

    def projected_ddim_sample_loop(self, *args, **kwargs):
        self.noises.append(kwargs["noise"].detach().cpu().numpy().copy())
        return super().projected_ddim_sample_loop(*args, **kwargs)


def _timeline(source: dict[str, np.ndarray], condition: str = "fixed_six"):
    frame_count = int(source[diagnostic.BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
    return build_isolated_condition_timeline(
        source_id="toy-sequence",
        frame_count=frame_count,
        condition=condition,
        global_seed=10,
    )


def test_diagnostic_cli_rejects_non_contract_history_and_invalid_horizons():
    args = diagnostic.build_arg_parser().parse_args(["--model_path", "model.pt"])
    assert args.history_length == 60
    assert args.horizons == [1, 4, 15, 30, 60]
    assert args.conditions == ["fixed_three", "fixed_six"]
    assert diagnostic.validate_diagnostic_options(60, [1, 4, 15, 30, 60], 2) == (
        1,
        4,
        15,
        30,
        60,
    )
    with pytest.raises(ValueError, match="只允许 60"):
        diagnostic.validate_diagnostic_options(59, [1], 1)
    with pytest.raises(ValueError, match="严格升序"):
        diagnostic.validate_diagnostic_options(60, [1, 15, 4], 1)
    with pytest.raises(ValueError, match="不能重复"):
        diagnostic.validate_diagnostic_options(60, [1, 1], 1)
    with pytest.raises(ValueError, match=r"\[1,60\]"):
        diagnostic.validate_diagnostic_options(60, [1, 61], 1)


def test_protocol_frame_metadata_excludes_warmup_and_incomplete_tail():
    teacher_mask, teacher_horizon, teacher_reset, blocks, tail = (
        diagnostic.build_protocol_frame_metadata(185, diagnostic.TEACHER_FORCED_PROTOCOL)
    )
    assert teacher_mask.sum() == 125
    assert not teacher_mask[:60].any()
    assert teacher_reset[60:].all()
    assert not teacher_horizon.any()
    assert (blocks, tail) == (0, 0)

    closed_mask, horizon, reset, blocks, tail = diagnostic.build_protocol_frame_metadata(
        185,
        diagnostic.CLOSED_LOOP_PROTOCOL,
    )
    assert (blocks, tail) == (2, 5)
    assert closed_mask[60:180].all()
    assert not closed_mask[:60].any()
    assert not closed_mask[180:].any()
    assert horizon[60] == 1
    assert horizon[63] == 4
    assert horizon[74] == 15
    assert horizon[89] == 30
    assert horizon[119] == 60
    assert horizon[120] == 1
    np.testing.assert_array_equal(np.flatnonzero(reset), [60, 120])


def test_gt_pose_history_override_is_exact_and_preserves_non_pose_runtime_state():
    source = build_toy_realtime_source(frame_count=61)
    model = _RecordingModel()
    runtime = RealtimePoseRuntime(
        model,
        _OneStepProjectedDiffusion(),
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
    timeline = _timeline(source)
    for frame in range(60):
        runtime.step(
            source["tracker_pos_world"][frame],
            source["tracker_rot_world_6d"][frame],
            timeline.configured[frame],
            timeline.measured_valid[frame],
            float(source["root_pos_world"][frame, 1]),
        )

    tracker_before = [
        (
            state.tracker_pos_world.copy(),
            state.tracker_rot_world_6d.copy(),
            state.configured.copy(),
            state.measured_valid.copy(),
            state.d_off.copy(),
            state.d_on.copy(),
            state.head_yaw_world,
            state.floor_y,
        )
        for state in runtime.tracker_history
    ]
    trajectory_before = [value.copy() for value in runtime.trajectory_history]
    d_off_before = runtime.previous_d_off.copy()
    d_on_before = runtime.previous_d_on.copy()
    head_yaw_before = runtime.previous_head_yaw
    head_position_before = runtime.previous_head_position.copy()
    states = diagnostic.build_ground_truth_world_pose_states(source)
    runtime.replace_pose_history_for_diagnostic(states[:60])

    for actual, expected in zip(runtime.tracker_history, tracker_before):
        np.testing.assert_array_equal(actual.tracker_pos_world, expected[0])
        np.testing.assert_array_equal(actual.tracker_rot_world_6d, expected[1])
        np.testing.assert_array_equal(actual.configured, expected[2])
        np.testing.assert_array_equal(actual.measured_valid, expected[3])
        np.testing.assert_array_equal(actual.d_off, expected[4])
        np.testing.assert_array_equal(actual.d_on, expected[5])
        assert actual.head_yaw_world == expected[6]
        assert actual.floor_y == expected[7]
    for actual, expected in zip(runtime.trajectory_history, trajectory_before):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(runtime.previous_d_off, d_off_before)
    np.testing.assert_array_equal(runtime.previous_d_on, d_on_before)
    assert runtime.previous_head_yaw == head_yaw_before
    np.testing.assert_array_equal(runtime.previous_head_position, head_position_before)

    prepared = runtime._prepare_batch_step(
        source["tracker_pos_world"][60],
        source["tracker_rot_world_6d"][60],
        timeline.configured[60],
        timeline.measured_valid[60],
        float(source["root_pos_world"][60, 1]),
    )
    expected_pose = build_pose_target_np(
        np.stack([state.joint_rotations_world for state in states[:60]]),
        runtime.tracker_history[-1].head_yaw_world,
    )
    np.testing.assert_allclose(
        prepared.conditioning["pose_history"].cpu().numpy()[0],
        expected_pose,
        atol=1e-6,
    )
    # Runtime 保存深拷贝，调用方后续改动不能污染诊断 history。
    states[0].joint_rotations_world[...] = 0.0
    assert not np.allclose(runtime.pose_history[0].joint_rotations_world, 0.0)


def test_teacher_forced_replaces_prediction_each_frame_and_protocols_share_noise():
    source = build_toy_realtime_source(frame_count=122)
    # 让 GT pelvis 明确偏离 fake diffusion 的 identity 输出。
    pelvis_yaw = 0.7
    source[diagnostic.BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY][:, :6] = np.asarray(
        [np.sin(pelvis_yaw), 0.0, np.cos(pelvis_yaw), 0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    timeline = _timeline(source)
    teacher_model = _HistoryRecordingModel()
    diffusion = _NoiseRecordingDiffusion()
    teacher = diagnostic.rollout_history_protocol_sources(
        teacher_model,
        diffusion,
        [source],
        [timeline],
        diagnostic.TEACHER_FORCED_PROTOCOL,
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[1234],
    )[0]
    teacher_noises = [value.copy() for value in diffusion.noises]

    closed_model = _HistoryRecordingModel()
    diffusion.noises.clear()
    closed = diagnostic.rollout_history_protocol_sources(
        closed_model,
        diffusion,
        [source],
        [timeline],
        diagnostic.CLOSED_LOOP_PROTOCOL,
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[1234],
    )[0]
    assert len(teacher_noises) == len(diffusion.noises) == 122
    for teacher_noise, closed_noise in zip(teacher_noises, diffusion.noises):
        np.testing.assert_array_equal(teacher_noise, closed_noise)

    gt_states = diagnostic.build_ground_truth_world_pose_states(source)
    expected_t61 = build_pose_target_np(
        np.stack([state.joint_rotations_world for state in gt_states[1:61]]),
        diagnostic.extract_rotation_heading_np(
            diagnostic.rotation_6d_to_matrix_np(source["tracker_rot_world_6d"][60, 0])
        ),
    )
    # prepare 调用索引与绝对帧一致；t=61 的 teacher history 必须全部来自 GT。
    np.testing.assert_allclose(
        teacher_model.pose_histories[61][0],
        expected_t61,
        atol=1e-5,
    )
    assert not np.allclose(
        closed_model.pose_histories[61][0, -1],
        expected_t61[-1],
    )
    pair = diagnostic.compare_teacher_forced_h1(teacher, closed)
    assert pair["samples"] == 1
    assert pair["matches"]


def test_diagnostic_batch_matches_single_sequence():
    sources = [build_toy_realtime_source(frame_count=121) for _ in range(2)]
    timelines = [_timeline(source) for source in sources]
    batch = diagnostic.rollout_history_protocol_sources(
        _RecordingModel(),
        _OneStepProjectedDiffusion(),
        sources,
        timelines,
        diagnostic.CLOSED_LOOP_PROTOCOL,
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[11, 22],
    )
    single = diagnostic.rollout_history_protocol_sources(
        _RecordingModel(),
        _OneStepProjectedDiffusion(),
        [sources[0]],
        [timelines[0]],
        diagnostic.CLOSED_LOOP_PROTOCOL,
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[11],
    )[0]
    for name in (
        "deployed_pred_target_raw",
        "predicted_joints_world",
        "predicted_root_position_world",
        "diagnostic_horizon_frame",
        "eval_frame_mask",
    ):
        np.testing.assert_allclose(batch[0][name], single[name], atol=1e-6)


def test_eval_mask_override_only_narrows_saved_mask(tmp_path):
    source = build_toy_realtime_source(frame_count=121)
    payload = diagnostic.rollout_history_protocol_sources(
        _RecordingModel(),
        _OneStepProjectedDiffusion(),
        [source],
        [_timeline(source)],
        diagnostic.CLOSED_LOOP_PROTOCOL,
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[10],
    )[0]
    path = tmp_path / "diagnostic.npz"
    np.savez(path, **payload)
    horizon_4 = evaluate_file(
        path,
        eval_frame_mask_override=payload["diagnostic_horizon_frame"] == 4,
    )
    assert horizon_4["samples"] == 1
    with pytest.raises(ValueError, match=r"必须为 \[T\] 或 \[N,T\]"):
        evaluate_file(path, eval_frame_mask_override=np.ones((2, 2), dtype=bool))


def test_entry_diagnostic_writes_protocols_curve_endpoints_and_b0_prior_null(tmp_path):
    eval_set_dir = tmp_path / "eval_set"
    eval_set_dir.mkdir()
    source_path = eval_set_dir / "source.npz"
    np.savez(source_path, **build_toy_realtime_source(frame_count=121))
    entry = {
        "sequence_id": "toy-sequence",
        "source_path": source_path.name,
        "source_relative_path": source_path.name,
    }
    (eval_set_dir / "manifest.jsonl").write_text(
        json.dumps(entry) + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model000005000.pt"
    checkpoint.write_bytes(b"smoke-checkpoint-identity")

    summary = diagnostic.evaluate_history_diagnostic_entries(
        entries=[entry],
        eval_set_dir=eval_set_dir,
        output_dir=tmp_path / "output",
        model=_RecordingModel(),
        diffusion=_OneStepProjectedDiffusion(),
        device=torch.device("cpu"),
        normalizer=None,
        model_path=checkpoint,
        weights="model",
        limit=1,
        sequence_batch_size=2,
        conditions=["fixed_three", "fixed_six"],
        horizons=[1, 4, 15, 30, 60],
        show_progress=False,
    )

    assert Path(summary["summary_path"]).exists()
    curve_path = Path(summary["metadata"]["curve_path"])
    assert curve_path.exists()
    assert len(summary["files"]) == 4
    assert set(summary["summary"]["teacher_forced"]["by_condition"]) == {
        "fixed_three",
        "fixed_six",
    }
    assert set(summary["summary"]["closed_loop_endpoints"]["fixed_six"]) == {
        "1",
        "4",
        "15",
        "30",
        "60",
    }
    assert summary["summary"]["teacher_vs_closed_h1"]["overall"]["matches"]
    assert all(result["taid_prior_available_ratio"] == 0.0 for result in summary["files"])
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    assert set(curve["curve"]["fixed_three"]) == {str(value) for value in range(1, 61)}
    assert curve["curve"]["fixed_six"]["60"]["samples"] == 1
    for protocol in diagnostic.DIAGNOSTIC_PROTOCOLS:
        result_path = (
            tmp_path / "output" / protocol / "f3" / "toy-sequence" / "diagnostic_result.npz"
        )
        assert result_path.exists()
        with np.load(result_path, allow_pickle=False) as data:
            assert "diagnostic_horizon_frame" in data.files
            assert "gt_pose_history_reset" in data.files
            assert np.isnan(data["taid_prior_root_head"]).all()


def test_decision_gate_selects_exposure_bias_branch_and_prior_audit():
    teacher = {
        "by_condition": {
            "fixed_six": {
                "mpjpe_cm": 6.0,
                "deployed_hard_tracker_rotation_deg": 1e-6,
            },
            "fixed_three": {
                "root_yaw_diagnostics": {
                    "error_over_150_ratio": 0.1,
                    "pi_majority_sequence_count": 0,
                }
            },
        }
    }
    endpoints = {
        "fixed_six": {
            "1": {"mpjpe_cm": 6.0},
            "15": {"mpjpe_cm": 12.0},
            "30": {"mpjpe_cm": 15.0},
            "60": {"mpjpe_cm": 18.0},
        }
    }
    decision = diagnostic.decide_next_branch(teacher, endpoints, {"matches": True})
    assert decision["eligible_for_15_frame_experiment"]
    assert decision["branch"] == "plan_15_frame_rollout_experiment"

    teacher["by_condition"]["fixed_six"]["mpjpe_cm"] = 12.0
    decision = diagnostic.decide_next_branch(teacher, endpoints, {"matches": True})
    assert not decision["eligible_for_15_frame_experiment"]
    assert decision["branch"] == "audit_prior_supervision_capacity"
