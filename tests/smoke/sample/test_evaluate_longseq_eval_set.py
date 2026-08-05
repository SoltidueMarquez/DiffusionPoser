from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import sample.evaluate_longseq_eval_set as longseq
from data_loaders.sensor_masking import (
    SCENARIO_FIXED_SIX,
    SCENARIO_THREE_TO_SIX,
    SCENARIO_TWO_POINT_DROPOUT_RECONNECT,
)
from data_loaders.tracker_timeline import TrackerTimeline
from eval.evaluate_realtime_pose import evaluate_file
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source
from tests.smoke.sample.test_realtime_pose_runtime import (
    _OneStepProjectedDiffusion,
    _RecordingModel,
)


def _timeline(configured: np.ndarray, measured: np.ndarray) -> TrackerTimeline:
    shape = configured.shape
    return TrackerTimeline(
        configured=configured,
        measured_valid=measured,
        d_off=np.zeros(shape, dtype=np.uint8),
        d_on=np.zeros(shape, dtype=np.uint8),
        hard_rotation_state=np.zeros(shape, dtype=bool),
    )


def test_longseq_defaults_to_five_steps_and_latency_summary_excludes_warmup():
    args = longseq.build_arg_parser().parse_args(["--model_path", "model.pt"])
    assert args.inference_steps == 5
    assert args.sequence_batch_size == 4
    summary = longseq.summarize_latency(np.asarray([100.0, 10.0, 20.0]), warmup_frames=1)
    assert summary["frames"] == 2
    assert summary["mean_ms"] == 15.0
    assert summary["p95_ms"] == 19.5


def test_longseq_default_output_path_is_short_stable_and_collision_resistant(tmp_path):
    eval_set = tmp_path / "very_long_eval_set_directory_name"
    checkpoint = tmp_path / "very_long_training_run_directory_name" / "model000120000.pt"
    output = longseq.build_default_output_dir(
        eval_set_dir=eval_set,
        model_path=checkpoint,
        weights="ema",
        projected_ddim_mode="all_steps",
        sequence_batch_size=4,
    )

    assert output.parent == Path("output") / "l"
    assert output.name.startswith("120000e-a-b4-")
    assert len(output.as_posix()) <= 40
    assert output == longseq.build_default_output_dir(
        eval_set_dir=eval_set,
        model_path=checkpoint,
        weights="ema",
        projected_ddim_mode="all_steps",
        sequence_batch_size=4,
    )
    assert output != longseq.build_default_output_dir(
        eval_set_dir=tmp_path / "another_eval_set",
        model_path=checkpoint,
        weights="ema",
        projected_ddim_mode="all_steps",
        sequence_batch_size=4,
    )


def test_longseq_classification_distinguishes_configuration_addition_from_reconnect():
    configured = np.ones((21, 6), dtype=bool)
    configured[:5, 3:] = False
    transition_timeline = _timeline(configured, configured.copy())
    assert all(
        longseq._classify_timeline_frame(transition_timeline, index) == SCENARIO_THREE_TO_SIX
        for index in range(5, 20)
    )
    assert longseq._classify_timeline_frame(transition_timeline, 20) == SCENARIO_FIXED_SIX

    configured = np.ones((21, 6), dtype=bool)
    measured = configured.copy()
    measured[2:5, [1, 4]] = False
    reconnect_timeline = _timeline(configured, measured)
    assert all(
        longseq._classify_timeline_frame(reconnect_timeline, index)
        == SCENARIO_TWO_POINT_DROPOUT_RECONNECT
        for index in range(2, 20)
    )
    assert longseq._classify_timeline_frame(reconnect_timeline, 20) == SCENARIO_FIXED_SIX


def test_longseq_forwards_projected_ddim_ablation_to_runtime():
    class RecordingDiffusion(_OneStepProjectedDiffusion):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def projected_ddim_sample_loop(self, *args, **kwargs):
            self.calls.append(dict(kwargs))
            return super().projected_ddim_sample_loop(*args, **kwargs)

    source = build_toy_realtime_source(frame_count=1)
    configured = np.ones((1, 6), dtype=bool)
    diffusion = RecordingDiffusion()
    longseq.rollout_long_sequence_source(
        _RecordingModel(),
        diffusion,
        source,
        _timeline(configured, configured.copy()),
        torch.device("cpu"),
        normalizer=None,
        projected_ddim_mode="late_steps",
        projected_ddim_late_steps=2,
    )
    assert diffusion.calls[-1]["projection_mode"] == "late_steps"
    assert diffusion.calls[-1]["late_steps"] == 2


def test_longseq_runtime_cold_starts_and_emits_new_eval_contract(tmp_path):
    source = build_toy_realtime_source(frame_count=5)
    # 给 pelvis local rotation 加固定 yaw，使 actor heading 与 pelvis forward 明确不同。
    pelvis_yaw_offset = 0.7
    source[longseq.BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY][:, :6] = np.asarray(
        [np.sin(pelvis_yaw_offset), 0.0, np.cos(pelvis_yaw_offset), 0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    configured = np.ones((5, 6), dtype=bool)
    measured = configured.copy()
    timeline = TrackerTimeline(
        configured=configured,
        measured_valid=measured,
        d_off=np.zeros((5, 6), dtype=np.uint8),
        d_on=np.tile(np.arange(1, 6, dtype=np.uint8)[:, None], (1, 6)),
        hard_rotation_state=np.tile(
            np.asarray([True, False, False, False, False, False]), (5, 1)
        ),
    )
    payload = longseq.rollout_long_sequence_source(
        _RecordingModel(),
        _OneStepProjectedDiffusion(),
        source,
        timeline,
        torch.device("cpu"),
        normalizer=None,
    )
    assert payload["reference_target_raw"].shape == (1, 5, 144)
    assert payload["current_tracker_raw"].shape == (1, 5, 6, 13)
    assert payload["raw_pred_target_raw"].shape == (1, 5, 144)
    assert payload["deployed_pred_target_raw"].shape == (1, 5, 144)
    assert "current_trajectory" not in payload
    reference_pelvis_yaw = longseq.extract_rotation_heading_np(
        longseq.compute_source_joint_rotations_world(source)[:, 0]
    )
    np.testing.assert_allclose(
        payload["reference_root_yaw_world"][0],
        reference_pelvis_yaw,
        atol=1e-6,
    )
    assert not np.allclose(payload["reference_root_yaw_world"][0], source["root_yaw"])
    np.testing.assert_array_equal(payload["history_length"], np.arange(5, dtype=np.int64)[None])
    result_path = tmp_path / "longseq.npz"
    np.savez(result_path, **payload)
    result = evaluate_file(result_path)
    assert result["samples"] == 5
    assert result["by_history_phase"]["cold_start_0_59"]["samples"] == 5
    assert result["by_history_phase"]["steady_state_60_plus"]["samples"] == 0


def test_longseq_cross_sequence_batch_supports_different_lengths_and_matches_single():
    sources = [
        build_toy_realtime_source(frame_count=4),
        build_toy_realtime_source(frame_count=2),
    ]
    timelines = []
    for frame_count in (4, 2):
        configured = np.ones((frame_count, 6), dtype=bool)
        timelines.append(_timeline(configured, configured.copy()))

    batch_model = _RecordingModel()
    payloads = longseq.rollout_long_sequence_sources(
        batch_model,
        _OneStepProjectedDiffusion(),
        sources,
        timelines,
        torch.device("cpu"),
        normalizer=None,
    )

    assert batch_model.batch_sizes == [2, 2, 1, 1]
    assert payloads[0]["deployed_pred_target_raw"].shape == (1, 4, 144)
    assert payloads[1]["deployed_pred_target_raw"].shape == (1, 2, 144)
    assert "current_trajectory" not in payloads[0]
    assert "current_trajectory" not in payloads[1]
    np.testing.assert_array_equal(payloads[0]["history_length"], [[0, 1, 2, 3]])
    np.testing.assert_array_equal(payloads[1]["history_length"], [[0, 1]])

    single_payload = longseq.rollout_long_sequence_source(
        _RecordingModel(),
        _OneStepProjectedDiffusion(),
        sources[0],
        timelines[0],
        torch.device("cpu"),
        normalizer=None,
    )
    np.testing.assert_allclose(
        payloads[0]["deployed_pred_target_raw"],
        single_payload["deployed_pred_target_raw"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        payloads[0]["predicted_joints_world"],
        single_payload["predicted_joints_world"],
        atol=1e-6,
    )


def test_isolated_conditions_share_framewise_diffusion_noise_for_fair_comparison():
    class NoiseRecordingDiffusion(_OneStepProjectedDiffusion):
        def __init__(self) -> None:
            self.noises: list[np.ndarray] = []

        def projected_ddim_sample_loop(self, *args, **kwargs):
            self.noises.append(kwargs["noise"].detach().cpu().numpy().copy())
            return super().projected_ddim_sample_loop(*args, **kwargs)

    source = build_toy_realtime_source(frame_count=3)
    timelines = [
        longseq.build_isolated_condition_timeline("same-source", 3, condition)
        for condition in ("fixed_six", "fixed_three")
    ]
    diffusion = NoiseRecordingDiffusion()
    longseq.rollout_long_sequence_sources(
        _RecordingModel(),
        diffusion,
        [source, source],
        timelines,
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[1234, 1234],
    )

    assert len(diffusion.noises) == 3
    for noise in diffusion.noises:
        np.testing.assert_array_equal(noise[0], noise[1])


def test_longseq_entry_evaluation_uses_sequence_batch_and_writes_each_result(tmp_path):
    eval_set_dir = tmp_path / "eval_set"
    eval_set_dir.mkdir()
    entries = []
    for index, frame_count in enumerate((4, 2)):
        source_path = eval_set_dir / f"source_{index}.npz"
        np.savez(source_path, **build_toy_realtime_source(frame_count=frame_count))
        entries.append(
            {
                "sequence_id": f"sequence_{index}",
                "source_path": source_path.name,
                "source_relative_path": source_path.name,
            }
        )

    model = _RecordingModel()
    summary = longseq.evaluate_longseq_entries(
        entries=entries,
        eval_set_dir=eval_set_dir,
        output_dir=tmp_path / "output",
        model=model,
        diffusion=_OneStepProjectedDiffusion(),
        device=torch.device("cpu"),
        normalizer=None,
        sequence_batch_size=2,
        conditions=["fixed_six", "fixed_three"],
        show_progress=False,
    )

    assert model.batch_sizes == [2, 2, 1, 1, 2, 2, 1, 1]
    assert summary["metadata"]["sequence_batch_size"] == 2
    assert summary["metadata"]["evaluation_protocol"] == "isolated_condition_cold_start"
    assert summary["metadata"]["conditions"] == ["fixed_six", "fixed_three"]
    assert summary["metadata"]["latency_scope"] == "active_batch_wall_time_per_stream"
    assert len(summary["files"]) == 4
    assert set(summary["summary"]["by_condition"]) == {"fixed_six", "fixed_three"}
    assert all(
        (
            tmp_path
            / "output"
            / condition_tag
            / f"sequence_{index}"
            / "rollout_result.npz"
        ).exists()
        for condition_tag in ("f6", "f3")
        for index in range(2)
    )
