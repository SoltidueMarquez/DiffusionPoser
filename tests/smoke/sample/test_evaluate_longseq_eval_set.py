from __future__ import annotations

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
    summary = longseq.summarize_latency(np.asarray([100.0, 10.0, 20.0]), warmup_frames=1)
    assert summary["frames"] == 2
    assert summary["mean_ms"] == 15.0
    assert summary["p95_ms"] == 19.5


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
    np.testing.assert_array_equal(payload["history_length"], np.arange(5, dtype=np.int64)[None])
    result_path = tmp_path / "longseq.npz"
    np.savez(result_path, **payload)
    result = evaluate_file(result_path)
    assert result["samples"] == 5
    assert result["by_history_phase"]["cold_start_0_59"]["samples"] == 5
    assert result["by_history_phase"]["steady_state_60_plus"]["samples"] == 0
