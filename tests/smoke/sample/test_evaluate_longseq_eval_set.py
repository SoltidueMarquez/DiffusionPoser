from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import sample.evaluate_longseq_eval_set as longseq
from data_loaders.sensor_masking import (
    SCENARIO_FIXED_SIX,
    SCENARIO_THREE_TO_SIX,
    SCENARIO_TWO_POINT_DROPOUT_RECONNECT,
)
from data_loaders.tracker_timeline import TrackerTimeline
from eval.evaluate_realtime_pose import evaluate_file
from tests.smoke.realtime_pose_fixtures import (
    build_toy_realtime_source,
)
from tests.smoke.sample.test_realtime_pose_runtime import (
    _OneStepProjectedDiffusion,
    _RecordingModel,
)


@pytest.fixture(autouse=True)
def _inject_calibrated_ik_parameters(monkeypatch):
    """长序列测试显式使用固定测试参数，不把它们冒充生产校准默认值。"""

    original_init = longseq.RealtimePoseRuntime.__init__

    def calibrated_init(self, *args, **kwargs):
        if kwargs.get("ik_direction_only_quality") is None:
            kwargs["ik_direction_only_quality"] = 0.4
        if kwargs.get("ik_residual_scale") is None:
            kwargs["ik_residual_scale"] = 0.1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(longseq.RealtimePoseRuntime, "__init__", calibrated_init)


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
    assert args.gt_history_warmup_frames == 0
    assert args.use_future_rolling_prior is False
    assert args.future_confidence_decay == pytest.approx(0.9)
    summary = longseq.summarize_latency(np.asarray([100.0, 10.0, 20.0]), warmup_frames=1)
    assert summary["frames"] == 2
    assert summary["mean_ms"] == 15.0
    assert summary["p95_ms"] == 19.5
    with pytest.raises(SystemExit):
        longseq.build_arg_parser().parse_args(
            ["--model_path", "model.pt", "--ik_reliability_lut_path", "removed.npz"]
        )


def test_longseq_default_output_path_is_short_stable_and_collision_resistant(tmp_path):
    eval_set = tmp_path / "very_long_eval_set_directory_name"
    checkpoint = tmp_path / "very_long_training_run_directory_name" / "model000120000.pt"
    output = longseq.build_default_output_dir(
        source_dir=eval_set,
        model_path=checkpoint,
        weights="ema",
        sequence_batch_size=4,
    )

    assert output.parent == Path("output") / "l"
    assert output.name.startswith("120000e-b4-")
    assert len(output.as_posix()) <= 40
    assert output == longseq.build_default_output_dir(
        source_dir=eval_set,
        model_path=checkpoint,
        weights="ema",
        sequence_batch_size=4,
    )
    assert output != longseq.build_default_output_dir(
        source_dir=tmp_path / "another_eval_set",
        model_path=checkpoint,
        weights="ema",
        sequence_batch_size=4,
    )
    rolling_output = longseq.build_default_output_dir(
        source_dir=eval_set,
        model_path=checkpoint,
        weights="ema",
        sequence_batch_size=4,
        use_future_rolling_prior=True,
        future_confidence_decay=0.9,
    )
    assert rolling_output != output
    assert "rp0" in output.name
    assert "rp1g0p9" in rolling_output.name


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


def test_longseq_has_no_projected_ddim_ablation_arguments():
    class RecordingDiffusion(_OneStepProjectedDiffusion):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def projected_ddim_sample_loop(self, *args, **kwargs):
            self.calls.append(dict(kwargs))
            return super().projected_ddim_sample_loop(*args, **kwargs)

    source = build_toy_realtime_source(frame_count=1)
    source["stationary_prob_5"][0, 1:3] = 1.0
    source["joints_world"][0, 10:12, 1] = [0.20, 0.025]
    configured = np.ones((1, 6), dtype=bool)
    diffusion = RecordingDiffusion()
    payload = longseq.rollout_long_sequence_source(
        _RecordingModel(),
        diffusion,
        source,
        _timeline(configured, configured.copy()),
        torch.device("cpu"),
        normalizer=None,
    )
    assert "projection_mode" not in diffusion.calls[-1]
    assert "late_steps" not in diffusion.calls[-1]
    np.testing.assert_allclose(payload["contact_target"][0, 0], [0.0, 1.0])


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
    assert payload["reference_pose_horizon_raw"].shape == (1, 5, 11, 144)
    assert payload["raw_pred_pose_horizon_raw"].shape == (1, 5, 11, 144)
    assert payload["deployed_pred_pose_horizon_raw"].shape == (1, 5, 11, 144)
    assert payload["pose_horizon_valid_mask"].shape == (1, 5, 11)
    assert payload["inpaint_confidence"].shape == (1, 5, 11, 24)
    assert "tracker_combination" not in payload
    assert "ik_joint_reliability" not in payload
    assert "joint_online_confidence" not in payload
    np.testing.assert_array_equal(
        payload["pose_horizon_valid_mask"].sum(axis=-1), [[5, 4, 3, 2, 1]]
    )
    assert np.isnan(payload["reference_pose_horizon_raw"][0, -1, 1:]).all()
    assert "current_trajectory" not in payload
    reference_pelvis_yaw = longseq._continuous_source_root_yaws(
        source,
        longseq.compute_source_joint_rotations_world(source),
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
    assert payloads[0]["deployed_pred_pose_horizon_raw"].shape == (1, 4, 11, 144)
    assert payloads[1]["deployed_pred_pose_horizon_raw"].shape == (1, 2, 11, 144)
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


def test_ground_truth_history_is_the_only_previous_pose_prior(monkeypatch):
    instances = []

    class PriorRecordingRuntime(longseq.RealtimePoseRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prior_inputs = []
            instances.append(self)

        def _previous_pose_for_ik(self, current_head_yaw):
            values = super()._previous_pose_for_ik(current_head_yaw)
            self.prior_inputs.append(
                (float(current_head_yaw), values[0].copy(), values[1])
            )
            return values

    monkeypatch.setattr(longseq, "RealtimePoseRuntime", PriorRecordingRuntime)
    source = build_toy_realtime_source(frame_count=2)
    configured = np.ones((2, 6), dtype=bool)
    timeline = TrackerTimeline(
        configured=configured,
        measured_valid=configured.copy(),
        d_off=np.zeros((2, 6), dtype=np.uint8),
        d_on=np.ones((2, 6), dtype=np.uint8),
        hard_rotation_state=np.zeros((2, 6), dtype=bool),
    )
    payload = longseq.rollout_long_sequence_sources(
        _RecordingModel(),
        _OneStepProjectedDiffusion(),
        [source],
        [timeline],
        torch.device("cpu"),
        normalizer=None,
        tracker_confidence_warmup=1,
        pose_history_mode="ground_truth",
    )[0]

    runtime = instances[0]
    second_prior = runtime.prior_inputs[1]
    expected_previous = longseq.build_pose_target_np(
        longseq.compute_source_joint_rotations_world(source)[0:1], second_prior[0]
    )[0]
    assert second_prior[2]
    np.testing.assert_allclose(second_prior[1], expected_previous, atol=1e-6)
    assert not hasattr(runtime, "previous_deployed_horizon_world")
    assert not (payload["inpaint_confidence"][0, 1, 1:] > 0.0).any()


def test_longseq_gt_history_warmup_starts_prediction_after_sixty_frames():
    source = build_toy_realtime_source(frame_count=65)
    configured = np.ones((65, 6), dtype=bool)
    model = _RecordingModel()
    payload = longseq.rollout_long_sequence_sources(
        model,
        _OneStepProjectedDiffusion(),
        [source],
        [_timeline(configured, configured.copy())],
        torch.device("cpu"),
        normalizer=None,
        gt_history_warmup_frames=60,
    )[0]

    assert model.batch_sizes == [1] * 5
    assert model.history_valid_counts == [10] * 5
    assert payload["deployed_pred_target_raw"].shape == (1, 5, 144)
    np.testing.assert_array_equal(payload["absolute_frame_index"], np.arange(60, 65))
    np.testing.assert_array_equal(payload["history_length"], np.full((1, 5), 60))
    np.testing.assert_array_equal(payload["eval_frame_mask"], np.ones((1, 5), dtype=bool))


def test_isolated_conditions_share_framewise_diffusion_noise_for_fair_comparison():
    class NoiseRecordingDiffusion(_OneStepProjectedDiffusion):
        def __init__(self) -> None:
            self.noises: list[np.ndarray] = []
            self.known_noises: list[np.ndarray] = []

        def projected_ddim_sample_loop(self, *args, **kwargs):
            self.noises.append(kwargs["noise"].detach().cpu().numpy().copy())
            self.known_noises.append(
                kwargs["known_noise"].detach().cpu().numpy().copy()
            )
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
    assert len(diffusion.known_noises) == 3
    for noise, known_noise in zip(diffusion.noises, diffusion.known_noises):
        assert noise.shape == (2, 11, 144)
        assert known_noise.shape == (2, 11, 144)
        np.testing.assert_array_equal(noise[0], noise[1])
        np.testing.assert_array_equal(known_noise[0], known_noise[1])


def test_sequence_noise_is_independent_of_batch_size_and_lane_order():
    class NoiseRecordingDiffusion(_OneStepProjectedDiffusion):
        def __init__(self) -> None:
            self.noises: list[torch.Tensor] = []
            self.known_noises: list[torch.Tensor] = []

        def projected_ddim_sample_loop(self, *args, **kwargs):
            self.noises.append(kwargs["noise"].detach().cpu().clone())
            self.known_noises.append(kwargs["known_noise"].detach().cpu().clone())
            return super().projected_ddim_sample_loop(*args, **kwargs)

    sources = [
        build_toy_realtime_source(frame_count=3),
        build_toy_realtime_source(frame_count=3),
    ]
    timelines = [
        longseq.build_isolated_condition_timeline("source-a", 3, "fixed_six"),
        longseq.build_isolated_condition_timeline("source-b", 3, "fixed_three"),
    ]

    batched = NoiseRecordingDiffusion()
    longseq.rollout_long_sequence_sources(
        _RecordingModel(),
        batched,
        sources,
        timelines,
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[31, 47],
    )
    reversed_batch = NoiseRecordingDiffusion()
    longseq.rollout_long_sequence_sources(
        _RecordingModel(),
        reversed_batch,
        sources[::-1],
        timelines[::-1],
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[47, 31],
    )
    single = NoiseRecordingDiffusion()
    longseq.rollout_long_sequence_sources(
        _RecordingModel(),
        single,
        sources[:1],
        timelines[:1],
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[31],
    )

    for frame_index in range(3):
        torch.testing.assert_close(
            batched.noises[frame_index][0], single.noises[frame_index][0]
        )
        torch.testing.assert_close(
            batched.known_noises[frame_index][0],
            single.known_noises[frame_index][0],
        )
        torch.testing.assert_close(
            reversed_batch.noises[frame_index][1], single.noises[frame_index][0]
        )
        torch.testing.assert_close(
            reversed_batch.known_noises[frame_index][1],
            single.known_noises[frame_index][0],
        )


def test_fixed_sequence_noise_reuses_full_horizon_noise():
    class NoiseRecordingDiffusion(_OneStepProjectedDiffusion):
        def __init__(self) -> None:
            self.noises: list[torch.Tensor] = []
            self.known_noises: list[torch.Tensor] = []

        def projected_ddim_sample_loop(self, *args, **kwargs):
            self.noises.append(kwargs["noise"].detach().cpu().clone())
            self.known_noises.append(kwargs["known_noise"].detach().cpu().clone())
            return super().projected_ddim_sample_loop(*args, **kwargs)

    source = build_toy_realtime_source(frame_count=3)
    timeline = longseq.build_isolated_condition_timeline("fixed", 3, "fixed_six")
    diffusion = NoiseRecordingDiffusion()
    longseq.rollout_long_sequence_sources(
        _RecordingModel(),
        diffusion,
        [source],
        [timeline],
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[17],
        diffusion_noise_mode="fixed_sequence",
    )

    assert all(noise.shape == (1, 11, 144) for noise in diffusion.noises)
    torch.testing.assert_close(diffusion.noises[0], diffusion.noises[1])
    torch.testing.assert_close(diffusion.noises[0], diffusion.noises[2])
    torch.testing.assert_close(diffusion.known_noises[0], diffusion.known_noises[1])
    torch.testing.assert_close(diffusion.known_noises[0], diffusion.known_noises[2])


def test_correlated_noise_uses_full_horizon_ar1_formula():
    class NoiseRecordingDiffusion(_OneStepProjectedDiffusion):
        def __init__(self) -> None:
            self.noises: list[torch.Tensor] = []
            self.known_noises: list[torch.Tensor] = []

        def projected_ddim_sample_loop(self, *args, **kwargs):
            self.noises.append(kwargs["noise"].detach().cpu().clone())
            self.known_noises.append(kwargs["known_noise"].detach().cpu().clone())
            return super().projected_ddim_sample_loop(*args, **kwargs)

    rho = 0.6
    source = build_toy_realtime_source(frame_count=2)
    timeline = longseq.build_isolated_condition_timeline("correlated", 2, "fixed_six")
    diffusion = NoiseRecordingDiffusion()
    longseq.rollout_long_sequence_sources(
        _RecordingModel(),
        diffusion,
        [source],
        [timeline],
        torch.device("cpu"),
        normalizer=None,
        diffusion_seeds=[23],
        diffusion_noise_mode="correlated",
        diffusion_noise_rho=rho,
    )

    generator = torch.Generator(device="cpu").manual_seed(23)
    first = torch.randn(11, 144, generator=generator)
    innovation = torch.randn(11, 144, generator=generator)
    expected_second = rho * first + np.sqrt(1.0 - rho**2) * innovation
    torch.testing.assert_close(diffusion.noises[0][0], first)
    torch.testing.assert_close(diffusion.noises[1][0], expected_second)

    known_seed = int(longseq.stable_context_seed(23, "longseq_inpaint_noise") % (2**63))
    known_generator = torch.Generator(device="cpu").manual_seed(known_seed)
    known_first = torch.randn(11, 144, generator=known_generator)
    known_innovation = torch.randn(11, 144, generator=known_generator)
    known_second = rho * known_first + np.sqrt(1.0 - rho**2) * known_innovation
    torch.testing.assert_close(diffusion.known_noises[0][0], known_first)
    torch.testing.assert_close(diffusion.known_noises[1][0], known_second)


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
                "source_path": str(source_path.resolve()),
                "source_relative_path": source_path.name,
            }
        )

    model = _RecordingModel()
    summary = longseq.evaluate_longseq_entries(
        entries=entries,
        source_dir=eval_set_dir,
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
    assert summary["metadata"]["use_future_rolling_prior"] is False
    assert summary["metadata"]["future_confidence_decay"] == pytest.approx(0.9)
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


def test_longseq_entry_evaluation_records_gt_history_warmup_protocol(tmp_path):
    eval_set_dir = tmp_path / "eval_set"
    eval_set_dir.mkdir()
    source_path = eval_set_dir / "source.npz"
    np.savez(source_path, **build_toy_realtime_source(frame_count=65))
    summary = longseq.evaluate_longseq_entries(
        entries=[
            {
                "sequence_id": "sequence",
                "source_path": str(source_path.resolve()),
                "source_relative_path": source_path.name,
            }
        ],
        source_dir=eval_set_dir,
        output_dir=tmp_path / "output",
        model=_RecordingModel(),
        diffusion=_OneStepProjectedDiffusion(),
        device=torch.device("cpu"),
        normalizer=None,
        conditions=["fixed_six"],
        gt_history_warmup_frames=60,
        show_progress=False,
    )

    assert summary["metadata"]["evaluation_protocol"] == (
        "isolated_condition_gt_history_warmup"
    )
    assert summary["metadata"]["gt_history_warmup_frames"] == 60
    assert summary["files"][0]["num_frames"] == 65
    assert summary["files"][0]["evaluated_frames"] == 5
