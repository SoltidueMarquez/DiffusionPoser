from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch

from data_loaders.build_realtime_longseq_eval_set import build_realtime_longseq_eval_set, read_longseq_manifest
from data_loaders.longseq_eval_dropout import (
    LongseqDropoutConfig,
    apply_seeded_hip_dropout_timeline,
    build_longseq_sensor_valid,
)
from data_loaders.sensor_masking import (
    HIP_TRACKER_INDEX,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_TARGET_START,
    get_schema_spec,
)
from sample.evaluate_longseq_eval_set import build_arg_parser, evaluate_longseq_entries, parse_longseq_eval_args
from sample.simulate_unity_stream import IDENTITY_6D
from tests.smoke.longseq_eval_fixtures import write_toy_longseq_task_run


CANONICAL_SCHEMA_NAME = "realtime_pose_stationary5_v1"
LEGACY_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"
LEGACY_PARENT_LOCAL_MARKERS = (
    "dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
    "dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
)


def normalized_path_text(value) -> str:
    return str(value).replace("\\", "/")


def assert_generated_layout_path(value, expected_suffix: str) -> None:
    text = normalized_path_text(value)
    expected = normalized_path_text(expected_suffix)
    assert text.endswith(expected)
    for marker in LEGACY_PARENT_LOCAL_MARKERS:
        assert marker not in text


def test_seeded_hip_dropout_timeline_has_exact_runs_and_reconnect_frames():
    base = np.ones((1000, 6), dtype=bool)
    first = apply_seeded_hip_dropout_timeline(
        sensor_valid=base,
        sequence_id="sequence_a",
        seed=10,
        duration_frames=10,
        interval_frames=100,
    )
    second = apply_seeded_hip_dropout_timeline(
        sensor_valid=base,
        sequence_id="sequence_a",
        seed=10,
        duration_frames=10,
        interval_frames=100,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, :HIP_TRACKER_INDEX], base[:, :HIP_TRACKER_INDEX])
    np.testing.assert_array_equal(first[:, HIP_TRACKER_INDEX + 1 :], base[:, HIP_TRACKER_INDEX + 1 :])
    missing = np.flatnonzero(~first[:, HIP_TRACKER_INDEX])
    runs = np.split(missing, np.where(np.diff(missing) > 1)[0] + 1)
    assert runs
    assert all(run.size == 10 for run in runs)
    assert all(run[0] >= REALTIME_POSE_TARGET_START for run in runs)
    assert all(first[run[-1] + 1, HIP_TRACKER_INDEX] for run in runs)


def test_seeded_hip_dropout_timeline_preserves_three_tracker_minimum():
    base = np.zeros((1000, 6), dtype=bool)
    non_hip_indices = [index for index in range(6) if index != HIP_TRACKER_INDEX][:2]
    base[:, non_hip_indices] = True
    base[:, HIP_TRACKER_INDEX] = True

    result = apply_seeded_hip_dropout_timeline(
        sensor_valid=base,
        sequence_id="three_tracker_sequence",
        seed=10,
        duration_frames=8,
        interval_frames=300,
    )

    np.testing.assert_array_equal(result, base)
    assert result.sum(axis=1).min() == 3


def test_longseq_dynamic_dropout_segments_keep_temporal_dropout_semantics():
    config = LongseqDropoutConfig(
        preset="tracker_mask_train",
        tracker_mask_policy="dynamic_categories",
        tracker_mask_categories=("dynamic_dropout",),
        tracker_mask_segment_frames=61,
    )

    valid = build_longseq_sensor_valid(
        frame_count=244,
        sequence_id="dynamic_sequence",
        base_sensor_valid=np.ones((244, 6), dtype=bool),
        config=config,
    )

    assert valid[:, 0].all()
    assert valid.sum(axis=1).min() >= 3
    assert np.any(valid.sum(axis=1) < 6)
    assert np.any(valid.sum(axis=1) == 6)


def patch_generated_root(monkeypatch, tmp_path):
    generated_root = tmp_path / "configured_generated"
    monkeypatch.setattr(
        "utils.default_artifact_paths.load_artifact_roots",
        lambda: SimpleNamespace(generated_root=generated_root),
    )
    return generated_root


def expected_normalizer_root(generated_root, schema_name: str):
    return generated_root / "normalizers" / schema_name / "amass_60hz_train"


def expected_longseq_root(generated_root, schema_name: str):
    return generated_root / "longseq_eval" / schema_name / "amass_60hz_test_stress_long"


class FixedLongseqDiffusion:
    def __init__(self):
        self.calls = 0

    def p_sample_loop(self, model, shape, noise, clip_denoised, model_kwargs):
        del model, shape, noise, clip_denoised
        self.calls += 1
        sample = model_kwargs["y"]["inpainted_motion"].clone()
        schema = get_schema_spec(model_kwargs["y"]["schema_name"])
        sample[:, schema.body_pose_slice(), REALTIME_POSE_TARGET_START] = torch.from_numpy(
            np.tile(IDENTITY_6D, 24)
        ).to(sample.device)
        sample[:, schema.root_yaw_delta_slice(), REALTIME_POSE_TARGET_START] = torch.tensor(
            [0.0, 1.0],
            dtype=sample.dtype,
            device=sample.device,
        )
        sample[:, schema.root_delta_xz_slice(), REALTIME_POSE_TARGET_START] = 0.0
        sample[:, schema.root_height_slice(), REALTIME_POSE_TARGET_START] = 0.0
        sample[:, schema.stationary_prob_slice(), REALTIME_POSE_TARGET_START] = 0.0
        return sample


def test_evaluate_longseq_parser_defaults_use_generated_artifact_layout(monkeypatch, tmp_path):
    generated_root = patch_generated_root(monkeypatch, tmp_path)
    args = build_arg_parser().parse_args(["--model_path", "model000000000.pt"])

    assert_generated_layout_path(args.eval_root, str(expected_longseq_root(generated_root, CANONICAL_SCHEMA_NAME)))
    assert_generated_layout_path(args.normalizer_dir, str(expected_normalizer_root(generated_root, CANONICAL_SCHEMA_NAME)))


def test_evaluate_longseq_parser_defaults_follow_explicit_legacy_schema(monkeypatch, tmp_path):
    generated_root = patch_generated_root(monkeypatch, tmp_path)
    args = build_arg_parser().parse_args(
        [
            "--model_path",
            "model000000000.pt",
            "--schema",
            LEGACY_SCHEMA_NAME,
        ]
    )

    assert_generated_layout_path(args.eval_root, str(expected_longseq_root(generated_root, LEGACY_SCHEMA_NAME)))
    assert_generated_layout_path(args.normalizer_dir, str(expected_normalizer_root(generated_root, LEGACY_SCHEMA_NAME)))


def test_evaluate_longseq_parser_preserves_explicit_path_overrides():
    args = build_arg_parser().parse_args(
        [
            "--model_path",
            "model000000000.pt",
            "--eval_root",
            "custom/longseq_eval",
            "--normalizer_dir",
            "custom/normalizer",
        ]
    )

    assert normalized_path_text(args.eval_root) == "custom/longseq_eval"
    assert normalized_path_text(args.normalizer_dir) == "custom/normalizer"


def test_evaluate_longseq_checkpoint_only_restores_model_and_diffusion_args(tmp_path):
    model_path = tmp_path / "model000000100.pt"
    model_path.touch()
    (tmp_path / "args.json").write_text(
        json.dumps(
            {
                "layers": 4,
                "diffusion_steps": 100,
                "seed": 0,
                "tracker_mask_seed": 0,
                "tracker_latency_max_frames": 2,
                "tracker_burst_dropout_prob": 0.05,
                "tracker_outlier_prob": 0.01,
            }
        ),
        encoding="utf-8",
    )

    args = parse_longseq_eval_args(
        build_arg_parser(),
        argv=[
            "--model_path",
            str(model_path),
            "--tracker_mask_policy",
            "fixed_categories",
            "--tracker_mask_categories",
            "standard_three",
        ],
    )

    assert args.layers == 4
    assert args.diffusion_steps == 100
    assert args.seed == 10
    assert args.tracker_mask_seed == 10
    assert args.tracker_latency_max_frames == 0
    assert args.tracker_burst_dropout_prob == 0.0
    assert args.tracker_outlier_prob == 0.0


def test_evaluate_longseq_entries_writes_per_sequence_and_aggregate_summary(tmp_path):
    task_root, _task_run = write_toy_longseq_task_run(tmp_path)
    eval_set_dir = build_realtime_longseq_eval_set(
        argparse.Namespace(
            task_dir=str(task_root),
            task_run="latest",
            output_root=str(tmp_path / "longseq_eval"),
            run_name="stress",
            preset="stress_long",
            split="test",
            min_frames=60,
            include_mirror=False,
            schema=REALTIME_POSE_SCHEMA_NAME,
            overwrite=True,
        )
    )
    entries = read_longseq_manifest(eval_set_dir)
    output_dir = tmp_path / "output"

    summary = evaluate_longseq_entries(
        entries=entries,
        eval_set_dir=eval_set_dir,
        output_dir=output_dir,
        model=object(),
        diffusion=FixedLongseqDiffusion(),
        device=torch.device("cpu"),
        normalizer=None,
        use_ddim=False,
        model_path="model000000000.pt",
        weights="model",
        limit=1,
        root_correction=False,
        tracker_ik=False,
        render_mp4=False,
    )

    assert summary["summary"]["file_count"] == 1
    assert summary["metadata"]["sequence_count"] == 1
    assert (output_dir / "longseq_eval_summary.json").exists()
    sequence_dir = output_dir / entries[0]["sequence_id"]
    assert (sequence_dir / "unity_stream_long_sequence_result.npz").exists()
    assert (sequence_dir / "unity_stream_eval_summary.json").exists()


def test_evaluate_longseq_entries_applies_dropout_to_sensor_valid(tmp_path):
    task_root, _task_run = write_toy_longseq_task_run(tmp_path)
    eval_set_dir = build_realtime_longseq_eval_set(
        argparse.Namespace(
            task_dir=str(task_root),
            task_run="latest",
            output_root=str(tmp_path / "longseq_eval"),
            run_name="stress",
            preset="stress_long",
            split="test",
            min_frames=60,
            include_mirror=False,
            schema=REALTIME_POSE_SCHEMA_NAME,
            overwrite=True,
        )
    )
    entries = read_longseq_manifest(eval_set_dir)
    output_dir = tmp_path / "output_dropout"

    summary = evaluate_longseq_entries(
        entries=entries,
        eval_set_dir=eval_set_dir,
        output_dir=output_dir,
        model=object(),
        diffusion=FixedLongseqDiffusion(),
        device=torch.device("cpu"),
        normalizer=None,
        use_ddim=False,
        model_path="model000000000.pt",
        weights="model",
        limit=1,
        root_correction=False,
        tracker_ik=False,
        render_mp4=False,
        dropout_config=LongseqDropoutConfig(
            preset="tracker_mask_train",
            tracker_mask_policy="fixed_categories",
            tracker_mask_categories=("standard_three",),
        ),
    )

    result_path = output_dir / entries[0]["sequence_id"] / "unity_stream_long_sequence_result.npz"
    with np.load(result_path, allow_pickle=True) as data:
        sensor_valid = np.asarray(data["sensor_valid"], dtype=bool)[0]
    assert not sensor_valid.all()
    assert sensor_valid[:, 0].all()
    assert sensor_valid.sum(axis=1).min() >= 3
    assert summary["files"][0]["valid_tracker_ratio"] < 1.0
