from __future__ import annotations

import argparse

import numpy as np
import torch

from data_loaders.build_realtime_longseq_eval_set import build_realtime_longseq_eval_set, read_longseq_manifest
from data_loaders.longseq_eval_dropout import LongseqDropoutConfig
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_TARGET_START, get_schema_spec
from sample.evaluate_longseq_eval_set import build_arg_parser, evaluate_longseq_entries
from sample.simulate_unity_stream import IDENTITY_6D
from tests.smoke.longseq_eval_fixtures import write_toy_longseq_task_run


GENERATED_NORMALIZER_ROOT = "dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_train"
GENERATED_LONGSEQ_EVAL_ROOT = "dataset/generated/longseq_eval/realtime_pose_stationary5_v1/amass_60hz_test_stress_long"
LEGACY_PARENT_LOCAL_MARKERS = (
    "dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
    "dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
)


def normalized_path_text(value) -> str:
    return str(value).replace("\\", "/")


def assert_generated_layout_path(value, expected_suffix: str) -> None:
    text = normalized_path_text(value)
    assert text.endswith(expected_suffix)
    for marker in LEGACY_PARENT_LOCAL_MARKERS:
        assert marker not in text


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


def test_evaluate_longseq_parser_defaults_use_generated_artifact_layout():
    args = build_arg_parser().parse_args(["--model_path", "model000000000.pt"])

    assert_generated_layout_path(args.eval_root, GENERATED_LONGSEQ_EVAL_ROOT)
    assert_generated_layout_path(args.normalizer_dir, GENERATED_NORMALIZER_ROOT)


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
            tracker_mask_categories=("upper-body",),
        ),
    )

    result_path = output_dir / entries[0]["sequence_id"] / "unity_stream_long_sequence_result.npz"
    with np.load(result_path, allow_pickle=True) as data:
        sensor_valid = np.asarray(data["sensor_valid"], dtype=bool)[0]
    assert not sensor_valid.all()
    assert sensor_valid[:, 3].all()
    assert sensor_valid.sum(axis=1).min() >= 3
    assert summary["files"][0]["valid_tracker_ratio"] < 1.0
