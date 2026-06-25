from __future__ import annotations

import argparse
import json

from data_loaders.build_realtime_longseq_eval_set import (
    build_arg_parser,
    build_realtime_longseq_eval_set,
    read_longseq_manifest,
    resolve_manifest_source_path,
)
from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from tests.smoke.longseq_eval_fixtures import write_toy_longseq_task_run
from utils.parser_util import add_data_options
from utils.run_dirs import read_latest_pointer


GENERATED_TASK_ROOT = "dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_tasks"
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


def test_longseq_eval_builder_defaults_use_generated_artifact_layout():
    args = build_arg_parser().parse_args([])

    assert_generated_layout_path(args.task_dir, GENERATED_TASK_ROOT)
    assert_generated_layout_path(args.output_root, GENERATED_LONGSEQ_EVAL_ROOT)


def test_longseq_eval_builder_preserves_explicit_path_overrides():
    args = build_arg_parser().parse_args(
        [
            "--task_dir",
            "custom/tasks",
            "--output_root",
            "custom/longseq_eval",
        ]
    )

    assert normalized_path_text(args.task_dir) == "custom/tasks"
    assert normalized_path_text(args.output_root) == "custom/longseq_eval"


def test_data_options_default_normalizer_uses_generated_artifact_layout():
    parser = argparse.ArgumentParser()
    add_data_options(parser)
    args = parser.parse_args(["--data_dir", "custom/tasks"])

    assert_generated_layout_path(args.normalizer_dir, GENERATED_NORMALIZER_ROOT)


def test_data_options_preserves_explicit_normalizer_override():
    parser = argparse.ArgumentParser()
    add_data_options(parser)
    args = parser.parse_args(
        [
            "--data_dir",
            "custom/tasks",
            "--normalizer_dir",
            "custom/normalizer",
        ]
    )

    assert normalized_path_text(args.normalizer_dir) == "custom/normalizer"


def test_longseq_eval_builder_selects_test_non_mirror_long_sources(tmp_path):
    task_root, _task_run = write_toy_longseq_task_run(tmp_path)
    output_root = tmp_path / "longseq_eval"

    output_dir = build_realtime_longseq_eval_set(
        argparse.Namespace(
            task_dir=str(task_root),
            task_run="latest",
            output_root=str(output_root),
            run_name="v1_test_stress_long_seed10",
            preset="stress_long",
            split="test",
            min_frames=60,
            include_mirror=False,
            schema=REALTIME_POSE_SCHEMA_NAME,
            overwrite=True,
        )
    )

    entries = read_longseq_manifest(output_dir)
    assert [entry["source_relative_path"] for entry in entries] == [
        "KIT/442/long_b_poses.npz",
        "CMU/55/long_a_poses.npz",
    ]
    assert {entry["preset"] for entry in entries} == {"stress_long"}
    assert {entry["is_mirrored"] for entry in entries} == {False}
    assert all(entry["split"] == "test" for entry in entries)

    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    for entry in entries:
        copied_path = resolve_manifest_source_path(output_dir, entry)
        assert copied_path.exists()
        source = load_realtime_source(copied_path, schema_name=schema.name)
        assert source[schema.body_pose_key].shape[0] == entry["num_frames"]
        assert entry["schema_name"] == schema.name
        assert entry["pose_representation"] == schema.pose_representation
        assert entry["root_y_policy"] == schema.root_y_policy
        assert entry["pelvis_height_mode"] == schema.pelvis_height_mode

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["sequence_count"] == 2
    assert summary["total_frames"] == 145
    assert read_latest_pointer(output_root, "longseq_eval") == output_dir
