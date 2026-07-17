from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from data_loaders.build_realtime_longseq_eval_set import (
    build_copied_source_filename,
    build_arg_parser,
    build_realtime_longseq_eval_set,
    read_longseq_manifest,
    resolve_manifest_source_path,
)
from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from data_loaders.stationary_label_config import stationary_label_metadata
from data_loaders.generate_realtime_pose_tasks import DEFAULT_SOURCE_SET_NAME, DEFAULT_TASK_SET_NAME
from data_loaders.compute_realtime_pose_normalizer import DEFAULT_NORMALIZER_NAME
from tests.smoke.longseq_eval_fixtures import write_toy_longseq_task_run
from utils.default_artifact_paths import (
    DEFAULT_REALTIME_POSE_NORMALIZER_NAME,
    DEFAULT_REALTIME_POSE_SOURCE_SET_NAME,
    DEFAULT_REALTIME_POSE_TASK_SET_NAME,
)
from utils.parser_util import add_data_options, parse_and_load_runtime_schema_from_model
from utils.run_dirs import read_latest_pointer


CANONICAL_SCHEMA_NAME = "realtime_pose_stationary5_v1"
LEGACY_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"
LEGACY_PARENT_LOCAL_MARKERS = (
    "dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
    "dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
)


def normalized_path_text(value) -> str:
    return str(value).replace("\\", "/")


def test_longseq_copied_filename_respects_windows_safe_path_budget():
    sequence_dir = Path("D:/") / ("nested_" * 28)
    filename = build_copied_source_filename(
        sequence_dir=sequence_dir,
        sequence_id="BioMotionLab_NTroje_rub010_0029_scamper_poses",
        frame_count=2586,
    )

    assert filename.endswith("__2586f.npz")
    assert len(str(sequence_dir / filename)) <= 248


def assert_generated_layout_path(value, expected_suffix: str) -> None:
    text = normalized_path_text(value)
    expected = normalized_path_text(expected_suffix)
    assert text.endswith(expected)
    for marker in LEGACY_PARENT_LOCAL_MARKERS:
        assert marker not in text


def patch_generated_root(monkeypatch, tmp_path):
    generated_root = tmp_path / "configured_generated"
    monkeypatch.setattr(
        "utils.default_artifact_paths.load_artifact_roots",
        lambda: SimpleNamespace(generated_root=generated_root),
    )
    return generated_root


def expected_task_root(generated_root, schema_name: str):
    return generated_root / "tasks" / schema_name / "amass_60hz_tasks"


def expected_normalizer_root(generated_root, schema_name: str):
    return generated_root / "normalizers" / schema_name / "amass_60hz_train"


def expected_longseq_root(generated_root, schema_name: str):
    return generated_root / "longseq_eval" / schema_name / "amass_60hz_test_stress_long"


def test_default_artifact_path_names_match_pipeline_entrypoints():
    assert DEFAULT_REALTIME_POSE_SOURCE_SET_NAME == DEFAULT_SOURCE_SET_NAME
    assert DEFAULT_REALTIME_POSE_TASK_SET_NAME == DEFAULT_TASK_SET_NAME
    assert DEFAULT_REALTIME_POSE_NORMALIZER_NAME == DEFAULT_NORMALIZER_NAME


def test_longseq_eval_builder_defaults_use_generated_artifact_layout(monkeypatch, tmp_path):
    generated_root = patch_generated_root(monkeypatch, tmp_path)
    args = build_arg_parser().parse_args([])

    assert_generated_layout_path(args.task_dir, str(expected_task_root(generated_root, CANONICAL_SCHEMA_NAME)))
    assert_generated_layout_path(args.output_root, str(expected_longseq_root(generated_root, CANONICAL_SCHEMA_NAME)))


def test_longseq_eval_builder_defaults_follow_explicit_legacy_schema(monkeypatch, tmp_path):
    generated_root = patch_generated_root(monkeypatch, tmp_path)
    args = build_arg_parser().parse_args(["--schema", LEGACY_SCHEMA_NAME])

    assert_generated_layout_path(args.task_dir, str(expected_task_root(generated_root, LEGACY_SCHEMA_NAME)))
    assert_generated_layout_path(args.output_root, str(expected_longseq_root(generated_root, LEGACY_SCHEMA_NAME)))


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


def test_data_options_default_normalizer_uses_generated_artifact_layout(monkeypatch, tmp_path):
    generated_root = patch_generated_root(monkeypatch, tmp_path)
    parser = argparse.ArgumentParser()
    add_data_options(parser)
    args = parser.parse_args(["--data_dir", "custom/tasks"])

    assert_generated_layout_path(args.normalizer_dir, str(expected_normalizer_root(generated_root, CANONICAL_SCHEMA_NAME)))


def test_data_options_default_normalizer_follows_explicit_legacy_schema(monkeypatch, tmp_path):
    generated_root = patch_generated_root(monkeypatch, tmp_path)
    parser = argparse.ArgumentParser()
    add_data_options(parser)
    args = parser.parse_args(["--data_dir", "custom/tasks", "--schema", LEGACY_SCHEMA_NAME])

    assert_generated_layout_path(args.normalizer_dir, str(expected_normalizer_root(generated_root, LEGACY_SCHEMA_NAME)))


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


def test_runtime_schema_parse_updates_default_normalizer_after_checkpoint_schema(monkeypatch, tmp_path):
    generated_root = patch_generated_root(monkeypatch, tmp_path)
    model_path = tmp_path / "model000000000.pt"
    (tmp_path / "args.json").write_text(
        json.dumps({"schema": LEGACY_SCHEMA_NAME, "schema_name": LEGACY_SCHEMA_NAME}),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_data_options(parser)
    parser.add_argument("--model_path", required=True)

    args = parse_and_load_runtime_schema_from_model(
        parser,
        argv=["--data_dir", "custom/tasks", "--model_path", str(model_path)],
    )

    assert args.schema == LEGACY_SCHEMA_NAME
    assert_generated_layout_path(args.normalizer_dir, str(expected_normalizer_root(generated_root, LEGACY_SCHEMA_NAME)))


def test_runtime_schema_parse_preserves_explicit_normalizer_override(monkeypatch, tmp_path):
    patch_generated_root(monkeypatch, tmp_path)
    model_path = tmp_path / "model000000000.pt"
    (tmp_path / "args.json").write_text(
        json.dumps({"schema": LEGACY_SCHEMA_NAME, "schema_name": LEGACY_SCHEMA_NAME}),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_data_options(parser)
    parser.add_argument("--model_path", required=True)

    args = parse_and_load_runtime_schema_from_model(
        parser,
        argv=[
            "--data_dir",
            "custom/tasks",
            "--model_path",
            str(model_path),
            "--normalizer_dir",
            "custom/normalizer",
        ],
    )

    assert args.schema == LEGACY_SCHEMA_NAME
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
        for key, value in stationary_label_metadata().items():
            assert entry[key] == value

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["sequence_count"] == 2
    assert summary["total_frames"] == 145
    for key, value in stationary_label_metadata().items():
        assert summary["config"][key] == value
    assert read_latest_pointer(output_root, "longseq_eval") == output_dir
