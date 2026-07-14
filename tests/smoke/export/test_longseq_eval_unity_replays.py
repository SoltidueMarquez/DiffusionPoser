from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np

from data_loaders.build_realtime_longseq_eval_set import build_realtime_longseq_eval_set
from data_loaders.longseq_eval_dropout import LongseqDropoutConfig
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, TRACKER_COUNT, get_schema_spec
from export.write_longseq_eval_unity_replays import build_arg_parser, export_longseq_eval_unity_replays
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


def patch_generated_root(monkeypatch, tmp_path):
    generated_root = tmp_path / "configured_generated"
    monkeypatch.setattr(
        "utils.default_artifact_paths.load_data_roots",
        lambda: SimpleNamespace(generated_root=generated_root),
    )
    return generated_root


def expected_longseq_root(generated_root, schema_name: str):
    return generated_root / "longseq_eval" / schema_name / "amass_60hz_test_stress_long"


def test_longseq_replay_export_parser_default_eval_root_uses_generated_layout(monkeypatch, tmp_path):
    generated_root = patch_generated_root(monkeypatch, tmp_path)
    args = build_arg_parser().parse_args([])

    assert_generated_layout_path(args.eval_root, str(expected_longseq_root(generated_root, CANONICAL_SCHEMA_NAME)))


def test_longseq_replay_export_parser_default_eval_root_follows_explicit_legacy_schema(monkeypatch, tmp_path):
    generated_root = patch_generated_root(monkeypatch, tmp_path)
    args = build_arg_parser().parse_args(["--schema", LEGACY_SCHEMA_NAME])

    assert_generated_layout_path(args.eval_root, str(expected_longseq_root(generated_root, LEGACY_SCHEMA_NAME)))


def test_longseq_replay_export_parser_preserves_explicit_eval_root_override():
    args = build_arg_parser().parse_args(
        [
            "--schema",
            LEGACY_SCHEMA_NAME,
            "--eval_root",
            "custom/longseq_eval",
        ]
    )

    assert normalized_path_text(args.eval_root) == "custom/longseq_eval"


def test_longseq_eval_unity_replay_export_writes_one_json_per_sequence(tmp_path):
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

    summary = export_longseq_eval_unity_replays(eval_set_dir=eval_set_dir)

    assert summary["file_count"] == 2
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    for item in summary["files"]:
        payload = json.loads(open(item["output_json"], "r", encoding="utf-8").read())
        assert payload["schemaName"] == schema.name
        assert payload["frameCount"] == item["num_frames"]
        assert payload["targetFeatureLength"] == schema.target_dim
        assert payload["trackerCount"] == TRACKER_COUNT
    assert json.loads(open(summary["summary_path"], "r", encoding="utf-8").read())["file_count"] == 2


def test_longseq_eval_unity_replay_export_can_apply_dropout(tmp_path):
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

    summary = export_longseq_eval_unity_replays(
        eval_set_dir=eval_set_dir,
        dropout_config=LongseqDropoutConfig(
            preset="tracker_mask_train",
            tracker_mask_policy="fixed_categories",
            tracker_mask_categories=("standard_three",),
        ),
    )

    payload = json.loads(open(summary["files"][0]["output_json"], "r", encoding="utf-8").read())
    valid = np.asarray(payload["sensorValid"], dtype=np.int32).reshape(payload["frameCount"], TRACKER_COUNT)
    assert not valid.all()
    assert valid[:, 0].all()
    assert valid.sum(axis=1).min() >= 3
    assert summary["files"][0]["valid_tracker_ratio"] < 1.0
