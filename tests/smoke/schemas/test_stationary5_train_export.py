from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data_loaders.generate_realtime_pose_tasks import load_realtime_source, main as generate_realtime_pose_tasks_main
from data_loaders.realtime_pose_dataset import (
    RealtimePoseTaskDataset,
    encode_realtime_pose_features,
    load_materialized_task_npz,
    load_realtime_task_arrays,
)
from data_loaders.sensor_masking import POSE_REPRESENTATION_KEY, get_schema_spec
from export.write_unity_runtime_assets import write_runtime_assets
from tests.smoke.realtime_pose_fixtures import build_toy_source_metadata, write_toy_source_dataset
from train.training_loop import validate_realtime_pose_training_args
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import read_latest_pointer


CANONICAL_SCHEMA_NAME = "realtime_pose_stationary5_v1"
LEGACY_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"
STATIONARY5_SCHEMA_NAMES = (CANONICAL_SCHEMA_NAME, LEGACY_SCHEMA_NAME)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_artifact_dir(root, kind):
    latest = read_latest_pointer(root, kind=kind)
    assert latest is not None
    return latest


def assert_runtime_payload_schema(payload, schema_name, schema):
    assert payload["schemaName"] == schema_name
    if "poseRepresentation" in payload:
        assert payload["poseRepresentation"] == schema.pose_representation
    if "rootYPolicy" in payload:
        assert payload["rootYPolicy"] == schema.root_y_policy
    if "pelvisHeightMode" in payload:
        assert payload["pelvisHeightMode"] == schema.pelvis_height_mode


@pytest.mark.parametrize("schema_name", STATIONARY5_SCHEMA_NAMES)
def test_stationary5_schema_toy_source_task_normalizer_export(tmp_path, schema_name):
    schema = get_schema_spec(schema_name)
    assert schema.name == schema_name
    assert schema.canonical_name == CANONICAL_SCHEMA_NAME

    source_dir = tmp_path / schema_name / "sources"
    source_path = write_toy_source_dataset(source_dir, frame_count=schema.seq_len, schema_name=schema_name)

    fixture_metadata = build_toy_source_metadata(frame_count=schema.seq_len, schema_name=schema_name)
    assert fixture_metadata["schema_name"] == schema_name
    assert fixture_metadata["schema_canonical_name"] == CANONICAL_SCHEMA_NAME

    source_manifest = read_jsonl(source_dir / "manifest.jsonl")
    assert source_manifest[0]["schema_name"] == schema_name
    assert source_manifest[0]["schema_canonical_name"] == CANONICAL_SCHEMA_NAME

    with np.load(source_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        assert metadata["schema_name"] == schema_name
        assert metadata["schema_canonical_name"] == CANONICAL_SCHEMA_NAME

    source = load_realtime_source(source_path, schema_name=schema_name)
    assert schema.body_pose_key in source
    assert str(source[POSE_REPRESENTATION_KEY].item()) == schema.pose_representation

    task_root = tmp_path / schema_name / "tasks"
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_root),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--schema",
            schema_name,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    task_dir = latest_artifact_dir(task_root, kind="tasks")
    task_manifest_path = task_dir / "train" / "manifest.jsonl"
    task_entry = read_jsonl(task_manifest_path)[0]
    assert task_entry["schema_name"] == schema_name
    assert task_entry["schema_canonical_name"] == CANONICAL_SCHEMA_NAME
    with np.load(task_manifest_path.parent / task_entry["task_path"], allow_pickle=False) as task:
        assert str(task["schema_name"].item()) == schema_name
        assert str(task["task_format"].item()) == schema.task_format

    normalizer_dir = tmp_path / schema_name / "normalizer"
    RealtimePoseNormalizer(normalizer_dir, disable=True, schema_name=schema_name).save(
        torch.zeros(schema.feature_dim),
        torch.ones(schema.feature_dim),
    )
    normalizer_meta = json.loads((normalizer_dir / "normalizer_meta.json").read_text(encoding="utf-8"))
    assert normalizer_meta["schema_name"] == schema_name
    loaded_normalizer = RealtimePoseNormalizer(normalizer_dir, schema_name=schema_name)
    assert loaded_normalizer.schema.name == schema_name
    assert loaded_normalizer.mean.numel() == schema.feature_dim

    dataset = RealtimePoseTaskDataset(
        task_root,
        split="train",
        normalizer_dir=normalizer_dir,
        normalize_input=True,
        tracker_mask_policy="task",
        schema_name=schema_name,
    )
    assert dataset.schema.name == schema_name
    item = dataset[0]
    assert tuple(item["x"].shape) == (schema.feature_dim, schema.seq_len)
    assert tuple(item["conditioned_x"].shape) == (schema.feature_dim, schema.seq_len)
    assert tuple(item["inpaint_mask"].shape) == (schema.feature_dim, schema.seq_len)
    assert item["inpaint_mask"][: schema.target_dim, schema.target_start].all()
    assert not item["inpaint_mask"][schema.target_dim :, :].any()

    task = load_materialized_task_npz(task_manifest_path.parent, task_entry["task_path"], schema_name=schema_name)
    arrays = load_realtime_task_arrays(task=task, seq_len=schema.seq_len, schema_name=schema_name)
    features = encode_realtime_pose_features(arrays, schema_name=schema_name)
    assert features.shape == (schema.seq_len, schema.feature_dim)

    assets = write_runtime_assets(
        output_dir=tmp_path / schema_name / "runtime_assets",
        normalize_input=True,
        normalizer_dir=normalizer_dir,
        schema_name=schema_name,
    )
    feature_schema = json.loads(assets["feature_schema"].read_text(encoding="utf-8"))
    assert_runtime_payload_schema(feature_schema, schema_name, schema)
    assert feature_schema["featureDim"] == schema.feature_dim
    normalizer_payload = json.loads(assets["normalizer"].read_text(encoding="utf-8"))
    assert_runtime_payload_schema(normalizer_payload, schema_name, schema)
    assert normalizer_payload["featureDim"] == schema.feature_dim
    ddim_schedule = json.loads(assets["ddim_schedule"].read_text(encoding="utf-8"))
    assert_runtime_payload_schema(ddim_schedule, schema_name, schema)


@pytest.mark.parametrize("schema_name", STATIONARY5_SCHEMA_NAMES)
def test_stationary5_schema_training_args_accept_exact_name(schema_name):
    schema = get_schema_spec(schema_name)
    args = Namespace(
        schema=schema_name,
        input_feats=schema.feature_dim,
        seq_len=schema.seq_len,
        max_seq_len=schema.seq_len,
    )

    resolved = validate_realtime_pose_training_args(args)

    assert resolved.name == schema_name
    assert resolved.canonical_name == CANONICAL_SCHEMA_NAME
