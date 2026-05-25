from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from data_loaders.generate_realtime_pose_tasks import main as generate_realtime_pose_tasks_main
from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset, encode_realtime_pose_features
from data_loaders.realtime_pose_kinematics import integrate_root_delta_xz_ref
from data_loaders.sensor_masking import (
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    get_schema_spec,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source, write_toy_source_dataset
from utils.normalizer import REALTIME_POSE_MIN_NORMALIZER_STD, RealtimePoseNormalizer


def test_v2_contact_feature_layout_and_root_integration():
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    source = build_toy_realtime_source(frame_count=REALTIME_POSE_SEQ_LEN)
    source["sensor_valid"] = np.ones((REALTIME_POSE_SEQ_LEN, 6), dtype=bool)
    features = encode_realtime_pose_features(source, schema_name=schema.name)
    assert features.shape == (REALTIME_POSE_SEQ_LEN, schema.feature_dim)
    assert features[:, schema.root_delta_xz_slice()].shape == (REALTIME_POSE_SEQ_LEN, 2)
    assert set(np.unique(features[:, schema.foot_contact_slice()]).tolist()).issubset({0.0, 1.0})

    prev_pos = source["root_pos_world"][REALTIME_POSE_TARGET_START - 1:REALTIME_POSE_TARGET_START]
    prev_yaw = source["root_yaw"][REALTIME_POSE_TARGET_START - 1:REALTIME_POSE_TARGET_START]
    delta = source["root_delta_xz_ref"][REALTIME_POSE_TARGET_START:REALTIME_POSE_TARGET_START + 1]
    integrated = integrate_root_delta_xz_ref(prev_pos, prev_yaw, delta)
    np.testing.assert_allclose(integrated[0, [0, 2]], source["root_pos_world"][REALTIME_POSE_TARGET_START, [0, 2]], atol=1e-6)


def test_v2_contact_task_dataset_and_normalizer_contract(tmp_path):
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    normalizer_dir = tmp_path / "meta"
    write_toy_source_dataset(source_dir, schema_name=REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--schema",
            REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    manifest_path = task_dir / "train" / "manifest.jsonl"
    entry = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    assert entry["feature_dim"] == schema.feature_dim

    mean = torch.zeros(schema.feature_dim)
    std = torch.ones(schema.feature_dim)
    normalizer = RealtimePoseNormalizer(normalizer_dir, disable=True, schema_name=schema.name)
    normalizer.save(mean, std)
    dataset = RealtimePoseTaskDataset(
        task_dir,
        split="train",
        normalizer_dir=normalizer_dir,
        normalize_input=True,
        schema_name=schema.name,
    )
    item = dataset[0]
    assert tuple(item["x"].shape) == (schema.feature_dim, REALTIME_POSE_SEQ_LEN)
    assert item["inpaint_mask"][:schema.target_dim, REALTIME_POSE_TARGET_START].all()
    assert "target_root_delta_xz_ref" in item
    assert "target_foot_contact" in item
    assert item["target_tracker_pos_ref"].shape == (6, 3)
    assert item["target_sensor_valid"].shape == (6,)


def test_realtime_normalizer_does_not_amplify_near_constant_channels(tmp_path):
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    mean = torch.zeros(schema.feature_dim)
    std = torch.ones(schema.feature_dim)
    std[0] = REALTIME_POSE_MIN_NORMALIZER_STD / 10.0
    normalizer = RealtimePoseNormalizer(tmp_path / "meta", disable=True, schema_name=schema.name)
    normalizer.save(mean, std)

    loaded = RealtimePoseNormalizer(tmp_path / "meta", schema_name=schema.name)
    assert float(loaded.std[0]) == 1.0
    features = torch.zeros(1, schema.feature_dim)
    features[0, 0] = 1e-4
    normalized = loaded.normalize(features)
    assert torch.isfinite(normalized).all()
    assert float(normalized[0, 0]) < 1e-3


def test_predicted_history_cache_requires_normalized_feature_space(tmp_path):
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    cache_dir = tmp_path / "cache"
    write_toy_source_dataset(source_dir, schema_name=REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--schema",
            REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    base_dataset = RealtimePoseTaskDataset(
        task_dir,
        split="train",
        normalize_input=False,
        schema_name=schema.name,
        tracker_mask_policy="task",
    )
    base_item = base_dataset[0]
    task_id = base_dataset.entries[0]["task_id"]
    cached = base_item["x"].T.numpy().copy()
    cached[:REALTIME_POSE_TARGET_START, schema.target_slice()] += 0.25
    cache_dir.mkdir()
    np.savez(
        cache_dir / f"{task_id}.npz",
        predicted_features_normalized=cached,
        schema_name=np.asarray(schema.name),
        feature_space=np.asarray("normalized"),
    )
    cached_dataset = RealtimePoseTaskDataset(
        task_dir,
        split="train",
        normalize_input=False,
        schema_name=schema.name,
        tracker_mask_policy="task",
        predicted_history_cache_dir=cache_dir,
        predicted_history_prob=1.0,
    )
    cached_item = cached_dataset[0]
    assert torch.allclose(cached_item["x"], base_item["x"])
    assert not torch.allclose(
        cached_item["conditioned_x"][schema.target_slice(), :REALTIME_POSE_TARGET_START],
        base_item["conditioned_x"][schema.target_slice(), :REALTIME_POSE_TARGET_START],
    )

    np.savez(
        cache_dir / f"{task_id}.npz",
        predicted_features_raw=cached,
        schema_name=np.asarray(schema.name),
        feature_space=np.asarray("raw"),
    )
    with pytest.raises(KeyError, match="predicted_features_normalized"):
        _ = cached_dataset[0]
