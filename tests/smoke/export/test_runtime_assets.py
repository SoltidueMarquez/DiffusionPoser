from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from data_loaders.sensor_masking import (
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    get_schema_spec,
)
from export.export_sentis_denoiser import validate_normalizer_export_contract
from export.write_unity_runtime_assets import build_normalizer, write_runtime_assets
from utils.normalizer import RealtimePoseNormalizer


def write_legacy_policyless_normalizer(normalizer_dir, schema):
    normalizer_dir.mkdir()
    torch.save(torch.zeros(REALTIME_POSE_INPUT_DIM), normalizer_dir / "mean.pt")
    torch.save(torch.ones(REALTIME_POSE_INPUT_DIM), normalizer_dir / "std.pt")
    (normalizer_dir / "normalizer_meta.json").write_text(
        json.dumps(
            {
                "schema_name": schema.name,
                POSE_REPRESENTATION_KEY: schema.pose_representation,
                "feature_dim": schema.feature_dim,
                "eps": 1e-8,
            }
        ),
        encoding="utf-8",
    )


def test_runtime_assets_are_realtime_pose_only(tmp_path):
    assets = write_runtime_assets(output_dir=tmp_path, normalize_input=False, schema_name=REALTIME_POSE_SCHEMA_NAME)
    schema = json.loads(assets["feature_schema"].read_text(encoding="utf-8"))
    schema_spec = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    assert schema["schemaVersion"] == 4
    assert schema["schemaName"] == REALTIME_POSE_SCHEMA_NAME
    assert schema["poseRepresentation"] == schema_spec.pose_representation
    assert schema["featureDim"] == REALTIME_POSE_INPUT_DIM
    assert schema["sequenceLength"] == REALTIME_POSE_SEQ_LEN
    assert schema["targetStart"] == 60
    assert schema["targetLength"] == 1
    assert schema["targetFeatureLength"] == schema_spec.target_dim
    assert schema["bodyPoseBodyFbxLocalDelta6d"] == {"name": schema_spec.body_pose_key, "start": 0, "length": 144}
    assert schema["rootHeadingDeltaSinCos"] == {"name": "root_heading_delta_sincos", "start": 144, "length": 2}
    assert schema["rootDeltaXZReference"] == {"name": "root_delta_xz_ref", "start": 146, "length": 2}
    assert schema["pelvisHeight"] == {"name": "pelvis_height", "start": 148, "length": 1}
    assert schema["stationaryProb5"]["name"] == "stationary_prob_5"
    assert schema["stationaryProb5"]["start"] == 149
    assert schema["stationaryProb5"]["length"] == 5
    assert schema["stationaryProb5"]["jointIndices"] == [0, 10, 11, 22, 23]
    assert schema["trackerPositionReference"] == {"name": "tracker_pos_ref", "start": 154, "length": 18}
    assert schema["trackerRotation6dReference"] == {"name": "tracker_rot_ref_6d", "start": 172, "length": 36}
    assert schema["sensorValid"] == {"name": "sensor_valid", "start": 208, "length": 6}
    assert schema["runtimeRules"]["poseRepresentation"] == schema_spec.pose_representation
    assert schema["runtimeRules"]["rootPositionY"] == schema_spec.root_y_policy
    assert schema["runtimeRules"]["pelvisHeightApplication"] == schema_spec.pelvis_height_mode
    assert schema["runtimeRules"]["onnxDummyInputShape"] == [1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN]


def test_realtime_normalizer_requires_schema_dimensions(tmp_path):
    normalizer_dir = tmp_path / "meta"
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    RealtimePoseNormalizer(normalizer_dir, disable=True, schema_name=schema.name).save(
        torch.zeros(REALTIME_POSE_INPUT_DIM),
        torch.ones(REALTIME_POSE_INPUT_DIM),
    )
    payload = build_normalizer(
        feature_dim=REALTIME_POSE_INPUT_DIM,
        normalizer_dir=normalizer_dir,
        normalize_input=True,
        strict=True,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
    )
    assert payload["enabled"] is True
    assert payload["featureDim"] == REALTIME_POSE_INPUT_DIM
    assert payload["poseRepresentation"] == schema.pose_representation
    assert payload["rootYPolicy"] == schema.root_y_policy
    assert payload["pelvisHeightMode"] == schema.pelvis_height_mode
    assert len(payload["mean"]) == REALTIME_POSE_INPUT_DIM


def test_realtime_normalizer_rejects_missing_pose_metadata(tmp_path):
    normalizer_dir = tmp_path / "old_meta"
    normalizer_dir.mkdir()
    torch.save(torch.zeros(REALTIME_POSE_INPUT_DIM), normalizer_dir / "mean.pt")
    torch.save(torch.ones(REALTIME_POSE_INPUT_DIM), normalizer_dir / "std.pt")
    with pytest.raises(FileNotFoundError, match="normalizer_meta.json"):
        build_normalizer(
            feature_dim=REALTIME_POSE_INPUT_DIM,
            normalizer_dir=normalizer_dir,
            normalize_input=True,
            strict=True,
            schema_name=REALTIME_POSE_SCHEMA_NAME,
        )


def test_realtime_normalizer_rejects_missing_root_y_policy_metadata(tmp_path):
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    normalizer_dir = tmp_path / "old_meta"
    write_legacy_policyless_normalizer(normalizer_dir, schema)

    with pytest.raises(ValueError, match="root_y_policy"):
        RealtimePoseNormalizer(normalizer_dir, schema_name=schema.name)


def test_runtime_export_rejects_policyless_normalizer(tmp_path):
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    normalizer_dir = tmp_path / "old_meta"
    write_legacy_policyless_normalizer(normalizer_dir, schema)

    with pytest.raises(ValueError, match="root_y_policy"):
        write_runtime_assets(
            output_dir=tmp_path / "assets",
            normalize_input=True,
            normalizer_dir=normalizer_dir,
            schema_name=schema.name,
        )


def test_normalized_runtime_asset_export_requires_normalizer_dir():
    with pytest.raises(FileNotFoundError):
        build_normalizer(
            feature_dim=REALTIME_POSE_INPUT_DIM,
            normalizer_dir=None,
            normalize_input=True,
            strict=False,
            schema_name=REALTIME_POSE_SCHEMA_NAME,
        )


def test_runtime_assets_default_to_body_fbx_local(tmp_path):
    schema_spec = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    assets = write_runtime_assets(output_dir=tmp_path, normalize_input=False)
    schema = json.loads(assets["feature_schema"].read_text(encoding="utf-8"))
    assert schema["schemaName"] == REALTIME_POSE_SCHEMA_NAME
    assert schema["poseRepresentation"] == schema_spec.pose_representation
    assert schema["featureDim"] == schema_spec.feature_dim
    assert schema["targetFeatureLength"] == schema_spec.target_dim
    assert schema["rootDeltaXZReference"]["start"] == schema_spec.root_delta_xz_start
    assert schema["pelvisHeight"]["start"] == schema_spec.root_height_start
    assert schema["stationaryProb5"]["start"] == schema_spec.stationary_prob_start


def test_sentis_export_rejects_normalized_checkpoint_without_normalizer():
    args = SimpleNamespace(normalize_input=True, normalizer_dir="")
    with pytest.raises(FileNotFoundError):
        validate_normalizer_export_contract(args, {"normalize_input": True})
