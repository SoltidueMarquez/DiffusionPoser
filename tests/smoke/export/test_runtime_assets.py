from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from data_loaders.sensor_masking import REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_SEQ_LEN
from export.export_sentis_denoiser import validate_normalizer_export_contract
from export.write_unity_runtime_assets import build_normalizer, write_runtime_assets


def test_runtime_assets_are_realtime_pose_only(tmp_path):
    assets = write_runtime_assets(output_dir=tmp_path, normalize_input=False)
    schema = json.loads(assets["feature_schema"].read_text(encoding="utf-8"))
    assert schema["schemaName"] == REALTIME_POSE_SCHEMA_NAME
    assert schema["featureDim"] == REALTIME_POSE_INPUT_DIM
    assert schema["sequenceLength"] == REALTIME_POSE_SEQ_LEN
    assert schema["targetStart"] == 60
    assert schema["targetLength"] == 1
    assert schema["targetFeatureLength"] == 146
    assert schema["bodyPoseParent6d"] == {"name": "body_pose_parent_6d", "start": 0, "length": 144}
    assert schema["rootYawDeltaSinCos"] == {"name": "root_yaw_delta_sincos", "start": 144, "length": 2}
    assert schema["trackerPositionReference"] == {"name": "tracker_pos_ref", "start": 146, "length": 18}
    assert schema["trackerRotation6dReference"] == {"name": "tracker_rot_ref_6d", "start": 164, "length": 36}
    assert schema["sensorValid"] == {"name": "sensor_valid", "start": 200, "length": 6}
    assert schema["runtimeRules"]["onnxDummyInputShape"] == [1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN]


def test_realtime_normalizer_requires_206_dimensions(tmp_path):
    normalizer_dir = tmp_path / "meta"
    normalizer_dir.mkdir()
    torch.save(torch.zeros(REALTIME_POSE_INPUT_DIM), normalizer_dir / "mean.pt")
    torch.save(torch.ones(REALTIME_POSE_INPUT_DIM), normalizer_dir / "std.pt")
    payload = build_normalizer(
        feature_dim=REALTIME_POSE_INPUT_DIM,
        normalizer_dir=normalizer_dir,
        normalize_input=True,
        strict=True,
    )
    assert payload["enabled"] is True
    assert payload["featureDim"] == REALTIME_POSE_INPUT_DIM
    assert len(payload["mean"]) == REALTIME_POSE_INPUT_DIM


def test_normalized_runtime_asset_export_requires_normalizer_dir():
    with pytest.raises(FileNotFoundError):
        build_normalizer(
            feature_dim=REALTIME_POSE_INPUT_DIM,
            normalizer_dir=None,
            normalize_input=True,
            strict=False,
        )


def test_sentis_export_rejects_normalized_checkpoint_without_normalizer():
    args = SimpleNamespace(normalize_input=True, normalizer_dir="")
    with pytest.raises(FileNotFoundError):
        validate_normalizer_export_contract(args, {"normalize_input": True})
