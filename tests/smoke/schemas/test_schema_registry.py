from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from schemas.registry import (
    get_default_schema_name,
    get_schema_adapter,
    get_schema_spec,
    list_schema_names,
    register_schema,
)


def test_stationary5_default_schema_registered():
    assert get_default_schema_name() == "realtime_pose_stationary5_v1"
    assert "realtime_pose_stationary5_v1" in list_schema_names(trainable_only=True)
    assert "realtime_pose_body_fbx_local_root_y0_v1" in list_schema_names(trainable_only=True)


def test_legacy_schema_name_keeps_exact_identity():
    spec = get_schema_spec("realtime_pose_body_fbx_local_root_y0_v1")
    assert spec.name == "realtime_pose_body_fbx_local_root_y0_v1"
    assert spec.canonical_name == "realtime_pose_stationary5_v1"
    assert spec.feature_dim == 214


def test_adapter_round_trip_lookup():
    adapter = get_schema_adapter("realtime_pose_stationary5_v1")
    assert adapter.spec.name == "realtime_pose_stationary5_v1"
    assert adapter.spec.pose_representation == "body_fbx_local_delta_6d"


def test_stationary5_channel_slices_match_contract():
    for schema_name in ("realtime_pose_stationary5_v1", "realtime_pose_body_fbx_local_root_y0_v1"):
        spec = get_schema_spec(schema_name)
        assert spec.target_slice() == slice(0, 154)
        assert spec.tracker_pos_slice() == slice(154, 172)
        assert spec.tracker_rot_slice() == slice(172, 208)
        assert spec.sensor_valid_slice() == slice(208, 214)


def test_stationary5_inpaint_mask_only_covers_target_frame():
    adapter = get_schema_adapter("realtime_pose_stationary5_v1")
    mask = adapter.build_inpaint_mask()

    assert mask.shape == (61, 214)
    assert mask.dtype == bool
    assert int(mask.sum()) == 154
    assert mask[60, 0:154].all()


def test_stationary5_inpaint_mask_rejects_invalid_seq_len():
    adapter = get_schema_adapter("realtime_pose_stationary5_v1")

    with pytest.raises(ValueError):
        adapter.build_inpaint_mask(62)
    with pytest.raises(ValueError):
        adapter.build_inpaint_mask(0)


def test_missing_schema_name_raises_value_error():
    with pytest.raises(ValueError):
        get_schema_spec("missing")


def test_duplicate_schema_registration_raises_value_error():
    adapter = get_schema_adapter("realtime_pose_stationary5_v1")

    with pytest.raises(ValueError):
        register_schema(adapter)
