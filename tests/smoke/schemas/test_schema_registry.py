from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from schemas.registry import get_default_schema_name, get_schema_adapter, get_schema_spec, list_schema_names


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
