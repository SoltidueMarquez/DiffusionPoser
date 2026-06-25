from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from schemas.registry import get_schema_adapter, get_schema_spec


CANONICAL_SCHEMA_NAME = "realtime_pose_stationary5_v1"
LEGACY_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"


def test_stationary5_and_legacy_specs_keep_expected_identity():
    canonical = get_schema_spec(CANONICAL_SCHEMA_NAME)
    legacy = get_schema_spec(LEGACY_SCHEMA_NAME)

    assert canonical.name == CANONICAL_SCHEMA_NAME
    assert canonical.canonical_name == CANONICAL_SCHEMA_NAME
    assert legacy.name == LEGACY_SCHEMA_NAME
    assert legacy.canonical_name == CANONICAL_SCHEMA_NAME
    assert canonical.feature_dim == 214
    assert canonical.target_dim == 154
    assert legacy.feature_dim == 214
    assert legacy.target_dim == 154


def test_stationary5_channel_slices_match_contract():
    for schema_name in (CANONICAL_SCHEMA_NAME, LEGACY_SCHEMA_NAME):
        spec = get_schema_spec(schema_name)

        assert spec.body_pose_slice() == slice(0, 144)
        assert spec.root_heading_delta_slice() == slice(144, 146)
        assert spec.root_delta_xz_slice() == slice(146, 148)
        assert spec.pelvis_height_slice() == slice(148, 149)
        assert spec.stationary_prob_slice() == slice(149, 154)
        assert spec.tracker_pos_slice() == slice(154, 172)
        assert spec.tracker_rot_slice() == slice(172, 208)
        assert spec.sensor_valid_slice() == slice(208, 214)


def test_stationary5_inpaint_mask_only_covers_target_frame_target_channels():
    adapter = get_schema_adapter(CANONICAL_SCHEMA_NAME)
    mask = adapter.build_inpaint_mask()

    assert mask.shape == (61, 214)
    assert mask.dtype == bool
    assert int(mask.sum()) == 154
    assert mask[60, 0:154].all()
    assert not mask[:60, :].any()
    assert not mask[60, 154:].any()


def test_stationary5_unity_feature_schema_matches_runtime_contract():
    for schema_name in (CANONICAL_SCHEMA_NAME, LEGACY_SCHEMA_NAME):
        adapter = get_schema_adapter(schema_name)
        unity_schema = adapter.build_unity_feature_schema()

        assert unity_schema["schemaName"] == adapter.spec.name
        assert unity_schema["featureDim"] == 214
        assert unity_schema["sequenceLength"] == 61
        assert unity_schema["targetFeatureLength"] == 154
        assert unity_schema["runtimeRules"]["rootPositionY"] == "fixed_zero"
        assert unity_schema["runtimeRules"]["pelvisHeightApplication"] == "pelvis_local_offset_y"
        assert unity_schema["stationaryProb5"]["length"] == 5


def test_stationary5_readme_first_line_documents_contract():
    readme_path = Path(__file__).resolve().parents[3] / "schemas" / "realtime_pose_stationary5_v1" / "README.md"

    first_line = readme_path.read_text(encoding="utf-8").splitlines()[0]

    assert (
        first_line
        == "realtime_pose_stationary5_v1：固定 body_fbx_local + root_y0 前提下，使用 61 帧窗口、214 维特征和 stationary_prob_5 的实时姿态重建契约。"
    )
