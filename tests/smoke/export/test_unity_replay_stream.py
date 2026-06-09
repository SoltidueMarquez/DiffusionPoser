from __future__ import annotations

import json

import numpy as np
import pytest

from data_loaders.sensor_masking import (
    BODY_POSE_DIM,
    FOOT_CONTACT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    ROOT_DELTA_XZ_DIM,
    ROOT_HEIGHT_DIM,
    ROOT_YAW_DELTA_DIM,
    TRACKER_COUNT,
    get_schema_spec,
)
from export.write_unity_replay_stream import (
    build_identity_body_pose_6d,
    build_identity_tracker_rotations_6d,
    build_unity_replay_stream_payload,
    compute_identity_6d_reference_joints,
    write_unity_replay_stream,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def write_source(path, frame_count: int = 8) -> dict[str, np.ndarray]:
    source = build_toy_realtime_source(frame_count=frame_count)
    sensor_valid = np.ones((frame_count, TRACKER_COUNT), dtype=bool)
    sensor_valid[2, 1] = False
    source["sensor_valid"] = sensor_valid
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **source)
    return source


def test_unity_replay_stream_exports_frame_major_flat_json(tmp_path):
    source_path = tmp_path / "toy_source.npz"
    source = write_source(source_path, frame_count=8)

    payload = build_unity_replay_stream_payload(
        source_npz=source_path,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
        fps=60.0,
        frame_start=1,
        frame_count=4,
    )

    assert payload["schemaName"] == REALTIME_POSE_SCHEMA_NAME
    assert payload["poseRepresentation"] == get_schema_spec(REALTIME_POSE_SCHEMA_NAME).pose_representation
    assert payload["frameStart"] == 1
    assert payload["frameCount"] == 4
    assert payload["trackerCount"] == TRACKER_COUNT
    assert len(payload["trackerPositions"]) == 4 * TRACKER_COUNT * 3
    assert len(payload["trackerRotations6d"]) == 4 * TRACKER_COUNT * 6
    assert len(payload["sensorValid"]) == 4 * TRACKER_COUNT
    assert len(payload["referenceJointsWorld"]) == 4 * 24 * 3
    assert len(payload["rootHeading"]) == 4
    assert len(payload["rootPosWorld"]) == 4 * 3
    assert len(payload["footContact"]) == 4 * 2
    assert payload["targetFeatureLength"] == 151
    assert len(payload["targetFeaturesRaw"]) == 4 * 151
    assert payload["metadata"]["targetFeaturesRawShape"] == [4, 151]
    assert payload["metadata"]["poseRepresentation"] == payload["poseRepresentation"]

    positions = np.asarray(payload["trackerPositions"], dtype=np.float32).reshape(4, TRACKER_COUNT, 3)
    rotations = np.asarray(payload["trackerRotations6d"], dtype=np.float32).reshape(4, TRACKER_COUNT, 6)
    valid = np.asarray(payload["sensorValid"], dtype=np.int32).reshape(4, TRACKER_COUNT)
    joints = np.asarray(payload["referenceJointsWorld"], dtype=np.float32).reshape(4, 24, 3)
    target_features = np.asarray(payload["targetFeaturesRaw"], dtype=np.float32).reshape(4, 151)
    np.testing.assert_allclose(positions, source["tracker_pos_world"][1:5])
    np.testing.assert_allclose(rotations, source["tracker_rot_world_6d"][1:5])
    np.testing.assert_array_equal(valid, source["sensor_valid"][1:5].astype(np.int32))
    np.testing.assert_allclose(joints, source["joints_world"][1:5])

    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    np.testing.assert_allclose(target_features[:, 0:BODY_POSE_DIM], source[schema.body_pose_key][1:5])
    cursor = BODY_POSE_DIM
    np.testing.assert_allclose(
        target_features[:, cursor : cursor + ROOT_YAW_DELTA_DIM],
        source[schema.root_heading_delta_key][1:5],
    )
    cursor += ROOT_YAW_DELTA_DIM
    np.testing.assert_allclose(
        target_features[:, cursor : cursor + ROOT_DELTA_XZ_DIM],
        source["root_delta_xz_ref"][1:5],
    )
    cursor += ROOT_DELTA_XZ_DIM
    np.testing.assert_allclose(
        target_features[:, cursor : cursor + ROOT_HEIGHT_DIM],
        source[schema.pelvis_height_key][1:5],
    )
    cursor += ROOT_HEIGHT_DIM
    np.testing.assert_allclose(
        target_features[:, cursor : cursor + FOOT_CONTACT_DIM],
        source["foot_contact"][1:5],
    )
    assert cursor + FOOT_CONTACT_DIM == schema.target_dim


def test_unity_replay_stream_writes_json_file(tmp_path):
    source_path = tmp_path / "toy_source.npz"
    write_source(source_path, frame_count=6)
    output_path = tmp_path / "unity" / "toy_replay.json"

    written = write_unity_replay_stream(source_path, output_path, frame_count=3)

    assert written == output_path.resolve()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["frameCount"] == 3
    assert payload["metadata"]["layout"] == "frame-major-flat"


def test_unity_replay_stream_can_export_identity_6d_debug_resource(tmp_path):
    source_path = tmp_path / "toy_source.npz"
    source = write_source(source_path, frame_count=5)
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)

    # 先把源里的 rotation 和 joints 改成非 identity，确保 debug 开关真的覆盖它们。
    source[schema.body_pose_key] = np.full((5, BODY_POSE_DIM), 0.25, dtype=np.float32)
    source["tracker_rot_world_6d"] = np.full((5, TRACKER_COUNT, 6), -0.5, dtype=np.float32)
    source["joints_world"] = source["joints_world"] + np.asarray([10.0, 0.0, -5.0], dtype=np.float32)
    np.savez(source_path, **source)

    payload = build_unity_replay_stream_payload(
        source_npz=source_path,
        frame_start=1,
        frame_count=3,
        identity_6d_rotations=True,
    )

    target_features = np.asarray(payload["targetFeaturesRaw"], dtype=np.float32).reshape(3, schema.target_dim)
    tracker_rotations = np.asarray(payload["trackerRotations6d"], dtype=np.float32).reshape(3, TRACKER_COUNT, 6)
    joints = np.asarray(payload["referenceJointsWorld"], dtype=np.float32).reshape(3, 24, 3)
    expected_joints = compute_identity_6d_reference_joints(
        root_pos_world=source["root_pos_world"][1:4],
        root_yaw=source["root_yaw"][1:4],
        joint_offsets_parent=source["joint_offsets_parent"],
    )

    np.testing.assert_allclose(target_features[:, 0:BODY_POSE_DIM], build_identity_body_pose_6d(3))
    np.testing.assert_allclose(tracker_rotations, build_identity_tracker_rotations_6d(3))
    np.testing.assert_allclose(joints, expected_joints)
    assert payload["metadata"]["debugIdentity6dRotations"] is True
    assert payload["metadata"]["debugReferenceJointsWorld"] == "identity_6d_fk_joint_offsets_parent"


def test_unity_replay_stream_rejects_invalid_frame_range(tmp_path):
    source_path = tmp_path / "toy_source.npz"
    write_source(source_path, frame_count=4)

    with pytest.raises(ValueError):
        build_unity_replay_stream_payload(source_npz=source_path, frame_start=3, frame_count=2)
