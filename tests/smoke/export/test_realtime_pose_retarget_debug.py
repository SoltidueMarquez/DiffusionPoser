from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES
from data_loaders.realtime_pose_kinematics import SMPL_PARENTS
from data_loaders.sensor_masking import (
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    SMPL_JOINT_COUNT,
    get_schema_spec,
)
from scripts.debug_realtime_pose_retarget import (
    IDENTITY_6D,
    build_debug_report,
    compute_fk_joints,
    parse_body_fbx_offsets_from_meta,
    write_report,
)
from data_converter.amass_smpl_utils import AMASS_TO_UNITY
from scripts.realtime_pose_retarget_debug import source_rest


# DEBUG_RETARGET_PROBE: 覆盖离线 retarget 诊断脚本的最小可运行路径。


def test_retarget_debug_report_classifies_body_fbx_offset_mismatch(tmp_path: Path):
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    frame_count = 4
    source_offsets = build_source_offsets()
    features = np.zeros((frame_count, schema.target_dim), dtype=np.float32)
    features[:, schema.body_pose_slice()] = np.tile(IDENTITY_6D, (frame_count, len(SMPL_JOINT_NAMES)))
    features[:, schema.root_yaw_delta_slice()] = np.asarray([0.0, 1.0], dtype=np.float32)
    features[:, schema.root_height_slice()] = source_offsets[0, 1]

    root_pos = np.zeros((frame_count, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(0.0, 0.3, frame_count, dtype=np.float32)
    root_yaw = np.linspace(0.0, 0.2, frame_count, dtype=np.float32)
    joints = compute_fk_joints(
        target_features_raw=features,
        root_pos_world=root_pos,
        root_yaw=root_yaw,
        joint_offsets_parent=source_offsets,
    )

    source_npz = tmp_path / "source.npz"
    np.savez(source_npz, joint_offsets_parent=source_offsets, **{POSE_REPRESENTATION_KEY: np.asarray(schema.pose_representation)})
    replay_json = tmp_path / "replay.json"
    replay_json.write_text(
        json.dumps(
            {
                "schemaName": REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
                "poseRepresentation": schema.pose_representation,
                "frameCount": frame_count,
                "targetFeatureLength": schema.target_dim,
                "targetFeaturesRaw": features.reshape(-1).astype(float).tolist(),
                "referenceJointsWorld": joints.reshape(-1).astype(float).tolist(),
                "rootYaw": root_yaw.astype(float).tolist(),
                "rootPosWorld": root_pos.reshape(-1).astype(float).tolist(),
                "metadata": {"sourcePath": str(source_npz)},
            }
        ),
        encoding="utf-8",
    )
    body_meta = tmp_path / "body.fbx.meta"
    write_body_meta(body_meta, source_offsets * 1.35)

    parsed_offsets = parse_body_fbx_offsets_from_meta(body_meta)
    np.testing.assert_allclose(parsed_offsets, source_offsets * 1.35)

    report = build_debug_report(replay_json=replay_json, body_fbx_meta=body_meta)
    assert report["classification"]["sourceRoundtripOk"] is True
    assert report["classification"]["bodyFbxOffsetsFail"] is True
    assert report["sourceRoundtrip"]["mean_m"] < 1e-5
    assert report["bodyFbxOffsetReplay"]["mean_m"] > 0.05
    assert report["sourceBoneDirectionAngles"]["max_deg"] < 1e-4
    assert "bodyFbxBoneDirectionAngles" in report

    paths = write_report(report, output_dir=tmp_path / "debug_out")
    assert paths["summary"].exists()
    assert paths["per_joint_errors"].exists()
    assert "synthetic_probe_replay" not in paths


def test_build_tpose_rest_local_offsets_grounded_y_up():
    rest_joints = build_zero_pose_tpose_joints()
    rest_vertices = np.asarray([[0.0, -0.02, 0.0], [-0.2, 0.02, 0.08], [0.2, 0.02, 0.08]], dtype=np.float32)

    offsets = source_rest.build_tpose_rest_local_offsets(rest_joints=rest_joints, rest_vertices=rest_vertices)

    assert offsets.shape == (SMPL_JOINT_COUNT, 3)
    np.testing.assert_allclose(offsets[0], [0.0, 1.02, 0.0], atol=1e-6)
    for joint_index in range(1, SMPL_JOINT_COUNT):
        parent_index = int(SMPL_PARENTS[joint_index])
        np.testing.assert_allclose(offsets[joint_index], rest_joints[joint_index] - rest_joints[parent_index])

    left_wrist = SMPL_JOINT_NAMES.index("left_wrist")
    right_wrist = SMPL_JOINT_NAMES.index("right_wrist")
    left_foot = SMPL_JOINT_NAMES.index("left_foot")
    right_foot = SMPL_JOINT_NAMES.index("right_foot")
    neck = SMPL_JOINT_NAMES.index("neck")
    assert offsets[left_wrist, 0] < 0.0
    assert offsets[right_wrist, 0] > 0.0
    assert offsets[left_foot, 2] > 0.0
    assert offsets[right_foot, 2] > 0.0
    assert rest_joints[neck, 1] > rest_joints[0, 1]


def test_export_source_rest_pose_json_uses_smpl_zero_pose_tpose(tmp_path: Path, monkeypatch):
    source_offsets = build_source_offsets()
    source_npz = tmp_path / "source.npz"
    amass_path = tmp_path / "raw_amass.npz"
    smpl_model_dir = tmp_path / "body_models"
    smpl_model_dir.mkdir()
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    metadata = {
        "schema_name": schema.name,
        "pose_representation": schema.pose_representation,
        "source_path": str(amass_path),
        "is_mirrored": True,
    }
    np.savez(
        source_npz,
        joint_offsets_parent=source_offsets,
        metadata=json.dumps(metadata, ensure_ascii=False),
        **{POSE_REPRESENTATION_KEY: np.asarray(schema.pose_representation)},
    )
    np.savez(amass_path, betas=np.zeros(16, dtype=np.float32), gender=np.asarray("female"))

    rest_joints = build_zero_pose_tpose_joints()
    rest_vertices = np.asarray([[0.0, -0.02, 0.0], [-0.2, 0.02, 0.08], [0.2, 0.02, 0.08]], dtype=np.float32)

    def fake_load_smpl_zero_pose_joints(*, amass_path: Path, smpl_model_dir: Path):
        assert amass_path == amass_path_expected
        assert smpl_model_dir == smpl_model_dir_expected
        return rest_joints, rest_vertices

    amass_path_expected = amass_path.resolve()
    smpl_model_dir_expected = smpl_model_dir
    monkeypatch.setattr(source_rest, "load_smpl_zero_pose_joints", fake_load_smpl_zero_pose_joints)

    payload = source_rest.build_source_rest_pose_payload(source_npz, smpl_model_dir=smpl_model_dir)
    assert payload["debugMarker"] == "DEBUG_RETARGET_SOURCE_REST"
    assert payload["schemaName"] == REALTIME_POSE_V2_CONTACT_SCHEMA_NAME
    assert payload["poseRepresentation"] == schema.pose_representation
    assert payload["sourceAmass"] == str(amass_path.resolve())
    assert payload["bodyModelDir"] == str(smpl_model_dir.resolve())
    assert payload["restPoseSource"] == "smpl_zero_pose_tpose"
    assert payload["isMirrored"] is True
    assert payload["boneCount"] == len(SMPL_JOINT_NAMES)
    assert payload["boneNames"] == list(SMPL_JOINT_NAMES)
    assert payload["parentIndices"] == [int(value) for value in SMPL_PARENTS.tolist()]

    local_offsets = np.asarray(
        [[item["x"], item["y"], item["z"]] for item in payload["restLocalOffsets"]],
        dtype=np.float32,
    )
    expected_offsets = source_rest.build_tpose_rest_local_offsets(rest_joints=rest_joints, rest_vertices=rest_vertices)
    np.testing.assert_allclose(local_offsets, expected_offsets)
    assert not np.allclose(local_offsets, source_offsets)

    source_fk_offsets = np.asarray(
        [[item["x"], item["y"], item["z"]] for item in payload["sourceFkLocalOffsets"]],
        dtype=np.float32,
    )
    expected_fk_offsets = expected_offsets.copy()
    expected_fk_offsets[1:] = expected_fk_offsets[1:] @ AMASS_TO_UNITY.T
    np.testing.assert_allclose(source_fk_offsets, expected_fk_offsets)
    np.testing.assert_allclose(source_fk_offsets[0], expected_offsets[0])
    assert not np.allclose(source_fk_offsets[1:], expected_offsets[1:])
    for key in ("restLocalRotations", "restWorldRotations"):
        rotations = payload[key]
        assert len(rotations) == len(SMPL_JOINT_NAMES)
        for rotation in rotations:
            assert rotation == {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

    output_json = source_rest.export_source_rest_pose_json(
        source_npz=source_npz,
        output_json=tmp_path / "smpl_source_rest.json",
        smpl_model_dir=smpl_model_dir,
    )
    written = json.loads(output_json.read_text(encoding="utf-8"))
    assert written["boneNames"] == list(SMPL_JOINT_NAMES)
    assert written["restPoseSource"] == "smpl_zero_pose_tpose"


def build_source_offsets() -> np.ndarray:
    offsets = np.zeros((len(SMPL_JOINT_NAMES), 3), dtype=np.float32)
    offsets[0] = [0.0, 1.0, 0.0]
    for index in range(1, len(SMPL_JOINT_NAMES)):
        sign = -1.0 if "left" in SMPL_JOINT_NAMES[index] else 1.0
        offsets[index] = [0.03 * sign, 0.04 + 0.01 * (index % 3), 0.02 * (index % 2)]
    return offsets


def build_zero_pose_tpose_joints() -> np.ndarray:
    joints = np.zeros((len(SMPL_JOINT_NAMES), 3), dtype=np.float32)
    positions = {
        "pelvis": [0.0, 1.0, 0.0],
        "left_hip": [-0.12, 0.92, 0.0],
        "right_hip": [0.12, 0.92, 0.0],
        "spine1": [0.0, 1.12, 0.0],
        "left_knee": [-0.12, 0.52, 0.02],
        "right_knee": [0.12, 0.52, 0.02],
        "spine2": [0.0, 1.28, 0.0],
        "left_ankle": [-0.12, 0.10, 0.0],
        "right_ankle": [0.12, 0.10, 0.0],
        "spine3": [0.0, 1.43, 0.0],
        "left_foot": [-0.12, 0.02, 0.16],
        "right_foot": [0.12, 0.02, 0.16],
        "neck": [0.0, 1.56, 0.0],
        "left_collar": [-0.10, 1.50, 0.0],
        "right_collar": [0.10, 1.50, 0.0],
        "head": [0.0, 1.75, 0.0],
        "left_shoulder": [-0.28, 1.50, 0.0],
        "right_shoulder": [0.28, 1.50, 0.0],
        "left_elbow": [-0.58, 1.50, 0.0],
        "right_elbow": [0.58, 1.50, 0.0],
        "left_wrist": [-0.86, 1.50, 0.0],
        "right_wrist": [0.86, 1.50, 0.0],
        "left_hand": [-0.96, 1.50, 0.0],
        "right_hand": [0.96, 1.50, 0.0],
    }
    for joint_name, position in positions.items():
        joints[SMPL_JOINT_NAMES.index(joint_name)] = position
    return joints


def write_body_meta(path: Path, offsets: np.ndarray) -> None:
    unity_names = [
        "m_avg_Pelvis",
        "m_avg_L_Hip",
        "m_avg_R_Hip",
        "m_avg_Spine1",
        "m_avg_L_Knee",
        "m_avg_R_Knee",
        "m_avg_Spine2",
        "m_avg_L_Ankle",
        "m_avg_R_Ankle",
        "m_avg_Spine3",
        "m_avg_L_Foot",
        "m_avg_R_Foot",
        "m_avg_Neck",
        "m_avg_L_Collar",
        "m_avg_R_Collar",
        "m_avg_Head",
        "m_avg_L_Shoulder",
        "m_avg_R_Shoulder",
        "m_avg_L_Elbow",
        "m_avg_R_Elbow",
        "m_avg_L_Wrist",
        "m_avg_R_Wrist",
        "m_avg_L_Hand",
        "m_avg_R_Hand",
    ]
    lines = ["humanDescription:", "  skeleton:"]
    for name, offset in zip(unity_names, offsets, strict=True):
        lines.extend(
            [
                f"    - name: {name}",
                "      parentName: body(Clone)",
                f"      position: {{x: {offset[0]}, y: {offset[1]}, z: {offset[2]}}}",
                "      rotation: {x: 0, y: 0, z: 0, w: 1}",
                "      scale: {x: 1, y: 1, z: 1}",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")

