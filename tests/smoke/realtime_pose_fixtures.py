from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    TRACKER_JOINT_INDICES,
    derive_stationary_prob_5,
    encode_root_delta_xz_ref,
)
from data_loaders.sensor_masking import (
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_SCHEMA_NAME,
    STATIONARY_JOINT_INDICES,
    STATIONARY_JOINT_NAMES,
    get_schema_spec,
)


IDENTITY_6D = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)


def build_toy_source_metadata(frame_count: int = 70, schema_name: str = REALTIME_POSE_SCHEMA_NAME) -> dict:
    schema = get_schema_spec(schema_name)
    metadata = {
        "schema_name": schema.name,
        "pose_representation": schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "status": "converted",
        "source_relative_path": "ACCAD/toy_realtime.npz",
        "stablemotion_split_key": "ACCAD/toy_realtime",
        "output_path": "ACCAD/toy_realtime.npz",
        "frames": int(frame_count),
    }
    if schema.supports_stationary_prob:
        metadata["stationary_joint_indices"] = [int(index) for index in STATIONARY_JOINT_INDICES]
        metadata["stationary_joint_names"] = list(STATIONARY_JOINT_NAMES)
    return metadata


def build_toy_realtime_source(frame_count: int = 70, schema_name: str = REALTIME_POSE_SCHEMA_NAME) -> dict[str, np.ndarray]:
    schema = get_schema_spec(schema_name)
    body_pose = np.tile(IDENTITY_6D, (frame_count, 24)).astype(np.float32)
    root_pos = np.zeros((frame_count, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(0.0, 0.3, frame_count, dtype=np.float32)
    root_yaw = np.linspace(0.0, 0.2, frame_count, dtype=np.float32)
    yaw_delta = np.zeros((frame_count,), dtype=np.float32)
    yaw_delta[1:] = root_yaw[1:] - root_yaw[:-1]
    root_heading_delta_sincos = np.stack([np.sin(yaw_delta), np.cos(yaw_delta)], axis=-1).astype(np.float32)

    joints = np.zeros((frame_count, 24, 3), dtype=np.float32)
    for frame in range(frame_count):
        joints[frame, :, 0] = root_pos[frame, 0]
        joints[frame, :, 1] = np.linspace(0.0, 1.7, 24, dtype=np.float32)
        joints[frame, :, 2] = np.linspace(0.0, 0.2, 24, dtype=np.float32)
    joints[:, JOINT_INDEX["left_foot"], 1] = 0.02
    joints[:, JOINT_INDEX["right_foot"], 1] = 0.02

    tracker_pos = joints[:, TRACKER_JOINT_INDICES].copy()
    tracker_pos[:, :, 0] += np.asarray([0.0, -0.25, 0.25, 0.0, -0.1, 0.1], dtype=np.float32)
    tracker_rot = np.tile(IDENTITY_6D, (frame_count, 6, 1)).astype(np.float32)
    offsets = np.zeros((24, 3), dtype=np.float32)
    offsets[:, 1] = 0.05
    rest_rotations_6d = np.tile(IDENTITY_6D, (24, 1)).astype(np.float32)
    root_delta_xz_ref = encode_root_delta_xz_ref(root_pos_world=root_pos, root_yaw=root_yaw)
    pelvis_height = joints[:, 0, 1:2].astype(np.float32)
    source = {
        schema.body_pose_key: body_pose,
        POSE_REPRESENTATION_KEY: np.asarray(schema.pose_representation),
        "root_pos_world": root_pos,
        "root_yaw": root_yaw,
        schema.root_heading_delta_key: root_heading_delta_sincos,
        "root_delta_xz_ref": root_delta_xz_ref,
        schema.pelvis_height_key: pelvis_height,
        "tracker_pos_world": tracker_pos,
        "tracker_rot_world_6d": tracker_rot,
        "joints_world": joints,
        "joint_offsets_parent": offsets,
    }
    if schema.supports_stationary_prob:
        source["stationary_prob_5"] = derive_stationary_prob_5(joints_world=joints)
    if schema.pose_representation == "body_fbx_local_delta_6d":
        source["joint_rest_local_rotations_6d"] = rest_rotations_6d
    source["metadata"] = np.asarray(json.dumps(build_toy_source_metadata(frame_count=frame_count, schema_name=schema.name)))
    return source


def write_toy_source_dataset(source_dir: Path, frame_count: int = 70, schema_name: str = REALTIME_POSE_SCHEMA_NAME) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    source = build_toy_realtime_source(frame_count=frame_count, schema_name=schema_name)
    source_path = source_dir / "ACCAD" / "toy_realtime.npz"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(source_path, **source)
    manifest_entry = build_toy_source_metadata(frame_count=frame_count, schema_name=schema_name)
    with (source_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
    return source_path
