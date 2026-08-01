from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    TRACKER_JOINT_INDICES,
    derive_stationary_prob_5,
    encode_root_delta_xz_ref,
    rotation_6d_forward_up_np,
)
from data_loaders.body_fbx_kinematics import build_synthetic_body_fbx_rest, fk_body_fbx_local_delta_root_y0
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    STATIONARY_JOINT_INDICES,
    STATIONARY_JOINT_NAMES,
)


IDENTITY_6D = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)


def build_toy_source_metadata(frame_count: int = 70) -> dict:
    metadata = {
        "status": "converted",
        "source_relative_path": "ACCAD/toy_realtime.npz",
        "stablemotion_split_key": "ACCAD/toy_realtime",
        "output_path": "ACCAD/toy_realtime.npz",
        "frames": int(frame_count),
        "target_fps": 60.0,
        "is_mirrored": False,
    }
    metadata["stationary_joint_indices"] = [int(index) for index in STATIONARY_JOINT_INDICES]
    metadata["stationary_joint_names"] = list(STATIONARY_JOINT_NAMES)
    return metadata


def build_toy_realtime_source(frame_count: int = 70) -> dict[str, np.ndarray]:
    body_pose = np.tile(IDENTITY_6D, (frame_count, 24)).astype(np.float32)
    root_pos = np.zeros((frame_count, 3), dtype=np.float32)
    root_pos[:, 0] = np.linspace(0.0, 0.3, frame_count, dtype=np.float32)
    root_yaw = np.linspace(0.0, 0.2, frame_count, dtype=np.float32)
    yaw_delta = np.zeros((frame_count,), dtype=np.float32)
    yaw_delta[1:] = root_yaw[1:] - root_yaw[:-1]
    root_heading_delta_sincos = np.stack([np.sin(yaw_delta), np.cos(yaw_delta)], axis=-1).astype(np.float32)

    rest = build_synthetic_body_fbx_rest()
    pelvis_height = np.full((frame_count, 1), 0.9, dtype=np.float32)
    joints, joint_rotations = fk_body_fbx_local_delta_root_y0(
        body_pose_local_delta_6d=body_pose,
        actor_root_pos_world=root_pos,
        root_heading=root_yaw,
        pelvis_height=pelvis_height,
        rest=rest,
    )
    tracker_pos = joints[:, TRACKER_JOINT_INDICES].copy()
    tracker_rot = rotation_6d_forward_up_np(joint_rotations[:, TRACKER_JOINT_INDICES]).astype(np.float32)
    offsets = rest.rest_local_positions.copy()
    rest_rotations_6d = rotation_6d_forward_up_np(rest.rest_local_rotations).astype(np.float32)
    root_delta_xz_ref = encode_root_delta_xz_ref(root_pos_world=root_pos, root_yaw=root_yaw)
    pelvis_height = joints[:, 0, 1:2].astype(np.float32)
    source = {
        BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY: body_pose,
        "root_pos_world": root_pos,
        "root_yaw": root_yaw,
        "root_heading_delta_sincos": root_heading_delta_sincos,
        "root_delta_xz_ref": root_delta_xz_ref,
        "pelvis_height": pelvis_height,
        "tracker_pos_world": tracker_pos,
        "tracker_rot_world_6d": tracker_rot,
        "joints_world": joints,
        "joint_offsets_parent": offsets,
    }
    source["stationary_prob_5"] = derive_stationary_prob_5(joints_world=joints)
    source["joint_rest_local_rotations_6d"] = rest_rotations_6d
    source["metadata"] = np.asarray(json.dumps(build_toy_source_metadata(frame_count=frame_count)))
    return source


def write_toy_source_dataset(source_dir: Path, frame_count: int = 70) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    source = build_toy_realtime_source(frame_count=frame_count)
    source_path = source_dir / "ACCAD" / "toy_realtime.npz"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(source_path, **source)
    manifest_entry = build_toy_source_metadata(frame_count=frame_count)
    with (source_dir / "manifest.jsonl").open("w", encoding="utf-8") as file:
        file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
    return source_path
