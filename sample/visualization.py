from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_np,
    pelvis_relative_joint_positions_np,
)
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    ROOT_YAW_RELATIVE_DIM,
    ROOT_YAW_RELATIVE_START,
)


def decode_realtime_pose_joints(
    features: np.ndarray,
    root_pos_world: np.ndarray,
    root_yaw: np.ndarray,
    joint_offsets_parent: np.ndarray,
    joint_rest_local_rotations_6d: np.ndarray | None = None,
) -> np.ndarray:
    """把 `[T,140]` Head-yaw 参考系姿态还原成 `[T,24,3]` 世界坐标关节。"""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError(f"features 应为 [T,{REALTIME_POSE_TARGET_DIM}]，实际为 {features.shape}")
    root_pos = np.asarray(root_pos_world, dtype=np.float32)
    yaw = np.asarray(root_yaw, dtype=np.float32)
    offsets = np.asarray(joint_offsets_parent, dtype=np.float32)
    if root_pos.shape != (features.shape[0], 3):
        raise ValueError(f"root_pos_world 应为 [T,3]，实际为 {root_pos.shape}")
    if yaw.shape != (features.shape[0],):
        raise ValueError(f"root_yaw 应为 [T]，实际为 {yaw.shape}")
    if offsets.shape != (24, 3):
        raise ValueError(f"joint_offsets_parent 应为 [24,3]，实际为 {offsets.shape}")

    if joint_rest_local_rotations_6d is None:
        rest_rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 24, axis=0)
    else:
        rest_rotations = rotation_6d_to_matrix_np(joint_rest_local_rotations_6d)

    rotations_head, _ = decode_target_head_rotations_np(features, rest_rotations)
    relative_yaw = np.arctan2(
        features[:, ROOT_YAW_RELATIVE_START],
        features[:, ROOT_YAW_RELATIVE_START + 1],
    )
    head_to_world = make_yaw_rotation_np(yaw + relative_yaw)
    rotations_world = np.einsum("tij,tajk->taik", head_to_world, rotations_head)
    joints_relative = pelvis_relative_joint_positions_np(rotations_world, offsets)
    pelvis_offset = np.einsum("tij,j->ti", make_yaw_rotation_np(yaw), offsets[0])
    return (joints_relative + root_pos[:, None] + pelvis_offset[:, None]).astype(np.float32)


def decode_root_yaw_from_relative(
    current_head_yaw: float,
    root_yaw_relative_sincos: np.ndarray,
) -> float:
    relative = np.asarray(root_yaw_relative_sincos, dtype=np.float32)
    if relative.shape != (ROOT_YAW_RELATIVE_DIM,):
        raise ValueError(
            f"root_yaw_relative_sincos 应为 [{ROOT_YAW_RELATIVE_DIM}]，实际为 {relative.shape}"
        )
    norm = max(float(np.linalg.norm(relative)), 1e-8)
    sin_relative, cos_relative = relative / norm
    return float(current_head_yaw - np.arctan2(sin_relative, cos_relative))


def extract_target_yaw_relative(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.shape[-1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError(
            f"features 最后一维应为 {REALTIME_POSE_TARGET_DIM}，实际为 {features.shape[-1]}"
        )
    return features[
        ..., ROOT_YAW_RELATIVE_START : ROOT_YAW_RELATIVE_START + ROOT_YAW_RELATIVE_DIM
    ]
