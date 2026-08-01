from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_np,
    pelvis_relative_joint_positions_np,
)
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np
from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM


def decode_realtime_pose_joints(
    features: np.ndarray,
    root_pos_world: np.ndarray,
    root_yaw: np.ndarray,
    joint_offsets_parent: np.ndarray,
) -> np.ndarray:
    """把 `[T,144]` Head-yaw 参考系姿态还原成 `[T,24,3]` 世界坐标关节。"""

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

    rotations_head, pelvis_heading_head = decode_target_head_rotations_np(features)
    # 已知 actor root world heading 时，可由 pelvis 的 Head-frame heading 反解当前 Head yaw。
    head_to_world = make_yaw_rotation_np(yaw - pelvis_heading_head)
    rotations_world = np.einsum("tij,tajk->taik", head_to_world, rotations_head)
    joints_relative = pelvis_relative_joint_positions_np(rotations_world, offsets)
    pelvis_offset = np.einsum("tij,j->ti", make_yaw_rotation_np(yaw), offsets[0])
    return (joints_relative + root_pos[:, None] + pelvis_offset[:, None]).astype(np.float32)
