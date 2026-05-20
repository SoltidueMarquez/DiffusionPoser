from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_kinematics import fk_parent_local_torch
from data_loaders.sensor_masking import (
    BODY_POSE_DIM,
    BODY_POSE_START,
    REALTIME_POSE_INPUT_DIM,
    ROOT_YAW_DELTA_DIM,
    ROOT_YAW_DELTA_START,
)


def decode_realtime_pose_joints(
    features: np.ndarray,
    root_pos_world: np.ndarray,
    root_yaw: np.ndarray,
    joint_offsets_parent: np.ndarray,
) -> np.ndarray:
    """
    用 realtime_pose_v1 的 `body_pose_parent_6d + root_yaw` 做轻量 FK。
    输入特征为 `[T,206]`，输出 joints 为 `[T,24,3]`，供可视化和 smoke test 使用。
    """

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != REALTIME_POSE_INPUT_DIM:
        raise ValueError(f"features 应为 [T,{REALTIME_POSE_INPUT_DIM}]，实际为 {features.shape}")
    pose = torch.from_numpy(features[:, BODY_POSE_START:BODY_POSE_START + BODY_POSE_DIM]).float()
    root_pos = torch.from_numpy(np.asarray(root_pos_world, dtype=np.float32)).float()
    yaw = torch.from_numpy(np.asarray(root_yaw, dtype=np.float32)).float()
    offsets = torch.from_numpy(np.asarray(joint_offsets_parent, dtype=np.float32)).float()
    offsets = offsets.unsqueeze(0).expand(features.shape[0], -1, -1)
    with torch.no_grad():
        joints = fk_parent_local_torch(pose, root_pos, yaw, offsets)
    return joints.cpu().numpy().astype(np.float32)


def decode_root_yaw_from_delta(prev_root_yaw: float, root_yaw_delta_sincos: np.ndarray) -> float:
    delta = np.asarray(root_yaw_delta_sincos, dtype=np.float32)
    if delta.shape != (ROOT_YAW_DELTA_DIM,):
        raise ValueError(f"root_yaw_delta_sincos 应为 [{ROOT_YAW_DELTA_DIM}]，实际为 {delta.shape}")
    norm = max(float(np.linalg.norm(delta)), 1e-8)
    sin_delta, cos_delta = delta / norm
    return float(prev_root_yaw + np.arctan2(sin_delta, cos_delta))


def extract_target_yaw_delta(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.shape[-1] != REALTIME_POSE_INPUT_DIM:
        raise ValueError(f"features 最后一维应为 {REALTIME_POSE_INPUT_DIM}，实际为 {features.shape[-1]}")
    return features[..., ROOT_YAW_DELTA_START:ROOT_YAW_DELTA_START + ROOT_YAW_DELTA_DIM]
