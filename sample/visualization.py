from __future__ import annotations

import numpy as np

from data_loaders.sensor_masking import (
    BODY_POSE_DIM,
    BODY_POSE_START,
    REALTIME_POSE_SCHEMA_NAME,
    ROOT_YAW_DELTA_DIM,
    ROOT_YAW_DELTA_START,
    get_schema_spec,
)
from sample.simulate_unity_stream import fk_joints_from_target


def decode_realtime_pose_joints(
    features: np.ndarray,
    root_pos_world: np.ndarray,
    root_yaw: np.ndarray,
    joint_offsets_parent: np.ndarray,
    joint_rest_local_rotations_6d: np.ndarray | None = None,
) -> np.ndarray:
    """
    按 registry 默认 schema 的目标 pose feature + root_yaw 做轻量 FK。
    输入特征为 `[T,C]`，输出 joints 为 `[T,24,3]`，供可视化和 smoke test 使用。
    """

    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != schema.feature_dim:
        raise ValueError(f"features 应为 [T,{schema.feature_dim}]，实际为 {features.shape}")
    root_pos = np.asarray(root_pos_world, dtype=np.float32)
    yaw = np.asarray(root_yaw, dtype=np.float32)
    if root_pos.shape != (features.shape[0], 3):
        raise ValueError(f"root_pos_world 应为 [T,3]，实际为 {root_pos.shape}")
    if yaw.shape != (features.shape[0],):
        raise ValueError(f"root_yaw 应为 [T]，实际为 {yaw.shape}")
    joints = [
        fk_joints_from_target(
            target_raw=features[frame_index],
            root_pos_world=root_pos[frame_index],
            root_yaw=float(yaw[frame_index]),
            joint_offsets_parent=joint_offsets_parent,
            schema_name=schema.name,
            joint_rest_local_rotations_6d=joint_rest_local_rotations_6d,
        )
        for frame_index in range(features.shape[0])
    ]
    return np.stack(joints, axis=0).astype(np.float32)


def decode_root_yaw_from_delta(prev_root_yaw: float, root_yaw_delta_sincos: np.ndarray) -> float:
    delta = np.asarray(root_yaw_delta_sincos, dtype=np.float32)
    if delta.shape != (ROOT_YAW_DELTA_DIM,):
        raise ValueError(f"root_yaw_delta_sincos 应为 [{ROOT_YAW_DELTA_DIM}]，实际为 {delta.shape}")
    norm = max(float(np.linalg.norm(delta)), 1e-8)
    sin_delta, cos_delta = delta / norm
    return float(prev_root_yaw + np.arctan2(sin_delta, cos_delta))


def extract_target_yaw_delta(features: np.ndarray) -> np.ndarray:
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    features = np.asarray(features, dtype=np.float32)
    if features.shape[-1] != schema.feature_dim:
        raise ValueError(f"features 最后一维应为 {schema.feature_dim}，实际为 {features.shape[-1]}")
    return features[..., ROOT_YAW_DELTA_START:ROOT_YAW_DELTA_START + ROOT_YAW_DELTA_DIM]
