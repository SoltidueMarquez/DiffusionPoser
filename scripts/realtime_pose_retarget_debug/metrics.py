from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_kinematics import (
    SMPL_PARENTS,
    fk_root_global_torch,
    rotation_6d_to_matrix_np,
)
from data_loaders.sensor_masking import REALTIME_POSE_V2_CONTACT_SCHEMA_NAME, SMPL_JOINT_COUNT, get_schema_spec

from .replay_io import ReplayArrays


def compute_fk_joints(
    target_features_raw: np.ndarray,
    root_pos_world: np.ndarray,
    root_yaw: np.ndarray,
    joint_offsets_parent: np.ndarray,
    use_root_yaw: bool = True,
    use_root_pos: bool = True,
) -> np.ndarray:
    """从 raw target feature 做 `[T,24,3]` FK，用于和 Unity/JSON joints 对齐。"""

    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    features = np.asarray(target_features_raw, dtype=np.float32)
    roots = np.asarray(root_pos_world, dtype=np.float32).copy()
    yaw = np.asarray(root_yaw, dtype=np.float32).copy()
    if features.ndim != 2 or features.shape[1] != schema.target_dim:
        raise ValueError(f"target_features_raw must be [T,{schema.target_dim}], got {features.shape}")
    if roots.shape != (features.shape[0], 3):
        raise ValueError(f"root_pos_world must be [T,3], got {roots.shape}")
    if yaw.shape != (features.shape[0],):
        raise ValueError(f"root_yaw must be [T], got {yaw.shape}")

    offsets = np.repeat(np.asarray(joint_offsets_parent, dtype=np.float32)[None], features.shape[0], axis=0)
    if offsets.shape != (features.shape[0], SMPL_JOINT_COUNT, 3):
        raise ValueError(f"joint offsets must broadcast to [T,24,3], got {offsets.shape}")

    # root_pos_world 只承载地面 XZ；pelvis 高度来自 root_height。
    roots[:, 1] = 0.0
    offsets[:, 0, 1] = features[:, schema.root_height_slice()].reshape(-1)
    if not use_root_pos:
        roots[:] = 0.0
    if not use_root_yaw:
        yaw[:] = 0.0

    with torch.no_grad():
        joints = fk_root_global_torch(
            body_pose_root_global_6d=torch.from_numpy(features[:, schema.body_pose_slice()]).float(),
            root_pos_world=torch.from_numpy(roots).float(),
            root_yaw=torch.from_numpy(yaw).float(),
            parent_offsets=torch.from_numpy(offsets).float(),
        )
    return joints.cpu().numpy().astype(np.float32)


def error_stats(errors_m: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors_m, dtype=np.float64)
    return {
        "mean_m": float(values.mean()),
        "mean_cm": float(values.mean() * 100.0),
        "max_m": float(values.max()),
        "max_cm": float(values.max() * 100.0),
        "p95_m": float(np.percentile(values, 95)),
        "p95_cm": float(np.percentile(values, 95) * 100.0),
    }


def compute_joint_errors(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if predicted.shape != reference.shape:
        raise ValueError(f"joint shape mismatch: predicted={predicted.shape}, reference={reference.shape}")
    return np.linalg.norm(np.asarray(predicted) - np.asarray(reference), axis=-1)


def compute_bone_direction_angles_degrees(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if predicted.shape != reference.shape:
        raise ValueError(f"joint shape mismatch: predicted={predicted.shape}, reference={reference.shape}")
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    angles = np.zeros(predicted.shape[:2], dtype=np.float64)
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            continue
        pred_vec = predicted[:, joint_index] - predicted[:, parent_index]
        ref_vec = reference[:, joint_index] - reference[:, parent_index]
        pred_norm = np.linalg.norm(pred_vec, axis=-1)
        ref_norm = np.linalg.norm(ref_vec, axis=-1)
        valid = (pred_norm > 1e-8) & (ref_norm > 1e-8)
        if not np.any(valid):
            continue
        dot = np.sum(pred_vec[valid] * ref_vec[valid], axis=-1) / np.maximum(pred_norm[valid] * ref_norm[valid], 1e-8)
        angles[valid, joint_index] = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return angles


def angle_stats(angles_degrees: np.ndarray) -> dict[str, float]:
    values = np.asarray(angles_degrees, dtype=np.float64)[:, 1:].reshape(-1)
    return {
        "mean_deg": float(values.mean()),
        "max_deg": float(values.max()),
        "p95_deg": float(np.percentile(values, 95)),
    }


def compute_6d_validity(target_features_raw: np.ndarray) -> dict[str, float]:
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    pose = np.asarray(target_features_raw, dtype=np.float32)[:, schema.body_pose_slice()].reshape(-1, SMPL_JOINT_COUNT, 6)
    forward = pose[..., 0:3]
    up = pose[..., 3:6]
    forward_norm = np.linalg.norm(forward, axis=-1)
    up_norm = np.linalg.norm(up, axis=-1)
    dot = np.sum(forward * up, axis=-1) / np.maximum(forward_norm * up_norm, 1e-8)
    rotations = rotation_6d_to_matrix_np(pose)
    det = np.linalg.det(rotations)
    return {
        "forward_norm_min": float(forward_norm.min()),
        "forward_norm_max": float(forward_norm.max()),
        "up_norm_min": float(up_norm.min()),
        "up_norm_max": float(up_norm.max()),
        "max_abs_forward_up_dot": float(np.abs(dot).max()),
        "determinant_min": float(det.min()),
        "determinant_max": float(det.max()),
    }


def compute_root_variant_stats(replay: ReplayArrays, offsets: np.ndarray) -> dict[str, dict[str, float]]:
    variants = {
        "full_root_yaw_and_pos": (True, True),
        "root_pos_only_no_yaw": (False, True),
        "root_yaw_only_no_pos": (True, False),
        "no_root_yaw_no_pos": (False, False),
    }
    stats: dict[str, dict[str, float]] = {}
    for name, (use_yaw, use_pos) in variants.items():
        joints = compute_fk_joints(
            target_features_raw=replay.target_features_raw,
            root_pos_world=replay.root_pos_world,
            root_yaw=replay.root_yaw,
            joint_offsets_parent=offsets,
            use_root_yaw=use_yaw,
            use_root_pos=use_pos,
        )
        stats[name] = error_stats(compute_joint_errors(joints, replay.reference_joints_world))
    return stats
