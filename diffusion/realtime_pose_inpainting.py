from __future__ import annotations

from dataclasses import dataclass

import torch

from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_ik import (
    DIRECT_ROTATION,
    DIRECTION_ONLY,
    POSITION_SOLVED,
    RealtimePoseIKResult,
    build_current_ik,
)
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_torch
from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_AVAILABLE_OFFSET,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
)


@dataclass(frozen=True)
class RealtimePoseInpaintingCondition:
    """由 IK 生成的 Predictor residual 方向与逐关节去噪门控。"""

    ik_residual: torch.Tensor  # [B,24,6]，normalized IK - Predictor
    ik_gap: torch.Tensor  # [B,24]，SO(3) geodesic angle，单位为弧度
    ik_confidence: torch.Tensor  # [B,24]
    denoise_strength: torch.Tensor  # [B,24]
    constraint_type: torch.Tensor  # [B,24]


def build_realtime_pose_inpainting_condition(
    ik_result: RealtimePoseIKResult,
    initial_pose_raw: torch.Tensor,
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
    config: IKInpaintingConfig,
) -> RealtimePoseInpaintingCondition:
    """比较 Predictor prior 与 IK，生成可解释的逐关节 residual 门控。"""

    cfg = config
    batch_size = initial_pose_raw.shape[0]
    if tuple(initial_pose_raw.shape) != (batch_size, REALTIME_POSE_TARGET_DIM):
        raise ValueError("initial_pose_raw 必须为 [B,144]。")
    if tuple(ik_result.pose.shape) != (
        batch_size,
        SMPL_JOINT_COUNT,
        ROTATION_6D_DIM,
    ):
        raise ValueError("ik_result.pose 必须为 [B,24,6]。")

    predictor_pose = initial_pose_raw.reshape(
        batch_size, SMPL_JOINT_COUNT, ROTATION_6D_DIM
    )
    ik_pose = ik_result.pose
    predictor_rotation = rotation_6d_to_matrix_torch(predictor_pose)
    ik_rotation = rotation_6d_to_matrix_torch(ik_pose)
    relative = predictor_rotation.transpose(-1, -2) @ ik_rotation
    cosine = (
        (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    ).clamp(-1.0, 1.0)
    ik_gap = torch.acos(cosine)

    predictor_normalized = _normalize_pose(initial_pose_raw, pose_mean, pose_scale)
    ik_normalized = _normalize_pose(
        ik_pose.reshape(batch_size, REALTIME_POSE_TARGET_DIM),
        pose_mean,
        pose_scale,
    )
    ik_residual = (ik_normalized - predictor_normalized).reshape(
        batch_size,
        SMPL_JOINT_COUNT,
        ROTATION_6D_DIM,
    )

    constraint_type = ik_result.constraint_type.long()
    support = torch.zeros_like(ik_gap)
    support = torch.where(
        constraint_type == DIRECT_ROTATION,
        torch.ones_like(support),
        support,
    )
    indirect = (constraint_type == DIRECTION_ONLY) | (
        constraint_type == POSITION_SOLVED
    )
    support = torch.where(
        indirect,
        torch.full_like(support, float(cfg.direction_support)),
        support,
    )
    gap_low = float(cfg.gap_low)
    gap_high = float(cfg.gap_high)
    normalized_gap = ((ik_gap - gap_low) / (gap_high - gap_low)).clamp(0.0, 1.0)
    correction_demand = normalized_gap.square() * (3.0 - 2.0 * normalized_gap)
    floor = float(cfg.untracked_strength)
    denoise_strength = floor + (1.0 - floor) * (
        support * ik_result.confidence * correction_demand
    )
    denoise_strength = denoise_strength.clamp(floor, 1.0)

    return RealtimePoseInpaintingCondition(
        ik_residual=ik_residual,
        ik_gap=ik_gap,
        ik_confidence=ik_result.confidence,
        denoise_strength=denoise_strength,
        constraint_type=constraint_type,
    )


def build_current_realtime_pose_conditions(
    initial_pose_raw: torch.Tensor,
    current_tracker_raw: torch.Tensor,
    joint_offsets_parent: torch.Tensor,
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
    tracker_mean: torch.Tensor | None,
    tracker_scale: torch.Tensor | None,
    config: IKInpaintingConfig,
) -> tuple[RealtimePoseIKResult, RealtimePoseInpaintingCondition, torch.Tensor]:
    """由 Predictor current 与当前 Tracker 构造 IK、门控和 Tracker K/V 几何。"""

    ik_result = build_current_ik(
        initial_pose_raw=initial_pose_raw,
        current_tracker_raw=current_tracker_raw,
        joint_offsets_parent=joint_offsets_parent,
        config=config,
    )
    condition = build_realtime_pose_inpainting_condition(
        ik_result=ik_result,
        initial_pose_raw=initial_pose_raw,
        pose_mean=pose_mean,
        pose_scale=pose_scale,
        config=config,
    )
    tracker_geometry = build_tracker_geometry_condition(
        current_tracker_raw=current_tracker_raw,
        tracker_mean=tracker_mean,
        tracker_scale=tracker_scale,
    )
    return ik_result, condition, tracker_geometry


def build_tracker_geometry_condition(
    current_tracker_raw: torch.Tensor,
    tracker_mean: torch.Tensor | None,
    tracker_scale: torch.Tensor | None,
) -> torch.Tensor:
    """返回供 cross-attention 使用的当前 Tracker normalized 9D 几何。"""

    batch_size = current_tracker_raw.shape[0]
    if tuple(current_tracker_raw.shape) != (
        batch_size,
        TRACKER_COUNT,
        TRACKER_FEATURE_DIM,
    ):
        raise ValueError("current_tracker_raw 必须为 [B,6,10]。")
    if (tracker_mean is None) != (tracker_scale is None):
        raise ValueError("Tracker mean 与 scale 必须同时提供或同时省略。")
    geometry = current_tracker_raw[..., :TRACKER_CONTINUOUS_DIM]
    if tracker_mean is not None:
        mean = tracker_mean.to(geometry)
        scale = tracker_scale.to(geometry)
        if tuple(mean.shape) != (TRACKER_COUNT, TRACKER_CONTINUOUS_DIM):
            raise ValueError("Tracker mean 必须为 [6,9]。")
        if tuple(scale.shape) != tuple(mean.shape):
            raise ValueError("Tracker scale 必须为 [6,9]。")
        geometry = (geometry - mean) / scale
    available = current_tracker_raw[..., TRACKER_AVAILABLE_OFFSET] > 0.5
    geometry = torch.where(available[..., None], geometry, torch.zeros_like(geometry))
    return geometry


def gate_realtime_pose_residual(
    residual: torch.Tensor,
    denoise_strength: torch.Tensor,
) -> torch.Tensor:
    """把 `[B,24]` 门控广播到 normalized rotation6D residual。"""

    batch_size = residual.shape[0]
    if tuple(residual.shape) != (batch_size, REALTIME_POSE_TARGET_DIM):
        raise ValueError("residual 必须为 [B,144]。")
    if tuple(denoise_strength.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("denoise_strength 必须为 [B,24]。")
    return (
        residual.reshape(batch_size, SMPL_JOINT_COUNT, ROTATION_6D_DIM)
        * denoise_strength.to(residual)[..., None]
    ).reshape_as(residual)


def _normalize_pose(
    value: torch.Tensor,
    mean: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if (mean is None) != (scale is None):
        raise ValueError("Pose mean 与 scale 必须同时提供或同时省略。")
    if mean is None:
        return value
    return (value - mean.to(value)) / scale.to(value)
