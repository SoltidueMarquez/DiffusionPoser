from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_ik import RealtimePoseIKResult, build_current_ik
from data_loaders.sensor_masking import (
    CURRENT_JOINT_CONDITION_DIM,
    CURRENT_JOINT_CONSTRAINT_TYPE_COUNT,
    REALTIME_POSE_TARGET_DIM,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_AVAILABLE_OFFSET,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_TO_JOINT,
)


@dataclass(frozen=True)
class RealtimePoseInpaintingCondition:
    """单帧 IK-inpainting 条件；known noise 在整条 DDIM 轨迹中保持不变。"""

    pose: torch.Tensor  # [B,144]
    valid: torch.Tensor  # [B,24]
    release_level: torch.Tensor  # [B,24]
    known_noise: torch.Tensor  # [B,144]


def confidence_to_release_level(confidence: torch.Tensor) -> torch.Tensor:
    """高 confidence 对应更低释放阈值，因此能在更长的去噪区间保持约束。"""

    corruption = 1.0 - confidence.clamp(0.0, 1.0)
    return torch.sin(corruption * torch.pi * 0.5)


def build_realtime_pose_inpainting_condition(
    ik_result: RealtimePoseIKResult,
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
    known_noise: torch.Tensor | None = None,
) -> RealtimePoseInpaintingCondition:
    batch_size = ik_result.pose.shape[0]
    if tuple(ik_result.pose.shape) != (
        batch_size,
        SMPL_JOINT_COUNT,
        ROTATION_6D_DIM,
    ):
        raise ValueError("ik_result.pose 必须为 [B,24,6]。")
    if tuple(ik_result.updated_mask.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("ik_result.updated_mask 必须为 [B,24]。")
    pose_raw = ik_result.pose.reshape(batch_size, REALTIME_POSE_TARGET_DIM)
    if pose_mean is None or pose_scale is None:
        pose = pose_raw
    else:
        mean = pose_mean.to(pose_raw)
        scale = pose_scale.to(pose_raw)
        pose = (pose_raw - mean) / scale
    if known_noise is None:
        known_noise = torch.randn_like(pose)
    condition = RealtimePoseInpaintingCondition(
        pose=pose,
        valid=ik_result.updated_mask.bool(),
        release_level=confidence_to_release_level(ik_result.confidence),
        known_noise=known_noise,
    )
    validate_realtime_pose_inpainting_condition(condition)
    return condition


def build_current_realtime_pose_conditions(
    initial_pose_raw: torch.Tensor,
    current_tracker_raw: torch.Tensor,
    joint_offsets_parent: torch.Tensor,
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
    tracker_position_mean: torch.Tensor | None,
    tracker_position_scale: torch.Tensor | None,
    config: IKInpaintingConfig,
    known_noise: torch.Tensor | None = None,
) -> tuple[RealtimePoseIKResult, RealtimePoseInpaintingCondition, torch.Tensor]:
    """由 Predictor current 与当前 Tracker 构造 IK、inpainting 和 joint condition。"""

    ik_result = build_current_ik(
        initial_pose_raw=initial_pose_raw,
        current_tracker_raw=current_tracker_raw,
        joint_offsets_parent=joint_offsets_parent,
        config=config,
    )
    condition = build_realtime_pose_inpainting_condition(
        ik_result=ik_result,
        pose_mean=pose_mean,
        pose_scale=pose_scale,
        known_noise=known_noise,
    )
    current_joint_condition = build_current_joint_condition(
        ik_result=ik_result,
        current_tracker_raw=current_tracker_raw,
        tracker_position_mean=tracker_position_mean,
        tracker_position_scale=tracker_position_scale,
    )
    return ik_result, condition, current_joint_condition


def build_current_joint_condition(
    ik_result: RealtimePoseIKResult,
    current_tracker_raw: torch.Tensor,
    tracker_position_mean: torch.Tensor | None,
    tracker_position_scale: torch.Tensor | None,
) -> torch.Tensor:
    """把当前逐 Tracker 几何与 IK 语义对齐为 `[B,24,10]`。"""

    batch_size = ik_result.pose.shape[0]
    if tuple(current_tracker_raw.shape) != (
        batch_size,
        TRACKER_COUNT,
        TRACKER_FEATURE_DIM,
    ):
        raise ValueError("current_tracker_raw 必须为 [B,6,10]。")
    if (tracker_position_mean is None) != (tracker_position_scale is None):
        raise ValueError("Tracker position mean 与 scale 必须同时提供或同时省略。")
    tracker_positions = current_tracker_raw[..., :3]
    if tracker_position_mean is not None:
        mean = tracker_position_mean.to(tracker_positions)
        scale = tracker_position_scale.to(tracker_positions)
        if tuple(mean.shape) != (TRACKER_COUNT, 3) or tuple(scale.shape) != (
            TRACKER_COUNT,
            3,
        ):
            raise ValueError("Tracker position normalizer 必须为 [6,3]。")
        tracker_positions = (tracker_positions - mean) / scale

    available = current_tracker_raw[..., TRACKER_AVAILABLE_OFFSET] > 0.5
    tracker_positions = torch.where(
        available[..., None], tracker_positions, torch.zeros_like(tracker_positions)
    )
    tracker_joint_indices = torch.as_tensor(
        TRACKER_TO_JOINT,
        device=current_tracker_raw.device,
        dtype=torch.long,
    )
    joint_positions = tracker_positions.new_zeros(
        batch_size, SMPL_JOINT_COUNT, 3
    )
    joint_position_valid = tracker_positions.new_zeros(
        batch_size, SMPL_JOINT_COUNT, 1
    )
    joint_positions[:, tracker_joint_indices] = tracker_positions
    joint_position_valid[:, tracker_joint_indices, 0] = available.to(
        tracker_positions.dtype
    )

    constraint_type = ik_result.constraint_type
    if bool(
        (
            (constraint_type < 0)
            | (constraint_type >= CURRENT_JOINT_CONSTRAINT_TYPE_COUNT)
        ).any()
    ):
        raise ValueError("constraint_type 必须位于 [0,3]。")
    constraint_one_hot = F.one_hot(
        constraint_type,
        num_classes=CURRENT_JOINT_CONSTRAINT_TYPE_COUNT,
    ).to(tracker_positions.dtype)
    result = torch.cat(
        [
            joint_positions,
            joint_position_valid,
            ik_result.updated_mask[..., None].to(tracker_positions.dtype),
            ik_result.confidence[..., None].to(tracker_positions.dtype),
            constraint_one_hot,
        ],
        dim=-1,
    )
    if tuple(result.shape) != (
        batch_size,
        SMPL_JOINT_COUNT,
        CURRENT_JOINT_CONDITION_DIM,
    ):
        raise RuntimeError("current_joint_condition 内部布局错误。")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("current_joint_condition 包含 NaN 或 Inf。")
    return result


def apply_realtime_pose_inpainting(
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: RealtimePoseInpaintingCondition,
    alphas_cumprod: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按当前噪声水平注入单帧条件，返回 `[B,144]` 与 `[B,24]` active mask。"""

    batch_size = x_t.shape[0]
    if tuple(x_t.shape) != (batch_size, REALTIME_POSE_TARGET_DIM):
        raise ValueError("x_t 必须为 [B,144]。")
    if tuple(t.shape) != (batch_size,):
        raise ValueError("t 必须为 [B]。")
    validate_realtime_pose_inpainting_condition(condition, check_values=False)
    alpha_values = torch.as_tensor(
        alphas_cumprod, device=x_t.device, dtype=x_t.dtype
    )
    alpha_bar_scalar = alpha_values.index_select(0, t.long())
    alpha_bar = alpha_bar_scalar[:, None, None]
    pose = condition.pose.to(x_t).reshape(
        batch_size, SMPL_JOINT_COUNT, ROTATION_6D_DIM
    )
    known_noise = condition.known_noise.to(x_t).reshape_as(pose)
    condition_at_t = alpha_bar.sqrt() * pose + (1.0 - alpha_bar).sqrt() * known_noise
    noise_level = torch.sqrt(1.0 - alpha_bar_scalar)[:, None]
    active = condition.valid.to(x_t.device) & (
        noise_level >= condition.release_level.to(x_t)
    )
    state = x_t.reshape_as(pose)
    return torch.where(active[..., None], condition_at_t, state).reshape_as(x_t), active


def validate_realtime_pose_inpainting_condition(
    condition: RealtimePoseInpaintingCondition,
    check_values: bool = True,
) -> None:
    batch_size = condition.pose.shape[0]
    if tuple(condition.pose.shape) != (batch_size, REALTIME_POSE_TARGET_DIM):
        raise ValueError("inpaint pose 必须为 [B,144]。")
    if tuple(condition.valid.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("inpaint valid 必须为 [B,24]。")
    if condition.valid.dtype != torch.bool:
        raise ValueError("inpaint valid 必须为 bool。")
    if tuple(condition.release_level.shape) != tuple(condition.valid.shape):
        raise ValueError("release_level 必须为 [B,24]。")
    if tuple(condition.known_noise.shape) != tuple(condition.pose.shape):
        raise ValueError("known_noise 必须为 [B,144]。")
    if check_values:
        tensors = (condition.pose, condition.release_level, condition.known_noise)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("inpainting condition 包含 NaN 或 Inf。")
        release = condition.release_level
        if bool(((release < 0.0) | (release > 1.0)).any()):
            raise ValueError("release_level 必须位于 [0,1]。")
