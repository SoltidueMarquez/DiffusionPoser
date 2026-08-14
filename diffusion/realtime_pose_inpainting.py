from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_loaders.sensor_masking import (
    REALTIME_POSE_FUTURE_FRAME_COUNT,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_LENGTH,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
)


@dataclass(frozen=True)
class RealtimePoseInpaintingCondition:
    """联合扩散 horizon 上唯一的 Pose 与逐关节置信度条件。"""

    pose: torch.Tensor
    confidence: torch.Tensor


def confidence_to_t_soft(
    confidence: torch.Tensor,
    max_timestep: int,
) -> torch.Tensor:
    """按老师的线性公式返回逐帧、逐关节连续释放阈值。"""

    if int(max_timestep) < 0:
        raise ValueError("max_timestep 不能小于 0。")
    return (1.0 - confidence.clamp(0.0, 1.0)) * float(max_timestep)


def build_realtime_pose_inpainting_condition(
    current_pose_raw: torch.Tensor,
    current_confidence: torch.Tensor,
    future_prior_raw: torch.Tensor,
    future_prior_valid: torch.Tensor,
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
    future_confidence_decay: float = 0.9,
) -> RealtimePoseInpaintingCondition:
    """构造 `[B,11,144]` 条件，未来置信度只继承当前值并按 `gamma^k` 衰减。"""

    if not 0.0 < float(future_confidence_decay) <= 1.0:
        raise ValueError("future_confidence_decay 必须位于 (0,1]。")
    batch_size = current_pose_raw.shape[0]
    if tuple(current_pose_raw.shape) != (batch_size, REALTIME_POSE_TARGET_DIM):
        raise ValueError("current_pose_raw 必须为 [B,144]。")
    if tuple(current_confidence.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("current_confidence 必须为 [B,24]。")
    if tuple(future_prior_raw.shape) != (
        batch_size,
        REALTIME_POSE_FUTURE_FRAME_COUNT,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError("future_prior_raw 必须为 [B,10,144]。")
    if tuple(future_prior_valid.shape) != (batch_size,):
        raise ValueError("future_prior_valid 必须为 [B]。")

    pose_raw = torch.cat([current_pose_raw[:, None], future_prior_raw], dim=1)
    if pose_mean is None or pose_scale is None:
        pose = pose_raw
    else:
        mean = pose_mean.to(device=pose_raw.device, dtype=pose_raw.dtype)
        scale = pose_scale.to(device=pose_raw.device, dtype=pose_raw.dtype)
        pose = (pose_raw - mean) / scale

    frame_indices = torch.arange(
        REALTIME_POSE_TARGET_LENGTH,
        device=current_confidence.device,
        dtype=current_confidence.dtype,
    )
    decay = torch.pow(
        torch.full_like(frame_indices, float(future_confidence_decay)), frame_indices
    )
    confidence = current_confidence[:, None] * decay[None, :, None]
    confidence[:, 1:] *= future_prior_valid[:, None, None].to(confidence.dtype)
    confidence = confidence.clamp(0.0, 1.0)
    validate_realtime_pose_inpainting_condition(
        RealtimePoseInpaintingCondition(pose=pose, confidence=confidence)
    )
    return RealtimePoseInpaintingCondition(pose=pose, confidence=confidence)


def apply_realtime_pose_inpainting(
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: RealtimePoseInpaintingCondition,
    known_noise: torch.Tensor,
    alphas_cumprod: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按逐关节 `T_soft` 注入条件，返回模型输入与 `[B,11,24]` 连续阈值。"""

    batch_size = x_t.shape[0]
    if tuple(x_t.shape) != (
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError("x_t 必须为 [B,11,144]。")
    if tuple(t.shape) != (batch_size,):
        raise ValueError("t 必须为 [B]。")
    if tuple(known_noise.shape) != tuple(x_t.shape):
        raise ValueError("known_noise 必须与 x_t 同形。")
    validate_realtime_pose_inpainting_condition(condition, check_values=False)

    max_timestep = int(len(alphas_cumprod) - 1)
    confidence = condition.confidence.to(device=x_t.device, dtype=x_t.dtype)
    global_t = t.to(device=x_t.device, dtype=torch.long)[:, None, None]
    t_soft = confidence_to_t_soft(confidence, max_timestep)

    alpha_values = torch.as_tensor(
        alphas_cumprod, device=x_t.device, dtype=x_t.dtype
    )
    alpha_bar = alpha_values.index_select(0, global_t[:, 0, 0])[:, None, None, None]
    pose = condition.pose.to(device=x_t.device, dtype=x_t.dtype).reshape(
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        SMPL_JOINT_COUNT,
        ROTATION_6D_DIM,
    )
    noise = known_noise.reshape_as(pose)
    condition_at_t = alpha_bar.sqrt() * pose + (1.0 - alpha_bar).sqrt() * noise

    state = x_t.reshape_as(pose)
    # t=0 统一释放 IK 条件，避免最高置信度再次覆盖此前步骤已经完成的修正。
    inpaint_mask = (
        (confidence > 0.0)
        & (global_t.to(confidence.dtype) >= t_soft)
        & (global_t > 0)
    )
    x_model = torch.where(inpaint_mask[..., None], condition_at_t, state)
    return x_model.reshape_as(x_t), t_soft


def validate_realtime_pose_inpainting_condition(
    condition: RealtimePoseInpaintingCondition,
    check_values: bool = True,
) -> None:
    batch_size = condition.pose.shape[0]
    if tuple(condition.pose.shape) != (
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError("inpaint_pose 必须为 [B,11,144]。")
    if tuple(condition.confidence.shape) != (
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        SMPL_JOINT_COUNT,
    ):
        raise ValueError("inpaint_confidence 必须为 [B,11,24]。")
    if check_values:
        if not bool(torch.isfinite(condition.pose).all()):
            raise ValueError("inpaint_pose 必须为有限数值。")
        if not bool(torch.isfinite(condition.confidence).all()) or bool(
            ((condition.confidence < 0.0) | (condition.confidence > 1.0)).any()
        ):
            raise ValueError("inpaint_confidence 必须为有限的 [0,1] 数值。")
