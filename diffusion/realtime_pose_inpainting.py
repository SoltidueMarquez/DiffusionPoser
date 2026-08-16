from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_ik import build_current_ik_pose
from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_LENGTH,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
)
from data_loaders.tracker_reliability import (
    compute_tracker_online_confidence_torch,
    map_tracker_confidence_to_joints_torch,
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
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
) -> RealtimePoseInpaintingCondition:
    """构造只约束 horizon 0 的 `[B,11,144]` 当前帧 IK 条件。"""

    batch_size = current_pose_raw.shape[0]
    if tuple(current_pose_raw.shape) != (batch_size, REALTIME_POSE_TARGET_DIM):
        raise ValueError("current_pose_raw 必须为 [B,144]。")
    if tuple(current_confidence.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("current_confidence 必须为 [B,24]。")
    pose_raw = torch.zeros(
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        REALTIME_POSE_TARGET_DIM,
        device=current_pose_raw.device,
        dtype=current_pose_raw.dtype,
    )
    pose_raw[:, 0] = current_pose_raw
    if pose_mean is None or pose_scale is None:
        pose = pose_raw
    else:
        mean = pose_mean.to(device=pose_raw.device, dtype=pose_raw.dtype)
        scale = pose_scale.to(device=pose_raw.device, dtype=pose_raw.dtype)
        # 未使用的未来帧保持数值零；其 confidence 恒为零，不参与注入。
        pose = torch.zeros_like(pose_raw)
        pose[:, 0] = (current_pose_raw - mean) / scale

    confidence = torch.zeros(
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        SMPL_JOINT_COUNT,
        device=current_confidence.device,
        dtype=current_confidence.dtype,
    )
    confidence[:, 0] = current_confidence.clamp(0.0, 1.0)
    condition = RealtimePoseInpaintingCondition(pose=pose, confidence=confidence)
    validate_realtime_pose_inpainting_condition(condition, require_current_only=True)
    return condition


def build_current_realtime_pose_inpainting_condition(
    previous_pose_raw: torch.Tensor,
    previous_pose_valid: torch.Tensor,
    current_tracker_raw: torch.Tensor,
    configured: torch.Tensor,
    measured_valid: torch.Tensor,
    d_on: torch.Tensor,
    joint_offsets_parent: torch.Tensor,
    joint_rest_local_rotations_6d: torch.Tensor,
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
    config: IKInpaintingConfig | None = None,
) -> RealtimePoseInpaintingCondition:
    """用训练/runtime 完全相同的流程构造当前帧 IK 与逐关节置信度。"""

    cfg = (config or IKInpaintingConfig()).validate()
    current_pose_raw = build_current_ik_pose(
        previous_pose_raw=previous_pose_raw,
        previous_pose_valid=previous_pose_valid,
        current_tracker_raw=current_tracker_raw,
        joint_offsets_parent=joint_offsets_parent,
        joint_rest_local_rotations_6d=joint_rest_local_rotations_6d,
        fabrik_iterations=cfg.fabrik_iterations,
    )
    tracker_confidence = compute_tracker_online_confidence_torch(
        tracker_valid=configured.bool() & measured_valid.bool(),
        d_on=d_on,
        warmup_frames=cfg.tracker_confidence_warmup,
    )
    return build_realtime_pose_inpainting_condition(
        current_pose_raw=current_pose_raw,
        current_confidence=map_tracker_confidence_to_joints_torch(tracker_confidence),
        pose_mean=pose_mean,
        pose_scale=pose_scale,
    )


def add_future_rolling_prior_to_condition(
    current_condition: RealtimePoseInpaintingCondition,
    aligned_future_prior_raw: torch.Tensor,
    future_prior_valid: torch.Tensor,
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
    confidence_decay: float = 0.9,
) -> RealtimePoseInpaintingCondition:
    """把严格对齐的旧 horizon 2..10 注入新 horizon 1..9。

    `aligned_future_prior_raw` 为 `[B,9,144]`。新 horizon 10 没有旧预测可与之
    对齐，因此 Pose 和 confidence 始终保持零，禁止重复最远帧。
    """

    validate_realtime_pose_inpainting_condition(
        current_condition,
        require_current_only=True,
    )
    if not 0.0 < float(confidence_decay) <= 1.0:
        raise ValueError("future_confidence_decay 必须位于 (0,1]。")
    batch_size = current_condition.pose.shape[0]
    expected_shape = (
        batch_size,
        REALTIME_POSE_TARGET_LENGTH - 2,
        REALTIME_POSE_TARGET_DIM,
    )
    if tuple(aligned_future_prior_raw.shape) != expected_shape:
        raise ValueError("aligned_future_prior_raw 必须为 [B,9,144]。")
    if tuple(future_prior_valid.shape) != (batch_size,):
        raise ValueError("future_prior_valid 必须为 [B]。")

    pose = current_condition.pose.clone()
    prior = aligned_future_prior_raw.to(device=pose.device, dtype=pose.dtype)
    if pose_mean is not None and pose_scale is not None:
        mean = pose_mean.to(device=pose.device, dtype=pose.dtype)
        scale = pose_scale.to(device=pose.device, dtype=pose.dtype)
        prior = (prior - mean) / scale
    valid = future_prior_valid.to(device=pose.device, dtype=torch.bool)
    prior = torch.where(valid[:, None, None], prior, torch.zeros_like(prior))
    pose[:, 1:-1] = prior

    confidence = current_condition.confidence.clone()
    future_steps = torch.arange(
        1,
        REALTIME_POSE_TARGET_LENGTH - 1,
        device=confidence.device,
        dtype=confidence.dtype,
    )
    decay = torch.pow(
        torch.full_like(future_steps, float(confidence_decay)),
        future_steps,
    )
    confidence[:, 1:-1] = (
        confidence[:, :1]
        * decay[None, :, None]
        * valid.to(device=confidence.device, dtype=confidence.dtype)[:, None, None]
    )
    condition = RealtimePoseInpaintingCondition(pose=pose, confidence=confidence)
    validate_realtime_pose_inpainting_condition(condition)
    return condition


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
    require_current_only: bool = False,
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
    if bool((condition.pose[:, -1] != 0.0).any()):
        raise ValueError("inpaint_pose 的 horizon 10 必须恒为零。")
    if bool((condition.confidence[:, -1] != 0.0).any()):
        raise ValueError("inpaint_confidence 的 horizon 10 必须恒为零。")
    if require_current_only and bool((condition.pose[:, 1:] != 0.0).any()):
        raise ValueError("训练/current-only inpaint_pose 的未来帧必须恒为零。")
    if require_current_only and bool((condition.confidence[:, 1:] != 0.0).any()):
        raise ValueError("训练/current-only inpaint_confidence 的未来帧必须恒为零。")
    if check_values:
        if not bool(torch.isfinite(condition.pose).all()):
            raise ValueError("inpaint_pose 必须为有限数值。")
        if not bool(torch.isfinite(condition.confidence).all()) or bool(
            ((condition.confidence < 0.0) | (condition.confidence > 1.0)).any()
        ):
            raise ValueError("inpaint_confidence 必须为有限的 [0,1] 数值。")
