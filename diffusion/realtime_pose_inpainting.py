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
    REALTIME_POSE_TARGET_LENGTH,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_COUNT,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_FEATURE_DIM,
    TRACKER_MEASURED_VALID_OFFSET,
    TRACKER_TO_JOINT,
)
from data_loaders.tracker_reliability import (
    compute_tracker_online_confidence_torch,
)


@dataclass(frozen=True)
class RealtimePoseInpaintingCondition:
    """联合扩散 horizon 上的 Pose、有效性和物理噪声释放阈值。"""

    pose: torch.Tensor  # [B,11,144]
    valid: torch.Tensor  # [B,11,24]
    release_level: torch.Tensor  # [B,11,24]


def confidence_to_release_level(confidence: torch.Tensor) -> torch.Tensor:
    """把 `[0,1]` confidence 映射到与调度步数无关的噪声坐标。"""

    corruption = 1.0 - confidence.clamp(0.0, 1.0)
    return torch.sin(corruption * torch.pi * 0.5)


def build_realtime_pose_inpainting_condition(
    ik_result: RealtimePoseIKResult,
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
) -> RealtimePoseInpaintingCondition:
    """构造只约束 horizon 0 的 `[B,11,144]` 当前帧 IK 条件。"""

    batch_size = ik_result.pose.shape[0]
    if tuple(ik_result.pose.shape) != (
        batch_size,
        SMPL_JOINT_COUNT,
        ROTATION_6D_DIM,
    ):
        raise ValueError("ik_result.pose 必须为 [B,24,6]。")
    if tuple(ik_result.updated_mask.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("ik_result.updated_mask 必须为 [B,24]。")
    if tuple(ik_result.confidence.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("ik_result.confidence 必须为 [B,24]。")
    current_pose_raw = ik_result.pose.reshape(batch_size, REALTIME_POSE_TARGET_DIM)
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

    valid = torch.zeros(
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        SMPL_JOINT_COUNT,
        device=ik_result.updated_mask.device,
        dtype=torch.bool,
    )
    valid[:, 0] = ik_result.updated_mask.bool()
    release_level = torch.ones(
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        SMPL_JOINT_COUNT,
        device=ik_result.confidence.device,
        dtype=ik_result.confidence.dtype,
    )
    release_level[:, 0] = confidence_to_release_level(ik_result.confidence)
    condition = RealtimePoseInpaintingCondition(
        pose=pose,
        valid=valid,
        release_level=release_level,
    )
    validate_realtime_pose_inpainting_condition(condition)
    return condition


def build_current_realtime_pose_conditions(
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
    tracker_position_mean: torch.Tensor | None,
    tracker_position_scale: torch.Tensor | None,
    config: IKInpaintingConfig | None = None,
) -> tuple[RealtimePoseIKResult, RealtimePoseInpaintingCondition, torch.Tensor]:
    """统一构造 IK、soft inpainting 与 `[B,24,10]` 模型条件。"""

    cfg = (config or IKInpaintingConfig()).validate()
    raw_configured = current_tracker_raw[..., TRACKER_CONFIGURED_OFFSET] > 0.5
    raw_measured = current_tracker_raw[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5
    if not torch.equal(raw_configured, configured.bool()):
        raise ValueError("current_tracker_raw 与 configured 的当前帧状态不一致。")
    if not torch.equal(raw_measured, measured_valid.bool()):
        raise ValueError("current_tracker_raw 与 measured_valid 的当前帧状态不一致。")
    tracker_source_reliability = compute_tracker_online_confidence_torch(
        tracker_valid=configured.bool() & measured_valid.bool(),
        d_on=d_on,
        warmup_frames=cfg.tracker_confidence_warmup,
    )
    ik_result = build_current_ik(
        previous_pose_raw=previous_pose_raw,
        previous_pose_valid=previous_pose_valid,
        current_tracker_raw=current_tracker_raw,
        tracker_source_reliability=tracker_source_reliability,
        joint_offsets_parent=joint_offsets_parent,
        joint_rest_local_rotations_6d=joint_rest_local_rotations_6d,
        config=cfg,
    )
    condition = build_realtime_pose_inpainting_condition(
        ik_result=ik_result,
        pose_mean=pose_mean,
        pose_scale=pose_scale,
    )
    current_joint_condition = build_current_joint_condition(
        ik_result=ik_result,
        current_tracker_raw=current_tracker_raw,
        configured=configured,
        measured_valid=measured_valid,
        tracker_position_mean=tracker_position_mean,
        tracker_position_scale=tracker_position_scale,
    )
    return ik_result, condition, current_joint_condition


def build_current_joint_condition(
    ik_result: RealtimePoseIKResult,
    current_tracker_raw: torch.Tensor,
    configured: torch.Tensor,
    measured_valid: torch.Tensor,
    tracker_position_mean: torch.Tensor | None,
    tracker_position_scale: torch.Tensor | None,
) -> torch.Tensor:
    """把 Tracker 几何与 IK 语义对齐为 `[B,24,10]` joint condition。

    前三维位置只写入六个直接 Tracker 关节。IK 骨链父关节只接收
    valid/confidence/type，端点位置由当前帧 spatial self-attention 传播。
    """

    batch_size = ik_result.pose.shape[0]
    if tuple(current_tracker_raw.shape) != (
        batch_size,
        TRACKER_COUNT,
        TRACKER_FEATURE_DIM,
    ):
        raise ValueError("current_tracker_raw 必须为 [B,6,13]。")
    if tuple(configured.shape) != (batch_size, TRACKER_COUNT):
        raise ValueError("configured 必须为 [B,6]。")
    if tuple(measured_valid.shape) != (batch_size, TRACKER_COUNT):
        raise ValueError("measured_valid 必须为 [B,6]。")
    if tuple(ik_result.updated_mask.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("ik_result.updated_mask 必须为 [B,24]。")
    if tuple(ik_result.confidence.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("ik_result.confidence 必须为 [B,24]。")
    if tuple(ik_result.constraint_type.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("ik_result.constraint_type 必须为 [B,24]。")
    if ik_result.constraint_type.dtype != torch.int64:
        raise ValueError("ik_result.constraint_type 必须为 int64。")
    if (tracker_position_mean is None) != (tracker_position_scale is None):
        raise ValueError("Tracker position mean 与 scale 必须同时提供或同时省略。")

    tracker_positions = current_tracker_raw[..., :3]
    if tracker_position_mean is not None and tracker_position_scale is not None:
        if tuple(tracker_position_mean.shape) != (TRACKER_COUNT, 3):
            raise ValueError("tracker_position_mean 必须为 [6,3]。")
        if tuple(tracker_position_scale.shape) != (TRACKER_COUNT, 3):
            raise ValueError("tracker_position_scale 必须为 [6,3]。")
        mean = tracker_position_mean.to(
            device=tracker_positions.device,
            dtype=tracker_positions.dtype,
        )
        scale = tracker_position_scale.to(
            device=tracker_positions.device,
            dtype=tracker_positions.dtype,
        )
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("Tracker position normalizer 必须为有限数值。")
        if bool((scale <= 0.0).any()):
            raise ValueError("tracker_position_scale 必须全部大于零。")
        tracker_positions = (tracker_positions - mean) / scale

    position_valid = configured.bool() & measured_valid.bool()
    tracker_positions = torch.where(
        position_valid[..., None],
        tracker_positions,
        torch.zeros_like(tracker_positions),
    )
    tracker_joint_indices = torch.as_tensor(
        TRACKER_TO_JOINT,
        device=current_tracker_raw.device,
        dtype=torch.long,
    )
    joint_positions = tracker_positions.new_zeros(
        batch_size,
        SMPL_JOINT_COUNT,
        3,
    )
    joint_position_valid = tracker_positions.new_zeros(
        batch_size,
        SMPL_JOINT_COUNT,
        1,
    )
    joint_positions[:, tracker_joint_indices] = tracker_positions
    joint_position_valid[:, tracker_joint_indices, 0] = position_valid.to(
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
    current_joint_condition = torch.cat(
        [
            joint_positions,
            joint_position_valid,
            ik_result.updated_mask[..., None].to(tracker_positions.dtype),
            ik_result.confidence[..., None].to(tracker_positions.dtype),
            constraint_one_hot,
        ],
        dim=-1,
    )
    if tuple(current_joint_condition.shape) != (
        batch_size,
        SMPL_JOINT_COUNT,
        CURRENT_JOINT_CONDITION_DIM,
    ):
        raise RuntimeError("current_joint_condition 内部布局错误。")
    if not bool(torch.isfinite(current_joint_condition).all()):
        raise ValueError("current_joint_condition 必须为有限数值。")
    return current_joint_condition


def apply_realtime_pose_inpainting(
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: RealtimePoseInpaintingCondition,
    known_noise: torch.Tensor,
    alphas_cumprod: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按实际 `alpha_bar_t` 注入条件，返回模型输入与 `[B,11,24]` active mask。"""

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

    alpha_values = torch.as_tensor(
        alphas_cumprod, device=x_t.device, dtype=x_t.dtype
    )
    timestep = t.to(device=x_t.device, dtype=torch.long)
    alpha_bar_scalar = alpha_values.index_select(0, timestep)
    alpha_bar = alpha_bar_scalar[:, None, None, None]
    pose = condition.pose.to(device=x_t.device, dtype=x_t.dtype).reshape(
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        SMPL_JOINT_COUNT,
        ROTATION_6D_DIM,
    )
    noise = known_noise.reshape_as(pose)
    condition_at_t = alpha_bar.sqrt() * pose + (1.0 - alpha_bar).sqrt() * noise

    state = x_t.reshape_as(pose)
    valid = condition.valid.to(device=x_t.device, dtype=torch.bool)
    release_level = condition.release_level.to(device=x_t.device, dtype=x_t.dtype)
    noise_level = torch.sqrt(1.0 - alpha_bar_scalar)[:, None, None]
    active = (
        valid
        & (noise_level >= release_level)
    )
    x_model = torch.where(active[..., None], condition_at_t, state)
    return x_model.reshape_as(x_t), active


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
    if tuple(condition.valid.shape) != (
        batch_size,
        REALTIME_POSE_TARGET_LENGTH,
        SMPL_JOINT_COUNT,
    ):
        raise ValueError("inpaint_valid 必须为 [B,11,24]。")
    if condition.valid.dtype != torch.bool:
        raise ValueError("inpaint_valid 必须为 bool。")
    if tuple(condition.release_level.shape) != tuple(condition.valid.shape):
        raise ValueError("release_level 必须为 [B,11,24]。")
    if bool((condition.pose[:, 1:] != 0.0).any()):
        raise ValueError("第一轮 inpaint_pose 的未来帧必须恒为零。")
    if bool(condition.valid[:, 1:].any()):
        raise ValueError("第一轮未来帧 inpaint_valid 必须恒为 False。")
    if check_values:
        if not bool(torch.isfinite(condition.pose).all()):
            raise ValueError("inpaint_pose 必须为有限数值。")
        if not bool(torch.isfinite(condition.release_level).all()) or bool(
            ((condition.release_level < 0.0) | (condition.release_level > 1.0)).any()
        ):
            raise ValueError("release_level 必须为有限的 [0,1] 数值。")
