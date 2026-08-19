from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_torch,
    extract_rotation_heading_components_torch,
    ROOT_HEADING_OBSERVABILITY_EPS,
    resolve_root_head_reference_torch,
)
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    SMPL_PARENTS,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    NON_HEAD_TRACKER_INDICES,
    TRACKER_AVAILABLE_OFFSET,
    TRACKER_TO_JOINT,
)


ROOT_YAW_TOLERANCE_RADIANS = math.radians(15.0)
ROOT_POSITION_TOLERANCE_METERS = 0.05


def compute_raw_deployed_losses(
    raw_pred_pose: torch.Tensor,
    deployed_pred_pose: torch.Tensor,
    target_pose: torch.Tensor,
    batch: dict,
    tracker_pos_huber_beta: float,
) -> dict[str, torch.Tensor]:
    """按 Raw/Deployed 边界计算逐样本损失；返回值均为 `[B]`。"""

    raw_pred = _to_raw_pose(raw_pred_pose, batch)
    deployed_pred = _to_raw_pose(deployed_pred_pose, batch)
    target = _to_raw_pose(target_pose, batch)
    if raw_pred.ndim != 2:
        raise ValueError("单帧 loss 的 Pose 必须为 [B,144]。")
    raw_global, _ = decode_target_head_rotations_torch(raw_pred)
    deployed_global, deployed_root_yaw = decode_target_head_rotations_torch(deployed_pred)
    target_global, _ = decode_target_head_rotations_torch(target)

    global_rotation_loss = _rotation_angle(raw_global, target_global).square().flatten(1).mean(dim=1)
    parents = torch.as_tensor(SMPL_PARENTS[1:], device=raw_pred.device, dtype=torch.long)
    raw_local = raw_global.index_select(1, parents).transpose(-1, -2) @ raw_global[:, 1:]
    target_local = target_global.index_select(1, parents).transpose(-1, -2) @ target_global[:, 1:]
    local_rotation_loss = _rotation_angle(raw_local, target_local).square().flatten(1).mean(dim=1)

    previous_pose = _to_raw_pose(
        batch["previous_pose_target"].to(device=raw_pred.device, dtype=raw_pred.dtype),
        batch,
    )
    previous_global, _ = decode_target_head_rotations_torch(previous_pose)
    raw_velocity = previous_global.transpose(-1, -2) @ raw_global
    target_velocity = previous_global.transpose(-1, -2) @ target_global
    rotation_velocity_loss = (
        _rotation_angle(raw_velocity, target_velocity).square().flatten(1).mean(dim=1)
    )

    current_tracker = batch["current_tracker_raw"].to(
        device=raw_pred.device, dtype=raw_pred.dtype
    )
    tracker_rot = rotation_6d_to_matrix_torch(current_tracker[..., 3:9])
    measured = current_tracker[..., TRACKER_AVAILABLE_OFFSET] > 0.5
    tracker_joints = torch.as_tensor(TRACKER_TO_JOINT, device=raw_pred.device, dtype=torch.long)
    tracker_rotation_error = _rotation_angle(
        raw_global.index_select(1, tracker_joints), tracker_rot
    )
    tracker_rotation_loss = _masked_mean(tracker_rotation_error.square(), measured)

    offsets = batch["joint_offsets_parent"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    tracker_pos = current_tracker[..., :3]
    deployed_root_position, deployed_hip_height, deployed_joints = resolve_root_head_reference_torch(
        deployed_global,
        deployed_root_yaw,
        offsets,
        observed_head_height=tracker_pos[:, HEAD_TRACKER_INDEX, 1],
    )
    target_joints = batch["target_joints_head_ref"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    fk_loss = torch.square(deployed_joints - target_joints).flatten(1).mean(dim=1)
    head_ref_joint_distance_loss = torch.linalg.norm(
        deployed_joints - target_joints, dim=-1
    ).mean(dim=1)

    non_head_trackers = torch.as_tensor(NON_HEAD_TRACKER_INDICES, device=raw_pred.device, dtype=torch.long)
    non_head_joints = tracker_joints.index_select(0, non_head_trackers)
    tracker_distance = torch.linalg.norm(
        deployed_joints.index_select(1, non_head_joints) - tracker_pos.index_select(1, non_head_trackers),
        dim=-1,
    )
    tracker_position_loss = _masked_mean(
        _radial_huber_loss(tracker_distance, tracker_pos_huber_beta),
        measured.index_select(1, non_head_trackers),
    )

    predicted_root = deployed_root_position
    target_root = batch["target_root_position_head_ref"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    current_head_yaw = batch["current_head_yaw_world"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    target_root_yaw = batch["target_root_yaw_world"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    target_root_yaw_head = target_root_yaw - current_head_yaw
    # Actor Root 的 Y 固定在 floor；root_loss 只监督 yaw，避免与下面的
    # Head-to-Root 水平几何项重复惩罚同一个 XZ 偏移。
    root_loss = _root_yaw_circular_loss(
        deployed_global[:, JOINT_INDEX["pelvis"]],
        target_root_yaw_head,
    )
    head_to_root_xz_loss = _scaled_huber_loss(
        predicted_root[:, [0, 2]] - target_root[:, [0, 2]],
        tolerance=ROOT_POSITION_TOLERANCE_METERS,
    ).mean(dim=-1)
    target_hip_height = batch["target_hip_height"].to(
        device=raw_pred.device,
        dtype=raw_pred.dtype,
    )
    hip_height_loss = _scaled_huber_loss(
        deployed_hip_height - target_hip_height,
        tolerance=ROOT_POSITION_TOLERANCE_METERS,
    )

    return {
        "global_rotation_loss": global_rotation_loss,
        "local_rotation_loss": local_rotation_loss,
        "rotation_velocity_loss": rotation_velocity_loss,
        "tracker_rotation_loss": tracker_rotation_loss,
        "fk_loss": fk_loss,
        "tracker_position_loss": tracker_position_loss,
        "root_loss": root_loss,
        "head_ref_joint_distance_loss": head_ref_joint_distance_loss,
        "head_to_root_xz_loss": head_to_root_xz_loss,
        "hip_height_loss": hip_height_loss,
    }


def _to_raw_pose(value: torch.Tensor, batch: dict) -> torch.Tensor:
    mean = batch.get("pose_mean")
    scale = batch.get("pose_scale")
    if mean is None or scale is None:
        return value
    return value * scale.to(device=value.device, dtype=value.dtype) + mean.to(
        device=value.device, dtype=value.dtype
    )


def _rotation_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """返回 `[0,pi]` SO(3) 夹角，并保证相同旋转处梯度为有限零值。"""

    # AMP 下先升到 float32，避免平滑项平方后在 float16 中下溢为零。
    calculation_dtype = (
        torch.float32
        if first.dtype in (torch.float16, torch.bfloat16)
        else first.dtype
    )
    relative = first.to(calculation_dtype).transpose(-1, -2) @ second.to(calculation_dtype)
    skew = 0.5 * torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sin_squared = skew.square().sum(dim=-1)
    smoothing_eps = 1e-6
    sin_angle = torch.sqrt(sin_squared + smoothing_eps**2) - smoothing_eps
    cos_angle = (
        (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    ).clamp(-1.0, 1.0)
    return torch.atan2(sin_angle, cos_angle)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def _radial_huber_loss(distance: torch.Tensor, beta: float) -> torch.Tensor:
    """对三维欧氏距离应用 Huber，保持 tracker position loss 的旋转不变性。"""

    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("tracker_pos_huber_beta 必须是有限正数。")
    beta_tensor = distance.new_tensor(beta)
    return torch.where(
        distance < beta_tensor,
        0.5 * distance.square() / beta_tensor,
        distance - 0.5 * beta_tensor,
    )


def _scaled_huber_loss(error: torch.Tensor, tolerance: float) -> torch.Tensor:
    """把物理误差按容差无量纲化，再应用 `beta=1` 的 Huber。"""

    scale = float(tolerance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Huber 物理容差必须是有限正数。")
    normalized_error = error / scale
    return F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        beta=1.0,
        reduction="none",
    )


def _root_yaw_circular_loss(
    pelvis_rotation: torch.Tensor,
    target_yaw: torch.Tensor,
) -> torch.Tensor:
    """用二维 heading 比较 yaw，天然跨越 `-pi/pi` 数值边界。"""

    cosine_scale, sine_scale, _ = extract_rotation_heading_components_torch(
        pelvis_rotation
    )
    predicted_heading = F.normalize(
        torch.stack((cosine_scale, sine_scale), dim=-1),
        dim=-1,
        eps=ROOT_HEADING_OBSERVABILITY_EPS,
    )
    target_heading = torch.stack(
        (torch.cos(target_yaw), torch.sin(target_yaw)),
        dim=-1,
    )
    heading_similarity = (predicted_heading * target_heading).sum(dim=-1).clamp(
        -1.0,
        1.0,
    )
    return (1.0 - heading_similarity) / (
        1.0 - math.cos(ROOT_YAW_TOLERANCE_RADIANS)
    )
