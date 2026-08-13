from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_torch,
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
    REALTIME_POSE_FPS,
    TRACKER_TO_JOINT,
)


CONTACT_SLIDE_HUBER_BETA_MPS = 0.1


def compute_raw_deployed_losses(
    raw_pred_xstart: torch.Tensor,
    deployed_pred_xstart: torch.Tensor,
    target_xstart: torch.Tensor,
    batch: dict,
    auxiliary_outputs: dict[str, torch.Tensor],
    tracker_pos_huber_beta: float,
) -> dict[str, torch.Tensor]:
    """按 Raw/Deployed 边界计算逐样本损失；返回值均为 `[B]`。"""

    raw_pred = _to_raw_pose(raw_pred_xstart, batch)
    deployed_pred = _to_raw_pose(deployed_pred_xstart, batch)
    target = _to_raw_pose(target_xstart, batch)
    raw_global, _ = decode_target_head_rotations_torch(raw_pred)
    deployed_global, deployed_root_yaw = decode_target_head_rotations_torch(deployed_pred)
    target_global, _ = decode_target_head_rotations_torch(target)

    global_rotation_loss = _rotation_angle(raw_global, target_global).square().flatten(1).mean(dim=1)
    parents = torch.as_tensor(SMPL_PARENTS[1:], device=raw_pred.device, dtype=torch.long)
    raw_local = raw_global.index_select(2, parents).transpose(-1, -2) @ raw_global[:, :, 1:]
    target_local = target_global.index_select(2, parents).transpose(-1, -2) @ target_global[:, :, 1:]
    local_rotation_loss = _rotation_angle(raw_local, target_local).square().flatten(1).mean(dim=1)

    raw_velocity = raw_global[:, :-1].transpose(-1, -2) @ raw_global[:, 1:]
    target_velocity = target_global[:, :-1].transpose(-1, -2) @ target_global[:, 1:]
    rotation_velocity_loss = (
        _rotation_angle(raw_velocity, target_velocity).square().flatten(1).mean(dim=1)
    )

    raw_current_global = raw_global[:, 0]
    deployed_current_global = deployed_global[:, 0]
    deployed_current_root_yaw = deployed_root_yaw[:, 0]

    current_tracker = batch["tracker_window_raw"][:, -1].to(
        device=raw_pred.device, dtype=raw_pred.dtype
    )
    tracker_rot = rotation_6d_to_matrix_torch(current_tracker[..., 3:9])
    measured = current_tracker[..., 10] > 0.5
    tracker_joints = torch.as_tensor(TRACKER_TO_JOINT, device=raw_pred.device, dtype=torch.long)
    tracker_rotation_error = _rotation_angle(
        raw_current_global.index_select(1, tracker_joints), tracker_rot
    )
    tracker_rotation_loss = _masked_mean(tracker_rotation_error.square(), measured)

    offsets = batch["joint_offsets_parent"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    tracker_pos = current_tracker[..., :3]
    deployed_root_position, _, deployed_joints = resolve_root_head_reference_torch(
        deployed_current_global,
        deployed_current_root_yaw,
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
    root_yaw_error = torch.remainder(
        deployed_current_root_yaw + current_head_yaw - target_root_yaw + math.pi,
        2.0 * math.pi,
    ) - math.pi
    # Actor Root 的 Y 固定在 floor；root_loss 只监督 yaw，避免与下面的
    # Head-to-Root 水平几何项重复惩罚同一个 XZ 偏移。
    root_loss = root_yaw_error.square()
    head_to_root_xz_loss = torch.square(
        predicted_root[:, [0, 2]] - target_root[:, [0, 2]]
    ).mean(dim=-1)

    contact_weight = batch["contact_target"].to(
        device=raw_pred.device, dtype=raw_pred.dtype
    ).clamp(0.0, 1.0)
    previous_contact_weight = batch["previous_contact_target"].to(
        device=raw_pred.device, dtype=raw_pred.dtype
    )
    # 只有相邻两帧都保持接触时才约束脚底滑动，避免在落地和离地边沿
    # 把上一帧仍在摆动的脚错误地当作世界空间锚点。
    adjacent_contact_weight = _adjacent_contact_weight(
        current_contact_weight=contact_weight,
        previous_contact_weight=previous_contact_weight,
    )
    contact_target = (contact_weight >= 0.5).to(raw_pred.dtype)
    contact_logits = auxiliary_outputs["contact_logits"].to(dtype=raw_pred.dtype)
    contact_loss = F.binary_cross_entropy_with_logits(
        contact_logits,
        contact_target,
        reduction="none",
    ).mean(dim=-1)

    previous_pose = _to_raw_pose(
        batch["previous_pose_target"].to(
            device=raw_pred.device, dtype=raw_pred.dtype
        ),
        batch,
    )
    previous_global, previous_root_yaw = decode_target_head_rotations_torch(
        previous_pose
    )
    previous_head_position = batch["tracker_window_raw"][:, -2, HEAD_TRACKER_INDEX, :3].to(
        device=raw_pred.device, dtype=raw_pred.dtype
    )
    _, _, previous_joints = resolve_root_head_reference_torch(
        previous_global,
        previous_root_yaw,
        offsets,
        observed_head_height=previous_head_position[:, 1],
    )
    # FK resolver 会把每一帧的 Head XZ 放在原点；补回上一帧 Head 相对当前
    # Head 的水平位移后，两帧脚点才位于同一个当前 Head-reference 坐标系。
    previous_head_translation = torch.zeros_like(previous_head_position)
    previous_head_translation[:, [0, 2]] = previous_head_position[:, [0, 2]]
    previous_joints = previous_joints + previous_head_translation[:, None]

    foot_indices = torch.as_tensor(
        [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]],
        device=raw_pred.device,
        dtype=torch.long,
    )
    contact_slide_loss = _contact_slide_loss(
        predicted_feet=deployed_joints.index_select(1, foot_indices),
        previous_target_feet=previous_joints.index_select(1, foot_indices),
        contact_weight=adjacent_contact_weight,
        previous_frame_valid=batch["window_valid_mask"][:, -2].to(
            device=raw_pred.device
        ),
        fps=REALTIME_POSE_FPS,
        huber_beta_mps=CONTACT_SLIDE_HUBER_BETA_MPS,
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
        "contact_loss": contact_loss,
        "contact_slide_loss": contact_slide_loss,
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


def _adjacent_contact_weight(
    current_contact_weight: torch.Tensor,
    previous_contact_weight: torch.Tensor,
) -> torch.Tensor:
    """返回连续两帧都接触的逐脚 soft 权重 `[B,2]`。"""

    if current_contact_weight.shape != previous_contact_weight.shape:
        raise ValueError("previous_contact_target 与 contact_target 必须同为 [B,2]。")
    if current_contact_weight.ndim != 2 or current_contact_weight.shape[-1] != 2:
        raise ValueError("contact_target 必须为 [B,2]。")
    return torch.minimum(
        previous_contact_weight.clamp(0.0, 1.0),
        current_contact_weight.clamp(0.0, 1.0),
    )


def _contact_slide_loss(
    predicted_feet: torch.Tensor,
    previous_target_feet: torch.Tensor,
    contact_weight: torch.Tensor,
    previous_frame_valid: torch.Tensor,
    *,
    fps: float,
    huber_beta_mps: float,
) -> torch.Tensor:
    """计算逐样本两脚接触滑动损失。

    `predicted_feet` 与 `previous_target_feet` 均为同一当前 Head-reference
    下的 `[B,2,3]`。这里只约束 XZ 水平速度，避免把 AMASS 中不可靠的
    绝对地面高度解释成物理接触真值。
    """

    if predicted_feet.shape != previous_target_feet.shape or predicted_feet.ndim != 3:
        raise ValueError("当前预测脚与前一帧目标脚必须同为 [B,2,3]。")
    if tuple(predicted_feet.shape[1:]) != (2, 3):
        raise ValueError("脚点张量必须包含左右脚两个三维位置。")
    if tuple(contact_weight.shape) != tuple(predicted_feet.shape[:2]):
        raise ValueError("contact_weight 必须为 [B,2]。")
    if tuple(previous_frame_valid.shape) != (predicted_feet.shape[0],):
        raise ValueError("previous_frame_valid 必须为 [B]。")
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps 必须是有限正数。")

    horizontal_speed = torch.linalg.norm(
        predicted_feet[..., [0, 2]] - previous_target_feet[..., [0, 2]],
        dim=-1,
    ) * float(fps)
    penalty = _radial_huber_loss(horizontal_speed, beta=huber_beta_mps)
    weight = contact_weight.to(penalty.dtype).clamp(0.0, 1.0)
    weight = weight * previous_frame_valid.to(penalty.dtype)[:, None]
    return (penalty * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)
