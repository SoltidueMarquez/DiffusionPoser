from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_torch,
    resolve_root_head_reference_torch,
)
from data_loaders.realtime_pose_config import TARGET_JOINT_REGIONS
from data_loaders.realtime_pose_kinematics import SMPL_PARENTS, rotation_6d_to_matrix_torch
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    NON_HEAD_TRACKER_INDICES,
    SMPL_JOINT_COUNT,
    TRACKER_TO_JOINT,
)


def wrapped_angle_difference(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """返回 ``first-second`` 在 ``[-π,π]`` 上的连续有符号角差。"""

    difference = first - second
    return torch.atan2(torch.sin(difference), torch.cos(difference))


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

    global_rotation_loss = _rotation_angle(raw_global, target_global).square().mean(dim=1)
    parents = torch.as_tensor(SMPL_PARENTS[1:], device=raw_pred.device, dtype=torch.long)
    raw_local = raw_global[:, parents].transpose(-1, -2) @ raw_global[:, 1:]
    target_local = target_global[:, parents].transpose(-1, -2) @ target_global[:, 1:]
    local_rotation_loss = _rotation_angle(raw_local, target_local).square().mean(dim=1)

    current_tracker = batch["current_tracker_raw"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    tracker_rot = rotation_6d_to_matrix_torch(current_tracker[..., 3:9])
    measured = current_tracker[..., 10] > 0.5
    tracker_joints = torch.as_tensor(TRACKER_TO_JOINT, device=raw_pred.device, dtype=torch.long)
    tracker_rotation_error = _rotation_angle(raw_global.index_select(1, tracker_joints), tracker_rot)
    observation_weight = auxiliary_outputs.get("taid_observation_weight")
    if observation_weight is None:
        tracker_rotation_loss = _masked_mean(tracker_rotation_error.square(), measured)
    else:
        tracker_rotation_loss = _weighted_mean(
            tracker_rotation_error.square(),
            observation_weight.to(device=raw_pred.device, dtype=raw_pred.dtype),
        )

    offsets = batch["joint_offsets_parent"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    tracker_pos = current_tracker[..., :3]
    deployed_root_position, _, deployed_joints = resolve_root_head_reference_torch(
        deployed_global,
        deployed_root_yaw,
        offsets,
        observed_head_height=tracker_pos[:, HEAD_TRACKER_INDEX, 1],
    )
    target_joints = batch["target_joints_head_ref"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    fk_loss = torch.square(deployed_joints - target_joints).flatten(1).mean(dim=1)
    world_joint_loss = torch.linalg.norm(deployed_joints - target_joints, dim=-1).mean(dim=1)

    non_head_trackers = torch.as_tensor(NON_HEAD_TRACKER_INDICES, device=raw_pred.device, dtype=torch.long)
    non_head_joints = tracker_joints.index_select(0, non_head_trackers)
    tracker_distance = torch.linalg.norm(
        deployed_joints.index_select(1, non_head_joints) - tracker_pos.index_select(1, non_head_trackers),
        dim=-1,
    )
    tracker_position_error = _radial_huber_loss(tracker_distance, tracker_pos_huber_beta)
    if observation_weight is None:
        tracker_position_loss = _masked_mean(
            tracker_position_error,
            measured.index_select(1, non_head_trackers),
        )
    else:
        tracker_position_loss = _weighted_mean(
            tracker_position_error,
            observation_weight.to(device=raw_pred.device, dtype=raw_pred.dtype).index_select(
                1, non_head_trackers
            ),
        )

    predicted_root = deployed_root_position
    target_root = batch["target_root_position_head_ref"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    current_head_yaw = batch["current_head_yaw_world"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    target_root_yaw = batch["target_root_yaw_world"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    root_yaw_error = wrapped_angle_difference(
        deployed_root_yaw + current_head_yaw,
        target_root_yaw,
    )
    # Actor Root 的 Y 固定在 floor；root_loss 只监督 yaw，避免与下面的
    # Head-to-Root 水平几何项重复惩罚同一个 XZ 偏移。
    root_loss = root_yaw_error.square()
    head_to_root_xz_loss = torch.square(
        predicted_root[:, [0, 2]] - target_root[:, [0, 2]]
    ).mean(dim=-1)

    future_target = batch["future_leg_target"].to(device=raw_pred.device, dtype=raw_pred.dtype)
    future_prediction = auxiliary_outputs["future_leg"].to(dtype=raw_pred.dtype)
    if future_prediction.shape != future_target.shape:
        raise ValueError("future_leg 输出与监督必须同为 [B,3,8,6]。")
    future_leg_loss = torch.square(future_prediction - future_target).flatten(1).mean(dim=1)
    contact_target = (
        batch["contact_target"].to(device=raw_pred.device, dtype=raw_pred.dtype) >= 0.5
    ).to(raw_pred.dtype)
    contact_logits = auxiliary_outputs["contact_logits"].to(dtype=raw_pred.dtype)
    contact_loss = F.binary_cross_entropy_with_logits(
        contact_logits,
        contact_target,
        reduction="none",
    ).mean(dim=-1)
    return {
        "global_rotation_loss": global_rotation_loss,
        "local_rotation_loss": local_rotation_loss,
        "tracker_rotation_loss": tracker_rotation_loss,
        "fk_loss": fk_loss,
        "tracker_position_loss": tracker_position_loss,
        "root_loss": root_loss,
        "world_joint_loss": world_joint_loss,
        "head_to_root_xz_loss": head_to_root_xz_loss,
        "future_leg_loss": future_leg_loss,
        "contact_loss": contact_loss,
    }


def compute_anchor_prior_losses(
    target_xstart: torch.Tensor,
    deployed_pred_xstart: torch.Tensor,
    batch: dict,
    auxiliary_outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """计算 B1 Prior 监督；主 FK 严格复用部署 Resolver，内部 FK 仅诊断。"""

    target_raw = _to_raw_pose(target_xstart, batch)
    target_rotations, _ = decode_target_head_rotations_torch(target_raw)
    prior_raw = auxiliary_outputs["taid_prior_pose_raw"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    prior_rotations, _ = decode_target_head_rotations_torch(prior_raw)
    coverage = auxiliary_outputs["taid_alpha"].new_zeros(
        (target_raw.shape[0], 5), dtype=target_raw.dtype
    )
    # region coverage 已在 Prior 内按 Anchor alpha 聚合，直接从已冻结的输出诊断张量读取。
    if "taid_region_coverage" in auxiliary_outputs:
        coverage = auxiliary_outputs["taid_region_coverage"].to(
            device=target_raw.device, dtype=target_raw.dtype
        )
    else:
        # 兼容第一版辅助字段：按角色 alpha 与固定覆盖矩阵恢复同一语义。
        from data_loaders.tracker_roles import ANCHOR_REGION_COVERAGE

        routes = torch.as_tensor(
            ANCHOR_REGION_COVERAGE,
            device=target_raw.device,
            dtype=target_raw.dtype,
        )
        coverage = torch.einsum(
            "bt,rt->br",
            auxiliary_outputs["taid_alpha"].to(target_raw.dtype),
            routes,
        ).clamp(max=1.0)
    joint_regions = torch.tensor(TARGET_JOINT_REGIONS.copy(), device=target_raw.device)
    region_weight = 0.25 + 0.75 * coverage
    joint_weight = region_weight.index_select(1, joint_regions)
    rotation_error = _rotation_angle(prior_rotations, target_rotations)
    prior_rotation_loss = _weighted_mean(rotation_error, joint_weight)

    prior_internal_joints = auxiliary_outputs["taid_prior_joints_head"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    target_joints = batch["target_joints_head_ref"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )

    # B1 的主 FK 必须与部署一致：先使用 hard rotation projection 后的 144D pose，
    # 再由稳定的 Head-Anchored Resolver 恢复 Root 和关节。内部 Root MLP xyz 不参与此路径。
    deployed_raw = _to_raw_pose(deployed_pred_xstart, batch)
    deployed_rotations, deployed_root_yaw = decode_target_head_rotations_torch(
        deployed_raw
    )
    current_tracker = batch["current_tracker_raw"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    joint_offsets = batch["joint_offsets_parent"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    deployed_root, _, deployed_joints = resolve_root_head_reference_torch(
        deployed_rotations,
        deployed_root_yaw,
        joint_offsets,
        observed_head_height=current_tracker[:, HEAD_TRACKER_INDEX, 1],
    )
    deployed_joint_l1 = (deployed_joints - target_joints).abs().mean(dim=-1)
    prior_fk_loss = _weighted_mean(deployed_joint_l1, joint_weight)
    internal_joint_l1 = (prior_internal_joints - target_joints).abs().mean(dim=-1)
    prior_internal_fk_loss = _weighted_mean(internal_joint_l1, joint_weight)

    root_head = auxiliary_outputs["taid_prior_root_head"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    target_root = batch["target_root_position_head_ref"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    target_root_yaw = batch["target_root_yaw_world"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    current_head_yaw = batch["current_head_yaw_world"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    relative_target_yaw = wrapped_angle_difference(target_root_yaw, current_head_yaw)
    # root_head[:,3] 已从 prior_pose_raw 的 Pelvis rotation 派生；该 circular
    # 监督会直接回传到真正部署的 144D pose，而不是训练一个平行但未消费的 yaw。
    yaw_error = wrapped_angle_difference(root_head[:, 3], relative_target_yaw)
    prior_root_loss = (
        (root_head[:, :3] - target_root).abs().mean(dim=-1) + yaw_error.abs()
    )
    root_gap = root_head[:, :3] - deployed_root
    prior_root_pose_gap_m = torch.linalg.norm(root_gap, dim=-1)
    prior_root_pose_gap_xz_m = torch.linalg.norm(root_gap[:, [0, 2]], dim=-1)
    prior_joint_resolver_gap_m = torch.linalg.norm(
        prior_internal_joints - deployed_joints,
        dim=-1,
    ).mean(dim=-1)

    previous_joints = batch["prev_joints_head_ref"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    predicted_velocity = auxiliary_outputs["taid_prior_joint_velocity_head"].to(
        device=target_raw.device, dtype=target_raw.dtype
    )
    target_velocity = target_joints - previous_joints
    velocity_l1 = (predicted_velocity - target_velocity).abs().mean(dim=-1)
    prior_velocity_loss = _weighted_mean(velocity_l1, joint_weight)

    contact_target = (batch["contact_target"].to(target_raw.device) >= 0.5).to(target_raw.dtype)
    contact_logits = auxiliary_outputs["taid_prior_contact_logits"].to(target_raw.dtype)
    prior_contact_loss = F.binary_cross_entropy_with_logits(
        contact_logits, contact_target, reduction="none"
    ).mean(dim=-1)
    return {
        "prior_rotation_loss": prior_rotation_loss,
        "prior_fk_loss": prior_fk_loss,
        "prior_internal_fk_loss": prior_internal_fk_loss,
        "prior_root_pose_gap_m": prior_root_pose_gap_m,
        "prior_root_pose_gap_xz_m": prior_root_pose_gap_xz_m,
        "prior_joint_resolver_gap_m": prior_joint_resolver_gap_m,
        "prior_root_loss": prior_root_loss,
        "prior_velocity_loss": prior_velocity_loss,
        "prior_contact_loss": prior_contact_loss,
    }


def compute_foot_slide_loss(
    previous_joints: torch.Tensor,
    current_joints: torch.Tensor,
    previous_contact: torch.Tensor,
    current_contact: torch.Tensor,
) -> torch.Tensor:
    """相邻两帧都接触时才惩罚左右脚 deployed 世界位移。"""

    foot_indices = torch.as_tensor([10, 11], device=current_joints.device, dtype=torch.long)
    displacement = torch.linalg.norm(
        current_joints.index_select(1, foot_indices) - previous_joints.index_select(1, foot_indices),
        dim=-1,
    )
    active = (previous_contact >= 0.5) & (current_contact >= 0.5)
    return _masked_mean(displacement, active)


def _to_raw_pose(value: torch.Tensor, batch: dict) -> torch.Tensor:
    mean = batch.get("normalizer_mean")
    std = batch.get("normalizer_std")
    if mean is None or std is None:
        return value
    return value * std.to(device=value.device, dtype=value.dtype) + mean.to(
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


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(value.dtype)
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
