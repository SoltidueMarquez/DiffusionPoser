from __future__ import annotations

from typing import Any, Mapping

import torch

from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    TRACKER_JOINT_INDICES,
    make_yaw_rotation_torch,
    rotation_6d_forward_up_torch,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    HIP_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
    POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D,
    REALTIME_POSE_TARGET_START,
    RIGHT_FOOT_TRACKER_INDEX,
    get_schema_spec,
)
from diffusion.realtime_pose.config import (
    REALTIME_POSE_LOSS_TERM_TO_WEIGHT,
    RealtimePoseLossConfig,
)
from diffusion.realtime_pose.resolver import (
    resolve_realtime_pose_frame_torch,
    rotation_matrix_log_vector,
    wrap_angle_torch,
)


def _rotation_axis_cosine_loss(pred_rotation_6d: torch.Tensor, target_rotation_6d: torch.Tensor) -> torch.Tensor:
    """在 FP32 中计算 forward/up 方向余弦损失，避免混合精度使损失出现负值。"""

    if pred_rotation_6d.shape != target_rotation_6d.shape or pred_rotation_6d.shape[-1] != 6:
        raise ValueError(
            "rotation axis cosine loss expects matching [..., 6] tensors, "
            f"got {tuple(pred_rotation_6d.shape)} and {tuple(target_rotation_6d.shape)}"
        )
    pred_axes = pred_rotation_6d.float().reshape(*pred_rotation_6d.shape[:-1], 2, 3)
    target_axes = target_rotation_6d.float().reshape(*target_rotation_6d.shape[:-1], 2, 3)
    pred_axes = torch.nn.functional.normalize(pred_axes, dim=-1, eps=1e-8)
    target_axes = torch.nn.functional.normalize(target_axes, dim=-1, eps=1e-8)
    cosine = (pred_axes * target_axes).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cosine).mean(dim=-1)


def _smooth_l1(values: torch.Tensor, targets: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.nn.functional.smooth_l1_loss(values, targets, beta=float(beta), reduction="none")


def _normalize_masked_samples(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """返回 `[B]` 损失；无效样本为零且不进入 batch mean 的分母。"""

    if loss.ndim != 1 or mask.shape != loss.shape:
        raise ValueError(f"masked sample loss expects matching [B], got {loss.shape} and {mask.shape}")
    weight = mask.to(device=loss.device, dtype=loss.dtype)
    count = weight.sum()
    scale = float(loss.shape[0]) / count.clamp_min(1.0)
    return loss * weight * scale


def _normalize_weighted_feet(
    loss: torch.Tensor,
    confidence: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """按左右脚独立归一化，soft confidence 只缩放分子，不抵消其监督强度。"""

    if loss.ndim != 2 or loss.shape[1] != 2 or confidence.shape != loss.shape:
        raise ValueError(
            f"weighted foot loss expects matching [B,2], got {loss.shape} and {confidence.shape}"
        )
    if active_mask is None:
        active_mask = confidence > 0
    if active_mask.shape != loss.shape:
        raise ValueError(f"active foot mask expects {tuple(loss.shape)}, got {tuple(active_mask.shape)}")
    batch_size = int(loss.shape[0])
    result = torch.zeros(batch_size, device=loss.device, dtype=loss.dtype)
    active_feet = torch.zeros((), device=loss.device, dtype=loss.dtype)
    for foot_index in range(2):
        foot_active = active_mask[:, foot_index].to(device=loss.device, dtype=loss.dtype)
        foot_confidence = confidence[:, foot_index].to(device=loss.device, dtype=loss.dtype)
        denominator = foot_active.sum()
        is_active = (denominator > 0).to(dtype=loss.dtype)
        result = result + loss[:, foot_index] * foot_confidence * foot_active * (
            float(batch_size) / denominator.clamp_min(1.0)
        )
        active_feet = active_feet + is_active
    return result / active_feet.clamp_min(1.0)


class RealtimePoseAuxiliaryLoss:
    """第 61 帧 realtime pose 物理约束和 rollout 监督的纯损失计算。"""

    def __init__(self, config: RealtimePoseLossConfig, *, num_timesteps: int) -> None:
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        self.config = config
        self.num_timesteps = int(num_timesteps)

    def _realtime_pose_slice_to_raw(self, values, y, start, end):
        """把归一化后的 realtime 特征切片还原到物理尺度。"""

        mean = y.get("normalizer_mean")
        std = y.get("normalizer_std")
        if mean is None or std is None:
            return values
        return values * std[start:end].view(1, -1) + mean[start:end].view(1, -1)

    def _aux_timestep_weight(self, timesteps, batch_size, device, dtype):
        """低噪声阶段的 pred_xstart 更可信，因此统一衰减全部辅助监督。"""

        if timesteps is None:
            return torch.ones(batch_size, device=device, dtype=dtype)
        timesteps = timesteps.to(device=device, dtype=dtype).view(-1)
        if timesteps.shape[0] != batch_size:
            raise ValueError(f"aux timestep batch 不匹配：t={tuple(timesteps.shape)} batch_size={batch_size}")
        if self.num_timesteps <= 1:
            progress = torch.ones_like(timesteps)
        else:
            progress = 1.0 - timesteps / float(self.num_timesteps - 1)
        progress = progress.clamp(0.0, 1.0)
        return self.config.aux_timestep_min_weight + (
            1.0 - self.config.aux_timestep_min_weight
        ) * progress.pow(self.config.aux_timestep_gamma)

    def compute(self, pred_xstart, x_start, model_kwargs, timesteps=None):
        # FK、rotation log 和速度差分在 BF16 下会把接近单位旋转的 acos 上界舍入到 1，
        # loss 本身仍 finite，但反向导数可能发散。辅助几何统一以 FP32 计算并保留回模型的梯度。
        with torch.autocast(device_type=pred_xstart.device.type, enabled=False):
            return self._compute_fp32(
                pred_xstart.float(),
                x_start.float(),
                model_kwargs,
                timesteps=timesteps,
            )

    def _compute_fp32(self, pred_xstart, x_start, model_kwargs, timesteps=None):
        """计算第 61 帧与 RuntimeRootResolver 对齐的 realtime pose loss 辅助监督。"""
        y = model_kwargs.get("y", {}) if model_kwargs is not None else {}
        if "schema_name" not in y:
            return {}
        schema = get_schema_spec(y["schema_name"])
        if schema.pose_representation != POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D:
            raise ValueError(f"realtime pose loss 只支持 body FBX local delta 6D，实际为 {schema.pose_representation}")
        required = (
            "target_joints_world",
            "gt_prev_joints_world",
            "pred_prev_joints_world",
            "gt_prev_local_pose_6d",
            "pred_prev_local_pose_6d",
            "previous_state_is_predicted",
            "target_frame_dt_seconds",
            "target_root_yaw",
            "target_root_pos_world",
            "gt_prev_root_yaw",
            "prev_root_yaw",
            "target_sensor_valid",
            "target_stationary_prob_5",
            "target_floor_y",
            "target_tracker_pos_ref",
            "target_tracker_rot_ref_6d",
            "tracker_ref_root_pos_world",
            "tracker_ref_root_yaw",
            "joint_offsets_parent",
            "resolver_before_target_root_pos_world",
            "resolver_before_target_root_yaw",
            "resolver_before_target_pelvis_height",
            "resolver_before_target_hip_valid",
            "resolver_before_target_reconnect_start_root_pos_world",
            "resolver_before_target_reconnect_start_root_yaw",
            "resolver_before_target_reconnect_start_pelvis_height",
            "resolver_before_target_reconnect_elapsed_seconds",
            "resolver_before_target_last_timestamp_seconds",
            "resolver_before_target_tracking_origin_revision",
            "target_timestamp_seconds",
            "target_tracking_origin_revision",
        )
        missing = [name for name in required if name not in y]
        if missing:
            raise KeyError(f"{schema.name} realtime pose loss 缺少 batch 字段：{missing}")

        device = pred_xstart.device
        dtype = pred_xstart.dtype
        batch_size = int(pred_xstart.shape[0])
        frame = REALTIME_POSE_TARGET_START
        pose_slice = schema.body_pose_slice()
        yaw_slice = schema.root_yaw_delta_slice()
        root_delta_slice = schema.root_delta_xz_slice()
        height_slice = schema.root_height_slice()
        stationary_slice = schema.stationary_prob_slice()

        def raw_slice(values: torch.Tensor, feature_slice: slice) -> torch.Tensor:
            return self._realtime_pose_slice_to_raw(
                values[:, feature_slice, frame], y, feature_slice.start, feature_slice.stop
            )

        pred_pose = raw_slice(pred_xstart, pose_slice)
        gt_pose = raw_slice(x_start, pose_slice)
        pred_yaw_pair = raw_slice(pred_xstart, yaw_slice)
        pred_root_delta = raw_slice(pred_xstart, root_delta_slice)
        pred_height = raw_slice(pred_xstart, height_slice).view(-1)
        resolver = resolve_realtime_pose_frame_torch(
            pred_pose=pred_pose,
            pred_root_delta_xz_ref=pred_root_delta,
            pred_yaw_delta_sincos=pred_yaw_pair,
            pred_pelvis_height=pred_height,
            y=y,
        )

        target_joints = y["target_joints_world"].to(device=device, dtype=dtype)
        gt_prev_joints = y["gt_prev_joints_world"].to(device=device, dtype=dtype)
        pred_prev_joints = y["pred_prev_joints_world"].to(device=device, dtype=dtype)
        gt_root_yaw = y["target_root_yaw"].to(device=device, dtype=dtype).view(-1)
        gt_prev_root_yaw = y["gt_prev_root_yaw"].to(device=device, dtype=dtype).view(-1)
        pred_prev_root_yaw = y["prev_root_yaw"].to(device=device, dtype=dtype).view(-1)
        temporal_valid = y["previous_state_is_predicted"].to(device=device).bool().view(-1)
        frame_dt = y["target_frame_dt_seconds"].to(device=device, dtype=dtype).view(-1)
        if frame_dt.shape != (batch_size,) or torch.any(frame_dt <= 0.0):
            raise ValueError(f"target_frame_dt_seconds 必须为正的 [B]，实际为 {frame_dt}")
        valid = y["target_sensor_valid"].to(device=device).bool()
        if valid.shape != (batch_size, 6):
            raise ValueError(f"target_sensor_valid 应为 [B,6]，实际为 {tuple(valid.shape)}")
        hip_missing = ~valid[:, HIP_TRACKER_INDEX]
        left_missing = hip_missing | (~valid[:, LEFT_FOOT_TRACKER_INDEX])
        right_missing = hip_missing | (~valid[:, RIGHT_FOOT_TRACKER_INDEX])

        # 全身局部旋转为基础；缺失侧腿部再增加一份等权重监督。
        pred_local = pred_pose.reshape(batch_size, 24, 6)
        gt_local = gt_pose.reshape(batch_size, 24, 6)
        local_rotation_per_joint = _rotation_axis_cosine_loss(pred_local, gt_local)
        rotation_joint_weight = torch.ones_like(local_rotation_per_joint)
        left_leg_indices = torch.as_tensor(
            [JOINT_INDEX[name] for name in ("left_hip", "left_knee", "left_ankle", "left_foot")],
            device=device,
        )
        right_leg_indices = torch.as_tensor(
            [JOINT_INDEX[name] for name in ("right_hip", "right_knee", "right_ankle", "right_foot")],
            device=device,
        )
        rotation_joint_weight[:, left_leg_indices] += left_missing[:, None].to(dtype=rotation_joint_weight.dtype)
        rotation_joint_weight[:, right_leg_indices] += right_missing[:, None].to(dtype=rotation_joint_weight.dtype)
        local_rotation_loss = (
            local_rotation_per_joint * rotation_joint_weight
        ).sum(dim=1) / rotation_joint_weight.sum(dim=1).clamp_min(1.0)

        # 对齐 pelvis 和各自 root heading 后只比较身体几何，不混入 root 平移/朝向误差。
        pelvis_index = JOINT_INDEX["pelvis"]
        pred_centered = resolver.final_joints_world - resolver.final_joints_world[:, pelvis_index, None]
        gt_centered = target_joints - target_joints[:, pelvis_index, None]
        pred_local_joints = torch.einsum(
            "bij,bkj->bki",
            make_yaw_rotation_torch(resolver.final_root_yaw).transpose(-1, -2),
            pred_centered,
        )
        gt_local_joints = torch.einsum(
            "bij,bkj->bki",
            make_yaw_rotation_torch(gt_root_yaw).transpose(-1, -2),
            gt_centered,
        )
        geometry_distance = torch.linalg.norm(pred_local_joints - gt_local_joints, dim=-1)
        geometry_per_joint = _smooth_l1(
            geometry_distance,
            torch.zeros_like(geometry_distance),
            self.config.geometry_huber_beta,
        )
        geometry_joint_weight = torch.ones_like(geometry_per_joint)
        geometry_joint_weight[:, pelvis_index] = 0.0
        geometry_joint_weight[:, left_leg_indices] += left_missing[:, None].to(dtype=dtype)
        geometry_joint_weight[:, right_leg_indices] += right_missing[:, None].to(dtype=dtype)
        geometry_joint_weight[:, JOINT_INDEX["head"]] += hip_missing.to(dtype=dtype)
        body_geometry_loss = (
            geometry_per_joint * geometry_joint_weight
        ).sum(dim=1) / geometry_joint_weight.sum(dim=1).clamp_min(1.0)

        tracker_indices = torch.as_tensor(TRACKER_JOINT_INDICES, device=device, dtype=torch.long)
        pred_tracker_pos = resolver.final_joints_world[:, tracker_indices]
        pred_tracker_rot = resolver.final_joint_rot_world[:, tracker_indices]
        observed_tracker_pos = resolver.tracker_pos_world
        observed_tracker_rot = resolver.tracker_rot_world
        anchor_index = torch.where(
            valid[:, HIP_TRACKER_INDEX],
            torch.full((batch_size,), HIP_TRACKER_INDEX, device=device, dtype=torch.long),
            torch.zeros(batch_size, device=device, dtype=torch.long),
        )
        batch_index = torch.arange(batch_size, device=device)
        pred_anchor_pos = pred_tracker_pos[batch_index, anchor_index]
        pred_anchor_rot = pred_tracker_rot[batch_index, anchor_index]
        observed_anchor_pos = observed_tracker_pos[batch_index, anchor_index]
        observed_anchor_rot = observed_tracker_rot[batch_index, anchor_index]
        pred_relative_pos = torch.einsum(
            "bij,bkj->bki",
            pred_anchor_rot.transpose(-1, -2),
            pred_tracker_pos - pred_anchor_pos[:, None],
        )
        observed_relative_pos = torch.einsum(
            "bij,bkj->bki",
            observed_anchor_rot.transpose(-1, -2),
            observed_tracker_pos - observed_anchor_pos[:, None],
        )
        tracker_distance = torch.linalg.norm(pred_relative_pos - observed_relative_pos, dim=-1)
        tracker_pos_per = _smooth_l1(
            tracker_distance,
            torch.zeros_like(tracker_distance),
            self.config.tracker_relative_pos_huber_beta,
        )
        relative_valid = valid.clone()
        relative_valid[batch_index, anchor_index] = False
        relative_weight = relative_valid.to(dtype=dtype)
        tracker_relative_pos_loss = (
            tracker_pos_per * relative_weight
        ).sum(dim=1) / relative_weight.sum(dim=1).clamp_min(1.0)

        pred_relative_rot = torch.einsum(
            "bij,bkjl->bkil", pred_anchor_rot.transpose(-1, -2), pred_tracker_rot
        )
        observed_relative_rot = torch.einsum(
            "bij,bkjl->bkil", observed_anchor_rot.transpose(-1, -2), observed_tracker_rot
        )
        tracker_rot_per = _rotation_axis_cosine_loss(
            rotation_6d_forward_up_torch(pred_relative_rot),
            rotation_6d_forward_up_torch(observed_relative_rot),
        )
        tracker_relative_rot_loss = (
            tracker_rot_per * relative_weight
        ).sum(dim=1) / relative_weight.sum(dim=1).clamp_min(1.0)

        nohip_yaw_raw = 1.0 - torch.cos(wrap_angle_torch(resolver.final_root_yaw - gt_root_yaw))
        nohip_yaw_loss = _normalize_masked_samples(nohip_yaw_raw, hip_missing)
        target_root_pos = y["target_root_pos_world"].to(device=device, dtype=dtype)
        nohip_root_xz_raw = _smooth_l1(
            resolver.final_root_pos_world[:, [0, 2]],
            target_root_pos[:, [0, 2]],
            self.config.nohip_root_xz_huber_beta,
        ).mean(dim=1)
        nohip_root_xz_loss = _normalize_masked_samples(nohip_root_xz_raw, hip_missing)
        floor_y = y["target_floor_y"].to(device=device, dtype=dtype).view(-1)
        target_pelvis_height = target_joints[:, pelvis_index, 1] - floor_y
        nohip_height_raw = _smooth_l1(
            resolver.final_pelvis_height,
            target_pelvis_height,
            self.config.nohip_height_huber_beta,
        )
        nohip_height_loss = _normalize_masked_samples(nohip_height_raw, hip_missing)

        pred_stationary = raw_slice(pred_xstart, stationary_slice)
        target_stationary = y["target_stationary_prob_5"].to(device=device, dtype=dtype)
        if target_stationary.shape != (batch_size, 5):
            raise ValueError(f"target_stationary_prob_5 应为 [B,5]，实际为 {tuple(target_stationary.shape)}")
        stationary_regression_loss = (pred_stationary - target_stationary).square().mean(dim=1)
        threshold = float(self.config.stationary_runtime_threshold)
        margin = float(self.config.stationary_runtime_margin)
        confidence = (target_stationary - threshold).abs()
        active_target = target_stationary >= threshold
        inactive_target = ~active_target

        def normalized_margin(values: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            weights = confidence * active.to(dtype=dtype)
            denominator = weights.sum(dim=1)
            normalized = (values * weights).sum(dim=1) / denominator.clamp_min(1e-6)
            has_class = denominator > 0.0
            return torch.where(has_class, normalized, torch.zeros_like(normalized)), has_class

        active_margin_loss, has_active = normalized_margin(
            torch.relu((threshold + margin) - pred_stationary).square(),
            active_target,
        )
        inactive_margin_loss, has_inactive = normalized_margin(
            torch.relu(pred_stationary - (threshold - margin)).square(),
            inactive_target,
        )
        active_class_count = has_active.to(dtype=dtype) + has_inactive.to(dtype=dtype)
        stationary_margin_loss = (
            active_margin_loss + inactive_margin_loss
        ) / active_class_count.clamp_min(1.0)
        stationary_range_target = pred_stationary.detach().clamp(0.0, 1.0)
        stationary_range_loss = _smooth_l1(
            pred_stationary,
            stationary_range_target,
            self.config.stationary_range_huber_beta,
        ).mean(dim=1)

        dt = frame_dt[:, None, None]
        pred_velocity = (resolver.final_joints_world - pred_prev_joints) / dt
        gt_velocity = (target_joints - gt_prev_joints) / dt
        joint_velocity_raw = _smooth_l1(
            pred_velocity,
            gt_velocity,
            self.config.joint_velocity_huber_beta,
        ).flatten(1).mean(dim=1)
        joint_velocity_loss = _normalize_masked_samples(joint_velocity_raw, temporal_valid)

        pred_prev_local = y["pred_prev_local_pose_6d"].to(device=device, dtype=dtype).reshape(batch_size, 24, 6)
        gt_prev_local = y["gt_prev_local_pose_6d"].to(device=device, dtype=dtype).reshape(batch_size, 24, 6)
        pred_current_rot = rotation_6d_to_matrix_torch(pred_local)
        gt_current_rot = rotation_6d_to_matrix_torch(gt_local)
        pred_prev_rot = rotation_6d_to_matrix_torch(pred_prev_local)
        gt_prev_rot = rotation_6d_to_matrix_torch(gt_prev_local)
        pred_rotation_velocity = rotation_matrix_log_vector(
            pred_prev_rot.transpose(-1, -2) @ pred_current_rot
        ) / frame_dt[:, None, None]
        gt_rotation_velocity = rotation_matrix_log_vector(
            gt_prev_rot.transpose(-1, -2) @ gt_current_rot
        ) / frame_dt[:, None, None]
        rotation_velocity_raw = _smooth_l1(
            pred_rotation_velocity,
            gt_rotation_velocity,
            self.config.rotation_velocity_huber_beta,
        ).flatten(1).mean(dim=1)
        rotation_velocity_loss = _normalize_masked_samples(rotation_velocity_raw, temporal_valid)

        pred_yaw_velocity = wrap_angle_torch(resolver.final_root_yaw - pred_prev_root_yaw) / frame_dt
        gt_yaw_velocity = wrap_angle_torch(gt_root_yaw - gt_prev_root_yaw) / frame_dt
        yaw_velocity_raw = _smooth_l1(
            pred_yaw_velocity,
            gt_yaw_velocity,
            self.config.yaw_velocity_huber_beta,
        )
        yaw_velocity_loss = _normalize_masked_samples(
            yaw_velocity_raw, temporal_valid & hip_missing
        )

        foot_indices = torch.as_tensor(
            [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]],
            device=device,
            dtype=torch.long,
        )
        gt_foot_height = target_joints[:, foot_indices, 1] - floor_y[:, None]
        pred_foot_height = resolver.final_joints_world[:, foot_indices, 1] - floor_y[:, None]
        ground_contact = gt_foot_height <= self.config.foot_contact_height_threshold
        missing_side = torch.stack((left_missing, right_missing), dim=1)
        contact_confidence = target_stationary[:, 1:3].clamp(0.0, 1.0)
        # GT stationary 决定是否属于 contact；soft 值只缩放分子，二值 active mask 决定分母。
        contact_active = (
            missing_side
            & ground_contact
            & (contact_confidence >= self.config.stationary_runtime_threshold)
        )
        contact_height_raw = _smooth_l1(
            pred_foot_height,
            gt_foot_height,
            self.config.contact_height_huber_beta,
        )
        contact_height_loss = _normalize_weighted_feet(
            contact_height_raw,
            contact_confidence,
            contact_active,
        )
        contact_velocity_raw = _smooth_l1(
            pred_velocity[:, foot_indices][:, :, [0, 2]],
            gt_velocity[:, foot_indices][:, :, [0, 2]],
            self.config.contact_velocity_huber_beta,
        ).mean(dim=-1)
        contact_velocity_active = contact_active & temporal_valid[:, None]
        contact_velocity_loss = _normalize_weighted_feet(
            contact_velocity_raw,
            contact_confidence,
            contact_velocity_active,
        )

        return {
            "local_rotation_loss": local_rotation_loss,
            "body_geometry_loss": body_geometry_loss,
            "tracker_relative_pos_loss": tracker_relative_pos_loss,
            "tracker_relative_rot_loss": tracker_relative_rot_loss,
            "nohip_yaw_loss": nohip_yaw_loss,
            "nohip_root_xz_loss": nohip_root_xz_loss,
            "nohip_height_loss": nohip_height_loss,
            "stationary_regression_loss": stationary_regression_loss,
            "stationary_margin_loss": stationary_margin_loss,
            "stationary_range_loss": stationary_range_loss,
            "contact_height_loss": contact_height_loss,
            "contact_velocity_loss": contact_velocity_loss,
            "joint_velocity_loss": joint_velocity_loss,
            "rotation_velocity_loss": rotation_velocity_loss,
            "yaw_velocity_loss": yaw_velocity_loss,
            "aux_timestep_weight": self._aux_timestep_weight(
                timesteps,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            ),
            "hip_missing_fraction": hip_missing.to(dtype=dtype),
            "left_missing_fraction": left_missing.to(dtype=dtype),
            "right_missing_fraction": right_missing.to(dtype=dtype),
            "temporal_sample_fraction": temporal_valid.to(dtype=dtype),
            "contact_active_foot_count": contact_active.to(dtype=dtype).sum(dim=1),
        }

    def apply_weights(self, losses: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """为每个原始 Loss 添加 profile 权重，并构造统一 timestep 衰减后的辅助损失。"""

        result = dict(losses)
        weighted_terms = {
            f"{loss_name}_weighted": self.config.weight_for(loss_name) * losses[loss_name]
            for loss_name in REALTIME_POSE_LOSS_TERM_TO_WEIGHT
        }
        result.update(weighted_terms)
        aux_loss_before_timestep = torch.stack(tuple(weighted_terms.values()), dim=0).sum(dim=0)
        result["aux_loss_before_timestep"] = aux_loss_before_timestep
        aux_loss = losses["aux_timestep_weight"] * aux_loss_before_timestep
        result["aux_loss_before_global_weight"] = aux_loss
        result["aux_loss"] = self.config.aux_loss_weight * aux_loss
        return result
