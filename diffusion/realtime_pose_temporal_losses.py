from __future__ import annotations

import torch
import torch.nn.functional as F

from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_torch,
    resolve_root_head_reference_torch,
)
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_torch
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
)


def _to_raw_pose(
    value: torch.Tensor,
    normalizer_mean: torch.Tensor | None,
    normalizer_std: torch.Tensor | None,
) -> torch.Tensor:
    if normalizer_mean is None or normalizer_std is None:
        return value
    mean = normalizer_mean.to(device=value.device, dtype=value.dtype)
    std = normalizer_std.to(device=value.device, dtype=value.dtype)
    return value * std + mean


def decode_rollout_frame_world_geometry(
    pred_xstart: torch.Tensor,
    batch: dict,
    normalizer_mean: torch.Tensor | None,
    normalizer_std: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """把单步预测和真值解码到世界系，供相邻 rollout 帧计算时序损失。"""

    if pred_xstart.ndim != 2:
        raise ValueError(f"pred_xstart 应为 [B,D]，实际为 {tuple(pred_xstart.shape)}")

    pred_raw = _to_raw_pose(pred_xstart, normalizer_mean, normalizer_std)
    target_raw = _to_raw_pose(batch["x"], normalizer_mean, normalizer_std)
    device = pred_raw.device
    dtype = pred_raw.dtype

    offsets = batch["joint_offsets_parent"].to(device=device, dtype=dtype)
    pred_rot_head, pred_root_yaw_head = decode_target_head_rotations_torch(pred_raw)
    target_rot_head, _ = decode_target_head_rotations_torch(target_raw)

    tracker_pos_head = batch["current_tracker_pos_head_ref"].to(device=device, dtype=dtype)
    _, _, pred_joints_head = resolve_root_head_reference_torch(
        pred_rot_head,
        pred_root_yaw_head,
        offsets,
        observed_head_height=tracker_pos_head[:, HEAD_TRACKER_INDEX, 1],
    )
    target_joints_head = batch["target_joints_head_ref"].to(device=device, dtype=dtype)

    # 每个 task 都使用当前 Head 的水平位置和 yaw 作为参考；时序差分前必须还原到同一世界系。
    head_yaw_world = batch["current_head_yaw_world"].to(device=device, dtype=dtype).reshape(-1)
    head_pos_world = batch["current_head_position_world"].to(device=device, dtype=dtype)
    floor_y = batch["floor_y"].to(device=device, dtype=dtype).reshape(-1)
    head_reference_origin = torch.stack(
        [head_pos_world[:, 0], floor_y, head_pos_world[:, 2]],
        dim=-1,
    )
    head_to_world = make_yaw_rotation_torch(head_yaw_world)
    pred_joints_world = (
        torch.einsum("bij,baj->bai", head_to_world, pred_joints_head)
        + head_reference_origin[:, None]
    )
    target_joints_world = (
        torch.einsum("bij,baj->bai", head_to_world, target_joints_head)
        + head_reference_origin[:, None]
    )
    pred_rot_world = torch.einsum("bij,bajk->baik", head_to_world, pred_rot_head)
    target_rot_world = torch.einsum("bij,bajk->baik", head_to_world, target_rot_head)

    known_mask = batch["known_mask"].to(device=device).bool()
    rotation_unknown = ~known_mask.reshape(
        pred_xstart.shape[0], SMPL_JOINT_COUNT, ROTATION_6D_DIM
    ).all(dim=-1)
    return {
        "pred_joints_world": pred_joints_world,
        "target_joints_world": target_joints_world,
        "pred_rot_world": pred_rot_world,
        "target_rot_world": target_rot_world,
        "rotation_unknown": rotation_unknown,
    }


def rollout_temporal_losses_from_geometry(
    pred_joints_world: torch.Tensor,
    target_joints_world: torch.Tensor,
    pred_rot_world: torch.Tensor,
    target_rot_world: torch.Tensor,
    rotation_unknown: torch.Tensor,
    joint_huber_beta: float = 0.01,
) -> dict[str, torch.Tensor]:
    """计算内部预测帧之间的关节速度与相对旋转损失。

    输入形状分别为 `[B,K,J,3]`、`[B,K,J,3,3]` 和 `[B,K,J]`。
    这里只比较 rollout 内部的预测帧，不把 GT 历史边界伪装成速度监督。
    """

    if pred_joints_world.ndim != 4 or pred_joints_world.shape[1] < 2:
        raise ValueError("时序损失至少需要两个预测帧，且 joints 应为 [B,K,J,3]。")
    if target_joints_world.shape != pred_joints_world.shape:
        raise ValueError("pred/target joints 形状必须一致。")
    if pred_rot_world.ndim != 5 or target_rot_world.shape != pred_rot_world.shape:
        raise ValueError("pred/target rotations 应为同形的 [B,K,J,3,3]。")
    if rotation_unknown.shape != pred_rot_world.shape[:3]:
        raise ValueError("rotation_unknown 应为 [B,K,J]。")
    if joint_huber_beta <= 0.0:
        raise ValueError("joint_huber_beta 必须大于 0。")

    pred_joint_delta = pred_joints_world[:, 1:] - pred_joints_world[:, :-1]
    target_joint_delta = target_joints_world[:, 1:] - target_joints_world[:, :-1]
    joint_delta_error = pred_joint_delta - target_joint_delta
    joint_vel_loss = F.smooth_l1_loss(
        joint_delta_error,
        torch.zeros_like(joint_delta_error),
        beta=float(joint_huber_beta),
        reduction="none",
    ).mean(dim=(1, 2, 3))

    pred_relative = pred_rot_world[:, :-1].transpose(-1, -2) @ pred_rot_world[:, 1:]
    target_relative = target_rot_world[:, :-1].transpose(-1, -2) @ target_rot_world[:, 1:]
    relative_error = pred_relative.transpose(-1, -2) @ target_relative
    cosine = (relative_error.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    safe_cosine = cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    rotation_error = torch.acos(safe_cosine)
    rotation_error = torch.where(cosine >= 1.0 - 1e-6, torch.zeros_like(rotation_error), rotation_error)

    pair_unknown = rotation_unknown[:, 1:] & rotation_unknown[:, :-1]
    pair_weight = pair_unknown.to(dtype=rotation_error.dtype)
    rotation_vel_loss = (rotation_error * pair_weight).sum(dim=(1, 2)) / pair_weight.sum(
        dim=(1, 2)
    ).clamp_min(1.0)
    return {
        "joint_vel_loss": joint_vel_loss,
        "rotation_vel_loss": rotation_vel_loss,
    }


def compute_rollout_temporal_losses(
    predictions: list[torch.Tensor],
    step_batches: list[dict],
    normalizer_mean: torch.Tensor | None,
    normalizer_std: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """解码 K 个因果预测，并在统一世界系中计算真正的多帧时序损失。"""

    if len(predictions) != len(step_batches) or len(predictions) < 2:
        raise ValueError("predictions 和 step_batches 必须等长，且至少包含两个 rollout 帧。")
    decoded = [
        decode_rollout_frame_world_geometry(
            prediction,
            step_batch,
            normalizer_mean=normalizer_mean,
            normalizer_std=normalizer_std,
        )
        for prediction, step_batch in zip(predictions, step_batches)
    ]
    return rollout_temporal_losses_from_geometry(
        pred_joints_world=torch.stack([frame["pred_joints_world"] for frame in decoded], dim=1),
        target_joints_world=torch.stack([frame["target_joints_world"] for frame in decoded], dim=1),
        pred_rot_world=torch.stack([frame["pred_rot_world"] for frame in decoded], dim=1),
        target_rot_world=torch.stack([frame["target_rot_world"] for frame in decoded], dim=1),
        rotation_unknown=torch.stack([frame["rotation_unknown"] for frame in decoded], dim=1),
    )
