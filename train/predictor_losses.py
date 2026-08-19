from __future__ import annotations

import torch

from data_loaders.realtime_pose_geometry import pelvis_relative_joint_positions_torch
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    PREDICTOR_POSE_HORIZON_LENGTH,
    SMPL_JOINT_COUNT,
)


def compute_predictor_losses(
    prediction_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    motion_context_normalized: torch.Tensor,
    joint_offsets_parent: torch.Tensor,
    pose_mean: torch.Tensor,
    pose_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """计算逐样本 Predictor rotation、velocity、FK 与 FK velocity loss。"""

    batch_size = prediction_normalized.shape[0]
    expected = (batch_size, PREDICTOR_POSE_HORIZON_LENGTH, REALTIME_POSE_TARGET_DIM)
    if tuple(prediction_normalized.shape) != expected or tuple(
        target_normalized.shape
    ) != expected:
        raise ValueError("Predictor prediction/target 必须为 [B,11,144]。")
    if tuple(motion_context_normalized.shape) != (
        batch_size,
        REALTIME_POSE_HISTORY_LENGTH,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError("motion_context_normalized 必须为 [B,10,144]。")

    pose_mse = (prediction_normalized - target_normalized).square().flatten(1).mean(1)
    mean = pose_mean.to(prediction_normalized)
    scale = pose_scale.to(prediction_normalized)
    prediction_raw = prediction_normalized * scale + mean
    target_raw = target_normalized * scale + mean
    context_last_raw = motion_context_normalized[:, -1] * scale + mean

    prediction_rot = rotation_6d_to_matrix_torch(
        prediction_raw.reshape(batch_size, PREDICTOR_POSE_HORIZON_LENGTH, SMPL_JOINT_COUNT, 6)
    )
    target_rot = rotation_6d_to_matrix_torch(
        target_raw.reshape(batch_size, PREDICTOR_POSE_HORIZON_LENGTH, SMPL_JOINT_COUNT, 6)
    )
    context_last_rot = rotation_6d_to_matrix_torch(
        context_last_raw.reshape(batch_size, SMPL_JOINT_COUNT, 6)
    )
    prediction_sequence = torch.cat([context_last_rot[:, None], prediction_rot], dim=1)
    target_sequence = torch.cat([context_last_rot[:, None], target_rot], dim=1)
    prediction_velocity = (
        prediction_sequence[:, :-1].transpose(-1, -2) @ prediction_sequence[:, 1:]
    )
    target_velocity = (
        target_sequence[:, :-1].transpose(-1, -2) @ target_sequence[:, 1:]
    )
    rotation_velocity = _rotation_angle(
        prediction_velocity, target_velocity
    ).square().flatten(1).mean(1)

    offsets = joint_offsets_parent.to(prediction_raw)
    prediction_joints = _head_aligned_joints(prediction_rot, offsets)
    target_joints = _head_aligned_joints(target_rot, offsets)
    context_joints = _head_aligned_joints(context_last_rot[:, None], offsets)
    fk = (prediction_joints - target_joints).square().flatten(1).mean(1)
    prediction_joint_sequence = torch.cat([context_joints, prediction_joints], dim=1)
    target_joint_sequence = torch.cat([context_joints, target_joints], dim=1)
    fk_velocity = (
        (
            prediction_joint_sequence[:, 1:]
            - prediction_joint_sequence[:, :-1]
            - target_joint_sequence[:, 1:]
            + target_joint_sequence[:, :-1]
        )
        .square()
        .flatten(1)
        .mean(1)
    )
    total = pose_mse + rotation_velocity + fk + fk_velocity
    return {
        "loss": total,
        "pose_mse": pose_mse,
        "rotation_velocity_loss": rotation_velocity,
        "fk_loss": fk,
        "fk_velocity_loss": fk_velocity,
    }


def _head_aligned_joints(
    rotations: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    batch_size, time_count = rotations.shape[:2]
    flat_rotations = rotations.reshape(batch_size * time_count, SMPL_JOINT_COUNT, 3, 3)
    flat_offsets = offsets[:, None].expand(-1, time_count, -1, -1).reshape(
        batch_size * time_count, SMPL_JOINT_COUNT, 3
    )
    positions = pelvis_relative_joint_positions_torch(flat_rotations, flat_offsets)
    positions = positions - positions[:, JOINT_INDEX["head"] : JOINT_INDEX["head"] + 1]
    return positions.reshape(batch_size, time_count, SMPL_JOINT_COUNT, 3)


def _rotation_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    relative = first.float().transpose(-1, -2) @ second.float()
    skew = 0.5 * torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sin_angle = torch.linalg.norm(skew, dim=-1)
    cos_angle = (
        (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    ).clamp(-1.0, 1.0)
    return torch.atan2(sin_angle, cos_angle)
