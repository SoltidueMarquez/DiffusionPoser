from __future__ import annotations

import torch
import torch.nn.functional as F

from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_LENGTH,
    SMPL_JOINT_COUNT,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_TO_JOINT,
)


def project_rotation_6d_to_so3(
    rotation_6d: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """把任意 rotation6D 投影到稳定正交基，并以 forward/up 重新编码。"""

    if rotation_6d.shape[-1] != 6:
        raise ValueError("rotation6D 尾维必须为 6。")
    forward_raw = rotation_6d[..., :3]
    up_raw = rotation_6d[..., 3:6]
    forward_norm = torch.linalg.norm(forward_raw, dim=-1, keepdim=True)
    fallback_forward = torch.zeros_like(forward_raw)
    fallback_forward[..., 2] = 1.0
    forward = torch.where(
        forward_norm > eps,
        forward_raw / forward_norm.clamp_min(eps),
        fallback_forward,
    )
    up_orthogonal = up_raw - (up_raw * forward).sum(dim=-1, keepdim=True) * forward
    up_norm = torch.linalg.norm(up_orthogonal, dim=-1, keepdim=True)
    world_up = torch.zeros_like(up_raw)
    world_up[..., 1] = 1.0
    world_right = torch.zeros_like(up_raw)
    world_right[..., 0] = 1.0
    fallback_axis = torch.where(
        torch.abs((forward * world_up).sum(dim=-1, keepdim=True)) < 0.9,
        world_up,
        world_right,
    )
    fallback_up = fallback_axis - (fallback_axis * forward).sum(
        dim=-1, keepdim=True
    ) * forward
    fallback_up = F.normalize(fallback_up, dim=-1, eps=eps)
    up = torch.where(
        up_norm > eps,
        up_orthogonal / up_norm.clamp_min(eps),
        fallback_up,
    )
    right = F.normalize(torch.cross(up, forward, dim=-1), dim=-1, eps=eps)
    up = torch.cross(forward, right, dim=-1)
    return torch.cat([forward, up], dim=-1)


def project_realtime_pose_xstart(
    pred_xstart: torch.Tensor,
    current_tracker_raw: torch.Tensor,
    hard_rotation_state: torch.Tensor,
    pose_mean: torch.Tensor | None = None,
    pose_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """投影 11 帧全部关节，并只替换当前帧 hard Tracker 对应旋转。"""

    if pred_xstart.ndim != 3 or tuple(pred_xstart.shape[1:]) != (
        REALTIME_POSE_TARGET_LENGTH,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError(
            f"pred_xstart 必须为 [B,{REALTIME_POSE_TARGET_LENGTH},"
            f"{REALTIME_POSE_TARGET_DIM}]。"
        )
    batch_size = pred_xstart.shape[0]
    if tuple(current_tracker_raw.shape) != (
        batch_size,
        TRACKER_COUNT,
        TRACKER_FEATURE_DIM,
    ):
        raise ValueError("current_tracker_raw 必须为 [B,6,13]。")
    if tuple(hard_rotation_state.shape) != (batch_size, TRACKER_COUNT):
        raise ValueError("hard_rotation_state 必须为 [B,6]。")

    raw = _inverse_pose(pred_xstart, pose_mean, pose_scale)
    rotations = project_rotation_6d_to_so3(
        raw.reshape(
            batch_size,
            REALTIME_POSE_TARGET_LENGTH,
            SMPL_JOINT_COUNT,
            6,
        )
    )
    tracker_rotations = project_rotation_6d_to_so3(current_tracker_raw[..., 3:9])
    joint_indices = torch.as_tensor(
        TRACKER_TO_JOINT,
        device=pred_xstart.device,
        dtype=torch.long,
    )
    deployed = rotations.clone()
    # 一次更新全部 Tracker 关节，避免逐关节 Python 循环触发设备同步。
    deployed_current = deployed[:, 0]
    current_rotations = deployed_current.index_select(1, joint_indices)
    hard_current = hard_rotation_state.to(
        device=pred_xstart.device,
        dtype=torch.bool,
    )
    replacement = torch.where(
        hard_current[..., None],
        tracker_rotations,
        current_rotations,
    )
    deployed_current.index_copy_(1, joint_indices, replacement)
    deployed[:, 0] = deployed_current
    return _normalize_pose(
        deployed.reshape(
            batch_size,
            REALTIME_POSE_TARGET_LENGTH,
            REALTIME_POSE_TARGET_DIM,
        ),
        pose_mean,
        pose_scale,
    )


def _inverse_pose(
    value: torch.Tensor,
    mean: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or scale is None:
        return value
    return value * scale.to(device=value.device, dtype=value.dtype) + mean.to(
        device=value.device,
        dtype=value.dtype,
    )


def _normalize_pose(
    value: torch.Tensor,
    mean: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or scale is None:
        return value
    return (value - mean.to(device=value.device, dtype=value.dtype)) / scale.to(
        device=value.device,
        dtype=value.dtype,
    )
