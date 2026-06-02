from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from data_loaders.realtime_pose_kinematics import (
    SMPL_PARENTS,
    TRACKER_JOINT_INDICES,
    fk_parent_local_torch,
    rotation_6d_forward_up_torch,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    HIP_TRACKER_INDEX,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    SMPL_JOINT_COUNT,
    TRACKER_COUNT,
    get_schema_spec,
)


IK_INIT_MODE_RANDOM = "random"
IK_INIT_MODE_TRACKER_POSE = "tracker_pose"
IK_INIT_MODES = (IK_INIT_MODE_RANDOM, IK_INIT_MODE_TRACKER_POSE)

DEFAULT_IK_INIT_TIMESTEP = -1
DEFAULT_IK_INIT_RATIO = 0.4
DEFAULT_IK_INIT_ITERATIONS = 16
DEFAULT_IK_INIT_LR = 0.03
DEFAULT_IK_INIT_POS_WEIGHT = 1.0
DEFAULT_IK_INIT_ROT_WEIGHT = 0.2
DEFAULT_IK_INIT_REG_WEIGHT = 0.01
DEFAULT_IK_INIT_DELTA_LIMIT = 0.15


@dataclass(frozen=True)
class IkInitConfig:
    mode: str = IK_INIT_MODE_RANDOM
    timestep: int = DEFAULT_IK_INIT_TIMESTEP
    iterations: int = DEFAULT_IK_INIT_ITERATIONS
    lr: float = DEFAULT_IK_INIT_LR
    pos_weight: float = DEFAULT_IK_INIT_POS_WEIGHT
    rot_weight: float = DEFAULT_IK_INIT_ROT_WEIGHT
    reg_weight: float = DEFAULT_IK_INIT_REG_WEIGHT
    delta_limit: float = DEFAULT_IK_INIT_DELTA_LIMIT


def validate_ik_init_mode(mode: str) -> str:
    value = str(mode or IK_INIT_MODE_RANDOM)
    if value not in IK_INIT_MODES:
        raise ValueError(f"未知 ik_init_mode={value}，可选值为 {IK_INIT_MODES}")
    return value


def resolve_ik_init_timestep(diffusion: Any, timestep: int = DEFAULT_IK_INIT_TIMESTEP) -> int:
    num_timesteps = int(getattr(diffusion, "num_timesteps"))
    if num_timesteps <= 0:
        raise ValueError(f"diffusion.num_timesteps 必须大于 0，实际为 {num_timesteps}")
    if int(timestep) < 0:
        resolved = int(round(DEFAULT_IK_INIT_RATIO * float(num_timesteps - 1)))
    else:
        resolved = int(timestep)
    return max(0, min(num_timesteps - 1, resolved))


def skip_timesteps_from_start(diffusion: Any, start_timestep: int) -> int:
    num_timesteps = int(getattr(diffusion, "num_timesteps"))
    start = max(0, min(num_timesteps - 1, int(start_timestep)))
    return int(num_timesteps - start - 1)


def build_tracker_pose_init_image(
    conditioned_x: torch.Tensor,
    *,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
    normalizer: Any | None = None,
    joint_offsets_parent: torch.Tensor | np.ndarray | None = None,
    iterations: int = DEFAULT_IK_INIT_ITERATIONS,
    lr: float = DEFAULT_IK_INIT_LR,
    pos_weight: float = DEFAULT_IK_INIT_POS_WEIGHT,
    rot_weight: float = DEFAULT_IK_INIT_ROT_WEIGHT,
    reg_weight: float = DEFAULT_IK_INIT_REG_WEIGHT,
    delta_limit: float = DEFAULT_IK_INIT_DELTA_LIMIT,
) -> torch.Tensor:
    """
    从当前条件窗口构造 SDEdit 使用的 `init_image`，输入输出均为 `[B, C, 61]`。

    IK 在 raw feature 空间里求解，只覆盖第 61 帧 target 通道；历史帧和 tracker/sensor 条件
    保持调用方传入的模型输入空间数值，避免破坏 inpainting 条件。
    """

    schema = get_schema_spec(schema_name)
    if conditioned_x.ndim != 3 or tuple(conditioned_x.shape[1:]) != (schema.feature_dim, REALTIME_POSE_SEQ_LEN):
        raise ValueError(
            f"{schema.name} conditioned_x 应为 [B,{schema.feature_dim},{REALTIME_POSE_SEQ_LEN}]，"
            f"实际为 {tuple(conditioned_x.shape)}"
        )

    raw_btc = _to_raw_btc(conditioned_x=conditioned_x, normalizer=normalizer)
    offsets = _batch_joint_offsets(
        joint_offsets_parent=joint_offsets_parent,
        batch_size=conditioned_x.shape[0],
        device=conditioned_x.device,
        dtype=conditioned_x.dtype,
    )
    raw_init_frames = []
    for batch_index in range(conditioned_x.shape[0]):
        raw_init_frames.append(
            _solve_single_frame_tracker_pose_ik(
                previous_frame_raw=raw_btc[batch_index, REALTIME_POSE_TARGET_START - 1],
                current_frame_raw=raw_btc[batch_index, REALTIME_POSE_TARGET_START],
                joint_offsets_parent=None if offsets is None else offsets[batch_index],
                schema_name=schema.name,
                iterations=iterations,
                lr=lr,
                pos_weight=pos_weight,
                rot_weight=rot_weight,
                reg_weight=reg_weight,
                delta_limit=delta_limit,
            )
        )

    raw_init_frame = torch.stack(raw_init_frames, dim=0)
    if normalizer is None:
        model_frame = raw_init_frame
    else:
        model_frame = normalizer.normalize(raw_init_frame)

    init_image = conditioned_x.clone()
    init_image[:, schema.target_slice(), REALTIME_POSE_TARGET_START] = model_frame[:, schema.target_slice()]
    return init_image


def _to_raw_btc(conditioned_x: torch.Tensor, normalizer: Any | None) -> torch.Tensor:
    features_btc = conditioned_x.transpose(1, 2).contiguous()
    if normalizer is None:
        return features_btc.clone()
    return normalizer.inverse(features_btc).clone()


def _batch_joint_offsets(
    joint_offsets_parent: torch.Tensor | np.ndarray | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if joint_offsets_parent is None:
        return None
    offsets = torch.as_tensor(joint_offsets_parent, dtype=dtype, device=device)
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).repeat(int(batch_size), 1, 1)
    if tuple(offsets.shape) != (int(batch_size), SMPL_JOINT_COUNT, 3):
        raise ValueError(f"joint_offsets_parent 应为 [B,24,3] 或 [24,3]，实际为 {tuple(offsets.shape)}")
    return offsets


def _solve_single_frame_tracker_pose_ik(
    *,
    previous_frame_raw: torch.Tensor,
    current_frame_raw: torch.Tensor,
    joint_offsets_parent: torch.Tensor | None,
    schema_name: str,
    iterations: int,
    lr: float,
    pos_weight: float,
    rot_weight: float,
    reg_weight: float,
    delta_limit: float,
) -> torch.Tensor:
    schema = get_schema_spec(schema_name)
    init_frame = current_frame_raw.clone()
    init_frame[schema.target_slice()] = previous_frame_raw[schema.target_slice()]

    tracker_pos_ref = current_frame_raw[schema.tracker_pos_slice()].reshape(TRACKER_COUNT, 3)
    tracker_rot_ref_6d = current_frame_raw[schema.tracker_rot_slice()].reshape(TRACKER_COUNT, 6)
    sensor_valid = current_frame_raw[schema.sensor_valid_slice()] > 0.5

    yaw_delta = _estimate_yaw_delta_from_hip_tracker(
        hip_tracker_rot_ref_6d=tracker_rot_ref_6d[HIP_TRACKER_INDEX],
        hip_valid=bool(sensor_valid[HIP_TRACKER_INDEX].item()),
        fallback=previous_frame_raw[schema.root_yaw_delta_slice()],
    )
    init_frame[schema.root_yaw_delta_slice()] = torch.stack([torch.sin(yaw_delta), torch.cos(yaw_delta)])
    if schema.supports_root_motion:
        init_frame[schema.root_delta_xz_slice()] = previous_frame_raw[schema.root_delta_xz_slice()]
        init_frame[schema.root_height_slice()] = tracker_pos_ref[HIP_TRACKER_INDEX, 1:2]
    if schema.supports_contact:
        init_frame[schema.foot_contact_slice()] = previous_frame_raw[schema.foot_contact_slice()]

    valid_nonhip = sensor_valid.clone()
    valid_nonhip[HIP_TRACKER_INDEX] = False
    ik_tracker_indices = torch.nonzero(valid_nonhip, as_tuple=False).flatten()
    if (
        joint_offsets_parent is None
        or int(iterations) <= 0
        or ik_tracker_indices.numel() == 0
        or float(pos_weight) <= 0.0 and float(rot_weight) <= 0.0
    ):
        init_frame[schema.body_pose_slice()] = _normalize_pose_6d(init_frame[schema.body_pose_slice()])
        return init_frame

    trainable_joint_mask = _tracker_ik_joint_mask(ik_tracker_indices.detach().cpu().numpy())
    if not bool(trainable_joint_mask.any()):
        init_frame[schema.body_pose_slice()] = _normalize_pose_6d(init_frame[schema.body_pose_slice()])
        return init_frame

    offsets = joint_offsets_parent.clone()
    if schema.supports_root_motion:
        offsets[0, 1] = init_frame[schema.root_height_slice()][0]

    target_joint_indices = torch.as_tensor(
        TRACKER_JOINT_INDICES[ik_tracker_indices.detach().cpu().numpy()],
        dtype=torch.long,
        device=current_frame_raw.device,
    )
    target_positions = tracker_pos_ref[ik_tracker_indices]
    target_rot_6d = rotation_6d_forward_up_torch(rotation_6d_to_matrix_torch(tracker_rot_ref_6d[ik_tracker_indices]))
    trainable_mask = torch.as_tensor(trainable_joint_mask, dtype=torch.bool, device=current_frame_raw.device)
    reference_pose = init_frame[schema.body_pose_slice()].reshape(1, SMPL_JOINT_COUNT, 6).clone()
    pose = reference_pose.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pose], lr=float(lr))
    root_pos_ref = torch.zeros(1, 3, dtype=current_frame_raw.dtype, device=current_frame_raw.device)
    root_yaw_ref = yaw_delta.reshape(1)

    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        joints, global_rot = fk_parent_local_torch(
            body_pose_parent_6d=pose.reshape(1, -1),
            root_pos_world=root_pos_ref,
            root_yaw=root_yaw_ref,
            parent_offsets=offsets.reshape(1, SMPL_JOINT_COUNT, 3),
            return_global_rot=True,
        )
        pos_loss = (joints[:, target_joint_indices] - target_positions[None]).square().mean()
        pred_rot_6d = rotation_6d_forward_up_torch(global_rot[:, target_joint_indices])
        rot_cos = (pred_rot_6d.reshape(1, -1, 2, 3) * target_rot_6d.reshape(1, -1, 2, 3)).sum(dim=-1)
        rot_loss = 1.0 - rot_cos.mean()
        reg_loss = (pose[:, trainable_mask] - reference_pose[:, trainable_mask]).square().mean()
        loss = float(pos_weight) * pos_loss + float(rot_weight) * rot_loss + float(reg_weight) * reg_loss
        loss.backward()
        if pose.grad is not None:
            pose.grad[:, ~trainable_mask] = 0.0
        optimizer.step()
        with torch.no_grad():
            pose[:, ~trainable_mask] = reference_pose[:, ~trainable_mask]
            pose[:] = _normalize_and_clamp_pose(pose, reference_pose, float(delta_limit))

    init_frame[schema.body_pose_slice()] = _normalize_and_clamp_pose(
        pose.detach(),
        reference_pose,
        float(delta_limit),
    ).reshape(-1)
    return init_frame


def _estimate_yaw_delta_from_hip_tracker(
    hip_tracker_rot_ref_6d: torch.Tensor,
    hip_valid: bool,
    fallback: torch.Tensor,
) -> torch.Tensor:
    fallback_angle = torch.atan2(fallback[0], fallback[1])
    if not hip_valid:
        return fallback_angle
    rotation = rotation_6d_to_matrix_torch(hip_tracker_rot_ref_6d.reshape(1, 6))[0]
    forward = rotation[:, 2]
    horizontal_norm = torch.linalg.norm(forward[[0, 2]])
    if float(horizontal_norm.detach().cpu()) < 1e-6:
        return fallback_angle
    return torch.atan2(forward[0], forward[2])


def _tracker_ik_joint_mask(tracker_indices: np.ndarray) -> np.ndarray:
    mask = np.zeros((SMPL_JOINT_COUNT,), dtype=bool)
    for tracker_index in np.asarray(tracker_indices, dtype=np.int64).reshape(-1):
        joint_index = int(TRACKER_JOINT_INDICES[int(tracker_index)])
        while joint_index >= 0:
            if joint_index != 0:
                mask[joint_index] = True
            joint_index = int(SMPL_PARENTS[joint_index])
    return mask


def _normalize_pose_6d(body_pose_parent_6d: torch.Tensor) -> torch.Tensor:
    shape = body_pose_parent_6d.shape
    rotations = rotation_6d_to_matrix_torch(body_pose_parent_6d.reshape(-1, SMPL_JOINT_COUNT, 6))
    return rotation_6d_forward_up_torch(rotations).reshape(shape)


def _normalize_and_clamp_pose(
    pose: torch.Tensor,
    reference_pose: torch.Tensor,
    delta_limit: float,
) -> torch.Tensor:
    normalized = _normalize_pose_6d(pose)
    if float(delta_limit) <= 0.0:
        return normalized
    delta = normalized - reference_pose
    delta_norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(float(delta_limit) / torch.clamp(delta_norm, min=1e-8), max=1.0)
    return _normalize_pose_6d(reference_pose + delta * scale)
