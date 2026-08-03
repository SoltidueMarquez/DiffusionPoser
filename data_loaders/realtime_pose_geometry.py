from __future__ import annotations

import math

import numpy as np
import torch

from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    SMPL_PARENTS,
    make_yaw_rotation_np,
    make_yaw_rotation_torch,
    rotation_6d_forward_up_np,
    rotation_6d_forward_up_torch,
    rotation_6d_to_matrix_np,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_COUNT,
    TRACKER_D_OFF_OFFSET,
    TRACKER_D_ON_OFFSET,
    TRACKER_FEATURE_DIM,
    TRACKER_TO_JOINT,
)


HEAD_YAW_HORIZONTAL_EPS = 1e-4


def extract_forward_yaw_np(
    rotations: np.ndarray,
    horizontal_eps: float = HEAD_YAW_HORIZONTAL_EPS,
    initial_yaw: float = 0.0,
) -> np.ndarray:
    """从 forward 轴提取因果 yaw；近似竖直时沿用上一可靠值。"""

    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"rotations 必须为 [T,3,3]，实际为 {matrices.shape}")
    forward = matrices[:, :, 2]
    norms = np.linalg.norm(forward[:, [0, 2]], axis=-1)
    result = np.empty(matrices.shape[0], dtype=np.float64)
    previous = float(initial_yaw)
    for frame_index in range(matrices.shape[0]):
        if norms[frame_index] >= float(horizontal_eps):
            previous = math.atan2(float(forward[frame_index, 0]), float(forward[frame_index, 2]))
        result[frame_index] = previous
    return result.astype(np.float32)


def extract_forward_yaw_torch(
    rotations: torch.Tensor,
    horizontal_eps: float = HEAD_YAW_HORIZONTAL_EPS,
    initial_yaw: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Torch 版同一因果 fallback 规则；用于在线/测试保持数值契约一致。"""

    if rotations.ndim != 3 or tuple(rotations.shape[1:]) != (3, 3):
        raise ValueError(f"rotations 必须为 [T,3,3]，实际为 {tuple(rotations.shape)}")
    forward = rotations[:, :, 2]
    norms = torch.linalg.norm(forward[:, [0, 2]], dim=-1)
    previous = torch.as_tensor(initial_yaw, device=rotations.device, dtype=rotations.dtype).reshape(())
    result: list[torch.Tensor] = []
    for frame_index in range(rotations.shape[0]):
        measured = torch.atan2(forward[frame_index, 0], forward[frame_index, 2])
        previous = torch.where(norms[frame_index] >= float(horizontal_eps), measured, previous)
        result.append(previous)
    return torch.stack(result) if result else rotations.new_zeros((0,))


def build_pose_target_np(
    joint_rotations_world: np.ndarray,
    current_head_yaw: float,
) -> np.ndarray:
    """把一段世界姿态统一表达在同一个当前 Head-yaw 参考系中。"""

    rotations = np.asarray(joint_rotations_world, dtype=np.float64)
    if rotations.ndim != 4 or rotations.shape[1:] != (24, 3, 3):
        raise ValueError(f"joint_rotations_world 必须为 [T,24,3,3]，实际为 {rotations.shape}")

    head_yaw_inv = make_yaw_rotation_np(np.asarray([current_head_yaw], dtype=np.float64))[0].T
    rotations_head = np.einsum("ij,tajk->taik", head_yaw_inv, rotations)
    return rotation_6d_forward_up_np(rotations_head).reshape(
        rotations.shape[0], REALTIME_POSE_TARGET_DIM
    ).astype(np.float32)


def build_tracker_measurements_np(
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    reference_head_pos_world: np.ndarray,
    floor_y: float,
    reference_head_yaw: float,
) -> np.ndarray:
    """把任意连续帧 Tracker 世界变换统一表达在指定 Head-yaw 参考系。"""

    positions = np.asarray(tracker_pos_world, dtype=np.float64)
    rotations = rotation_6d_to_matrix_np(np.asarray(tracker_rot_world_6d, dtype=np.float64))
    if positions.ndim != 3 or positions.shape[1:] != (TRACKER_COUNT, 3):
        raise ValueError(f"tracker_pos_world 必须为 [T,6,3]，实际为 {positions.shape}")
    if rotations.shape != (*positions.shape[:2], 3, 3):
        raise ValueError("tracker_rot_world_6d 必须为 [T,6,6]。")

    origin = np.asarray(
        [reference_head_pos_world[0], float(floor_y), reference_head_pos_world[2]],
        dtype=np.float64,
    )
    head_yaw_inv = make_yaw_rotation_np(np.asarray([reference_head_yaw], dtype=np.float64))[0].T
    positions_head = np.einsum("ij,taj->tai", head_yaw_inv, positions - origin[None, None])
    rotations_head = np.einsum("ij,tajk->taik", head_yaw_inv, rotations)
    result = np.empty((positions.shape[0], TRACKER_COUNT, 9), dtype=np.float32)
    result[..., 0:3] = positions_head.astype(np.float32)
    result[..., 3:9] = rotation_6d_forward_up_np(rotations_head).astype(np.float32)
    return result


def assemble_tracker_features_np(
    measurements: np.ndarray,
    configured: np.ndarray,
    measured_valid: np.ndarray,
    d_off: np.ndarray,
    d_on: np.ndarray,
    duration_cap: int = 60,
) -> np.ndarray:
    """组合 `[T,6,13]`，且只归一化 duration 两个状态通道。"""

    continuous = np.asarray(measurements, dtype=np.float32)
    if continuous.ndim != 3 or continuous.shape[1:] != (TRACKER_COUNT, 9):
        raise ValueError("measurements 必须为 [T,6,9]。")
    state_shape = continuous.shape[:2]
    configured = np.asarray(configured, dtype=bool)
    measured = np.asarray(measured_valid, dtype=bool)
    d_off = np.asarray(d_off, dtype=np.float32)
    d_on = np.asarray(d_on, dtype=np.float32)
    if any(value.shape != state_shape for value in (configured, measured, d_off, d_on)):
        raise ValueError("Tracker 状态必须与 measurements 的 [T,6] 轴一致。")
    if np.any(measured & ~configured):
        raise ValueError("measured_valid 必须是 configured 子集。")
    if not configured[:, HEAD_TRACKER_INDEX].all() or not measured[:, HEAD_TRACKER_INDEX].all():
        raise ValueError("Head 必须始终 configured 且 measured_valid。")
    result = np.zeros((*state_shape, TRACKER_FEATURE_DIM), dtype=np.float32)
    result[..., :9] = continuous
    result[..., 9] = configured
    result[..., 10] = measured
    result[..., TRACKER_D_OFF_OFFSET] = np.clip(d_off, 0, duration_cap) / float(duration_cap)
    result[..., TRACKER_D_ON_OFFSET] = np.clip(d_on, 0, duration_cap) / float(duration_cap)
    result[..., :9] *= measured[..., None]
    validate_tracker_features_np(result)
    return result


def build_head_trajectory_np(
    head_pos_world: np.ndarray,
    head_yaw_world: np.ndarray,
    floor_y: float | np.ndarray,
    head_height_mean: float = 0.0,
    head_height_std: float = 1.0,
) -> np.ndarray:
    """构造逐帧 5D Head trajectory；位移使用上一帧 Head yaw 参考系。"""

    positions = np.asarray(head_pos_world, dtype=np.float64)
    yaws = np.asarray(head_yaw_world, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or yaws.shape != (positions.shape[0],):
        raise ValueError("Head trajectory 输入必须为 [T,3] 和 [T]。")
    floor = np.asarray(floor_y, dtype=np.float64)
    if floor.shape == ():
        floor = np.full(positions.shape[0], float(floor), dtype=np.float64)
    if floor.shape != (positions.shape[0],):
        raise ValueError("floor_y 必须是标量或 [T]。")
    if float(head_height_std) <= 0.0:
        raise ValueError("head_height_std 必须大于 0。")
    delta_world = np.zeros_like(positions)
    delta_yaw = np.zeros_like(yaws)
    if positions.shape[0] > 1:
        delta_world[1:] = positions[1:] - positions[:-1]
        delta_yaw[1:] = (yaws[1:] - yaws[:-1] + math.pi) % (2.0 * math.pi) - math.pi
    previous_yaw = np.concatenate([yaws[:1], yaws[:-1]], axis=0)
    previous_yaw_inv = np.swapaxes(make_yaw_rotation_np(previous_yaw), -1, -2)
    delta_ref = np.einsum("tij,tj->ti", previous_yaw_inv, delta_world)
    height = (positions[:, 1] - floor - float(head_height_mean)) / float(head_height_std)
    return np.stack(
        [delta_ref[:, 0], delta_ref[:, 2], height, np.sin(delta_yaw), np.cos(delta_yaw)],
        axis=-1,
    ).astype(np.float32)


def extract_rotation_heading_np(rotations: np.ndarray) -> np.ndarray:
    """从任意前导形状的旋转 forward 轴提取水平 heading。"""

    matrices = np.asarray(rotations, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError(f"rotations 尾部必须为 [3,3]，实际为 {matrices.shape}")
    forward = matrices[..., :, 2]
    right = matrices[..., :, 0]
    forward_norm = np.linalg.norm(forward[..., [0, 2]], axis=-1)
    use_forward = forward_norm >= HEAD_YAW_HORIZONTAL_EPS
    heading_x = np.where(use_forward, forward[..., 0], -right[..., 2])
    heading_z = np.where(use_forward, forward[..., 2], right[..., 0])
    return np.arctan2(heading_x, heading_z).astype(np.float32)


def extract_rotation_heading_torch(rotations: torch.Tensor) -> torch.Tensor:
    """Torch 版水平 heading 提取，保持 pelvis FK 路径可微。"""

    if tuple(rotations.shape[-2:]) != (3, 3):
        raise ValueError(f"rotations 尾部必须为 [3,3]，实际为 {tuple(rotations.shape)}")
    forward = rotations[..., :, 2]
    right = rotations[..., :, 0]
    forward_norm = torch.linalg.norm(forward[..., [0, 2]], dim=-1)
    use_forward = forward_norm >= HEAD_YAW_HORIZONTAL_EPS
    heading_x = torch.where(use_forward, forward[..., 0], -right[..., 2])
    heading_z = torch.where(use_forward, forward[..., 2], right[..., 0])
    return torch.atan2(heading_x, heading_z)


def so3_log_map_torch(rotations: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """把 ``[...,3,3]`` 相对旋转映射为弧度 axis-angle，含 0/π 邻域稳定分支。"""

    if tuple(rotations.shape[-2:]) != (3, 3):
        raise ValueError("SO(3) Log 输入尾部必须为 [3,3]。")
    matrix = rotations
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = torch.clamp((trace - 1.0) * 0.5, min=-1.0, max=1.0)
    skew = 0.5 * torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sine = torch.linalg.norm(skew, dim=-1)
    angle = torch.atan2(sine, cosine)
    safe_sine = sine.clamp_min(float(eps))
    regular = skew * (angle / safe_sine)[..., None]
    small = skew * (1.0 + angle.square() / 6.0)[..., None]

    # θ≈π 时反对称部分趋近 0，改从 (R+I)/2 的对角线恢复轴；符号由
    # 非零反对称分量确定。该分支只用于避免 180° residual 退化为零。
    diagonal_axis = torch.sqrt(
        torch.clamp((matrix.diagonal(dim1=-2, dim2=-1) + 1.0) * 0.5, min=0.0)
    )
    axis_sign = torch.sign(skew)
    axis_sign = torch.where(axis_sign == 0.0, torch.ones_like(axis_sign), axis_sign)
    diagonal_axis = diagonal_axis * axis_sign
    diagonal_axis = torch.nn.functional.normalize(diagonal_axis, dim=-1, eps=float(eps))
    near_pi = diagonal_axis * angle[..., None]
    result = torch.where((sine < 1e-4)[..., None], small, regular)
    result = torch.where(((cosine < -0.9999) & (sine < 1e-3))[..., None], near_pi, result)
    return result


def reexpress_previous_position_residual_torch(
    previous_residual: torch.Tensor,
    current_trajectory: torch.Tensor,
) -> torch.Tensor:
    """把上一帧 C_(n-1) 中的位置 residual 旋转到当前 C_n。"""

    if previous_residual.ndim != 3 or previous_residual.shape[-1] != 3:
        raise ValueError("previous_residual 必须为 [B,Tracker,3]。")
    batch_size = previous_residual.shape[0]
    if tuple(current_trajectory.shape) != (batch_size, 1, 5):
        raise ValueError("current_trajectory 必须为 [B,1,5]。")
    delta_yaw = torch.atan2(current_trajectory[:, 0, 3], current_trajectory[:, 0, 4])
    return torch.einsum(
        "bij,btj->bti",
        make_yaw_rotation_torch(-delta_yaw),
        previous_residual,
    )


def decode_target_head_rotations_np(
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从 144 维目标恢复 Head 参考系的 24 个全局旋转与 pelvis heading。"""

    values = np.asarray(target, dtype=np.float64)
    if values.shape[-1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError(f"target 最后一维必须为 {REALTIME_POSE_TARGET_DIM}，实际为 {values.shape}")
    leading = values.shape[:-1]
    rotations = rotation_6d_to_matrix_np(values.reshape(*leading, SMPL_JOINT_COUNT, ROTATION_6D_DIM))
    return rotations, extract_rotation_heading_np(rotations[..., 0, :, :])


def decode_target_root_yaw_world_np(
    target: np.ndarray,
    current_head_yaw_world: np.ndarray | float,
) -> np.ndarray:
    """从 144D target 与当前 Head yaw 组合出唯一的 Pelvis world heading。

    Source 中的 ``root_yaw`` 是 Actor Root 的分解 heading；当 Pelvis local
    residual 接近 180° 时，它可以与 Pelvis forward heading 相差约 π。训练与
    评估必须从 target 本身恢复后者，避免把两种合法但不同的 yaw 语义混用。
    """

    _, pelvis_heading_head = decode_target_head_rotations_np(target)
    head_yaw = np.asarray(current_head_yaw_world, dtype=np.float64)
    try:
        head_yaw = np.broadcast_to(head_yaw, pelvis_heading_head.shape)
    except ValueError as error:
        raise ValueError(
            "current_head_yaw_world 必须可广播到 target 的前导形状。"
        ) from error
    return (
        (head_yaw + pelvis_heading_head.astype(np.float64) + math.pi)
        % (2.0 * math.pi)
        - math.pi
    ).astype(np.float32)


def decode_target_head_rotations_torch(
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.shape[-1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError(f"target 最后一维必须为 {REALTIME_POSE_TARGET_DIM}，实际为 {tuple(target.shape)}")
    batch_size = target.shape[0]
    rotations = rotation_6d_to_matrix_torch(
        target.reshape(batch_size, SMPL_JOINT_COUNT, ROTATION_6D_DIM)
    )
    return rotations, extract_rotation_heading_torch(rotations[:, 0])


def reexpress_pose_target_between_head_yaws_torch(
    target: torch.Tensor,
    source_head_yaw_world: torch.Tensor,
    destination_head_yaw_world: torch.Tensor,
) -> torch.Tensor:
    """把已生成姿态从旧 Head-yaw 参考系精确转到下一帧 Head-yaw 参考系。"""

    if target.ndim != 2 or target.shape[1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError(f"target 必须为 [B,{REALTIME_POSE_TARGET_DIM}]。")
    batch_size = target.shape[0]
    rotations = rotation_6d_to_matrix_torch(
        target.reshape(batch_size, SMPL_JOINT_COUNT, ROTATION_6D_DIM)
    )
    source_yaw = source_head_yaw_world.to(device=target.device, dtype=target.dtype).reshape(-1)
    destination_yaw = destination_head_yaw_world.to(device=target.device, dtype=target.dtype).reshape(-1)
    reference_delta = make_yaw_rotation_torch(source_yaw - destination_yaw)
    rotations = torch.einsum("bij,bajk->baik", reference_delta, rotations)
    result = target.clone()
    result[:] = rotation_6d_forward_up_torch(rotations).reshape(batch_size, REALTIME_POSE_TARGET_DIM)
    return result


def advance_rollout_pose_history_torch(
    pose_history: torch.Tensor,
    prediction: torch.Tensor,
    source_head_yaw_world: torch.Tensor,
    destination_head_yaw_world: torch.Tensor,
    normalizer_mean: torch.Tensor | None = None,
    normalizer_std: torch.Tensor | None = None,
    detach_prediction: bool = True,
) -> torch.Tensor:
    """把旧历史换到下一历史参考系，并追加已位于该参考系的 deployed 预测。"""

    if pose_history.ndim != 3 or pose_history.shape[1:] != (
        REALTIME_POSE_HISTORY_LENGTH,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError(
            f"pose_history 应为 [B,{REALTIME_POSE_HISTORY_LENGTH},{REALTIME_POSE_TARGET_DIM}]，"
            f"实际为 {tuple(pose_history.shape)}"
        )
    if prediction.shape != pose_history[:, -1].shape:
        raise ValueError("prediction 必须与单帧 pose history 同形。")
    prediction = prediction.detach() if detach_prediction else prediction
    if normalizer_mean is not None and normalizer_std is not None:
        mean = normalizer_mean.to(device=pose_history.device, dtype=pose_history.dtype)
        std = normalizer_std.to(device=pose_history.device, dtype=pose_history.dtype)
        history_raw = pose_history * std + mean
        prediction_raw = prediction * std + mean
    else:
        mean = std = None
        history_raw = pose_history
        prediction_raw = prediction

    batch_size, history_length, target_dim = history_raw.shape
    source_yaw = source_head_yaw_world.to(device=pose_history.device).reshape(batch_size)
    destination_yaw = destination_head_yaw_world.to(device=pose_history.device).reshape(batch_size)
    history_reexpressed = reexpress_pose_target_between_head_yaws_torch(
        history_raw.reshape(batch_size * history_length, target_dim),
        source_yaw[:, None].expand(-1, history_length).reshape(-1),
        destination_yaw[:, None].expand(-1, history_length).reshape(-1),
    ).reshape(batch_size, history_length, target_dim)
    next_history = torch.cat(
        [history_reexpressed[:, 1:], prediction_raw[:, None]],
        dim=1,
    )
    if mean is None or std is None:
        return next_history
    return (next_history - mean) / std


def pelvis_relative_joint_positions_np(
    global_rotations_head: np.ndarray,
    rest_local_positions: np.ndarray,
) -> np.ndarray:
    rotations = np.asarray(global_rotations_head, dtype=np.float64)
    offsets = np.asarray(rest_local_positions, dtype=np.float64)
    if rotations.shape[-3:] != (24, 3, 3) or offsets.shape != (24, 3):
        raise ValueError("global rotations/rest positions 形状必须分别为 [...,24,3,3] 和 [24,3]。")
    leading = rotations.shape[:-3]
    positions = np.zeros((*leading, 24, 3), dtype=np.float64)
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            continue
        positions[..., joint_index, :] = positions[..., parent_index, :] + np.einsum(
            "...ij,j->...i", rotations[..., parent_index, :, :], offsets[joint_index]
        )
    return positions


def pelvis_relative_joint_positions_torch(
    global_rotations_head: torch.Tensor,
    rest_local_positions: torch.Tensor,
) -> torch.Tensor:
    batch_size = global_rotations_head.shape[0]
    offsets = rest_local_positions.to(device=global_rotations_head.device, dtype=global_rotations_head.dtype)
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(batch_size, -1, -1)
    positions: list[torch.Tensor] = []
    zero = torch.zeros((batch_size, 3), device=global_rotations_head.device, dtype=global_rotations_head.dtype)
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            joint_position = zero
        else:
            joint_position = positions[parent_index] + torch.einsum(
                "bij,bj->bi", global_rotations_head[:, parent_index], offsets[:, joint_index]
            )
        positions.append(joint_position)
    return torch.stack(positions, dim=1)


def derive_hip_height_from_head_np(
    global_rotations_head: np.ndarray,
    rest_local_positions: np.ndarray,
    observed_head_height: np.ndarray | float,
) -> np.ndarray:
    relative = pelvis_relative_joint_positions_np(global_rotations_head, rest_local_positions)
    head_y = relative[..., JOINT_INDEX["head"], 1]
    return np.asarray(observed_head_height, dtype=np.float64) - head_y


def derive_hip_height_from_head_torch(
    global_rotations_head: torch.Tensor,
    rest_local_positions: torch.Tensor,
    observed_head_height: torch.Tensor,
) -> torch.Tensor:
    relative = pelvis_relative_joint_positions_torch(global_rotations_head, rest_local_positions)
    return observed_head_height.reshape(-1) - relative[:, JOINT_INDEX["head"], 1]


def resolve_root_head_reference_np(
    global_rotations_head: np.ndarray,
    root_yaw_head: float,
    rest_local_positions: np.ndarray,
    observed_head_height: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """在当前 Head 参考系中解析 Root 平移、Pelvis 高度和全部关节位置。"""

    rotations = np.asarray(global_rotations_head, dtype=np.float64)
    if rotations.shape != (24, 3, 3):
        raise ValueError(f"global_rotations_head 必须为 [24,3,3]，实际为 {rotations.shape}")
    relative = pelvis_relative_joint_positions_np(rotations, rest_local_positions)
    hip_height = float(observed_head_height - relative[JOINT_INDEX["head"], 1])

    pelvis_zero = np.asarray(rest_local_positions[0], dtype=np.float64).copy()
    pelvis_zero[1] = hip_height
    # Pelvis bone 的位置只由 actor Root yaw 旋转 rest offset，不能把 rest rotation
    # 的 pitch/roll 错当成 actor Root 旋转。
    root_yaw_rotation = make_yaw_rotation_np(np.asarray([root_yaw_head], dtype=np.float64))[0]
    pelvis_at_zero = root_yaw_rotation @ pelvis_zero
    joints_at_zero = relative + pelvis_at_zero[None]
    head_index = JOINT_INDEX["head"]
    root_translation = np.asarray([-joints_at_zero[head_index, 0], 0.0, -joints_at_zero[head_index, 2]])
    joints = joints_at_zero + root_translation[None]
    return root_translation.astype(np.float32), hip_height, joints.astype(np.float32)


def resolve_root_head_reference_torch(
    global_rotations_head: torch.Tensor,
    root_yaw_head: torch.Tensor,
    rest_local_positions: torch.Tensor,
    observed_head_height: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """可微分的批量 Root Resolver；Root 平移始终硬对齐 Head。"""

    batch_size = global_rotations_head.shape[0]
    if tuple(global_rotations_head.shape[1:]) != (24, 3, 3):
        raise ValueError("global_rotations_head 必须为 [B,24,3,3]。")
    offsets = rest_local_positions.to(
        device=global_rotations_head.device,
        dtype=global_rotations_head.dtype,
    )
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(batch_size, -1, -1)
    relative = pelvis_relative_joint_positions_torch(global_rotations_head, offsets)
    head_index = JOINT_INDEX["head"]
    hip_height = observed_head_height.reshape(-1) - relative[:, head_index, 1]

    pelvis_offset = offsets[:, 0].clone()
    pelvis_offset[:, 1] = hip_height
    pelvis_at_zero = torch.einsum(
        "bij,bj->bi",
        make_yaw_rotation_torch(root_yaw_head.reshape(-1)),
        pelvis_offset,
    )
    joints_at_zero = relative + pelvis_at_zero[:, None]
    root_translation = torch.zeros(
        (batch_size, 3),
        device=global_rotations_head.device,
        dtype=global_rotations_head.dtype,
    )
    root_translation[:, 0] = -joints_at_zero[:, head_index, 0]
    root_translation[:, 2] = -joints_at_zero[:, head_index, 2]
    joints = joints_at_zero + root_translation[:, None]
    return root_translation, hip_height, joints


def global_head_rotations_to_local_delta_6d_np(
    global_rotations_head: np.ndarray,
    root_heading_head: np.ndarray | float,
    rest_local_rotations: np.ndarray,
) -> np.ndarray:
    rotations = np.asarray(global_rotations_head, dtype=np.float64)
    rest = np.asarray(rest_local_rotations, dtype=np.float64)
    if rotations.shape[-3:] != (24, 3, 3) or rest.shape != (24, 3, 3):
        raise ValueError("global/rest rotations 形状不正确。")
    local = np.empty_like(rotations)
    heading = np.asarray(root_heading_head, dtype=np.float64)
    expected_heading_shape = rotations.shape[:-3]
    if heading.shape != expected_heading_shape:
        heading = np.broadcast_to(heading, expected_heading_shape)
    heading_inv = np.swapaxes(make_yaw_rotation_np(heading.reshape(-1)), -1, -2).reshape(
        *expected_heading_shape, 3, 3
    )
    local[..., 0, :, :] = heading_inv @ rotations[..., 0, :, :]
    for joint_index in range(1, 24):
        parent = int(SMPL_PARENTS[joint_index])
        local[..., joint_index, :, :] = np.swapaxes(rotations[..., parent, :, :], -1, -2) @ rotations[
            ..., joint_index, :, :
        ]
    delta = np.swapaxes(rest, -1, -2) @ local
    return rotation_6d_forward_up_np(delta).reshape(*rotations.shape[:-3], 144).astype(np.float32)


def validate_tracker_features_np(tracker_window: np.ndarray) -> None:
    tracker = np.asarray(tracker_window)
    if tracker.shape[-2:] != (TRACKER_COUNT, TRACKER_FEATURE_DIM):
        raise ValueError(f"Tracker 特征尾部形状必须为 [6,13]，实际为 {tracker.shape}")
    d_off = tracker[..., TRACKER_D_OFF_OFFSET]
    d_on = tracker[..., TRACKER_D_ON_OFFSET]
    if np.any((d_off < 0.0) | (d_off > 1.0)) or np.any((d_on < 0.0) | (d_on > 1.0)):
        raise ValueError("d_off/d_on norm 必须在 [0,1]。")
    configured = tracker[..., 9] > 0.5
    measured = tracker[..., 10] > 0.5
    if np.any(measured & ~configured):
        raise ValueError("measured_valid 必须是 configured 子集。")
    if np.any((~configured | measured) & (np.abs(d_off) > 1e-7)):
        raise ValueError("未配置或有效 Tracker 的 d_off 必须为 0。")
    if np.any((~configured | ~measured) & (np.abs(d_on) > 1e-7)):
        raise ValueError("未配置或掉线 Tracker 的 d_on 必须为 0。")
    if np.any(np.abs(tracker[..., :9][~measured]) > 1e-7):
        raise ValueError("无效测量的前 9 维必须严格清零。")
