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
    rotation_6d_to_matrix_np,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
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


def build_head_path_window_np(
    head_pos_world: np.ndarray,
    head_yaw_world: np.ndarray,
    reference_head_pos_world: np.ndarray,
    floor_y: float,
    reference_head_yaw: float,
) -> np.ndarray:
    """把同步锚点构造成当前参考系下的绝对 Head 路径 `[T,5]`。

    与旧的逐帧增量 trajectory 不同，这里保存每个锚点相对当前 Head 原点的
    绝对 XZ 位置和相对当前 yaw。这样下采样后仍能直接表达整条历史路径。
    """

    positions = np.asarray(head_pos_world, dtype=np.float64)
    yaws = np.asarray(head_yaw_world, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or yaws.shape != (positions.shape[0],):
        raise ValueError("Head 路径输入必须为 [T,3] 和 [T]。")
    origin = np.asarray(
        [reference_head_pos_world[0], float(floor_y), reference_head_pos_world[2]],
        dtype=np.float64,
    )
    yaw_inverse = make_yaw_rotation_np(
        np.asarray([reference_head_yaw], dtype=np.float64)
    )[0].T
    positions_reference = np.einsum(
        "ij,tj->ti", yaw_inverse, positions - origin[None]
    )
    relative_yaw = (yaws - float(reference_head_yaw) + math.pi) % (2.0 * math.pi) - math.pi
    return np.stack(
        [
            positions_reference[:, 0],
            positions_reference[:, 2],
            positions_reference[:, 1],
            np.sin(relative_yaw),
            np.cos(relative_yaw),
        ],
        axis=-1,
    ).astype(np.float32)


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


def decode_target_head_rotations_torch(
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.shape[-1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError(f"target 最后一维必须为 {REALTIME_POSE_TARGET_DIM}，实际为 {tuple(target.shape)}")
    leading = target.shape[:-1]
    rotations = rotation_6d_to_matrix_torch(
        target.reshape(*leading, SMPL_JOINT_COUNT, ROTATION_6D_DIM)
    )
    return rotations, extract_rotation_heading_torch(rotations[..., 0, :, :])


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
