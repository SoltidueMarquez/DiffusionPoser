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
    wrap_radians,
)
from data_loaders.sensor_masking import (
    FOOT_TRACKER_INDICES,
    HAND_TRACKER_INDICES,
    HEAD_TRACKER_INDEX,
    HIP_TRACKER_INDEX,
    JOINT_GLOBAL_ROTATION_DIM,
    MISSING_AGE_CAP,
    REALTIME_POSE_TARGET_DIM,
    ROOT_YAW_RELATIVE_START,
    ROTATION_6D_DIM,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_MISSING_AGE_OFFSET,
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
    root_yaw_world: np.ndarray,
    current_head_yaw: float,
) -> np.ndarray:
    """把一段世界姿态统一表达在同一个当前 Head-yaw 参考系中。"""

    rotations = np.asarray(joint_rotations_world, dtype=np.float64)
    root_yaw = np.asarray(root_yaw_world, dtype=np.float64).reshape(-1)
    if rotations.ndim != 4 or rotations.shape[1:] != (24, 3, 3):
        raise ValueError(f"joint_rotations_world 必须为 [T,24,3,3]，实际为 {rotations.shape}")
    if root_yaw.shape != (rotations.shape[0],):
        raise ValueError(f"root_yaw_world 必须为 [T]，实际为 {root_yaw.shape}")

    head_yaw_inv = make_yaw_rotation_np(np.asarray([current_head_yaw], dtype=np.float64))[0].T
    rotations_head = np.einsum("ij,tajk->taik", head_yaw_inv, rotations)
    target = np.zeros((rotations.shape[0], REALTIME_POSE_TARGET_DIM), dtype=np.float32)
    target[:, :JOINT_GLOBAL_ROTATION_DIM] = rotation_6d_forward_up_np(rotations_head[:, 1:]).reshape(
        rotations.shape[0], JOINT_GLOBAL_ROTATION_DIM
    )
    relative = wrap_radians(float(current_head_yaw) - root_yaw)
    target[:, ROOT_YAW_RELATIVE_START:] = np.stack([np.sin(relative), np.cos(relative)], axis=-1)
    return target


def build_tracker_window_np(
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    current_head_pos_world: np.ndarray,
    floor_y: float,
    current_head_yaw: float,
    configured: np.ndarray,
    measured_valid: np.ndarray,
    missing_age_norm: np.ndarray,
) -> np.ndarray:
    """构造 `[61,6,12]` Tracker 条件，并彻底清除无效测量连续量。"""

    positions = np.asarray(tracker_pos_world, dtype=np.float64)
    rotations = rotation_6d_to_matrix_np(np.asarray(tracker_rot_world_6d, dtype=np.float64))
    configured = np.asarray(configured, dtype=bool)
    measured_valid = np.asarray(measured_valid, dtype=bool)
    missing_age_norm = np.asarray(missing_age_norm, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (TRACKER_COUNT, 3):
        raise ValueError(f"tracker_pos_world 必须为 [T,6,3]，实际为 {positions.shape}")
    expected_state_shape = positions.shape[:2]
    if configured.shape != expected_state_shape or measured_valid.shape != expected_state_shape:
        raise ValueError("Tracker 状态形状必须与位置的 [T,6] 一致。")
    if missing_age_norm.shape != expected_state_shape:
        raise ValueError("missing_age_norm 必须为 [T,6]。")

    origin = np.asarray(
        [current_head_pos_world[0], float(floor_y), current_head_pos_world[2]],
        dtype=np.float64,
    )
    head_yaw_inv = make_yaw_rotation_np(np.asarray([current_head_yaw], dtype=np.float64))[0].T
    positions_head = np.einsum("ij,taj->tai", head_yaw_inv, positions - origin[None, None])
    rotations_head = np.einsum("ij,tajk->taik", head_yaw_inv, rotations)

    result = np.zeros((positions.shape[0], TRACKER_COUNT, TRACKER_FEATURE_DIM), dtype=np.float32)
    result[..., 0:3] = positions_head.astype(np.float32)
    result[..., 3:9] = rotation_6d_forward_up_np(rotations_head).astype(np.float32)
    result[..., 9] = configured.astype(np.float32)
    result[..., 10] = measured_valid.astype(np.float32)
    result[..., TRACKER_MISSING_AGE_OFFSET] = np.clip(missing_age_norm, 0.0, 1.0)
    result[..., 0:9] *= measured_valid[..., None]
    return result


def build_known_target_np(
    current_tracker_features: np.ndarray,
    current_head_yaw: float,
    current_tracker_rot_world_6d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """只通过当前可信 Tracker 构造 hard inpainting 值和 mask。"""

    tracker = np.asarray(current_tracker_features, dtype=np.float32)
    tracker_world_rot = rotation_6d_to_matrix_np(np.asarray(current_tracker_rot_world_6d, dtype=np.float64))
    if tracker.shape != (TRACKER_COUNT, TRACKER_FEATURE_DIM):
        raise ValueError(f"current_tracker_features 必须为 [6,12]，实际为 {tracker.shape}")
    measured = tracker[:, 10] > 0.5
    known_target = np.zeros(REALTIME_POSE_TARGET_DIM, dtype=np.float32)
    known_mask = np.zeros(REALTIME_POSE_TARGET_DIM, dtype=bool)

    for tracker_index in (HEAD_TRACKER_INDEX, *HAND_TRACKER_INDICES, *FOOT_TRACKER_INDICES):
        if not measured[tracker_index]:
            continue
        joint_index = TRACKER_TO_JOINT[tracker_index]
        start = (joint_index - 1) * ROTATION_6D_DIM
        target_slice = slice(start, start + ROTATION_6D_DIM)
        known_target[target_slice] = tracker[tracker_index, 3:9]
        known_mask[target_slice] = True

    if measured[HIP_TRACKER_INDEX]:
        hip_yaw = float(extract_forward_yaw_np(tracker_world_rot[HIP_TRACKER_INDEX][None])[0])
        relative = float(wrap_radians(np.asarray([current_head_yaw - hip_yaw], dtype=np.float64))[0])
        known_target[ROOT_YAW_RELATIVE_START:] = (math.sin(relative), math.cos(relative))
        known_mask[ROOT_YAW_RELATIVE_START:] = True

    # Head 是模型的硬前提；该断言可发现 task 构造或 runtime gating 错误。
    head_start = (TRACKER_TO_JOINT[HEAD_TRACKER_INDEX] - 1) * ROTATION_6D_DIM
    head_slice = slice(head_start, head_start + ROTATION_6D_DIM)
    if not known_mask[head_slice].all():
        raise ValueError("Head rotation 必须始终写入 known_target。")
    return known_target, known_mask


def decode_target_head_rotations_np(
    target: np.ndarray,
    rest_local_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从 140 维目标恢复 Head 参考系的 24 个关节全局旋转与 Root yaw。"""

    values = np.asarray(target, dtype=np.float64)
    if values.shape[-1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError(f"target 最后一维必须为 {REALTIME_POSE_TARGET_DIM}，实际为 {values.shape}")
    rest = np.asarray(rest_local_rotations, dtype=np.float64)
    if rest.shape != (24, 3, 3):
        raise ValueError(f"rest_local_rotations 必须为 [24,3,3]，实际为 {rest.shape}")

    leading = values.shape[:-1]
    non_pelvis = rotation_6d_to_matrix_np(values[..., :JOINT_GLOBAL_ROTATION_DIM].reshape(*leading, 23, 6))
    relative_values = values[..., ROOT_YAW_RELATIVE_START:]
    relative = np.arctan2(relative_values[..., 0], relative_values[..., 1])
    root_yaw_head = -relative
    root_rotation_head = make_yaw_rotation_np(root_yaw_head.reshape(-1)).reshape(*leading, 3, 3)
    rotations = np.empty((*leading, 24, 3, 3), dtype=np.float64)
    rotations[..., 0, :, :] = root_rotation_head @ rest[0]
    rotations[..., 1:, :, :] = non_pelvis
    return rotations, root_yaw_head.astype(np.float32)


def decode_target_head_rotations_torch(
    target: torch.Tensor,
    rest_local_rotations_6d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.shape[-1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError(f"target 最后一维必须为 {REALTIME_POSE_TARGET_DIM}，实际为 {tuple(target.shape)}")
    batch_size = target.shape[0]
    non_pelvis = rotation_6d_to_matrix_torch(target[:, :JOINT_GLOBAL_ROTATION_DIM].reshape(batch_size, 23, 6))
    relative_values = target[:, ROOT_YAW_RELATIVE_START:]
    relative = torch.atan2(relative_values[:, 0], relative_values[:, 1])
    root_yaw_head = -relative
    root_rotation_head = make_yaw_rotation_torch(root_yaw_head)
    rest = rest_local_rotations_6d.to(device=target.device, dtype=target.dtype)
    if rest.ndim == 2:
        rest = rest.unsqueeze(0).expand(batch_size, -1, -1)
    rest_rot = rotation_6d_to_matrix_torch(rest.reshape(batch_size, 24, 6))
    pelvis = root_rotation_head @ rest_rot[:, 0]
    rotations = torch.cat([pelvis[:, None], non_pelvis], dim=1)
    return rotations, root_yaw_head


def reexpress_pose_target_between_head_yaws_torch(
    target: torch.Tensor,
    source_head_yaw_world: torch.Tensor,
    destination_head_yaw_world: torch.Tensor,
) -> torch.Tensor:
    """把已生成姿态从旧 Head-yaw 参考系精确转到下一帧 Head-yaw 参考系。"""

    if target.ndim != 2 or target.shape[1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError("target 必须为 [B,140]。")
    batch_size = target.shape[0]
    rotations = rotation_6d_to_matrix_torch(
        target[:, :JOINT_GLOBAL_ROTATION_DIM].reshape(batch_size, 23, 6)
    )
    source_yaw = source_head_yaw_world.to(device=target.device, dtype=target.dtype).reshape(-1)
    destination_yaw = destination_head_yaw_world.to(device=target.device, dtype=target.dtype).reshape(-1)
    reference_delta = make_yaw_rotation_torch(source_yaw - destination_yaw)
    rotations = torch.einsum("bij,bajk->baik", reference_delta, rotations)
    result = target.clone()
    result[:, :JOINT_GLOBAL_ROTATION_DIM] = rotation_6d_forward_up_torch(rotations).reshape(
        batch_size, JOINT_GLOBAL_ROTATION_DIM
    )
    relative = torch.atan2(
        target[:, ROOT_YAW_RELATIVE_START],
        target[:, ROOT_YAW_RELATIVE_START + 1],
    )
    destination_relative = relative + destination_yaw - source_yaw
    result[:, ROOT_YAW_RELATIVE_START] = torch.sin(destination_relative)
    result[:, ROOT_YAW_RELATIVE_START + 1] = torch.cos(destination_relative)
    return result


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
    hip_measured_valid: bool,
    observed_hip_position_head: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """在当前 Head 参考系中解析 Root 平移、Hip 高度和全部关节位置。"""

    rotations = np.asarray(global_rotations_head, dtype=np.float64)
    if rotations.shape != (24, 3, 3):
        raise ValueError(f"global_rotations_head 必须为 [24,3,3]，实际为 {rotations.shape}")
    relative = pelvis_relative_joint_positions_np(rotations, rest_local_positions)
    if hip_measured_valid:
        if observed_hip_position_head is None:
            raise ValueError("Hip 有效时必须提供 observed_hip_position_head。")
        hip_height = float(np.asarray(observed_hip_position_head)[1])
    else:
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
    if hip_measured_valid:
        observed_hip = np.asarray(observed_hip_position_head, dtype=np.float64)
        predicted_hip = root_translation + joints_at_zero[0]
        root_translation[[0, 2]] += observed_hip[[0, 2]] - predicted_hip[[0, 2]]
    joints = joints_at_zero + root_translation[None]
    return root_translation.astype(np.float32), hip_height, joints.astype(np.float32)


def resolve_root_head_reference_torch(
    global_rotations_head: torch.Tensor,
    root_yaw_head: torch.Tensor,
    rest_local_positions: torch.Tensor,
    observed_head_height: torch.Tensor,
    hip_measured_valid: torch.Tensor,
    observed_hip_position_head: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """可微分的批量 Root Resolver；仅用 Head，Hip 有效时做一次硬校正。"""

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
    derived_height = observed_head_height.reshape(-1) - relative[:, head_index, 1]
    hip_valid = hip_measured_valid.reshape(-1).bool()
    observed_hip = observed_hip_position_head.to(
        device=global_rotations_head.device,
        dtype=global_rotations_head.dtype,
    )
    hip_height = torch.where(hip_valid, observed_hip[:, 1], derived_height)

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
    predicted_hip = root_translation + joints_at_zero[:, 0]
    hip_correction = observed_hip[:, [0, 2]] - predicted_hip[:, [0, 2]]
    root_xz = torch.where(hip_valid[:, None], root_translation[:, [0, 2]] + hip_correction, root_translation[:, [0, 2]])
    root_translation = root_translation.clone()
    root_translation[:, [0, 2]] = root_xz
    joints = joints_at_zero + root_translation[:, None]
    return root_translation, hip_height, joints


def global_head_rotations_to_local_delta_6d_np(
    global_rotations_head: np.ndarray,
    rest_local_rotations: np.ndarray,
) -> np.ndarray:
    rotations = np.asarray(global_rotations_head, dtype=np.float64)
    rest = np.asarray(rest_local_rotations, dtype=np.float64)
    if rotations.shape[-3:] != (24, 3, 3) or rest.shape != (24, 3, 3):
        raise ValueError("global/rest rotations 形状不正确。")
    local = np.empty_like(rotations)
    local[..., 0, :, :] = rest[0]
    for joint_index in range(1, 24):
        parent = int(SMPL_PARENTS[joint_index])
        local[..., joint_index, :, :] = np.swapaxes(rotations[..., parent, :, :], -1, -2) @ rotations[
            ..., joint_index, :, :
        ]
    delta = np.swapaxes(rest, -1, -2) @ local
    delta[..., 0, :, :] = np.eye(3, dtype=np.float64)
    return rotation_6d_forward_up_np(delta).reshape(*rotations.shape[:-3], 144).astype(np.float32)


def validate_missing_age_feature(tracker_window: np.ndarray) -> None:
    tracker = np.asarray(tracker_window)
    if tracker.shape[-2:] != (TRACKER_COUNT, TRACKER_FEATURE_DIM):
        raise ValueError(f"tracker_window 尾部形状必须为 [6,12]，实际为 {tracker.shape}")
    age = tracker[..., TRACKER_MISSING_AGE_OFFSET]
    if np.any(age < 0.0) or np.any(age > 1.0):
        raise ValueError("missing_age_norm 必须在 [0,1]。")
    configured = tracker[..., 9] > 0.5
    measured = tracker[..., 10] > 0.5
    if np.any(measured & ~configured):
        raise ValueError("measured_valid 必须是 configured 子集。")
    if np.any((~configured | measured) & (np.abs(age) > 1e-7)):
        raise ValueError("未配置或有效 Tracker 的 missing_age_norm 必须为 0。")
