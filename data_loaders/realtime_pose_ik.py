from __future__ import annotations

import torch
import torch.nn.functional as F

from data_loaders.realtime_pose_geometry import pelvis_relative_joint_positions_torch
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    SMPL_PARENTS,
    rotation_6d_forward_up_torch,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    HIP_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    RIGHT_FOOT_TRACKER_INDEX,
    RIGHT_HAND_TRACKER_INDEX,
    SMPL_JOINT_COUNT,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_MEASURED_VALID_OFFSET,
    TRACKER_TO_JOINT,
)

_TORSO_CHAIN = (
    JOINT_INDEX["spine1"],
    JOINT_INDEX["spine2"],
    JOINT_INDEX["spine3"],
    JOINT_INDEX["neck"],
    JOINT_INDEX["head"],
)
_LEFT_ARM_CHAIN = (
    JOINT_INDEX["left_shoulder"],
    JOINT_INDEX["left_elbow"],
    JOINT_INDEX["left_wrist"],
)
_RIGHT_ARM_CHAIN = (
    JOINT_INDEX["right_shoulder"],
    JOINT_INDEX["right_elbow"],
    JOINT_INDEX["right_wrist"],
)
_LEFT_LEG_CHAIN = (
    JOINT_INDEX["left_hip"],
    JOINT_INDEX["left_knee"],
    JOINT_INDEX["left_ankle"],
    JOINT_INDEX["left_foot"],
)
_RIGHT_LEG_CHAIN = (
    JOINT_INDEX["right_hip"],
    JOINT_INDEX["right_knee"],
    JOINT_INDEX["right_ankle"],
    JOINT_INDEX["right_foot"],
)


def build_current_ik_pose(
    previous_pose_raw: torch.Tensor,
    previous_pose_valid: torch.Tensor,
    current_tracker_raw: torch.Tensor,
    joint_offsets_parent: torch.Tensor,
    joint_rest_local_rotations_6d: torch.Tensor,
    fabrik_iterations: int = 2,
) -> torch.Tensor:
    """用上一帧 Pose 和当前 Tracker 构造 `[B,144]` IK 姿态初值。

    输入 Pose 与 Tracker 都必须已经表达在当前 Head-yaw 参考系中。函数只把
    Tracker 覆盖的骨链更新为当前测量；其他关节保留上一帧或 rest pose。IK 结果
    允许存在误差，逐关节置信度由独立的 Tracker 区域 mapping 决定。
    """

    if int(fabrik_iterations) <= 0:
        raise ValueError("fabrik_iterations 必须大于 0。")
    batch_size = previous_pose_raw.shape[0]
    if tuple(previous_pose_raw.shape) != (batch_size, SMPL_JOINT_COUNT * 6):
        raise ValueError("previous_pose_raw 必须为 [B,144]。")
    if tuple(previous_pose_valid.shape) != (batch_size,):
        raise ValueError("previous_pose_valid 必须为 [B]。")
    if tuple(current_tracker_raw.shape) != (
        batch_size,
        TRACKER_COUNT,
        TRACKER_FEATURE_DIM,
    ):
        raise ValueError("current_tracker_raw 必须为 [B,6,13]。")
    if tuple(joint_offsets_parent.shape) != (batch_size, SMPL_JOINT_COUNT, 3):
        raise ValueError("joint_offsets_parent 必须为 [B,24,3]。")
    if tuple(joint_rest_local_rotations_6d.shape) != (
        batch_size,
        SMPL_JOINT_COUNT,
        6,
    ):
        raise ValueError("joint_rest_local_rotations_6d 必须为 [B,24,6]。")
    previous_global = rotation_6d_to_matrix_torch(
        previous_pose_raw.reshape(batch_size, SMPL_JOINT_COUNT, 6)
    )
    rest_global = _rest_local_to_global_rotations(joint_rest_local_rotations_6d)
    global_rotations = torch.where(
        previous_pose_valid[:, None, None, None], previous_global, rest_global
    )

    configured = current_tracker_raw[..., TRACKER_CONFIGURED_OFFSET] > 0.5
    measured = current_tracker_raw[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5
    tracker_valid = configured & measured
    tracker_rotations = rotation_6d_to_matrix_torch(current_tracker_raw[..., 3:9])
    tracker_positions = current_tracker_raw[..., :3]

    # 直接测量旋转先写入骨架，使 Pelvis 的已测旋转能参与后续躯干和腿部位置求解。
    for tracker_index, joint_index in enumerate(TRACKER_TO_JOINT):
        active = tracker_valid[:, tracker_index]
        global_rotations[:, joint_index] = torch.where(
            active[:, None, None],
            tracker_rotations[:, tracker_index],
            global_rotations[:, joint_index],
        )

    # 有 Hip 时用其位置对齐整棵骨架；否则只以始终有效的 Head 对齐上一帧形状。
    joints = _aligned_joint_positions(
        global_rotations,
        joint_offsets_parent,
        tracker_positions,
        tracker_valid[:, HIP_TRACKER_INDEX],
    )

    torso_active = tracker_valid[:, HIP_TRACKER_INDEX] & tracker_valid[:, HEAD_TRACKER_INDEX]
    solved = solve_fabrik_chain(
        joints[:, _TORSO_CHAIN],
        tracker_positions[:, HEAD_TRACKER_INDEX],
        torso_active,
        iterations=fabrik_iterations,
    )
    global_rotations = _apply_chain_directions(
        global_rotations, joints[:, _TORSO_CHAIN], solved, _TORSO_CHAIN, torso_active
    )
    joints = _aligned_joint_positions(
        global_rotations,
        joint_offsets_parent,
        tracker_positions,
        tracker_valid[:, HIP_TRACKER_INDEX],
    )

    for chain, tracker_index in (
        (_LEFT_ARM_CHAIN, LEFT_HAND_TRACKER_INDEX),
        (_RIGHT_ARM_CHAIN, RIGHT_HAND_TRACKER_INDEX),
    ):
        active = tracker_valid[:, tracker_index]
        solved = solve_fabrik_chain(
            joints[:, chain],
            tracker_positions[:, tracker_index],
            active,
            iterations=fabrik_iterations,
        )
        global_rotations = _apply_chain_directions(
            global_rotations, joints[:, chain], solved, chain, active
        )

    for chain, tracker_index in (
        (_LEFT_LEG_CHAIN, LEFT_FOOT_TRACKER_INDEX),
        (_RIGHT_LEG_CHAIN, RIGHT_FOOT_TRACKER_INDEX),
    ):
        active = tracker_valid[:, HIP_TRACKER_INDEX] & tracker_valid[:, tracker_index]
        solved = solve_fabrik_chain(
            joints[:, chain],
            tracker_positions[:, tracker_index],
            active,
            iterations=fabrik_iterations,
        )
        global_rotations = _apply_chain_directions(
            global_rotations, joints[:, chain], solved, chain, active
        )

    # 内部 IK 更新完成后再次覆盖直接 Tracker，保证叶节点旋转不会被任何链路修改。
    for tracker_index, joint_index in enumerate(TRACKER_TO_JOINT):
        active = tracker_valid[:, tracker_index]
        global_rotations[:, joint_index] = torch.where(
            active[:, None, None],
            tracker_rotations[:, tracker_index],
            global_rotations[:, joint_index],
        )

    return rotation_6d_forward_up_torch(global_rotations).reshape(batch_size, -1)


def _rest_local_to_global_rotations(rest_local_6d: torch.Tensor) -> torch.Tensor:
    local = rotation_6d_to_matrix_torch(rest_local_6d)
    global_values: list[torch.Tensor] = []
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            value = local[:, joint_index]
        else:
            value = global_values[parent_index] @ local[:, joint_index]
        global_values.append(value)
    return torch.stack(global_values, dim=1)


def _aligned_joint_positions(
    global_rotations: torch.Tensor,
    offsets: torch.Tensor,
    tracker_positions: torch.Tensor,
    hip_valid: torch.Tensor,
) -> torch.Tensor:
    relative = pelvis_relative_joint_positions_torch(global_rotations, offsets)
    pelvis_target = tracker_positions[:, HIP_TRACKER_INDEX]
    head_target = tracker_positions[:, HEAD_TRACKER_INDEX]
    head_index = JOINT_INDEX["head"]
    head_aligned_translation = head_target - relative[:, head_index]
    translation = torch.where(
        hip_valid[:, None], pelvis_target - relative[:, 0], head_aligned_translation
    )
    return relative + translation[:, None]


def solve_fabrik_chain(
    points: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """固定轮数 FABRIK；输入输出均为 `[B,L,3]`，骨长严格保持。"""

    root = points[:, 0].clone()
    lengths = torch.linalg.norm(points[:, 1:] - points[:, :-1], dim=-1).clamp_min(1e-8)
    total_length = lengths.sum(dim=-1)
    root_to_target = target - root
    target_distance = torch.linalg.norm(root_to_target, dim=-1)
    direction = _safe_normalize(root_to_target, points[:, -1] - root)

    cumulative = torch.cumsum(lengths, dim=-1)
    straight = root[:, None] + direction[:, None] * cumulative[..., None]
    straight = torch.cat([root[:, None], straight], dim=1)

    work = points.clone()
    for _ in range(int(iterations)):
        work[:, -1] = target
        for index in range(work.shape[1] - 2, -1, -1):
            fallback = points[:, index] - points[:, index + 1]
            backward = _safe_normalize(work[:, index] - work[:, index + 1], fallback)
            work[:, index] = work[:, index + 1] + backward * lengths[:, index, None]
        work[:, 0] = root
        for index in range(1, work.shape[1]):
            fallback = points[:, index] - points[:, index - 1]
            forward = _safe_normalize(work[:, index] - work[:, index - 1], fallback)
            work[:, index] = work[:, index - 1] + forward * lengths[:, index - 1, None]

    unreachable = target_distance >= total_length
    solved = torch.where(unreachable[:, None, None], straight, work)
    return torch.where(active[:, None, None], solved, points)


def _apply_chain_directions(
    global_rotations: torch.Tensor,
    old_points: torch.Tensor,
    new_points: torch.Tensor,
    chain: tuple[int, ...],
    active: torch.Tensor,
) -> torch.Tensor:
    result = global_rotations.clone()
    for edge_index, joint_index in enumerate(chain[:-1]):
        old_direction = old_points[:, edge_index + 1] - old_points[:, edge_index]
        new_direction = new_points[:, edge_index + 1] - new_points[:, edge_index]
        alignment = shortest_arc_rotation(old_direction, new_direction)
        updated = alignment @ result[:, joint_index]
        result[:, joint_index] = torch.where(
            active[:, None, None], updated, result[:, joint_index]
        )
    return result


def shortest_arc_rotation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_norm = torch.linalg.norm(source, dim=-1)
    target_norm = torch.linalg.norm(target, dim=-1)
    valid = (source_norm > 1e-8) & (target_norm > 1e-8)
    source_unit = F.normalize(source, dim=-1, eps=1e-8)
    target_unit = F.normalize(target, dim=-1, eps=1e-8)
    cross = torch.cross(source_unit, target_unit, dim=-1)
    cosine = (source_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    skew = _skew_matrix(cross)
    identity = torch.eye(3, device=source.device, dtype=source.dtype).expand(
        source.shape[0], -1, -1
    )
    normal = identity + skew + (skew @ skew) / (1.0 + cosine).clamp_min(1e-6)[:, None, None]

    # 反向共线时 shortest arc 的轴不唯一；选择与源方向最不平行的世界轴，保证确定性。
    basis_x = torch.zeros_like(source_unit)
    basis_x[:, 0] = 1.0
    basis_y = torch.zeros_like(source_unit)
    basis_y[:, 1] = 1.0
    use_x = torch.abs(source_unit[:, 0]) < 0.9
    basis = torch.where(use_x[:, None], basis_x, basis_y)
    opposite_axis = F.normalize(torch.cross(source_unit, basis, dim=-1), dim=-1, eps=1e-8)
    opposite = 2.0 * opposite_axis[:, :, None] * opposite_axis[:, None, :] - identity
    rotation = torch.where((cosine < -0.9999)[:, None, None], opposite, normal)
    return torch.where(valid[:, None, None], rotation, identity)


def _skew_matrix(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(vector.shape[0], 3, 3)


def _safe_normalize(value: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    value_norm = torch.linalg.norm(value, dim=-1, keepdim=True)
    fallback_unit = F.normalize(fallback, dim=-1, eps=1e-8)
    return torch.where(
        value_norm > 1e-8,
        value / value_norm.clamp_min(1e-8),
        fallback_unit,
    )
