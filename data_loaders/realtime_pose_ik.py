from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_geometry import pelvis_relative_joint_positions_torch
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
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
    TRACKER_AVAILABLE_OFFSET,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_TO_JOINT,
)


DIRECT_ROTATION = 0
POSITION_SOLVED = 1
DIRECTION_ONLY = 2
INHERITED = 3


@dataclass(frozen=True)
class RealtimePoseIKResult:
    """当前帧部分 IK 结果；所有逐关节张量的关节轴均为 24。"""

    pose: torch.Tensor  # [B,24,6]
    updated_mask: torch.Tensor  # [B,24]
    direct_rotation_mask: torch.Tensor  # [B,24]
    constraint_type: torch.Tensor  # [B,24]
    position_residual: torch.Tensor  # [B,24]
    confidence: torch.Tensor  # [B,24]

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


def build_current_ik(
    initial_pose_raw: torch.Tensor,
    current_tracker_raw: torch.Tensor,
    joint_offsets_parent: torch.Tensor,
    config: IKInpaintingConfig,
) -> RealtimePoseIKResult:
    """用初始化 Pose 和当前 Tracker 构造 `[B,24,6]` 部分 IK 结果。

    输入 Pose 与 Tracker 都必须已经表达在当前 Head-yaw 参考系中。函数只把
    Tracker 真正约束的骨链标为已更新；上一帧/rest pose 只负责初始化，绝不会
    自动变为 inpainting 条件。当前 FABRIK 只确定骨骼方向，因此所有非直接更新
    都属于 DIRECTION_ONLY，不产生 POSITION_SOLVED。
    """

    cfg = config.validate()
    batch_size = initial_pose_raw.shape[0]
    if tuple(initial_pose_raw.shape) != (batch_size, SMPL_JOINT_COUNT * 6):
        raise ValueError("initial_pose_raw 必须为 Predictor current `[B,144]`。")
    if tuple(current_tracker_raw.shape) != (
        batch_size,
        TRACKER_COUNT,
        TRACKER_FEATURE_DIM,
    ):
        raise ValueError("current_tracker_raw 必须为 [B,6,10]。")
    if tuple(joint_offsets_parent.shape) != (batch_size, SMPL_JOINT_COUNT, 3):
        raise ValueError("joint_offsets_parent 必须为 [B,24,3]。")
    global_rotations = rotation_6d_to_matrix_torch(
        initial_pose_raw.reshape(batch_size, SMPL_JOINT_COUNT, 6)
    )
    updated_mask = torch.zeros(
        batch_size, SMPL_JOINT_COUNT, device=global_rotations.device, dtype=torch.bool
    )
    direct_rotation_mask = torch.zeros_like(updated_mask)
    constraint_type = torch.full(
        (batch_size, SMPL_JOINT_COUNT),
        INHERITED,
        device=global_rotations.device,
        dtype=torch.long,
    )
    position_residual = torch.zeros(
        batch_size,
        SMPL_JOINT_COUNT,
        device=global_rotations.device,
        dtype=global_rotations.dtype,
    )
    chain_length = build_ik_joint_chain_length(joint_offsets_parent)

    tracker_valid = current_tracker_raw[..., TRACKER_AVAILABLE_OFFSET] > 0.5
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
        updated_mask[:, joint_index] |= active
        direct_rotation_mask[:, joint_index] |= active
        constraint_type[:, joint_index] = torch.where(
            active,
            torch.full_like(constraint_type[:, joint_index], DIRECT_ROTATION),
            constraint_type[:, joint_index],
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
        iterations=cfg.fabrik_iterations,
    )
    global_rotations = _apply_chain_directions(
        global_rotations, joints[:, _TORSO_CHAIN], solved, _TORSO_CHAIN, torso_active
    )
    _mark_direction_only_chain(
        updated_mask,
        constraint_type,
        _TORSO_CHAIN,
        torso_active,
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
            iterations=cfg.fabrik_iterations,
        )
        global_rotations = _apply_chain_directions(
            global_rotations, joints[:, chain], solved, chain, active
        )
        _mark_direction_only_chain(
            updated_mask,
            constraint_type,
            chain,
            active,
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
            iterations=cfg.fabrik_iterations,
        )
        global_rotations = _apply_chain_directions(
            global_rotations, joints[:, chain], solved, chain, active
        )
        _mark_direction_only_chain(
            updated_mask,
            constraint_type,
            chain,
            active,
        )

    # 内部 IK 更新完成后再次覆盖直接 Tracker，保证叶节点旋转不会被任何链路修改。
    for tracker_index, joint_index in enumerate(TRACKER_TO_JOINT):
        active = tracker_valid[:, tracker_index]
        global_rotations[:, joint_index] = torch.where(
            active[:, None, None],
            tracker_rotations[:, tracker_index],
            global_rotations[:, joint_index],
        )

    final_joints = _aligned_joint_positions(
        global_rotations,
        joint_offsets_parent,
        tracker_positions,
        tracker_valid[:, HIP_TRACKER_INDEX],
    )
    for chain, tracker_index, active in (
        (_TORSO_CHAIN, HEAD_TRACKER_INDEX, torso_active),
        (_LEFT_ARM_CHAIN, LEFT_HAND_TRACKER_INDEX, tracker_valid[:, LEFT_HAND_TRACKER_INDEX]),
        (_RIGHT_ARM_CHAIN, RIGHT_HAND_TRACKER_INDEX, tracker_valid[:, RIGHT_HAND_TRACKER_INDEX]),
        (
            _LEFT_LEG_CHAIN,
            LEFT_FOOT_TRACKER_INDEX,
            tracker_valid[:, HIP_TRACKER_INDEX] & tracker_valid[:, LEFT_FOOT_TRACKER_INDEX],
        ),
        (
            _RIGHT_LEG_CHAIN,
            RIGHT_FOOT_TRACKER_INDEX,
            tracker_valid[:, HIP_TRACKER_INDEX] & tracker_valid[:, RIGHT_FOOT_TRACKER_INDEX],
        ),
    ):
        chain_points = final_joints[:, chain]
        endpoint_residual = torch.linalg.norm(
            chain_points[:, -1] - tracker_positions[:, tracker_index], dim=-1
        )
        for joint_index in chain:
            position_residual[:, joint_index] = torch.where(
                active, endpoint_residual, position_residual[:, joint_index]
            )

    joint_source_reliability = build_ik_joint_source_reliability(
        tracker_source_reliability=tracker_valid.to(global_rotations.dtype),
        constraint_type=constraint_type,
    )
    confidence = compute_ik_joint_confidence_torch(
        joint_source_reliability=joint_source_reliability,
        constraint_type=constraint_type,
        updated_mask=updated_mask,
        position_residual=position_residual,
        chain_length=chain_length,
        direction_only_quality=float(cfg.direction_only_quality),
        residual_scale=float(cfg.residual_scale),
        position_solved_quality=cfg.position_solved_quality,
    )
    if bool((constraint_type == POSITION_SOLVED).any()):
        raise RuntimeError("当前 shortest-arc FABRIK 不应产生 POSITION_SOLVED。")
    if bool((updated_mask[constraint_type == INHERITED]).any()):
        raise RuntimeError("INHERITED 关节不得标记为已更新。")
    return RealtimePoseIKResult(
        pose=rotation_6d_forward_up_torch(global_rotations),
        updated_mask=updated_mask,
        direct_rotation_mask=direct_rotation_mask,
        constraint_type=constraint_type,
        position_residual=position_residual,
        confidence=confidence,
    )


def build_ik_joint_chain_length(joint_offsets_parent: torch.Tensor) -> torch.Tensor:
    """按当前五条 FABRIK 链返回 `[B,24]` 可动链总长。"""

    if joint_offsets_parent.ndim != 3 or tuple(joint_offsets_parent.shape[1:]) != (
        SMPL_JOINT_COUNT,
        3,
    ):
        raise ValueError("joint_offsets_parent 必须为 [B,24,3]。")
    result = torch.zeros(
        joint_offsets_parent.shape[0],
        SMPL_JOINT_COUNT,
        device=joint_offsets_parent.device,
        dtype=joint_offsets_parent.dtype,
    )
    offset_lengths = torch.linalg.norm(joint_offsets_parent, dim=-1)
    for chain in (
        _TORSO_CHAIN,
        _LEFT_ARM_CHAIN,
        _RIGHT_ARM_CHAIN,
        _LEFT_LEG_CHAIN,
        _RIGHT_LEG_CHAIN,
    ):
        total_length = offset_lengths[:, list(chain[1:])].sum(dim=-1)
        result[:, list(chain)] = total_length[:, None]
    return result


def build_ik_joint_source_reliability(
    tracker_source_reliability: torch.Tensor,
    constraint_type: torch.Tensor,
) -> torch.Tensor:
    """按真实约束依赖把 `[B,6]` Tracker 来源可靠度映射到 `[B,24]`。"""

    batch_size = tracker_source_reliability.shape[0]
    if tuple(tracker_source_reliability.shape) != (batch_size, TRACKER_COUNT):
        raise ValueError("tracker_source_reliability 必须为 [B,6]。")
    if tuple(constraint_type.shape) != (batch_size, SMPL_JOINT_COUNT):
        raise ValueError("constraint_type 必须为 [B,24]。")
    source = tracker_source_reliability.float()
    if not bool(torch.isfinite(source).all()) or bool(
        ((source < 0.0) | (source > 1.0)).any()
    ):
        raise ValueError("tracker_source_reliability 必须为有限的 [0,1] 数值。")
    result = torch.zeros(
        batch_size,
        SMPL_JOINT_COUNT,
        device=source.device,
        dtype=source.dtype,
    )
    for tracker_index, joint_index in enumerate(TRACKER_TO_JOINT):
        result[:, joint_index] = torch.where(
            constraint_type[:, joint_index] == DIRECT_ROTATION,
            source[:, tracker_index],
            result[:, joint_index],
        )
    chain_sources = (
        (_TORSO_CHAIN, torch.minimum(source[:, HEAD_TRACKER_INDEX], source[:, HIP_TRACKER_INDEX])),
        (_LEFT_ARM_CHAIN, source[:, LEFT_HAND_TRACKER_INDEX]),
        (_RIGHT_ARM_CHAIN, source[:, RIGHT_HAND_TRACKER_INDEX]),
        (
            _LEFT_LEG_CHAIN,
            torch.minimum(source[:, HIP_TRACKER_INDEX], source[:, LEFT_FOOT_TRACKER_INDEX]),
        ),
        (
            _RIGHT_LEG_CHAIN,
            torch.minimum(source[:, HIP_TRACKER_INDEX], source[:, RIGHT_FOOT_TRACKER_INDEX]),
        ),
    )
    for chain, chain_source in chain_sources:
        for joint_index in chain[:-1]:
            result[:, joint_index] = torch.where(
                constraint_type[:, joint_index] == DIRECTION_ONLY,
                chain_source,
                result[:, joint_index],
            )
    return result


def compute_ik_joint_confidence_torch(
    joint_source_reliability: torch.Tensor,
    constraint_type: torch.Tensor,
    updated_mask: torch.Tensor,
    position_residual: torch.Tensor,
    chain_length: torch.Tensor,
    direction_only_quality: float,
    residual_scale: float,
    position_solved_quality: float | None = None,
) -> torch.Tensor:
    """按约束类型、FABRIK 端点残差和固定 source reliability 计算 confidence。"""

    source = joint_source_reliability.float()
    expected = source.shape
    if source.ndim != 2 or source.shape[1] != SMPL_JOINT_COUNT:
        raise ValueError("joint_source_reliability 必须为 [B,24]。")
    if any(
        value.shape != expected
        for value in (constraint_type, updated_mask, position_residual, chain_length)
    ):
        raise ValueError("IK confidence 的逐关节输入必须同为 [B,24]。")
    if not 0.0 < float(direction_only_quality) < 1.0:
        raise ValueError("direction_only_quality 必须位于 (0,1)。")
    if float(residual_scale) <= 0.0:
        raise ValueError("residual_scale 必须大于 0。")
    constraint = constraint_type.long()
    position_solved = constraint == POSITION_SOLVED
    if bool(position_solved.any()) and position_solved_quality is None:
        raise ValueError("POSITION_SOLVED 必须提供 position_solved_quality。")
    quality = torch.zeros_like(source)
    quality = torch.where(constraint == DIRECT_ROTATION, torch.ones_like(quality), quality)
    if position_solved_quality is not None:
        quality = torch.where(
            position_solved,
            torch.full_like(quality, float(position_solved_quality)),
            quality,
        )
    quality = torch.where(
        constraint == DIRECTION_ONLY,
        torch.full_like(quality, float(direction_only_quality)),
        quality,
    )
    residual_constrained = updated_mask.bool() & (
        position_solved | (constraint == DIRECTION_ONLY)
    )
    if bool((residual_constrained & (chain_length <= 0.0)).any()):
        raise ValueError("位置/方向约束必须有正的 chain_length。")
    residual_quality = torch.exp(
        -(position_residual / chain_length.clamp_min(1e-8)) / float(residual_scale)
    )
    residual_quality = torch.where(
        constraint == DIRECT_ROTATION, torch.ones_like(source), residual_quality
    )
    confidence = (source * quality * residual_quality).clamp(0.0, 1.0)
    return torch.where(updated_mask.bool(), confidence, torch.zeros_like(confidence))


def _mark_direction_only_chain(
    updated_mask: torch.Tensor,
    constraint_type: torch.Tensor,
    chain: tuple[int, ...],
    active: torch.Tensor,
) -> None:
    """只标记 FABRIK 实际旋转的父关节；链末端旋转仍由 Tracker 直接提供。"""

    for joint_index in chain[:-1]:
        updated_mask[:, joint_index] |= active
        constraint_type[:, joint_index] = torch.where(
            active,
            torch.full_like(constraint_type[:, joint_index], DIRECTION_ONLY),
            constraint_type[:, joint_index],
        )


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
