from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_loaders.realtime_pose_geometry import (
    build_pose_target_np,
    build_tracker_measurements_np,
)
from data_loaders.realtime_pose_kinematics import (
    make_yaw_rotation_np,
    make_yaw_rotation_torch,
    rotation_6d_forward_up_np,
    rotation_6d_forward_up_torch,
    rotation_6d_to_matrix_np,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    CORE_TRACKER_INDICES,
    HEAD_TRACKER_INDEX,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH,
    PREDICTOR_POSE_HORIZON_LENGTH,
    PREDICTOR_SPARSE_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
)


@dataclass(frozen=True)
class PredictorStepFeatures:
    """单个 rolling step 的 Predictor 输入与监督，全部表达在当前 `C_n`。"""

    motion_context: np.ndarray  # [10,144]
    core_tracker_context: np.ndarray  # [11,54]
    pose_target_horizon: np.ndarray | None  # [11,144]
    current_head_yaw_world: float


def build_predictor_sparse_features_np(
    tracker_continuous_with_previous: np.ndarray,
) -> np.ndarray:
    """从 `[-11,...,0]` 的 12 帧 Tracker 构造官方语义的 `[11,54]`。

    输入的 12 帧必须已经表达在同一个当前 Head-yaw 参考系中。输出通道依次
    为三点 rotation6D、相对 rotation6D、position、position delta。
    """

    tracker = np.asarray(tracker_continuous_with_previous, dtype=np.float64)
    expected = (
        PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH + 1,
        TRACKER_COUNT,
        TRACKER_CONTINUOUS_DIM,
    )
    if tracker.shape != expected:
        raise ValueError(f"tracker_continuous_with_previous 必须为 {expected}。")
    core = tracker[:, list(CORE_TRACKER_INDICES)]
    rotations = rotation_6d_to_matrix_np(core[..., 3:9])
    relative = np.swapaxes(rotations[:-1], -1, -2) @ rotations[1:]
    result = np.concatenate(
        [
            core[1:, :, 3:9].reshape(PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, -1),
            rotation_6d_forward_up_np(relative).reshape(
                PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, -1
            ),
            core[1:, :, :3].reshape(PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, -1),
            (core[1:, :, :3] - core[:-1, :, :3]).reshape(
                PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, -1
            ),
        ],
        axis=-1,
    )
    if result.shape != (PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, PREDICTOR_SPARSE_DIM):
        raise RuntimeError("Predictor sparse feature 内部布局错误。")
    return result.astype(np.float32)


def build_predictor_sparse_features_torch(
    tracker_continuous_with_previous: torch.Tensor,
) -> torch.Tensor:
    """Torch 版 54D 特征构造，输入 `[B,12,6,9]`。"""

    if tracker_continuous_with_previous.ndim != 4 or tuple(
        tracker_continuous_with_previous.shape[1:]
    ) != (
        PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH + 1,
        TRACKER_COUNT,
        TRACKER_CONTINUOUS_DIM,
    ):
        raise ValueError("tracker_continuous_with_previous 必须为 [B,12,6,9]。")
    core_indices = torch.as_tensor(
        CORE_TRACKER_INDICES,
        device=tracker_continuous_with_previous.device,
        dtype=torch.long,
    )
    core = tracker_continuous_with_previous.index_select(2, core_indices)
    rotations = rotation_6d_to_matrix_torch(core[..., 3:9])
    relative = rotations[:, :-1].transpose(-1, -2) @ rotations[:, 1:]
    batch_size = core.shape[0]
    result = torch.cat(
        [
            core[:, 1:, :, 3:9].reshape(batch_size, PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, -1),
            rotation_6d_forward_up_torch(relative).reshape(
                batch_size, PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, -1
            ),
            core[:, 1:, :, :3].reshape(batch_size, PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, -1),
            (core[:, 1:, :, :3] - core[:, :-1, :, :3]).reshape(
                batch_size, PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, -1
            ),
        ],
        dim=-1,
    )
    if tuple(result.shape[1:]) != (PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, PREDICTOR_SPARSE_DIM):
        raise RuntimeError("Predictor sparse feature 内部布局错误。")
    return result


def build_predictor_step_features_np(
    motion_rotations_world: np.ndarray,
    tracker_positions_world_with_previous: np.ndarray,
    tracker_rotations_world_6d_with_previous: np.ndarray,
    floor_y: float,
    pose_target_rotations_world: np.ndarray | None = None,
) -> PredictorStepFeatures:
    """为一个预测时刻构造 Predictor 输入；不读取当前之后的 Tracker。"""

    motion_world = np.asarray(motion_rotations_world, dtype=np.float64)
    if motion_world.shape != (
        REALTIME_POSE_HISTORY_LENGTH,
        SMPL_JOINT_COUNT,
        3,
        3,
    ):
        raise ValueError("motion_rotations_world 必须为 [10,24,3,3]。")
    tracker_positions = np.asarray(
        tracker_positions_world_with_previous, dtype=np.float64
    )
    tracker_rotations_6d = np.asarray(
        tracker_rotations_world_6d_with_previous, dtype=np.float64
    )
    if tracker_positions.shape != (
        PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH + 1,
        TRACKER_COUNT,
        3,
    ):
        raise ValueError("tracker_positions_world_with_previous 必须为 [12,6,3]。")
    if tracker_rotations_6d.shape != (
        PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH + 1,
        TRACKER_COUNT,
        6,
    ):
        raise ValueError("tracker_rotations_world_6d_with_previous 必须为 [12,6,6]。")

    tracker_rotations = rotation_6d_to_matrix_np(tracker_rotations_6d)
    current_head_rotation = tracker_rotations[-1, HEAD_TRACKER_INDEX]
    current_head_forward = current_head_rotation[:, 2]
    current_head_yaw = float(
        np.arctan2(current_head_forward[0], current_head_forward[2])
    )
    current_head_position = tracker_positions[-1, HEAD_TRACKER_INDEX]
    motion = build_pose_target_np(motion_world, current_head_yaw)
    tracker_continuous = build_tracker_measurements_np(
        tracker_positions,
        tracker_rotations_6d,
        current_head_position,
        float(floor_y),
        current_head_yaw,
    )
    sparse = build_predictor_sparse_features_np(tracker_continuous)

    target = None
    if pose_target_rotations_world is not None:
        target_world = np.asarray(pose_target_rotations_world, dtype=np.float64)
        if target_world.shape != (
            PREDICTOR_POSE_HORIZON_LENGTH,
            SMPL_JOINT_COUNT,
            3,
            3,
        ):
            raise ValueError("pose_target_rotations_world 必须为 [11,24,3,3]。")
        target = build_pose_target_np(target_world, current_head_yaw)
    return PredictorStepFeatures(
        motion_context=motion,
        core_tracker_context=sparse,
        pose_target_horizon=target,
        current_head_yaw_world=current_head_yaw,
    )


def build_predictor_step_features_torch(
    motion_rotations_world: torch.Tensor,
    tracker_positions_world_with_previous: torch.Tensor,
    tracker_rotations_world_6d_with_previous: torch.Tensor,
    floor_y: torch.Tensor,
    pose_target_rotations_world: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """批量 Torch 版 step builder，供 free-running 训练在 GPU 上重建 `C_n`。"""

    if motion_rotations_world.ndim != 5 or tuple(motion_rotations_world.shape[1:]) != (
        REALTIME_POSE_HISTORY_LENGTH,
        SMPL_JOINT_COUNT,
        3,
        3,
    ):
        raise ValueError("motion_rotations_world 必须为 [B,10,24,3,3]。")
    batch_size = motion_rotations_world.shape[0]
    if tuple(tracker_positions_world_with_previous.shape) != (
        batch_size,
        PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH + 1,
        TRACKER_COUNT,
        3,
    ):
        raise ValueError("tracker_positions_world_with_previous 必须为 [B,12,6,3]。")
    if tuple(tracker_rotations_world_6d_with_previous.shape) != (
        batch_size,
        PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH + 1,
        TRACKER_COUNT,
        6,
    ):
        raise ValueError("tracker_rotations_world_6d_with_previous 必须为 [B,12,6,6]。")

    tracker_rotations_world = rotation_6d_to_matrix_torch(
        tracker_rotations_world_6d_with_previous
    )
    head_forward = tracker_rotations_world[:, -1, HEAD_TRACKER_INDEX, :, 2]
    head_yaw = torch.atan2(head_forward[:, 0], head_forward[:, 2])
    yaw_inverse = make_yaw_rotation_torch(head_yaw).transpose(-1, -2)

    motion_head = torch.einsum(
        "bij,bthjk->bthik", yaw_inverse, motion_rotations_world
    )
    motion = rotation_6d_forward_up_torch(motion_head).reshape(
        batch_size, REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM
    )

    head_position = tracker_positions_world_with_previous[
        :, -1, HEAD_TRACKER_INDEX
    ]
    origin = head_position.clone()
    origin[:, 1] = floor_y.reshape(-1)
    tracker_positions_head = torch.einsum(
        "bij,btaj->btai",
        yaw_inverse,
        tracker_positions_world_with_previous - origin[:, None, None],
    )
    tracker_rotations_head = torch.einsum(
        "bij,btajk->btaik", yaw_inverse, tracker_rotations_world
    )
    tracker_continuous = torch.cat(
        [
            tracker_positions_head,
            rotation_6d_forward_up_torch(tracker_rotations_head),
        ],
        dim=-1,
    )
    sparse = build_predictor_sparse_features_torch(tracker_continuous)

    target = None
    if pose_target_rotations_world is not None:
        if tuple(pose_target_rotations_world.shape) != (
            batch_size,
            PREDICTOR_POSE_HORIZON_LENGTH,
            SMPL_JOINT_COUNT,
            3,
            3,
        ):
            raise ValueError("pose_target_rotations_world 必须为 [B,11,24,3,3]。")
        target_head = torch.einsum(
            "bij,bthjk->bthik", yaw_inverse, pose_target_rotations_world
        )
        target = rotation_6d_forward_up_torch(target_head).reshape(
            batch_size, PREDICTOR_POSE_HORIZON_LENGTH, REALTIME_POSE_TARGET_DIM
        )
    return motion, sparse, target, head_yaw


def pose_head_to_world_rotations_np(
    pose_head_6d: np.ndarray,
    head_yaw_world: float,
) -> np.ndarray:
    """把 `C_n` 下的 `[... ,144]` Pose 恢复为世界全局旋转矩阵。"""

    pose = np.asarray(pose_head_6d, dtype=np.float64)
    if pose.shape[-1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError("pose_head_6d 最后一维必须为 144。")
    rotations_head = rotation_6d_to_matrix_np(
        pose.reshape(*pose.shape[:-1], SMPL_JOINT_COUNT, 6)
    )
    yaw = make_yaw_rotation_np(np.asarray([head_yaw_world], dtype=np.float64))[0]
    return np.einsum("ij,...hjk->...hik", yaw, rotations_head).astype(np.float32)


def pose_head_to_world_rotations_torch(
    pose_head_6d: torch.Tensor,
    head_yaw_world: torch.Tensor,
) -> torch.Tensor:
    """Torch 版 `C_n → world`，输入 `[B,...,144]`。"""

    if pose_head_6d.shape[-1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError("pose_head_6d 最后一维必须为 144。")
    batch_size = pose_head_6d.shape[0]
    rotations_head = rotation_6d_to_matrix_torch(
        pose_head_6d.reshape(
            batch_size, *pose_head_6d.shape[1:-1], SMPL_JOINT_COUNT, 6
        )
    )
    yaw = make_yaw_rotation_torch(head_yaw_world.reshape(-1))
    return torch.einsum("bij,b...hjk->b...hik", yaw, rotations_head)
