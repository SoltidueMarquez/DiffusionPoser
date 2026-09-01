from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from data_loaders.realtime_pose_kinematics import (
    rotation_6d_forward_up_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.sensor_masking import TRACKER_COUNT, TRACKER_TO_JOINT


TrackerActivationRamps = dict[int, tuple[int, np.ndarray, np.ndarray]]


def smoothstep_activation_alpha(frame_offset: int, frame_count: int) -> float:
    """返回新 Tracker 第 ``frame_offset`` 帧的 smoothstep 渐入权重。"""

    if int(frame_count) <= 0:
        raise ValueError("frame_count 必须为正整数。")
    if int(frame_offset) < 0:
        raise ValueError("frame_offset 不能为负数。")
    unit = float(np.clip((int(frame_offset) + 1) / int(frame_count), 0.0, 1.0))
    return unit * unit * (3.0 - 2.0 * unit)


def interpolate_tracker_measurement(
    *,
    anchor_position: np.ndarray,
    anchor_rotation: np.ndarray,
    measured_position: np.ndarray,
    measured_rotation_6d: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """从切换前部署姿态平滑过渡到当前真实 Tracker 测量。"""

    weight = float(alpha)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"alpha 必须位于 [0,1]，实际为 {weight}")
    anchor_position = np.asarray(anchor_position, dtype=np.float64)
    measured_position = np.asarray(measured_position, dtype=np.float64)
    anchor_rotation = np.asarray(anchor_rotation, dtype=np.float64)
    measured_rotation = rotation_6d_to_matrix_np(measured_rotation_6d)
    if anchor_position.shape != (3,) or measured_position.shape != (3,):
        raise ValueError(
            f"Tracker 位置应为 [3]，实际为 {anchor_position.shape}/{measured_position.shape}"
        )
    if anchor_rotation.shape != (3, 3) or measured_rotation.shape != (3, 3):
        raise ValueError(
            f"Tracker 旋转应为 [3,3]，实际为 {anchor_rotation.shape}/{measured_rotation.shape}"
        )
    position = (1.0 - weight) * anchor_position + weight * measured_position
    key_rotations = Rotation.from_matrix(
        np.stack([anchor_rotation, measured_rotation], axis=0)
    )
    rotation = Slerp([0.0, 1.0], key_rotations)([weight]).as_matrix()[0]
    return (
        position.astype(np.float32),
        rotation_6d_forward_up_np(rotation).astype(np.float32),
    )


def apply_tracker_activation_blend(
    *,
    current_frame: int,
    blend_frames: int,
    previous_available: np.ndarray,
    current_available: np.ndarray,
    measured_positions: np.ndarray,
    measured_rotations_6d: np.ndarray,
    previous_joint_positions: np.ndarray | None,
    previous_joint_rotations: np.ndarray | None,
    activation_ramps: TrackerActivationRamps,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """对本帧新接入的 Tracker 应用项目统一的 soft-start 策略。

    每条 ramp 固定锚定在重连前一帧的部署关节姿态；位置使用 LERP，旋转使用
    SLERP，权重在 ``blend_frames`` 帧内按 smoothstep 从锚点渐入真实测量。
    ``activation_ramps`` 由调用方跨帧持有，并在本函数中原地更新。
    """

    frame = int(current_frame)
    frame_count = int(blend_frames)
    if frame_count < 0:
        raise ValueError("blend_frames 不能为负数。")
    previous = np.asarray(previous_available, dtype=bool)
    current = np.asarray(current_available, dtype=bool)
    positions = np.asarray(measured_positions, dtype=np.float32).copy()
    rotations = np.asarray(measured_rotations_6d, dtype=np.float32).copy()
    if previous.shape != (TRACKER_COUNT,) or current.shape != (TRACKER_COUNT,):
        raise ValueError("previous_available/current_available 必须为 [6]。")
    if positions.shape != (TRACKER_COUNT, 3) or rotations.shape != (TRACKER_COUNT, 6):
        raise ValueError("Tracker 测量必须分别为 [6,3]/[6,6]。")

    newly_added = tuple(np.flatnonzero(~previous & current).astype(int).tolist())
    if frame_count > 0 and newly_added:
        if previous_joint_positions is None or previous_joint_rotations is None:
            raise RuntimeError("Tracker 渐入缺少切换前一帧的部署姿态。")
        joint_positions = np.asarray(previous_joint_positions, dtype=np.float32)
        joint_rotations = np.asarray(previous_joint_rotations, dtype=np.float32)
        if joint_positions.ndim != 2 or joint_positions.shape[1] != 3:
            raise ValueError("previous_joint_positions 必须为 [J,3]。")
        if joint_rotations.ndim != 3 or joint_rotations.shape[1:] != (3, 3):
            raise ValueError("previous_joint_rotations 必须为 [J,3,3]。")
        for tracker_index in newly_added:
            joint_index = int(TRACKER_TO_JOINT[tracker_index])
            activation_ramps[tracker_index] = (
                frame,
                joint_positions[joint_index].copy(),
                joint_rotations[joint_index].copy(),
            )

    alpha = current.astype(np.float32)
    finished_ramps: list[int] = []
    for tracker_index, (start_frame, anchor_position, anchor_rotation) in tuple(
        activation_ramps.items()
    ):
        if not current[tracker_index]:
            finished_ramps.append(tracker_index)
            continue
        frame_offset = frame - int(start_frame)
        if frame_offset < 0:
            raise RuntimeError("Tracker activation ramp 起始帧晚于当前帧。")
        if frame_offset >= frame_count:
            finished_ramps.append(tracker_index)
            continue
        weight = smoothstep_activation_alpha(frame_offset, frame_count)
        positions[tracker_index], rotations[tracker_index] = (
            interpolate_tracker_measurement(
                anchor_position=anchor_position,
                anchor_rotation=anchor_rotation,
                measured_position=positions[tracker_index],
                measured_rotation_6d=rotations[tracker_index],
                alpha=weight,
            )
        )
        alpha[tracker_index] = weight
    for tracker_index in finished_ramps:
        del activation_ramps[tracker_index]
    return positions, rotations, alpha, newly_added
