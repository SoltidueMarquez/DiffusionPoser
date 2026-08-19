from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH,
    PREDICTOR_SPARSE_DIM,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    validate_tracker_available,
)


SOURCE_FRAME_SHAPES = {
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY: (144,),
    "root_pos_world": (3,),
    "root_yaw": (),
    "pelvis_height": (1,),
    "tracker_pos_world": (TRACKER_COUNT, 3),
    "tracker_rot_world_6d": (TRACKER_COUNT, 6),
    "joints_world": (24, 3),
    "stationary_prob_5": (5,),
}

SOURCE_STATIC_SHAPES = {
    "joint_offsets_parent": (24, 3),
    "joint_rest_local_rotations_6d": (24, 6),
}

TASK_SHAPES = {
    "x": (REALTIME_POSE_TARGET_DIM,),
    "motion_context": (REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM),
    "core_tracker_context": (PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH, PREDICTOR_SPARSE_DIM),
    "current_tracker_raw": (TRACKER_COUNT, TRACKER_FEATURE_DIM),
    "tracker_available": (TRACKER_COUNT,),
    "previous_pose_target": (REALTIME_POSE_TARGET_DIM,),
    "target_joints_head_ref": (24, 3),
    "target_root_position_head_ref": (3,),
    "current_head_position_world": (3,),
    "joint_offsets_parent": (24, 3),
    "joint_rest_local_rotations_6d": (24, 6),
}

TASK_SCALAR_FIELDS = (
    "target_root_yaw_world",
    "target_hip_height",
    "current_head_yaw_world",
    "floor_y",
    "current_frame",
    "task_id",
)


def validate_realtime_source_arrays(
    payload: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> int:
    """校验 30Hz source 数组并返回帧数。"""

    label = f"{path} " if path else ""
    if BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY not in payload:
        raise KeyError(f"{label}缺少 `{BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY}`")
    body_pose = np.asarray(payload[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY])
    if body_pose.ndim != 2:
        raise ValueError(
            f"{label}{BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY} 必须为 [T,144]，"
            f"实际为 {body_pose.shape}"
        )
    frame_count = int(body_pose.shape[0])
    if frame_count <= 0:
        raise ValueError(f"{label}source 帧数必须大于 0")
    for key, frame_shape in SOURCE_FRAME_SHAPES.items():
        _validate_array(payload, key, (frame_count, *frame_shape), label)
    for key, shape in SOURCE_STATIC_SHAPES.items():
        _validate_array(payload, key, shape, label)
    return frame_count


def validate_realtime_task_arrays(
    task: Mapping[str, Any],
    *,
    seq_len: int = REALTIME_POSE_SEQ_LEN,
    path: Path | None = None,
) -> None:
    """在 Dataset/测试边界校验新的单帧 batch 契约。"""

    label = f"{path} " if path else ""
    if int(seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(
            f"{label}当前任务固定 seq_len={REALTIME_POSE_SEQ_LEN}，实际为 {seq_len}"
        )
    for key, shape in TASK_SHAPES.items():
        _validate_array(task, key, shape, label)
    for key in TASK_SCALAR_FIELDS:
        if key not in task:
            raise KeyError(f"{label}缺少 `{key}`")
        if key != "task_id" and np.asarray(task[key]).shape != ():
            raise ValueError(f"{label}{key} 必须是标量。")

    available = validate_tracker_available(np.asarray(task["tracker_available"]))
    tracker_raw = np.asarray(task["current_tracker_raw"])
    if not np.array_equal(tracker_raw[:, -1] > 0.5, available):
        raise ValueError(f"{label}current_tracker_raw available 与独立 mask 不一致。")
    if not np.allclose(
        tracker_raw[:, :TRACKER_CONTINUOUS_DIM][~available], 0.0, atol=1e-7
    ):
        raise ValueError(f"{label}不可用 Tracker 的连续量必须为零。")


def _validate_array(
    payload: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
    label: str,
) -> None:
    if key not in payload:
        raise KeyError(f"{label}缺少 `{key}`")
    value = np.asarray(payload[key])
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{label}{key} 必须为 {shape}，实际为 {value.shape}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{label}{key} 包含 NaN 或 Inf")
