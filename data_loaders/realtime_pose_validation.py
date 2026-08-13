from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_CONDITION_WINDOW_LENGTH,
    REALTIME_POSE_MODEL_TOKEN_LENGTH,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_HISTORY_ANCHOR_COUNT,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_MEASURED_VALID_OFFSET,
    TRACKER_D_OFF_OFFSET,
    TRACKER_D_ON_OFFSET,
    validate_tracker_states,
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
    "x": (REALTIME_POSE_TARGET_LENGTH, REALTIME_POSE_TARGET_DIM),
    "history_pose_observation": (REALTIME_POSE_HISTORY_ANCHOR_COUNT, REALTIME_POSE_TARGET_DIM),
    "tracker_window": (REALTIME_POSE_CONDITION_WINDOW_LENGTH, TRACKER_COUNT, TRACKER_FEATURE_DIM),
    "head_path_window": (REALTIME_POSE_CONDITION_WINDOW_LENGTH, 5),
    "history_region_confidence": (REALTIME_POSE_HISTORY_ANCHOR_COUNT, 5),
    "window_valid_mask": (REALTIME_POSE_CONDITION_WINDOW_LENGTH,),
    "frame_offsets": (REALTIME_POSE_MODEL_TOKEN_LENGTH,),
    "tracker_window_raw": (REALTIME_POSE_CONDITION_WINDOW_LENGTH, TRACKER_COUNT, TRACKER_FEATURE_DIM),
    "hard_rotation_state_window": (REALTIME_POSE_CONDITION_WINDOW_LENGTH, TRACKER_COUNT),
    "target_joints_head_ref": (24, 3),
    "target_root_position_head_ref": (3,),
    "current_head_position_world": (3,),
    "joint_offsets_parent": (24, 3),
    "joint_rest_local_rotations_6d": (24, 6),
    "configured": (REALTIME_POSE_CONDITION_WINDOW_LENGTH, TRACKER_COUNT),
    "measured_valid": (REALTIME_POSE_CONDITION_WINDOW_LENGTH, TRACKER_COUNT),
    "d_off": (REALTIME_POSE_CONDITION_WINDOW_LENGTH, TRACKER_COUNT),
    "d_on": (REALTIME_POSE_CONDITION_WINDOW_LENGTH, TRACKER_COUNT),
    "previous_contact_target": (2,),
    "contact_target": (2,),
}

TASK_SCALAR_FIELDS = (
    "target_root_yaw_world",
    "target_hip_height",
    "current_head_yaw_world",
    "floor_y",
    "scenario",
    "start_frame",
    "scenario_id",
    "task_id",
)


def validate_realtime_source_arrays(
    payload: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> int:
    """校验 source 数组契约并返回帧数。

    source 的身份、镜像状态和划分都由目录决定，因此这里不再读取内嵌
    metadata。60 Hz 是转换与任务链路的固定接口，不再通过每个文件重复声明。
    """

    label = f"{path} " if path else ""
    if BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY not in payload:
        raise KeyError(f"{label}缺少 `{BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY}`")
    body_pose = np.asarray(payload[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY])
    if body_pose.ndim != 2:
        raise ValueError(f"{label}{BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY} 必须为 [T,144]，实际为 {body_pose.shape}")
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
    """在 Dataset 边界校验 materialized task 的形状与 mask 语义。"""

    label = f"{path} " if path else ""
    if int(seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"{label}当前任务固定 seq_len={REALTIME_POSE_SEQ_LEN}，实际为 {seq_len}")
    for key, shape in TASK_SHAPES.items():
        _validate_array(task, key, shape, label)
    for key in TASK_SCALAR_FIELDS:
        if key not in task:
            raise KeyError(f"{label}缺少 `{key}`")
        if np.asarray(task[key]).shape != ():
            raise ValueError(f"{label}{key} 必须是标量，实际为 {np.asarray(task[key]).shape}")

    start_frame = int(np.asarray(task["start_frame"]).item())
    if start_frame < 0:
        raise ValueError(f"{label}start_frame 不能为负数")

    configured = np.asarray(task["configured"], dtype=bool)
    measured_valid = np.asarray(task["measured_valid"], dtype=bool)
    window_valid = np.asarray(task["window_valid_mask"], dtype=bool)
    validate_tracker_states(configured[window_valid], measured_valid[window_valid])
    if np.any(configured[~window_valid]) or np.any(measured_valid[~window_valid]):
        raise ValueError(f"{label}padding Tracker 状态必须清零。")
    tracker = np.asarray(task["tracker_window"])
    tracker_raw = np.asarray(task["tracker_window_raw"])
    if not np.array_equal(tracker[..., TRACKER_CONFIGURED_OFFSET] > 0.5, configured):
        raise ValueError(f"{label}Tracker configured 与 task.configured 不一致")
    if not np.array_equal(tracker[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5, measured_valid):
        raise ValueError(f"{label}Tracker measured_valid 与 task.measured_valid 不一致")
    if not np.array_equal(tracker_raw[..., TRACKER_CONFIGURED_OFFSET] > 0.5, configured):
        raise ValueError(f"{label}Raw Tracker configured 与 task.configured 不一致")
    if not np.array_equal(tracker_raw[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5, measured_valid):
        raise ValueError(f"{label}Raw Tracker measured_valid 与 task.measured_valid 不一致")
    d_off = np.asarray(task["d_off"], dtype=np.int64)
    d_on = np.asarray(task["d_on"], dtype=np.int64)
    if np.any((d_off < 0) | (d_off > 60)) or np.any((d_on < 0) | (d_on > 60)):
        raise ValueError(f"{label}d_off/d_on 必须位于 [0,60]")
    if not np.allclose(tracker[..., TRACKER_D_OFF_OFFSET], d_off / 60.0, atol=1e-6):
        raise ValueError(f"{label}Tracker d_off 与 task.d_off 不一致")
    if not np.allclose(tracker[..., TRACKER_D_ON_OFFSET], d_on / 60.0, atol=1e-6):
        raise ValueError(f"{label}Tracker d_on 与 task.d_on 不一致")
    if not np.allclose(tracker[..., :9][~measured_valid], 0.0, atol=1e-7):
        raise ValueError(f"{label}无效测量的前 9 维必须清零")
    if not np.allclose(tracker_raw[..., :9][~measured_valid], 0.0, atol=1e-7):
        raise ValueError(f"{label}Raw Tracker 无效测量的前 9 维必须清零")
    hard = np.asarray(task["hard_rotation_state_window"], dtype=bool)
    if np.any(hard & ~(configured & measured_valid)):
        raise ValueError(f"{label}hard rotation 必须是 configured/measured_valid 的子集")
    if np.any(hard[~window_valid]):
        raise ValueError(f"{label}padding hard rotation 必须清零")


def _validate_array(payload: Mapping[str, Any], key: str, shape: tuple[int, ...], label: str) -> None:
    if key not in payload:
        raise KeyError(f"{label}缺少 `{key}`")
    value = np.asarray(payload[key])
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{label}{key} 必须为 {shape}，实际为 {value.shape}")
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise ValueError(f"{label}{key} 必须是数值或 bool 数组，实际为 {value.dtype}")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{label}{key} 包含 NaN 或 Inf")
