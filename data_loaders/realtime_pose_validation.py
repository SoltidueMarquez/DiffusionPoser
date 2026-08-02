from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_FPS,
    REALTIME_POSE_HISTORY_LENGTH,
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
    "root_heading_delta_sincos": (2,),
    "root_delta_xz_ref": (2,),
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
    "pose_history": (REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM),
    "tracker_history": (REALTIME_POSE_HISTORY_LENGTH, TRACKER_COUNT, TRACKER_FEATURE_DIM),
    "current_tracker": (TRACKER_COUNT, TRACKER_FEATURE_DIM),
    "current_tracker_raw": (TRACKER_COUNT, TRACKER_FEATURE_DIM),
    "trajectory_history": (REALTIME_POSE_HISTORY_LENGTH, 5),
    "current_trajectory": (1, 5),
    "current_target": (REALTIME_POSE_TARGET_DIM,),
    "valid_frame_mask": (REALTIME_POSE_HISTORY_LENGTH,),
    "hard_rotation_state": (TRACKER_COUNT,),
    "target_joints_head_ref": (24, 3),
    "prev_joints_head_ref": (24, 3),
    "target_root_position_head_ref": (3,),
    "current_head_position_world": (3,),
    "joint_offsets_parent": (24, 3),
    "joint_rest_local_rotations_6d": (24, 6),
    "configured": (REALTIME_POSE_SEQ_LEN, TRACKER_COUNT),
    "measured_valid": (REALTIME_POSE_SEQ_LEN, TRACKER_COUNT),
    "d_off": (REALTIME_POSE_SEQ_LEN, TRACKER_COUNT),
    "d_on": (REALTIME_POSE_SEQ_LEN, TRACKER_COUNT),
    "future_leg_target": (3, 8, 6),
    "contact_target": (2,),
}

TASK_SCALAR_FIELDS = (
    "target_root_yaw_world",
    "target_hip_height",
    "history_head_yaw_world",
    "current_head_yaw_world",
    "floor_y",
    "scenario",
    "source_path",
    "start_frame",
    "scenario_id",
    "task_id",
)


def load_realtime_metadata(payload: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    label = f"{path} " if path else ""
    if "metadata" not in payload:
        raise KeyError(f"{label}缺少 `metadata`")
    value = np.asarray(payload["metadata"])
    if value.shape != ():
        raise ValueError(f"{label}metadata 必须是 JSON 标量，实际为 {value.shape}")
    try:
        metadata = json.loads(str(value.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}metadata 不是合法 JSON object") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{label}metadata 必须解析为 JSON object")
    return metadata


def validate_realtime_source_arrays(
    payload: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    expected_fps: float | None = REALTIME_POSE_FPS,
    path: Path | None = None,
) -> int:
    """校验完整 source 契约并返回帧数，防止错误缓存进入 task 生成。"""

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

    source_metadata = dict(metadata) if metadata is not None else load_realtime_metadata(payload, path)
    declared_frames = int(source_metadata.get("frames", -1))
    if declared_frames != frame_count:
        raise ValueError(f"{label}metadata.frames={declared_frames}，实际帧数为 {frame_count}")
    if "target_fps" not in source_metadata:
        raise KeyError(f"{label}metadata 缺少 `target_fps`")
    target_fps = float(source_metadata["target_fps"])
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"{label}metadata.target_fps 必须为正数，实际为 {target_fps}")
    if expected_fps is not None and not np.isclose(target_fps, float(expected_fps), rtol=0.0, atol=1e-6):
        raise ValueError(f"{label}target_fps={target_fps:g}，当前链路要求 {float(expected_fps):g}Hz")
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
    validate_tracker_states(configured, measured_valid)
    tracker = np.concatenate(
        [np.asarray(task["tracker_history"]), np.asarray(task["current_tracker"])[None]],
        axis=0,
    )
    if not np.array_equal(tracker[..., TRACKER_CONFIGURED_OFFSET] > 0.5, configured):
        raise ValueError(f"{label}Tracker configured 与 task.configured 不一致")
    if not np.array_equal(tracker[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5, measured_valid):
        raise ValueError(f"{label}Tracker measured_valid 与 task.measured_valid 不一致")
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
