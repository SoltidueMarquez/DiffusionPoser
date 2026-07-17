from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from schemas.base import SchemaSpec
from schemas.registry import get_default_schema_name, get_schema_spec, list_schema_names


REALTIME_POSE_V2_MOTION_SCHEMA_NAME = "realtime_pose_v2_motion"
REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"
REALTIME_POSE_SCHEMA_NAME = get_default_schema_name()
DEFAULT_REALTIME_POSE_SCHEMA_NAME = REALTIME_POSE_SCHEMA_NAME
REALTIME_POSE_SCHEMA_NAMES = list_schema_names()
POSE_REPRESENTATION_KEY = "pose_representation"
POSE_REPRESENTATION_ROOT_YAW_GLOBAL_6D = "root_yaw_global_6d"
POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D = "body_fbx_local_delta_6d"
ROOT_Y_POLICY_ACTOR_ROOT_FROM_PELVIS = "actor_root_from_pelvis"
ROOT_Y_POLICY_FIXED_ZERO = "fixed_zero"
PELVIS_HEIGHT_MODE_ACTOR_ROOT_Y = "actor_root_y"
PELVIS_HEIGHT_MODE_PELVIS_LOCAL_OFFSET_Y = "pelvis_local_offset_y"
BODY_POSE_ROOT_GLOBAL_KEY = "body_pose_root_global_6d"
BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY = "body_pose_body_fbx_local_delta_6d"
LEGACY_BODY_POSE_PARENT_KEY = "body_pose_parent_6d"
TASK_FORMAT_REALTIME_POSE_V2_MOTION = "materialized_realtime_pose_v2_motion"
TASK_FORMAT_REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0 = "materialized_realtime_pose_body_fbx_local_root_y0_v1"
TASK_MODE_REALTIME_POSE = "realtime_pose_reconstruction"
TASK_MODES = (TASK_MODE_REALTIME_POSE,)

REALTIME_POSE_SEQ_LEN = 61
REALTIME_POSE_TARGET_START = 60
REALTIME_POSE_TARGET_LENGTH = 1

SMPL_JOINT_COUNT = 24
BODY_POSE_DIM = SMPL_JOINT_COUNT * 6
ROOT_YAW_DELTA_DIM = 2
ROOT_DELTA_XZ_DIM = 2
ROOT_HEIGHT_DIM = 1
STATIONARY_PROB_DIM = 5
STATIONARY_JOINT_INDICES = (0, 10, 11, 22, 23)
STATIONARY_JOINT_NAMES = ("pelvis", "left_foot", "right_foot", "left_hand", "right_hand")
TRACKER_COUNT = 6
TRACKER_POS_DIM = TRACKER_COUNT * 3
TRACKER_ROT_DIM = TRACKER_COUNT * 6
SENSOR_VALID_DIM = TRACKER_COUNT

BODY_POSE_START = 0
ROOT_YAW_DELTA_START = BODY_POSE_START + BODY_POSE_DIM

V2_MOTION_ROOT_DELTA_XZ_START = ROOT_YAW_DELTA_START + ROOT_YAW_DELTA_DIM
V2_MOTION_ROOT_HEIGHT_START = V2_MOTION_ROOT_DELTA_XZ_START + ROOT_DELTA_XZ_DIM
V2_MOTION_TRACKER_POS_REF_START = V2_MOTION_ROOT_HEIGHT_START + ROOT_HEIGHT_DIM
V2_MOTION_TRACKER_ROT_REF_START = V2_MOTION_TRACKER_POS_REF_START + TRACKER_POS_DIM
V2_MOTION_SENSOR_VALID_START = V2_MOTION_TRACKER_ROT_REF_START + TRACKER_ROT_DIM
REALTIME_POSE_V2_MOTION_INPUT_DIM = V2_MOTION_SENSOR_VALID_START + SENSOR_VALID_DIM
REALTIME_POSE_V2_MOTION_TARGET_DIM = BODY_POSE_DIM + ROOT_YAW_DELTA_DIM + ROOT_DELTA_XZ_DIM + ROOT_HEIGHT_DIM

BODY_FBX_LOCAL_ROOT_DELTA_XZ_START = ROOT_YAW_DELTA_START + ROOT_YAW_DELTA_DIM
BODY_FBX_LOCAL_PELVIS_HEIGHT_START = BODY_FBX_LOCAL_ROOT_DELTA_XZ_START + ROOT_DELTA_XZ_DIM
BODY_FBX_LOCAL_STATIONARY_PROB_START = BODY_FBX_LOCAL_PELVIS_HEIGHT_START + ROOT_HEIGHT_DIM
BODY_FBX_LOCAL_TRACKER_POS_REF_START = BODY_FBX_LOCAL_STATIONARY_PROB_START + STATIONARY_PROB_DIM
BODY_FBX_LOCAL_TRACKER_ROT_REF_START = BODY_FBX_LOCAL_TRACKER_POS_REF_START + TRACKER_POS_DIM
BODY_FBX_LOCAL_SENSOR_VALID_START = BODY_FBX_LOCAL_TRACKER_ROT_REF_START + TRACKER_ROT_DIM
REALTIME_POSE_BODY_FBX_LOCAL_INPUT_DIM = BODY_FBX_LOCAL_SENSOR_VALID_START + SENSOR_VALID_DIM
REALTIME_POSE_BODY_FBX_LOCAL_TARGET_DIM = (
    BODY_POSE_DIM + ROOT_YAW_DELTA_DIM + ROOT_DELTA_XZ_DIM + ROOT_HEIGHT_DIM + STATIONARY_PROB_DIM
)
TRACKER_POS_REF_START = BODY_FBX_LOCAL_TRACKER_POS_REF_START
TRACKER_ROT_REF_START = BODY_FBX_LOCAL_TRACKER_ROT_REF_START
SENSOR_VALID_START = BODY_FBX_LOCAL_SENSOR_VALID_START
REALTIME_POSE_INPUT_DIM = REALTIME_POSE_BODY_FBX_LOCAL_INPUT_DIM
REALTIME_POSE_TARGET_DIM = REALTIME_POSE_BODY_FBX_LOCAL_TARGET_DIM

TRACKER_NAMES = (
    "head",
    "left_wrist",
    "right_wrist",
    "waist",
    "left_foot",
    "right_foot",
)
HEAD_TRACKER_INDEX = 0
LEFT_HAND_TRACKER_INDEX = 1
RIGHT_HAND_TRACKER_INDEX = 2
HIP_TRACKER_INDEX = 3
LEFT_FOOT_TRACKER_INDEX = 4
RIGHT_FOOT_TRACKER_INDEX = 5
HAND_TRACKER_INDICES = (1, 2)
FOOT_TRACKER_INDICES = (4, 5)
STANDARD_THREE_TRACKER_INDICES = (
    HEAD_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    RIGHT_HAND_TRACKER_INDEX,
)

MIN_VALID_TRACKERS = 3
TRACKER_PATTERN_CATEGORIES = (
    "full_six",
    "standard_three",
    "static_sparse",
    "dynamic_dropout",
)
TRACKER_PATTERN_PROBABILITIES = {
    "full_six": 0.30,
    "standard_three": 0.30,
    "static_sparse": 0.20,
    "dynamic_dropout": 0.20,
}
TASK_MASK_POLICY_FULL = "full"
TASK_MASK_POLICY_FIXED_PATTERNS = "fixed_patterns"
TASK_MASK_POLICIES = (TASK_MASK_POLICY_FULL, TASK_MASK_POLICY_FIXED_PATTERNS)

TRACKER_MASK_POLICY_AUTO = "auto"
TRACKER_MASK_POLICY_TASK = "task"
TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES = "dynamic_categories"
TRACKER_MASK_POLICY_FIXED_CATEGORIES = "fixed_categories"
DATASET_TRACKER_MASK_POLICIES = (
    TRACKER_MASK_POLICY_AUTO,
    TRACKER_MASK_POLICY_TASK,
    TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
    TRACKER_MASK_POLICY_FIXED_CATEGORIES,
)
TRACKER_MASK_POLICIES = DATASET_TRACKER_MASK_POLICIES

TRACKER_MASK_FILL_ZERO = "zero"
TRACKER_MASK_FILL_MODES = (TRACKER_MASK_FILL_ZERO,)


@dataclass(frozen=True)
class TrackerPattern:
    category: str
    sensor_valid: tuple[bool, ...]

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "sensor_valid": [bool(value) for value in self.sensor_valid],
            "valid_tracker_count": int(sum(self.sensor_valid)),
            "valid_tracker_names": [
                TRACKER_NAMES[index] for index, value in enumerate(self.sensor_valid) if value
            ],
        }


def scalar_string(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(()).item())
    raise ValueError(f"{name} must be a scalar string, got shape={array.shape}")


def validate_pose_representation(
    value: object,
    schema_name: str | None = None,
    source: str = "payload",
) -> str:
    schema = get_schema_spec(schema_name)
    representation = scalar_string(value, POSE_REPRESENTATION_KEY)
    if representation != schema.pose_representation:
        raise ValueError(
            f"{source} pose_representation={representation!r}, "
            f"expected {schema.pose_representation!r} for {schema.name}."
        )
    return representation


def validate_realtime_seq_len(seq_len: int) -> None:
    if int(seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"realtime_pose 固定使用 {REALTIME_POSE_SEQ_LEN} 帧窗口，实际为 {seq_len}")


def validate_realtime_target(target_start: int, target_length: int) -> None:
    if int(target_start) != REALTIME_POSE_TARGET_START or int(target_length) != REALTIME_POSE_TARGET_LENGTH:
        raise ValueError(
            "realtime_pose 固定只补全第 61 帧："
            f"target_start={target_start}, target_length={target_length}"
        )


def validate_sensor_valid(sensor_valid: np.ndarray, min_valid_trackers: int = MIN_VALID_TRACKERS) -> np.ndarray:
    valid = np.asarray(sensor_valid, dtype=bool)
    if valid.ndim != 2 or valid.shape[1] != TRACKER_COUNT:
        raise ValueError(f"sensor_valid must be [T,{TRACKER_COUNT}], got {valid.shape}")
    if not valid[:, HEAD_TRACKER_INDEX].all():
        raise ValueError("realtime_pose requires the Head tracker to be valid in every frame")
    counts = valid.sum(axis=1)
    if np.any(counts < int(min_valid_trackers)):
        first_bad = int(np.where(counts < int(min_valid_trackers))[0][0])
        raise ValueError(
            f"every frame needs at least {min_valid_trackers} valid trackers; "
            f"frame {first_bad} has {int(counts[first_bad])}"
        )
    return valid


def create_realtime_inpaint_mask(
    seq_len: int = REALTIME_POSE_SEQ_LEN,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> np.ndarray:
    validate_realtime_seq_len(seq_len)
    schema = get_schema_spec(schema_name)
    mask = np.zeros((seq_len, schema.feature_dim), dtype=bool)
    mask[REALTIME_POSE_TARGET_START, schema.target_slice()] = True
    return mask


def _valid_tuple(indices: tuple[int, ...] | list[int]) -> tuple[bool, ...]:
    valid = [False] * TRACKER_COUNT
    valid[HEAD_TRACKER_INDEX] = True
    for index in indices:
        valid[int(index)] = True
    validate_sensor_valid(np.asarray(valid, dtype=bool)[None, :])
    return tuple(valid)


def make_tracker_pattern(category: str, rng: np.random.Generator) -> TrackerPattern:
    category = str(category)
    if category == "full_six":
        return TrackerPattern(category=category, sensor_valid=tuple([True] * TRACKER_COUNT))
    if category == "standard_three":
        return TrackerPattern(category=category, sensor_valid=_valid_tuple(list(STANDARD_THREE_TRACKER_INDICES)))
    if category == "static_sparse":
        standard_three = _valid_tuple(list(STANDARD_THREE_TRACKER_INDICES))
        while True:
            count = int(rng.integers(MIN_VALID_TRACKERS, TRACKER_COUNT))
            extras = rng.choice(
                [index for index in range(TRACKER_COUNT) if index != HEAD_TRACKER_INDEX],
                size=count - 1,
                replace=False,
            )
            valid = _valid_tuple([HEAD_TRACKER_INDEX, *[int(index) for index in extras]])
            if valid != standard_three:
                return TrackerPattern(category=category, sensor_valid=valid)
    if category == "dynamic_dropout":
        return TrackerPattern(category=category, sensor_valid=tuple([True] * TRACKER_COUNT))
    raise ValueError(f"未知 tracker pattern category: {category}")


def make_window_patterns(
    rng: np.random.Generator,
    patterns_per_window: int,
    ensure_pattern_categories: bool = True,
) -> list[TrackerPattern]:
    patterns: list[TrackerPattern] = []
    if ensure_pattern_categories and int(patterns_per_window) == 10:
        for category, count in (
            ("full_six", 3),
            ("standard_three", 3),
            ("static_sparse", 2),
            ("dynamic_dropout", 2),
        ):
            patterns.extend(make_tracker_pattern(category, rng) for _ in range(count))
    elif ensure_pattern_categories:
        patterns.extend(make_tracker_pattern(category, rng) for category in TRACKER_PATTERN_CATEGORIES)

    target_count = max(int(patterns_per_window), len(patterns))
    while len(patterns) < target_count:
        category = str(
            rng.choice(
                TRACKER_PATTERN_CATEGORIES,
                p=[TRACKER_PATTERN_PROBABILITIES[value] for value in TRACKER_PATTERN_CATEGORIES],
            )
        )
        patterns.append(make_tracker_pattern(category, rng))
    return patterns


def normalize_tracker_pattern_categories(categories: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if categories is None or len(categories) == 0:
        return TRACKER_PATTERN_CATEGORIES
    values = tuple(str(category) for category in categories)
    if "all" in values:
        return TRACKER_PATTERN_CATEGORIES
    unknown = [category for category in values if category not in TRACKER_PATTERN_CATEGORIES]
    if unknown:
        raise ValueError(f"未知 tracker pattern category: {unknown}")
    return values


def repeat_pattern_sensor_valid(pattern: TrackerPattern, seq_len: int = REALTIME_POSE_SEQ_LEN) -> np.ndarray:
    if int(seq_len) <= 0:
        raise ValueError("tracker pattern 的 seq_len 必须为正数。")
    sensor_valid = np.repeat(np.asarray(pattern.sensor_valid, dtype=bool)[None, :], seq_len, axis=0)
    validate_sensor_valid(sensor_valid)
    return sensor_valid


def make_dynamic_dropout_sensor_valid(
    rng: np.random.Generator,
    seq_len: int = REALTIME_POSE_SEQ_LEN,
    min_duration_frames: int = 2,
    max_duration_frames: int = 30,
    max_missing_trackers: int = 3,
) -> np.ndarray:
    """从完整六点开始生成可重叠的连续断线时间线。"""

    if int(seq_len) <= 0:
        raise ValueError("dynamic dropout 的 seq_len 必须为正数。")
    min_duration = max(1, int(min_duration_frames))
    first_dropout_frame = 1 if int(seq_len) > 1 else 0
    max_duration = min(int(seq_len) - first_dropout_frame, int(max_duration_frames))
    if min_duration > max_duration:
        raise ValueError("dynamic dropout 的最短持续帧不能大于最长持续帧。")
    max_missing = min(TRACKER_COUNT - MIN_VALID_TRACKERS, max(1, int(max_missing_trackers)))
    valid = np.ones((int(seq_len), TRACKER_COUNT), dtype=bool)
    droppable = np.asarray(
        [index for index in range(TRACKER_COUNT) if index != HEAD_TRACKER_INDEX],
        dtype=np.int64,
    )
    episode_count = max(1, int(rng.integers(1, max(2, seq_len // min_duration + 1))))
    inserted = 0
    for _ in range(episode_count * 12):
        if inserted >= episode_count:
            break
        tracker_index = int(rng.choice(droppable))
        duration = int(rng.integers(min_duration, max_duration + 1))
        start = int(rng.integers(first_dropout_frame, seq_len - duration + 1))
        end = start + duration
        # Do not merge two intervals for the same tracker; every resulting run
        # therefore remains independently bounded by [min_duration,max_duration].
        neighbor_start = max(first_dropout_frame, start - 1)
        neighbor_end = min(seq_len, end + 1)
        if not valid[neighbor_start:neighbor_end, tracker_index].all():
            continue
        if np.any((~valid[start:end]).sum(axis=1) >= max_missing):
            continue
        valid[start:end, tracker_index] = False
        inserted += 1
    if inserted == 0:
        duration = min_duration
        valid[first_dropout_frame : first_dropout_frame + duration, int(droppable[0])] = False
    valid[0] = True
    valid[:, HEAD_TRACKER_INDEX] = True
    validate_sensor_valid(valid)
    return valid
