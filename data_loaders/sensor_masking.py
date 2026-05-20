from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REALTIME_POSE_SCHEMA_NAME = "realtime_pose_v1"
TASK_FORMAT_REALTIME_POSE_V1 = "materialized_realtime_pose_v1"
TASK_MODE_REALTIME_POSE = "realtime_pose_reconstruction"
TASK_MODES = (TASK_MODE_REALTIME_POSE,)

REALTIME_POSE_SEQ_LEN = 61
REALTIME_POSE_TARGET_START = 60
REALTIME_POSE_TARGET_LENGTH = 1

SMPL_JOINT_COUNT = 24
BODY_POSE_DIM = SMPL_JOINT_COUNT * 6
ROOT_YAW_DELTA_DIM = 2
TRACKER_COUNT = 6
TRACKER_POS_DIM = TRACKER_COUNT * 3
TRACKER_ROT_DIM = TRACKER_COUNT * 6
SENSOR_VALID_DIM = TRACKER_COUNT

BODY_POSE_START = 0
ROOT_YAW_DELTA_START = BODY_POSE_START + BODY_POSE_DIM
TRACKER_POS_REF_START = ROOT_YAW_DELTA_START + ROOT_YAW_DELTA_DIM
TRACKER_ROT_REF_START = TRACKER_POS_REF_START + TRACKER_POS_DIM
SENSOR_VALID_START = TRACKER_ROT_REF_START + TRACKER_ROT_DIM
REALTIME_POSE_INPUT_DIM = SENSOR_VALID_START + SENSOR_VALID_DIM
REALTIME_POSE_TARGET_DIM = BODY_POSE_DIM + ROOT_YAW_DELTA_DIM

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
NON_HIP_TRACKER_INDICES = (0, 1, 2, 4, 5)
HAND_TRACKER_INDICES = (1, 2)
FOOT_TRACKER_INDICES = (4, 5)

MIN_VALID_TRACKERS = 3
TRACKER_PATTERN_CATEGORIES = (
    "head-present",
    "hand-present",
    "foot-present",
    "upper-body",
    "lower-body",
    "mixed-sparse",
    "full-trackers",
)
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


def validate_realtime_seq_len(seq_len: int) -> None:
    if int(seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"realtime_pose_v1 固定使用 {REALTIME_POSE_SEQ_LEN} 帧窗口，实际为 {seq_len}")


def validate_realtime_target(target_start: int, target_length: int) -> None:
    if int(target_start) != REALTIME_POSE_TARGET_START or int(target_length) != REALTIME_POSE_TARGET_LENGTH:
        raise ValueError(
            "realtime_pose_v1 固定只补全第 61 帧："
            f"target_start={target_start}, target_length={target_length}"
        )


def validate_sensor_valid(sensor_valid: np.ndarray, min_valid_trackers: int = MIN_VALID_TRACKERS) -> np.ndarray:
    valid = np.asarray(sensor_valid, dtype=bool)
    if valid.ndim != 2 or valid.shape[1] != TRACKER_COUNT:
        raise ValueError(f"sensor_valid 应为 [T, {TRACKER_COUNT}]，实际为 {valid.shape}")
    if not valid[:, HIP_TRACKER_INDEX].all():
        raise ValueError("realtime_pose_v1 要求 waist/hip tracker 在所有帧都有效。")
    counts = valid.sum(axis=1)
    if np.any(counts < int(min_valid_trackers)):
        first_bad = int(np.where(counts < int(min_valid_trackers))[0][0])
        raise ValueError(
            f"每帧至少需要 {min_valid_trackers} 个有效 tracker；"
            f"第 {first_bad} 帧只有 {int(counts[first_bad])} 个。"
        )
    return valid


def create_realtime_inpaint_mask(seq_len: int = REALTIME_POSE_SEQ_LEN) -> np.ndarray:
    validate_realtime_seq_len(seq_len)
    mask = np.zeros((seq_len, REALTIME_POSE_INPUT_DIM), dtype=bool)
    mask[REALTIME_POSE_TARGET_START, BODY_POSE_START:REALTIME_POSE_TARGET_DIM] = True
    return mask


def _valid_tuple(indices: tuple[int, ...] | list[int]) -> tuple[bool, ...]:
    valid = [False] * TRACKER_COUNT
    valid[HIP_TRACKER_INDEX] = True
    for index in indices:
        valid[int(index)] = True
    validate_sensor_valid(np.asarray(valid, dtype=bool)[None, :])
    return tuple(valid)


def make_tracker_pattern(category: str, rng: np.random.Generator) -> TrackerPattern:
    category = str(category)
    if category == "full-trackers":
        return TrackerPattern(category=category, sensor_valid=tuple([True] * TRACKER_COUNT))
    if category == "head-present":
        extra = int(rng.choice([*HAND_TRACKER_INDICES, *FOOT_TRACKER_INDICES]))
        return TrackerPattern(category=category, sensor_valid=_valid_tuple([HEAD_TRACKER_INDEX, extra]))
    if category == "hand-present":
        hand = int(rng.choice(HAND_TRACKER_INDICES))
        extra = int(rng.choice([HEAD_TRACKER_INDEX, *FOOT_TRACKER_INDICES]))
        return TrackerPattern(category=category, sensor_valid=_valid_tuple([hand, extra]))
    if category == "foot-present":
        foot = int(rng.choice(FOOT_TRACKER_INDICES))
        extra = int(rng.choice([HEAD_TRACKER_INDEX, *HAND_TRACKER_INDICES]))
        return TrackerPattern(category=category, sensor_valid=_valid_tuple([foot, extra]))
    if category == "upper-body":
        hand = int(rng.choice(HAND_TRACKER_INDICES))
        return TrackerPattern(category=category, sensor_valid=_valid_tuple([HEAD_TRACKER_INDEX, hand]))
    if category == "lower-body":
        if rng.random() < 0.5:
            return TrackerPattern(category=category, sensor_valid=_valid_tuple(list(FOOT_TRACKER_INDICES)))
        foot = int(rng.choice(FOOT_TRACKER_INDICES))
        return TrackerPattern(category=category, sensor_valid=_valid_tuple([HEAD_TRACKER_INDEX, foot]))
    if category == "mixed-sparse":
        count = int(rng.integers(2, len(NON_HIP_TRACKER_INDICES) + 1))
        indices = rng.choice(NON_HIP_TRACKER_INDICES, size=count, replace=False)
        return TrackerPattern(category=category, sensor_valid=_valid_tuple([int(index) for index in indices]))
    raise ValueError(f"未知 tracker pattern category: {category}")


def make_window_patterns(
    rng: np.random.Generator,
    patterns_per_window: int,
    ensure_pattern_categories: bool = True,
) -> list[TrackerPattern]:
    patterns: list[TrackerPattern] = []
    if ensure_pattern_categories:
        patterns.extend(make_tracker_pattern(category, rng) for category in TRACKER_PATTERN_CATEGORIES)

    target_count = max(int(patterns_per_window), len(patterns))
    while len(patterns) < target_count:
        patterns.append(make_tracker_pattern("mixed-sparse", rng))
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
    validate_realtime_seq_len(seq_len)
    sensor_valid = np.repeat(np.asarray(pattern.sensor_valid, dtype=bool)[None, :], seq_len, axis=0)
    validate_sensor_valid(sensor_valid)
    return sensor_valid
