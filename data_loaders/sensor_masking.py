from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY = "body_pose_body_fbx_local_delta_6d"

TASK_MODE_REALTIME_POSE = "realtime_pose_reconstruction"
TASK_MODES = (TASK_MODE_REALTIME_POSE,)

REALTIME_POSE_SEQ_LEN = 61
REALTIME_POSE_HISTORY_LENGTH = 60
REALTIME_POSE_TARGET_START = 60
REALTIME_POSE_FUTURE_FRAME_COUNT = 10
REALTIME_POSE_TARGET_LENGTH = REALTIME_POSE_FUTURE_FRAME_COUNT + 1
REALTIME_POSE_FPS = 60.0

# 模型仍从 60 帧密集运行时历史中取样，但真正进入时空 DiT 的只有 10 个
# 同步锚点和 1 个当前帧。固定索引能保证训练、离线采样和在线运行时使用
# 完全相同的时间语义，避免动态重采样造成不可复现的条件分布。
REALTIME_POSE_HISTORY_ANCHOR_INDICES = (0, 7, 13, 20, 26, 33, 39, 46, 52, 59)
REALTIME_POSE_HISTORY_FRAME_OFFSETS = (-60, -53, -47, -40, -34, -27, -21, -14, -8, -1)
REALTIME_POSE_HISTORY_ANCHOR_COUNT = len(REALTIME_POSE_HISTORY_ANCHOR_INDICES)
REALTIME_POSE_CONDITION_WINDOW_LENGTH = REALTIME_POSE_HISTORY_ANCHOR_COUNT + 1
REALTIME_POSE_TARGET_FRAME_OFFSETS = tuple(range(REALTIME_POSE_TARGET_LENGTH))
REALTIME_POSE_MODEL_TOKEN_LENGTH = (
    REALTIME_POSE_HISTORY_ANCHOR_COUNT + REALTIME_POSE_TARGET_LENGTH
)
REALTIME_POSE_FRAME_OFFSETS = (
    *REALTIME_POSE_HISTORY_FRAME_OFFSETS,
    *REALTIME_POSE_TARGET_FRAME_OFFSETS,
)

SMPL_JOINT_COUNT = 24
ROTATION_6D_DIM = 6
BODY_POSE_DIM = SMPL_JOINT_COUNT * ROTATION_6D_DIM  # source local pose: 144
JOINT_GLOBAL_ROTATION_DIM = BODY_POSE_DIM  # target: 24 个 Head-yaw 参考系全局旋转
ROOT_DELTA_XZ_DIM = 2  # 仅供读取现有 source；不进入新 task。
ROOT_HEIGHT_DIM = 1  # 仅供读取现有 source；不进入新 target。
STATIONARY_PROB_DIM = 5  # 保存在 Source；task 生成时取左右脚通道作为 contact supervision。
STATIONARY_JOINT_INDICES = (0, 10, 11, 22, 23)
STATIONARY_JOINT_NAMES = ("pelvis", "left_foot", "right_foot", "left_hand", "right_hand")

JOINT_GLOBAL_ROTATION_START = 0
REALTIME_POSE_TARGET_DIM = JOINT_GLOBAL_ROTATION_DIM
REALTIME_POSE_INPUT_DIM = REALTIME_POSE_TARGET_DIM

TRACKER_COUNT = 6
TRACKER_CONTINUOUS_DIM = 9
TRACKER_FEATURE_DIM = 13
TRACKER_CONFIGURED_OFFSET = 9
TRACKER_MEASURED_VALID_OFFSET = 10
TRACKER_D_OFF_OFFSET = 11
TRACKER_D_ON_OFFSET = 12
TRACKER_DURATION_CAP = 60

TRACKER_NAMES = (
    "head",
    "left_wrist",
    "right_wrist",
    "hip",
    "left_foot",
    "right_foot",
)
HEAD_TRACKER_INDEX = 0
LEFT_HAND_TRACKER_INDEX = 1
RIGHT_HAND_TRACKER_INDEX = 2
HIP_TRACKER_INDEX = 3
LEFT_FOOT_TRACKER_INDEX = 4
RIGHT_FOOT_TRACKER_INDEX = 5
HAND_TRACKER_INDICES = (LEFT_HAND_TRACKER_INDEX, RIGHT_HAND_TRACKER_INDEX)
FOOT_TRACKER_INDICES = (LEFT_FOOT_TRACKER_INDEX, RIGHT_FOOT_TRACKER_INDEX)
NON_HEAD_TRACKER_INDICES = (1, 2, 3, 4, 5)
NON_HIP_TRACKER_INDICES = (0, 1, 2, 4, 5)
TRACKER_TO_JOINT = (15, 20, 21, 0, 10, 11)

MIN_VALID_TRACKERS = 1
SENSOR_VALID_DIM = TRACKER_COUNT
TRACKER_POS_DIM = TRACKER_COUNT * 3
TRACKER_ROT_DIM = TRACKER_COUNT * 6

SCENARIO_FIXED_SIX = "fixed_six"
SCENARIO_FIXED_THREE = "fixed_three"
SCENARIO_THREE_TO_SIX = "three_to_six"
SCENARIO_SIX_TO_THREE = "six_to_three"
SCENARIO_TWO_POINT_DROPOUT_RECONNECT = "two_point_dropout_reconnect"
TRACKER_PATTERN_CATEGORIES = (
    SCENARIO_FIXED_SIX,
    SCENARIO_FIXED_THREE,
    SCENARIO_THREE_TO_SIX,
    SCENARIO_SIX_TO_THREE,
    SCENARIO_TWO_POINT_DROPOUT_RECONNECT,
)
DEFAULT_SCENARIO_WEIGHTS = (0.2, 0.2, 0.2, 0.2, 0.2)

TASK_MASK_POLICY_FULL = "full"
TASK_MASK_POLICY_FIXED_PATTERNS = "fixed_patterns"
TASK_MASK_POLICIES = (TASK_MASK_POLICY_FULL, TASK_MASK_POLICY_FIXED_PATTERNS)
TRACKER_MASK_POLICY_AUTO = "auto"
TRACKER_MASK_POLICY_TASK = "task"
TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES = "dynamic_categories"
TRACKER_MASK_POLICY_FIXED_CATEGORIES = "fixed_categories"
TRACKER_MASK_POLICIES = (
    TRACKER_MASK_POLICY_AUTO,
    TRACKER_MASK_POLICY_TASK,
    TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
    TRACKER_MASK_POLICY_FIXED_CATEGORIES,
)
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
        raise ValueError(f"realtime_pose 固定使用 {REALTIME_POSE_SEQ_LEN} 帧窗口，实际为 {seq_len}")


def validate_realtime_target(target_start: int, target_length: int) -> None:
    if int(target_start) != REALTIME_POSE_TARGET_START or int(target_length) != REALTIME_POSE_TARGET_LENGTH:
        raise ValueError(
            "realtime_pose 固定使用前 60 帧历史，并联合预测当前帧和未来 10 帧。"
        )


def validate_tracker_states(configured: np.ndarray, measured_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    configured = np.asarray(configured, dtype=bool)
    measured_valid = np.asarray(measured_valid, dtype=bool)
    if configured.shape != measured_valid.shape or configured.ndim != 2 or configured.shape[1] != TRACKER_COUNT:
        raise ValueError(
            f"configured/measured_valid 必须同为 [T,{TRACKER_COUNT}]，实际为 "
            f"{configured.shape}/{measured_valid.shape}"
        )
    if np.any(measured_valid & ~configured):
        raise ValueError("measured_valid 必须是 configured 的子集。")
    if not configured[:, HEAD_TRACKER_INDEX].all() or not measured_valid[:, HEAD_TRACKER_INDEX].all():
        raise ValueError("Head 必须在所有帧 configured 且 measured_valid。")
    return configured, measured_valid


def validate_sensor_valid(sensor_valid: np.ndarray, min_valid_trackers: int = MIN_VALID_TRACKERS) -> np.ndarray:
    """旧函数名保留为调用入口；新语义只要求 Head 有效，允许仅剩一个测量。"""

    valid = np.asarray(sensor_valid, dtype=bool)
    if valid.ndim != 2 or valid.shape[1] != TRACKER_COUNT:
        raise ValueError(f"sensor_valid 必须为 [T,{TRACKER_COUNT}]，实际为 {valid.shape}")
    if not valid[:, HEAD_TRACKER_INDEX].all():
        raise ValueError("Head 必须在所有帧有效。")
    return valid


def normalize_tracker_pattern_categories(categories: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not categories or "all" in categories:
        return TRACKER_PATTERN_CATEGORIES
    values = tuple(str(category) for category in categories)
    unknown = [value for value in values if value not in TRACKER_PATTERN_CATEGORIES]
    if unknown:
        raise ValueError(f"未知 Tracker 场景: {unknown}")
    return values


def make_tracker_pattern(category: str, rng: np.random.Generator) -> TrackerPattern:
    category = str(category)
    if category == SCENARIO_FIXED_SIX:
        valid = (True,) * TRACKER_COUNT
    elif category == SCENARIO_FIXED_THREE:
        valid = (True, True, True, False, False, False)
    else:
        # 切换与掉线必须由绝对帧时间线生成，不能退化成窗口级随机 Pattern。
        raise ValueError(f"场景 {category} 必须使用 deterministic tracker timeline。")
    return TrackerPattern(category=category, sensor_valid=valid)


def make_window_patterns(
    rng: np.random.Generator,
    patterns_per_window: int,
    ensure_pattern_categories: bool = True,
) -> list[TrackerPattern]:
    del rng, patterns_per_window, ensure_pattern_categories
    return [
        TrackerPattern(SCENARIO_FIXED_SIX, (True,) * TRACKER_COUNT),
        TrackerPattern(SCENARIO_FIXED_THREE, (True, True, True, False, False, False)),
    ]


def repeat_pattern_sensor_valid(pattern: TrackerPattern, seq_len: int = REALTIME_POSE_SEQ_LEN) -> np.ndarray:
    validate_realtime_seq_len(seq_len)
    valid = np.repeat(np.asarray(pattern.sensor_valid, dtype=bool)[None], seq_len, axis=0)
    return validate_sensor_valid(valid)
