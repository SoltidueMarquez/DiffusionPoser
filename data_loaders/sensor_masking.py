from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REALTIME_POSE_V2_MOTION_SCHEMA_NAME = "realtime_pose_v2_motion"
REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME = "realtime_pose_body_fbx_local_root_y0_v1"
REALTIME_POSE_SCHEMA_NAME = REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME
DEFAULT_REALTIME_POSE_SCHEMA_NAME = REALTIME_POSE_SCHEMA_NAME
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


@dataclass(frozen=True)
class SchemaSpec:
    """集中描述 realtime_pose 各版本的通道布局，避免调用侧硬编码切片。"""

    name: str
    task_format: str
    feature_dim: int
    target_dim: int
    body_pose_start: int
    root_yaw_delta_start: int
    tracker_pos_ref_start: int
    tracker_rot_ref_start: int
    sensor_valid_start: int
    pose_representation: str = POSE_REPRESENTATION_ROOT_YAW_GLOBAL_6D
    body_pose_key: str = BODY_POSE_ROOT_GLOBAL_KEY
    root_heading_delta_key: str = "root_yaw_delta_sincos"
    pelvis_height_key: str = "root_height"
    root_delta_xz_start: int | None = None
    root_height_start: int | None = None
    stationary_prob_start: int | None = None
    root_y_policy: str = ROOT_Y_POLICY_ACTOR_ROOT_FROM_PELVIS
    pelvis_height_mode: str = PELVIS_HEIGHT_MODE_ACTOR_ROOT_Y

    @property
    def supports_root_motion(self) -> bool:
        return self.root_delta_xz_start is not None and self.root_height_start is not None

    @property
    def supports_stationary_prob(self) -> bool:
        return self.stationary_prob_start is not None

    def target_slice(self) -> slice:
        return slice(BODY_POSE_START, self.target_dim)

    def body_pose_slice(self) -> slice:
        return slice(self.body_pose_start, self.body_pose_start + BODY_POSE_DIM)

    def root_yaw_delta_slice(self) -> slice:
        return slice(self.root_yaw_delta_start, self.root_yaw_delta_start + ROOT_YAW_DELTA_DIM)

    def root_heading_delta_slice(self) -> slice:
        return self.root_yaw_delta_slice()

    def root_delta_xz_slice(self) -> slice:
        if self.root_delta_xz_start is None:
            raise ValueError(f"{self.name} 不包含 root_delta_xz_ref。")
        return slice(self.root_delta_xz_start, self.root_delta_xz_start + ROOT_DELTA_XZ_DIM)

    def root_height_slice(self) -> slice:
        if self.root_height_start is None:
            raise ValueError(f"{self.name} 不包含 root_height。")
        return slice(self.root_height_start, self.root_height_start + ROOT_HEIGHT_DIM)

    def pelvis_height_slice(self) -> slice:
        return self.root_height_slice()

    def stationary_prob_slice(self) -> slice:
        if self.stationary_prob_start is None:
            raise ValueError(f"{self.name} 不包含 stationary_prob_5。")
        return slice(self.stationary_prob_start, self.stationary_prob_start + STATIONARY_PROB_DIM)

    def tracker_pos_slice(self, tracker_index: int | None = None) -> slice:
        if tracker_index is None:
            return slice(self.tracker_pos_ref_start, self.tracker_pos_ref_start + TRACKER_POS_DIM)
        start = self.tracker_pos_ref_start + int(tracker_index) * 3
        return slice(start, start + 3)

    def tracker_rot_slice(self, tracker_index: int | None = None) -> slice:
        if tracker_index is None:
            return slice(self.tracker_rot_ref_start, self.tracker_rot_ref_start + TRACKER_ROT_DIM)
        start = self.tracker_rot_ref_start + int(tracker_index) * 6
        return slice(start, start + 6)

    def sensor_valid_slice(self) -> slice:
        return slice(self.sensor_valid_start, self.sensor_valid_start + SENSOR_VALID_DIM)


SCHEMA_SPECS: dict[str, SchemaSpec] = {
    REALTIME_POSE_V2_MOTION_SCHEMA_NAME: SchemaSpec(
        name=REALTIME_POSE_V2_MOTION_SCHEMA_NAME,
        task_format=TASK_FORMAT_REALTIME_POSE_V2_MOTION,
        feature_dim=REALTIME_POSE_V2_MOTION_INPUT_DIM,
        target_dim=REALTIME_POSE_V2_MOTION_TARGET_DIM,
        body_pose_start=BODY_POSE_START,
        root_yaw_delta_start=ROOT_YAW_DELTA_START,
        root_delta_xz_start=V2_MOTION_ROOT_DELTA_XZ_START,
        root_height_start=V2_MOTION_ROOT_HEIGHT_START,
        tracker_pos_ref_start=V2_MOTION_TRACKER_POS_REF_START,
        tracker_rot_ref_start=V2_MOTION_TRACKER_ROT_REF_START,
        sensor_valid_start=V2_MOTION_SENSOR_VALID_START,
    ),
    REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME: SchemaSpec(
        name=REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME,
        task_format=TASK_FORMAT_REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0,
        feature_dim=REALTIME_POSE_BODY_FBX_LOCAL_INPUT_DIM,
        target_dim=REALTIME_POSE_BODY_FBX_LOCAL_TARGET_DIM,
        body_pose_start=BODY_POSE_START,
        root_yaw_delta_start=ROOT_YAW_DELTA_START,
        root_delta_xz_start=BODY_FBX_LOCAL_ROOT_DELTA_XZ_START,
        root_height_start=BODY_FBX_LOCAL_PELVIS_HEIGHT_START,
        stationary_prob_start=BODY_FBX_LOCAL_STATIONARY_PROB_START,
        tracker_pos_ref_start=BODY_FBX_LOCAL_TRACKER_POS_REF_START,
        tracker_rot_ref_start=BODY_FBX_LOCAL_TRACKER_ROT_REF_START,
        sensor_valid_start=BODY_FBX_LOCAL_SENSOR_VALID_START,
        pose_representation=POSE_REPRESENTATION_BODY_FBX_LOCAL_DELTA_6D,
        body_pose_key=BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
        root_heading_delta_key="root_heading_delta_sincos",
        pelvis_height_key="pelvis_height",
        root_y_policy=ROOT_Y_POLICY_FIXED_ZERO,
        pelvis_height_mode=PELVIS_HEIGHT_MODE_PELVIS_LOCAL_OFFSET_Y,
    ),
}
REALTIME_POSE_SCHEMA_NAMES = tuple(SCHEMA_SPECS.keys())


def get_schema_spec(schema_name: str | None) -> SchemaSpec:
    name = str(schema_name or DEFAULT_REALTIME_POSE_SCHEMA_NAME)
    try:
        return SCHEMA_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"未知 realtime pose schema: {name}，可选值为 {REALTIME_POSE_SCHEMA_NAMES}") from exc


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
        raise ValueError(f"sensor_valid 应为 [T, {TRACKER_COUNT}]，实际为 {valid.shape}")
    if not valid[:, HIP_TRACKER_INDEX].all():
        raise ValueError("realtime_pose 要求 waist/hip tracker 在所有帧都有效。")
    counts = valid.sum(axis=1)
    if np.any(counts < int(min_valid_trackers)):
        first_bad = int(np.where(counts < int(min_valid_trackers))[0][0])
        raise ValueError(
            f"每帧至少需要 {min_valid_trackers} 个有效 tracker；"
            f"第 {first_bad} 帧只有 {int(counts[first_bad])} 个。"
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
