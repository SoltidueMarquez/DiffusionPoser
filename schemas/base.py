from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


REALTIME_POSE_SEQ_LEN = 61
REALTIME_POSE_TARGET_START = 60
REALTIME_POSE_TARGET_LENGTH = 1

BODY_POSE_DIM = 24 * 6
ROOT_YAW_DELTA_DIM = 2
ROOT_DELTA_XZ_DIM = 2
ROOT_HEIGHT_DIM = 1
STATIONARY_PROB_DIM = 5
TRACKER_COUNT = 6
TRACKER_POS_DIM = TRACKER_COUNT * 3
TRACKER_ROT_DIM = TRACKER_COUNT * 6
SENSOR_VALID_DIM = TRACKER_COUNT

BODY_POSE_START = 0

POSE_REPRESENTATION_ROOT_YAW_GLOBAL_6D = "root_yaw_global_6d"
BODY_POSE_ROOT_GLOBAL_KEY = "body_pose_root_global_6d"
ROOT_Y_POLICY_ACTOR_ROOT_FROM_PELVIS = "actor_root_from_pelvis"
PELVIS_HEIGHT_MODE_ACTOR_ROOT_Y = "actor_root_y"


@dataclass(frozen=True)
class SchemaSpec:
    """描述一个 realtime pose schema 的稳定通道布局和运行时能力。"""

    name: str
    task_format: str
    feature_dim: int
    target_dim: int
    body_pose_start: int
    root_yaw_delta_start: int
    tracker_pos_ref_start: int
    tracker_rot_ref_start: int
    sensor_valid_start: int
    canonical_name: str | None = None
    one_line: str = ""
    seq_len: int = REALTIME_POSE_SEQ_LEN
    target_start: int = REALTIME_POSE_TARGET_START
    target_length: int = REALTIME_POSE_TARGET_LENGTH
    trainable: bool = True
    exportable: bool = True
    pose_representation: str = POSE_REPRESENTATION_ROOT_YAW_GLOBAL_6D
    body_pose_key: str = BODY_POSE_ROOT_GLOBAL_KEY
    root_heading_delta_key: str = "root_yaw_delta_sincos"
    pelvis_height_key: str = "root_height"
    root_delta_xz_start: int | None = None
    root_height_start: int | None = None
    stationary_prob_start: int | None = None
    root_y_policy: str = ROOT_Y_POLICY_ACTOR_ROOT_FROM_PELVIS
    pelvis_height_mode: str = PELVIS_HEIGHT_MODE_ACTOR_ROOT_Y

    def __post_init__(self) -> None:
        if self.canonical_name is None:
            object.__setattr__(self, "canonical_name", self.name)

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


class SchemaAdapter(Protocol):
    spec: SchemaSpec

    def validate_source(self, payload: Mapping[str, Any]) -> None:
        ...

    def validate_task(self, payload: Mapping[str, Any]) -> None:
        ...

    def build_inpaint_mask(self, seq_len: int | None = None) -> Any:
        ...

    def build_unity_feature_schema(self) -> Mapping[str, Any]:
        ...
