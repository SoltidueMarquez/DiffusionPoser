from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    HIP_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    RIGHT_FOOT_TRACKER_INDEX,
    RIGHT_HAND_TRACKER_INDEX,
    TRACKER_COUNT,
)


TARGET_REGION_NAMES = ("torso", "left_arm", "right_arm", "left_leg", "right_leg")
MOTION_REGION_NAMES = ("global", "pelvis", "left_leg", "right_leg")
PROJECTED_DDIM_MODES = ("all_steps", "late_steps", "final_step")


@dataclass(frozen=True)
class TrackerReliabilityConfig:
    """Tracker 可靠性恢复窗口和 hard 门槛的唯一配置入口。"""

    d_warm_pos: int = 15
    d_warm_rot: int = 15
    d_hard: int = 15
    duration_cap: int = 60

    def validate(self) -> "TrackerReliabilityConfig":
        if min(self.d_warm_pos, self.d_warm_rot, self.d_hard, self.duration_cap) <= 0:
            raise ValueError("可靠性时长参数必须全部大于 0。")
        return self

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class RealtimePoseLossWeights:
    global_rotation: float = 1.0
    local_rotation: float = 1.0
    fk: float = 2.0
    tracker_position: float = 10.0
    tracker_rotation: float = 1.0
    root: float = 1.0
    head_ref_joint_distance: float = 1.0
    head_to_root_xz: float = 1.0
    future_leg: float = 0.5
    contact: float = 0.1

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def build_target_joint_regions() -> np.ndarray:
    """返回 `[24]` 唯一区域编号，避免 collar/pelvis 被多区域重复消费。"""

    groups = (
        ("pelvis", "spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar"),
        ("left_shoulder", "left_elbow", "left_wrist", "left_hand"),
        ("right_shoulder", "right_elbow", "right_wrist", "right_hand"),
        ("left_hip", "left_knee", "left_ankle", "left_foot"),
        ("right_hip", "right_knee", "right_ankle", "right_foot"),
    )
    result = np.full(24, -1, dtype=np.int64)
    for region_index, names in enumerate(groups):
        result[[JOINT_INDEX[name] for name in names]] = region_index
    if np.any(result < 0):
        raise RuntimeError("24 个关节必须恰好映射到一个时空 DiT 目标区域。")
    return result


def build_tracker_coverage() -> tuple[np.ndarray, np.ndarray]:
    """构造 `[5,6]` position/rotation 固定二值覆盖矩阵。"""

    position = np.zeros((len(TARGET_REGION_NAMES), TRACKER_COUNT), dtype=np.float32)
    rotation = np.zeros_like(position)
    position[0, HIP_TRACKER_INDEX] = 1.0
    position[1, LEFT_HAND_TRACKER_INDEX] = 1.0
    position[2, RIGHT_HAND_TRACKER_INDEX] = 1.0
    position[3, LEFT_FOOT_TRACKER_INDEX] = 1.0
    position[4, RIGHT_FOOT_TRACKER_INDEX] = 1.0
    rotation[0, [HEAD_TRACKER_INDEX, HIP_TRACKER_INDEX]] = 1.0
    rotation[1, LEFT_HAND_TRACKER_INDEX] = 1.0
    rotation[2, RIGHT_HAND_TRACKER_INDEX] = 1.0
    rotation[3, LEFT_FOOT_TRACKER_INDEX] = 1.0
    rotation[4, RIGHT_FOOT_TRACKER_INDEX] = 1.0
    return position, rotation


TARGET_JOINT_REGIONS = build_target_joint_regions()
POSITION_COVERAGE, ROTATION_COVERAGE = build_tracker_coverage()
TRAJECTORY_REGION_MULTIPLIERS = np.asarray([1.0, 0.25, 0.25, 1.0, 1.0], dtype=np.float32)

for _constant in (TARGET_JOINT_REGIONS, POSITION_COVERAGE, ROTATION_COVERAGE, TRAJECTORY_REGION_MULTIPLIERS):
    _constant.setflags(write=False)
