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
TAID_ABLATIONS = ("B0", "B1", "B2", "B3", "B4", "B5", "B6")
TAID_PRIOR_TRACKER_AGGREGATIONS = ("fixed_slots",)


@dataclass(frozen=True)
class TrackerRoleConfig:
    """TAID 第一版确定性角色与重连权重配置。"""

    anchor_ramp_start: int = 5
    anchor_ramp_end: int = 15
    innovation_ramp_frames: int = 15

    def validate(self) -> "TrackerRoleConfig":
        if self.anchor_ramp_start < 0:
            raise ValueError("anchor_ramp_start 必须大于等于 0。")
        if self.anchor_ramp_end <= self.anchor_ramp_start:
            raise ValueError("anchor_ramp_end 必须严格大于 anchor_ramp_start。")
        if self.innovation_ramp_frames <= 0:
            raise ValueError("innovation_ramp_frames 必须大于 0。")
        return self

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TaIDConfig:
    """TAID B0～B6 的结构、角色和确定性几何默认值。"""

    ablation: str = "B0"
    prior_tracker_aggregation: str = "fixed_slots"
    role: TrackerRoleConfig = TrackerRoleConfig()
    innovation_dim: int = 64
    innovation_clip: float = 3.0
    # 顺序固定为 Head/LHand/RHand/Hip/LFoot/RFoot；Head 不进入 innovation。
    position_scales: tuple[float, ...] = (1.0, 0.25, 0.25, 0.20, 0.20, 0.20)
    rotation_scales: tuple[float, ...] = (1.0, 0.50, 0.50, 0.35, 0.35, 0.35)
    hip_leg_secondary_weight: float = 0.25
    hand_torso_weight: float = 0.10
    foot_root_contact_weight: float = 0.25

    def validate(self) -> "TaIDConfig":
        if self.ablation not in TAID_ABLATIONS:
            raise ValueError(f"TAID ablation 必须属于 {TAID_ABLATIONS}，实际为 {self.ablation}")
        if self.prior_tracker_aggregation not in TAID_PRIOR_TRACKER_AGGREGATIONS:
            raise ValueError(
                "TAID Prior Tracker 聚合必须属于 "
                f"{TAID_PRIOR_TRACKER_AGGREGATIONS}，实际为 {self.prior_tracker_aggregation}"
            )
        self.role.validate()
        if self.innovation_dim <= 0 or self.innovation_clip <= 0.0:
            raise ValueError("innovation_dim/innovation_clip 必须大于 0。")
        if len(self.position_scales) != TRACKER_COUNT or len(self.rotation_scales) != TRACKER_COUNT:
            raise ValueError("TAID residual scale 必须各提供 6 个 Tracker 值。")
        if min(self.position_scales) <= 0.0 or min(self.rotation_scales) <= 0.0:
            raise ValueError("TAID residual scale 必须全部大于 0。")
        route_weights = (
            self.hip_leg_secondary_weight,
            self.hand_torso_weight,
            self.foot_root_contact_weight,
        )
        if any(value < 0.0 or value > 1.0 for value in route_weights):
            raise ValueError("TAID 固定路由权重必须位于 [0,1]。")
        return self

    @property
    def enabled(self) -> bool:
        return self.ablation != "B0"

    @property
    def prior_only(self) -> bool:
        return self.ablation == "B1"

    @property
    def uses_prior_condition(self) -> bool:
        return self.ablation in {"B2", "B3", "B4", "B5", "B6"}

    @property
    def uses_absolute_uncertain(self) -> bool:
        return self.ablation == "B3"

    @property
    def uses_innovation(self) -> bool:
        return self.ablation in {"B4", "B5", "B6"}

    @property
    def uses_fixed_routing(self) -> bool:
        return self.ablation in {"B5", "B6"}

    @property
    def uses_continuous_transition(self) -> bool:
        return self.ablation == "B6"

    @property
    def uses_uncertain_condition(self) -> bool:
        return self.ablation in {"B3", "B4", "B5", "B6"}


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
    diffusion: float = 1.0
    global_rotation: float = 1.0
    local_rotation: float = 1.0
    fk: float = 2.0
    tracker_position: float = 10.0
    tracker_rotation: float = 1.0
    root: float = 1.0
    world_joint: float = 1.0
    head_to_root_xz: float = 1.0
    rollout: float = 1.0
    future_leg: float = 0.5
    contact: float = 0.1
    foot_slide: float = 0.5

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
        raise RuntimeError("24 个关节必须恰好映射到一个 TargetDiT 区域。")
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
