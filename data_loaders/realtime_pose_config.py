from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from data_loaders.realtime_pose_kinematics import JOINT_INDEX


@dataclass(frozen=True)
class IKInpaintingConfig:
    """训练与采样共享的 IK 可信度与 Predictor residual 门控参数。"""

    fabrik_iterations: int = 2
    direction_only_quality: float | None = None
    residual_scale: float | None = None
    position_solved_quality: float | None = None
    gap_low: float | None = None
    gap_high: float | None = None
    direction_support: float = 0.35
    untracked_strength: float = 0.05

    def validate(self) -> "IKInpaintingConfig":
        if int(self.fabrik_iterations) <= 0:
            raise ValueError("fabrik_iterations 必须大于 0。")
        if self.direction_only_quality is None:
            raise ValueError(
                "direction_only_quality 尚未校准；请先运行离线 IK 校准并显式提供该参数。"
            )
        if not 0.0 < float(self.direction_only_quality) < 1.0:
            raise ValueError("direction_only_quality 必须位于 (0,1)。")
        if self.residual_scale is None:
            raise ValueError(
                "residual_scale 尚未校准；请先运行离线 IK 校准并显式提供该参数。"
            )
        if float(self.residual_scale) <= 0.0:
            raise ValueError("residual_scale 必须大于 0。")
        if self.position_solved_quality is not None and not 0.0 < float(
            self.position_solved_quality
        ) < 1.0:
            raise ValueError("position_solved_quality 必须位于 (0,1)。")
        if self.gap_low is None or self.gap_high is None:
            raise ValueError(
                "IK/Predictor gap 尚未校准；请显式提供 gap_low 与 gap_high。"
            )
        if float(self.gap_low) < 0.0:
            raise ValueError("gap_low 必须为非负弧度值。")
        if float(self.gap_high) - float(self.gap_low) < 1e-4:
            raise ValueError("gap_high 必须至少比 gap_low 大 1e-4 弧度。")
        if not 0.0 < float(self.direction_support) < 1.0:
            raise ValueError("direction_support 必须位于 (0,1)。")
        if not 0.0 <= float(self.untracked_strength) < 1.0:
            raise ValueError("untracked_strength 必须位于 [0,1)。")
        return self

    def to_dict(self) -> dict[str, float | int | None]:
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
    hip_height: float = 1.0
    rotation_velocity: float = 1.0

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


TARGET_JOINT_REGIONS = build_target_joint_regions()
TARGET_JOINT_REGIONS.setflags(write=False)
