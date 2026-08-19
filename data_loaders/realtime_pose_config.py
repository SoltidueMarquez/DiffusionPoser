from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from data_loaders.realtime_pose_kinematics import JOINT_INDEX


@dataclass(frozen=True)
class IKInpaintingConfig:
    """训练与采样共享的当前帧 IK Inpainting 超参数。"""

    fabrik_iterations: int = 2
    direction_only_quality: float | None = None
    residual_scale: float | None = None
    position_solved_quality: float | None = None

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
    # 暂时关闭接触分类和脚底滑动监督，但保留 loss 字段与模型输出，
    # 后续恢复实验时只需重新设置权重，不需要改变 checkpoint 接口。
    contact: float = 0.0
    contact_slide: float = 0.0

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
