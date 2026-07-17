from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Mapping


@dataclass(frozen=True)
class RealtimePoseLossConfig:
    """实时姿态辅助损失的唯一配置来源。

    权重和阈值属于实验 profile；这里的默认值固定为 C04 训练时使用的数值，
    使没有显式覆盖项的训练仍能复现同一损失定义。
    """

    aux_loss_weight: float = 1.0
    local_rotation_loss_weight: float = 3.7483991512973005
    body_geometry_loss_weight: float = 0.20561741210310552
    tracker_relative_pos_loss_weight: float = 0.06649022851925018
    tracker_relative_rot_loss_weight: float = 0.05810694589283483
    nohip_yaw_loss_weight: float = 8.72134586652926
    nohip_height_loss_weight: float = 0.023221202705222908
    stationary_regression_loss_weight: float = 0.020235997785184236
    stationary_margin_loss_weight: float = 0.022660554661242633
    # target_dit 已将 stationary logits 映射到 [0,1]，range loss 只保留为诊断兼容项。
    stationary_range_loss_weight: float = 0.0
    contact_height_loss_weight: float = 0.015288775346708965
    contact_velocity_loss_weight: float = 2.0343500261020283e-05
    joint_velocity_loss_weight: float = 0.00038461258563439796
    rotation_velocity_loss_weight: float = 0.0006603078011493034
    yaw_velocity_loss_weight: float = 0.0009690314102330362
    geometry_huber_beta: float = 0.05
    tracker_relative_pos_huber_beta: float = 0.05
    nohip_height_huber_beta: float = 0.05
    contact_velocity_huber_beta: float = 0.05
    contact_height_huber_beta: float = 0.01
    joint_velocity_huber_beta: float = 0.10
    rotation_velocity_huber_beta: float = 1.0
    yaw_velocity_huber_beta: float = 1.0
    foot_contact_height_threshold: float = 0.05
    stationary_runtime_threshold: float = 0.70
    stationary_runtime_margin: float = 0.10
    stationary_range_huber_beta: float = 0.05
    aux_timestep_min_weight: float = 0.10
    aux_timestep_gamma: float = 2.0

    def __post_init__(self) -> None:
        values = self.as_dict()
        for name, value in values.items():
            object.__setattr__(self, name, float(value))
        for name in _POSITIVE_CONFIG_FIELDS:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be greater than zero")
        for name in REALTIME_POSE_LOSS_TERM_TO_WEIGHT.values():
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be non-negative")
        if not 0.0 < self.stationary_runtime_threshold < 1.0:
            raise ValueError("stationary_runtime_threshold must be in (0, 1)")
        if not 0.0 < self.stationary_runtime_margin < min(
            self.stationary_runtime_threshold,
            1.0 - self.stationary_runtime_threshold,
        ):
            raise ValueError("stationary_runtime_margin must stay inside the threshold probability range")
        if not 0.0 <= self.aux_timestep_min_weight <= 1.0:
            raise ValueError("aux_timestep_min_weight must be in [0, 1]")
        if self.aux_timestep_gamma < 0.0:
            raise ValueError("aux_timestep_gamma must be non-negative")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "RealtimePoseLossConfig":
        field_names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - field_names)
        if unknown:
            raise ValueError(f"unknown realtime pose loss options: {unknown}")
        return cls(**{name: float(value) for name, value in values.items()})

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    def weight_for(self, loss_name: str) -> float:
        try:
            return float(getattr(self, REALTIME_POSE_LOSS_TERM_TO_WEIGHT[loss_name]))
        except KeyError as error:
            raise KeyError(f"unknown realtime pose loss term: {loss_name}") from error


REALTIME_POSE_LOSS_TERM_TO_WEIGHT: dict[str, str] = {
    "local_rotation_loss": "local_rotation_loss_weight",
    "body_geometry_loss": "body_geometry_loss_weight",
    "tracker_relative_pos_loss": "tracker_relative_pos_loss_weight",
    "tracker_relative_rot_loss": "tracker_relative_rot_loss_weight",
    "nohip_yaw_loss": "nohip_yaw_loss_weight",
    "nohip_height_loss": "nohip_height_loss_weight",
    "stationary_regression_loss": "stationary_regression_loss_weight",
    "stationary_margin_loss": "stationary_margin_loss_weight",
    "stationary_range_loss": "stationary_range_loss_weight",
    "contact_height_loss": "contact_height_loss_weight",
    "contact_velocity_loss": "contact_velocity_loss_weight",
    "joint_velocity_loss": "joint_velocity_loss_weight",
    "rotation_velocity_loss": "rotation_velocity_loss_weight",
    "yaw_velocity_loss": "yaw_velocity_loss_weight",
}


REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS: dict[str, float] = {
    "local_rotation_loss": 0.20,
    "body_geometry_loss": 0.25,
    "tracker_relative_pos_loss": 0.15,
    "tracker_relative_rot_loss": 0.05,
    "nohip_yaw_loss": 0.05,
    "nohip_height_loss": 0.05,
    "stationary_regression_loss": 0.10,
    "stationary_margin_loss": 0.05,
    "contact_height_loss": 0.025,
    "contact_velocity_loss": 0.025,
    "joint_velocity_loss": 0.04,
    "rotation_velocity_loss": 0.04,
    "yaw_velocity_loss": 0.02,
}


_POSITIVE_CONFIG_FIELDS = (
    "geometry_huber_beta",
    "tracker_relative_pos_huber_beta",
    "nohip_height_huber_beta",
    "contact_velocity_huber_beta",
    "contact_height_huber_beta",
    "joint_velocity_huber_beta",
    "rotation_velocity_huber_beta",
    "yaw_velocity_huber_beta",
    "foot_contact_height_threshold",
    "stationary_range_huber_beta",
)


REALTIME_POSE_LOSS_DEFAULTS = RealtimePoseLossConfig().as_dict()
