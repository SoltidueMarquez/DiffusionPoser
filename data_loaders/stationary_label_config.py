from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STATIONARY_LABEL_METADATA_FIELDS = (
    "stationary_label_method",
    "stationary_speed_full_motion",
    "stationary_median_window",
)


@dataclass(frozen=True)
class StationaryLabelConfig:
    """关节中心速度 stationary 标签生成的单一配置入口。"""

    method: str
    speed_full_motion: float
    median_window: int

    def metadata(self) -> dict[str, Any]:
        return {
            "stationary_label_method": self.method,
            "stationary_speed_full_motion": float(self.speed_full_motion),
            "stationary_median_window": int(self.median_window),
        }

    def metadata_arrays(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self.metadata().items()}


DEFAULT_STATIONARY_LABEL_CONFIG = StationaryLabelConfig(
    method="joint_center_speed_only_v1",
    speed_full_motion=0.25,
    median_window=5,
)


def stationary_label_metadata(config: StationaryLabelConfig = DEFAULT_STATIONARY_LABEL_CONFIG) -> dict[str, Any]:
    return config.metadata()


def stationary_label_metadata_arrays(
    config: StationaryLabelConfig = DEFAULT_STATIONARY_LABEL_CONFIG,
) -> dict[str, np.ndarray]:
    return config.metadata_arrays()
