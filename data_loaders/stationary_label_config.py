from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


STATIONARY_LABEL_METADATA_FIELDS = (
    "stationary_label_method",
    "stationary_static_speed",
    "stationary_moving_speed",
    "stationary_causal_window",
    "stationary_release_mode",
)


@dataclass(frozen=True)
class StationaryLabelConfig:
    """关节中心速度 stationary 标签生成的单一配置入口。"""

    method: str
    static_speed: float
    moving_speed: float
    causal_window: int
    release_mode: str

    def metadata(self) -> dict[str, Any]:
        return {
            "stationary_label_method": self.method,
            "stationary_static_speed": float(self.static_speed),
            "stationary_moving_speed": float(self.moving_speed),
            "stationary_causal_window": int(self.causal_window),
            "stationary_release_mode": self.release_mode,
        }

    def metadata_arrays(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(value) for key, value in self.metadata().items()}


DEFAULT_STATIONARY_LABEL_CONFIG = StationaryLabelConfig(
    method="joint_center_speed_causal_fast_release_v2",
    static_speed=0.03,
    moving_speed=0.25,
    causal_window=5,
    release_mode="fast_release_min",
)


def stationary_label_metadata(config: StationaryLabelConfig = DEFAULT_STATIONARY_LABEL_CONFIG) -> dict[str, Any]:
    return config.metadata()


def stationary_label_metadata_arrays(
    config: StationaryLabelConfig = DEFAULT_STATIONARY_LABEL_CONFIG,
) -> dict[str, np.ndarray]:
    return config.metadata_arrays()
