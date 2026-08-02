from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_config import (
    POSITION_COVERAGE,
    ROTATION_COVERAGE,
    TrackerReliabilityConfig,
)
from data_loaders.sensor_masking import HEAD_TRACKER_INDEX, TRACKER_COUNT


def compute_tracker_reliability_np(
    configured: np.ndarray,
    measured_valid: np.ndarray,
    d_on: np.ndarray,
    config: TrackerReliabilityConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """由有效状态和连续恢复时长计算逐 Tracker、逐模态可靠性。"""

    cfg = (config or TrackerReliabilityConfig()).validate()
    configured_value = np.asarray(configured, dtype=np.float32)
    measured_value = np.asarray(measured_valid, dtype=np.float32)
    d_on_value = np.asarray(d_on, dtype=np.float32)
    expected = configured_value.shape
    if any(value.shape != expected for value in (measured_value, d_on_value)):
        raise ValueError("configured、measured_valid 和 d_on 必须同形。")
    common = configured_value * measured_value
    kappa_pos = common * np.minimum(1.0, d_on_value / float(cfg.d_warm_pos))
    kappa_rot = common * np.minimum(1.0, d_on_value / float(cfg.d_warm_rot))
    return kappa_pos.astype(np.float32), kappa_rot.astype(np.float32)


def compute_region_coverage_np(
    kappa_pos: np.ndarray,
    kappa_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把 `[... ,6]` Tracker 可靠性汇聚为 `[... ,5]` 区域覆盖度。"""

    return (
        _coverage_np(kappa_pos, POSITION_COVERAGE),
        _coverage_np(kappa_rot, ROTATION_COVERAGE),
    )


def compute_tracker_reliability_torch(
    configured: torch.Tensor,
    measured_valid: torch.Tensor,
    d_on: torch.Tensor,
    config: TrackerReliabilityConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = (config or TrackerReliabilityConfig()).validate()
    values = (configured, measured_valid, d_on)
    if any(value.shape != configured.shape for value in values[1:]):
        raise ValueError("configured、measured_valid 和 d_on 必须同形。")
    dtype = d_on.dtype
    common = configured.to(dtype) * measured_valid.to(dtype)
    kappa_pos = common * torch.clamp(d_on / float(cfg.d_warm_pos), max=1.0)
    kappa_rot = common * torch.clamp(d_on / float(cfg.d_warm_rot), max=1.0)
    return kappa_pos, kappa_rot


def compute_region_coverage_torch(
    kappa_pos: torch.Tensor,
    kappa_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kappa_pos.shape[-1] != TRACKER_COUNT or kappa_rot.shape != kappa_pos.shape:
        raise ValueError("kappa_pos/kappa_rot 尾维必须同为 6。")
    position = torch.tensor(POSITION_COVERAGE.copy(), device=kappa_pos.device, dtype=kappa_pos.dtype)
    rotation = torch.tensor(ROTATION_COVERAGE.copy(), device=kappa_rot.device, dtype=kappa_rot.dtype)
    return _coverage_torch(kappa_pos, position), _coverage_torch(kappa_rot, rotation)


def compute_hard_rotation_state_np(
    configured: np.ndarray,
    measured_valid: np.ndarray,
    d_on: np.ndarray,
    config: TrackerReliabilityConfig | None = None,
) -> np.ndarray:
    """由当前有效性和连续恢复时长计算 hard rotation 集合。"""

    cfg = (config or TrackerReliabilityConfig()).validate()
    configured_value = np.asarray(configured, dtype=bool)
    measured_value = np.asarray(measured_valid, dtype=bool)
    d_on_value = np.asarray(d_on, dtype=np.int64)
    if configured_value.ndim == 0 or configured_value.shape[-1] != TRACKER_COUNT:
        raise ValueError("hard rotation 状态尾维必须为 6。")
    if measured_value.shape != configured_value.shape or d_on_value.shape != configured_value.shape:
        raise ValueError("configured、measured_valid 和 d_on 必须同形。")
    if np.any(measured_value & ~configured_value):
        raise ValueError("measured_valid 必须是 configured 子集。")
    if not np.all(
        configured_value[..., HEAD_TRACKER_INDEX] & measured_value[..., HEAD_TRACKER_INDEX]
    ):
        raise ValueError("Head 必须始终 configured 且 measured_valid。")
    # 恢复期的连续权重已由 kappa 负责；hard 只在测量连续稳定足够久后开启。
    current = configured_value & measured_value & (d_on_value >= cfg.d_hard)
    current = current.copy()
    current[..., HEAD_TRACKER_INDEX] = True
    return current


def _coverage_np(kappa: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    value = np.asarray(kappa, dtype=np.float32)
    if value.shape[-1] != TRACKER_COUNT:
        raise ValueError("kappa 尾维必须为 6。")
    expanded = value[..., None, :] * coverage
    return (1.0 - np.prod(1.0 - expanded, axis=-1)).astype(np.float32)


def _coverage_torch(kappa: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.prod(1.0 - kappa.unsqueeze(-2) * coverage, dim=-1)
