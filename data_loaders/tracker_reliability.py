from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_config import (
    INPAINT_JOINT_COVERAGE,
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


def compute_tracker_online_confidence_np(
    tracker_valid: np.ndarray,
    d_on: np.ndarray,
    warmup_frames: int,
) -> np.ndarray:
    """按连续在线时长计算当前帧唯一的逐 Tracker 置信度。"""

    valid = np.asarray(tracker_valid, dtype=bool)
    duration = np.asarray(d_on, dtype=np.float32)
    if valid.ndim == 0 or valid.shape != duration.shape or valid.shape[-1] != TRACKER_COUNT:
        raise ValueError("tracker_valid 和 d_on 必须同形且尾维为 6。")
    if int(warmup_frames) <= 0:
        raise ValueError("warmup_frames 必须大于 0。")
    if not np.isfinite(duration).all() or np.any(duration < 0.0):
        raise ValueError("d_on 必须为有限非负数。")
    return (
        valid.astype(np.float32)
        * np.clip(duration / float(warmup_frames), 0.0, 1.0)
    ).astype(np.float32)


def compute_tracker_online_confidence_torch(
    tracker_valid: torch.Tensor,
    d_on: torch.Tensor,
    warmup_frames: int,
) -> torch.Tensor:
    """Torch 版本的逐 Tracker 在线置信度，输入输出尾维均为 6。"""

    valid = tracker_valid.bool()
    duration = d_on.float()
    if valid.ndim == 0 or valid.shape != duration.shape or valid.shape[-1] != TRACKER_COUNT:
        raise ValueError("tracker_valid 和 d_on 必须同形且尾维为 6。")
    if int(warmup_frames) <= 0:
        raise ValueError("warmup_frames 必须大于 0。")
    if not bool(torch.isfinite(duration).all()) or bool((duration < 0.0).any()):
        raise ValueError("d_on 必须为有限非负数。")
    return valid.to(duration.dtype) * torch.clamp(
        duration / float(warmup_frames), min=0.0, max=1.0
    )


def map_tracker_confidence_to_joints_torch(
    tracker_confidence: torch.Tensor,
) -> torch.Tensor:
    """用固定区域 mapping 将 `[B,6]` Tracker 置信度映射为 `[B,24]`。"""

    confidence = tracker_confidence.float()
    if confidence.ndim != 2 or confidence.shape[1] != TRACKER_COUNT:
        raise ValueError("tracker_confidence 必须为 [B,6]。")
    if not bool(torch.isfinite(confidence).all()) or bool(
        ((confidence < 0.0) | (confidence > 1.0)).any()
    ):
        raise ValueError("tracker_confidence 必须为有限的 [0,1] 数值。")
    coverage = torch.tensor(
        INPAINT_JOINT_COVERAGE.copy(),
        device=confidence.device,
        dtype=confidence.dtype,
    )
    # 每个关节只取所有覆盖 Tracker 中的最大置信度；禁止使用平均、noisy-or
    # 或骨链 min，确保公式与部署约定 c[j]=max_t(A[j,t]w_t) 完全一致。
    return (confidence[:, None, :] * coverage[None]).amax(dim=-1)


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
