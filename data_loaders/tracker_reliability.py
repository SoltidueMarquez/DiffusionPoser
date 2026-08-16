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


def compute_ik_joint_confidence_torch(
    joint_source_reliability: torch.Tensor,
    constraint_type: torch.Tensor,
    updated_mask: torch.Tensor,
    position_residual: torch.Tensor,
    chain_length: torch.Tensor,
    direction_only_quality: float,
    residual_scale: float,
    position_solved_quality: float | None = None,
) -> torch.Tensor:
    """由 IK 的真实约束来源、类型和残差计算 `[B,24]` 置信度。

    `position_residual` 使用骨架坐标系中的长度，`chain_length` 是相同单位的
    对应可动骨链总长。直接 Tracker rotation 不依赖位置残差；其余已求解类型
    使用归一化残差指数衰减。约束编号遵循 realtime_pose_ik 中的固定契约：
    0=direct、1=position solved、2=direction only、3=inherited。
    """

    source = joint_source_reliability.float()
    expected_shape = source.shape
    values = (constraint_type, updated_mask, position_residual, chain_length)
    if source.ndim != 2 or source.shape[1] != 24:
        raise ValueError("joint_source_reliability 必须为 [B,24]。")
    if any(value.shape != expected_shape for value in values):
        raise ValueError("IK confidence 的所有逐关节输入必须同为 [B,24]。")
    if not bool(torch.isfinite(source).all()) or bool(
        ((source < 0.0) | (source > 1.0)).any()
    ):
        raise ValueError("joint_source_reliability 必须为有限的 [0,1] 数值。")
    if not bool(torch.isfinite(position_residual).all()) or bool(
        (position_residual < 0.0).any()
    ):
        raise ValueError("position_residual 必须为有限非负数。")
    if not bool(torch.isfinite(chain_length).all()) or bool((chain_length < 0.0).any()):
        raise ValueError("chain_length 必须为有限非负数。")
    if not 0.0 < float(direction_only_quality) < 1.0:
        raise ValueError("direction_only_quality 必须位于 (0,1)。")
    if float(residual_scale) <= 0.0:
        raise ValueError("residual_scale 必须大于 0。")

    constraint = constraint_type.to(dtype=torch.long)
    if bool(((constraint < 0) | (constraint > 3)).any()):
        raise ValueError("constraint_type 只能取 0、1、2、3。")
    position_solved = constraint == 1
    if bool(position_solved.any()) and position_solved_quality is None:
        raise ValueError("出现 POSITION_SOLVED 时必须提供 position_solved_quality。")
    if position_solved_quality is not None and not 0.0 < float(
        position_solved_quality
    ) < 1.0:
        raise ValueError("position_solved_quality 必须位于 (0,1)。")

    quality = torch.zeros_like(source)
    quality = torch.where(constraint == 0, torch.ones_like(quality), quality)
    if position_solved_quality is not None:
        quality = torch.where(
            position_solved,
            torch.full_like(quality, float(position_solved_quality)),
            quality,
        )
    quality = torch.where(
        constraint == 2,
        torch.full_like(quality, float(direction_only_quality)),
        quality,
    )

    residual_constrained = updated_mask.bool() & ((constraint == 1) | (constraint == 2))
    if bool((residual_constrained & (chain_length <= 0.0)).any()):
        raise ValueError("位置/方向 IK 约束必须具有正的 chain_length。")
    residual_ratio = position_residual / chain_length.clamp_min(1e-8)
    residual_quality = torch.exp(-residual_ratio / float(residual_scale))
    # 直接测得的是旋转本身，端点位置误差不能降低这条旋转测量的可信度。
    residual_quality = torch.where(constraint == 0, torch.ones_like(source), residual_quality)
    confidence = (source * quality * residual_quality).clamp(0.0, 1.0)
    return torch.where(updated_mask.bool(), confidence, torch.zeros_like(confidence))


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
