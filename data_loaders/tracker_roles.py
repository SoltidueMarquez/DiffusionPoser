from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_loaders.realtime_pose_config import (
    POSITION_COVERAGE,
    ROTATION_COVERAGE,
    TrackerRoleConfig,
)
from data_loaders.sensor_masking import HEAD_TRACKER_INDEX, TRACKER_COUNT


TRACKER_ROLE_UNCONFIGURED = 0
TRACKER_ROLE_MISSING = 1
TRACKER_ROLE_UNCERTAIN = 2
TRACKER_ROLE_ANCHOR = 3
TRACKER_ROLE_NAMES = ("unconfigured", "missing", "uncertain", "anchor")

# Anchor coverage 同时考虑 position/rotation 的固定作用域。Head 虽不提供普通
# position token，但其 rotation 仍为 Torso 提供稳定锚点。
ANCHOR_REGION_COVERAGE = np.maximum(POSITION_COVERAGE, ROTATION_COVERAGE).astype(np.float32)
ANCHOR_REGION_COVERAGE.setflags(write=False)


@dataclass(frozen=True)
class TrackerRoleStateNP:
    roles: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    beta_hard: np.ndarray
    region_coverage: np.ndarray


@dataclass(frozen=True)
class TrackerRoleStateTorch:
    roles: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    beta_hard: torch.Tensor
    region_coverage: torch.Tensor


def compute_tracker_roles_np(
    configured: np.ndarray,
    measured_valid: np.ndarray,
    d_on: np.ndarray,
    config: TrackerRoleConfig | None = None,
) -> TrackerRoleStateNP:
    """由物理帧计数生成 Unconfigured/M/U/A 与连续互补权重。"""

    cfg = (config or TrackerRoleConfig()).validate()
    configured_value = np.asarray(configured, dtype=bool)
    measured_value = np.asarray(measured_valid, dtype=bool)
    d_on_value = np.asarray(d_on, dtype=np.float32)
    _validate_inputs(configured_value, measured_value, d_on_value)

    roles = np.full(configured_value.shape, TRACKER_ROLE_UNCONFIGURED, dtype=np.int64)
    roles[configured_value & ~measured_value] = TRACKER_ROLE_MISSING
    roles[configured_value & measured_value] = TRACKER_ROLE_UNCERTAIN
    roles[
        configured_value
        & measured_value
        & (d_on_value >= float(cfg.anchor_ramp_end))
    ] = TRACKER_ROLE_ANCHOR
    roles[..., HEAD_TRACKER_INDEX] = TRACKER_ROLE_ANCHOR

    valid = (configured_value & measured_value).astype(np.float32)
    alpha = valid * np.clip(
        (d_on_value - float(cfg.anchor_ramp_start))
        / float(cfg.anchor_ramp_end - cfg.anchor_ramp_start),
        0.0,
        1.0,
    )
    beta = valid * (1.0 - alpha) * np.minimum(
        1.0,
        d_on_value / float(cfg.innovation_ramp_frames),
    )
    beta_hard = (
        roles == TRACKER_ROLE_UNCERTAIN
    ).astype(np.float32)
    alpha[..., HEAD_TRACKER_INDEX] = 1.0
    beta[..., HEAD_TRACKER_INDEX] = 0.0
    beta_hard[..., HEAD_TRACKER_INDEX] = 0.0
    coverage = _coverage_np(alpha)
    return TrackerRoleStateNP(
        roles=roles,
        alpha=alpha.astype(np.float32),
        beta=beta.astype(np.float32),
        beta_hard=beta_hard,
        region_coverage=coverage,
    )


def compute_tracker_roles_torch(
    configured: torch.Tensor,
    measured_valid: torch.Tensor,
    d_on: torch.Tensor,
    config: TrackerRoleConfig | None = None,
) -> TrackerRoleStateTorch:
    """Torch 版角色计算；公式与 runtime NumPy 路径逐项一致。"""

    cfg = (config or TrackerRoleConfig()).validate()
    configured_value = configured.bool()
    measured_value = measured_valid.bool()
    d_on_value = d_on.to(dtype=torch.float32)
    _validate_inputs_torch(configured_value, measured_value, d_on_value)

    roles = torch.full_like(d_on_value, TRACKER_ROLE_UNCONFIGURED, dtype=torch.long)
    roles = torch.where(
        configured_value & ~measured_value,
        torch.full_like(roles, TRACKER_ROLE_MISSING),
        roles,
    )
    roles = torch.where(
        configured_value & measured_value,
        torch.full_like(roles, TRACKER_ROLE_UNCERTAIN),
        roles,
    )
    roles = torch.where(
        configured_value
        & measured_value
        & (d_on_value >= float(cfg.anchor_ramp_end)),
        torch.full_like(roles, TRACKER_ROLE_ANCHOR),
        roles,
    )
    roles = roles.clone()
    roles[..., HEAD_TRACKER_INDEX] = TRACKER_ROLE_ANCHOR

    valid = (configured_value & measured_value).to(d_on_value.dtype)
    alpha = valid * torch.clamp(
        (d_on_value - float(cfg.anchor_ramp_start))
        / float(cfg.anchor_ramp_end - cfg.anchor_ramp_start),
        min=0.0,
        max=1.0,
    )
    beta = valid * (1.0 - alpha) * torch.clamp(
        d_on_value / float(cfg.innovation_ramp_frames),
        min=0.0,
        max=1.0,
    )
    beta_hard = (roles == TRACKER_ROLE_UNCERTAIN).to(d_on_value.dtype)
    alpha = alpha.clone()
    beta = beta.clone()
    beta_hard = beta_hard.clone()
    alpha[..., HEAD_TRACKER_INDEX] = 1.0
    beta[..., HEAD_TRACKER_INDEX] = 0.0
    beta_hard[..., HEAD_TRACKER_INDEX] = 0.0
    coverage_matrix = torch.as_tensor(
        ANCHOR_REGION_COVERAGE.copy(),
        device=alpha.device,
        dtype=alpha.dtype,
    )
    coverage = 1.0 - torch.prod(
        1.0 - alpha.unsqueeze(-2) * coverage_matrix,
        dim=-1,
    )
    return TrackerRoleStateTorch(
        roles=roles,
        alpha=alpha,
        beta=beta,
        beta_hard=beta_hard,
        region_coverage=coverage,
    )


def _coverage_np(alpha: np.ndarray) -> np.ndarray:
    return (
        1.0
        - np.prod(
            1.0 - alpha[..., None, :] * ANCHOR_REGION_COVERAGE,
            axis=-1,
        )
    ).astype(np.float32)


def _validate_inputs(
    configured: np.ndarray,
    measured_valid: np.ndarray,
    d_on: np.ndarray,
) -> None:
    if configured.ndim == 0 or configured.shape[-1] != TRACKER_COUNT:
        raise ValueError("角色状态尾维必须为 6。")
    if measured_valid.shape != configured.shape or d_on.shape != configured.shape:
        raise ValueError("configured、measured_valid 和 d_on 必须同形。")
    if np.any(measured_valid & ~configured):
        raise ValueError("measured_valid 必须是 configured 子集。")
    if not np.all(configured[..., HEAD_TRACKER_INDEX] & measured_valid[..., HEAD_TRACKER_INDEX]):
        raise ValueError("Head 必须始终 configured 且 measured_valid。")


def _validate_inputs_torch(
    configured: torch.Tensor,
    measured_valid: torch.Tensor,
    d_on: torch.Tensor,
) -> None:
    if configured.ndim == 0 or configured.shape[-1] != TRACKER_COUNT:
        raise ValueError("角色状态尾维必须为 6。")
    if measured_valid.shape != configured.shape or d_on.shape != configured.shape:
        raise ValueError("configured、measured_valid 和 d_on 必须同形。")
    if torch.any(measured_valid & ~configured):
        raise ValueError("measured_valid 必须是 configured 子集。")
    if not torch.all(configured[..., HEAD_TRACKER_INDEX] & measured_valid[..., HEAD_TRACKER_INDEX]):
        raise ValueError("Head 必须始终 configured 且 measured_valid。")
