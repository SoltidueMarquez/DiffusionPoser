from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from data_loaders.realtime_pose_config import TARGET_JOINT_REGIONS
from data_loaders.realtime_pose_kinematics import (
    rotation_6d_forward_up_torch,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    REALTIME_POSE_HISTORY_ANCHOR_COUNT,
    REALTIME_POSE_HISTORY_FRAME_OFFSETS,
    REALTIME_POSE_TARGET_DIM,
    SMPL_JOINT_COUNT,
)


@dataclass(frozen=True)
class HistoryPoseNoiseConfig:
    probability: float = 0.8
    min_degrees: float = 2.0
    max_degrees: float = 10.0
    temporal_rho: float = 0.95
    region_ratio: float = 0.75
    joint_ratio: float = 0.25

    def validate(self) -> "HistoryPoseNoiseConfig":
        # NaN 会让常规大小比较全部返回 False，Inf 也可能落入开放上界；必须先拒绝非有限配置。
        for name, value in (
            ("probability", self.probability),
            ("min_degrees", self.min_degrees),
            ("max_degrees", self.max_degrees),
            ("temporal_rho", self.temporal_rho),
            ("region_ratio", self.region_ratio),
            ("joint_ratio", self.joint_ratio),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} 必须是有限浮点数。")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("history_noise_prob 必须位于 [0,1]。")
        if not 0.0 <= self.min_degrees <= self.max_degrees:
            raise ValueError("历史噪声角度范围无效。")
        if not 0.0 <= self.temporal_rho < 1.0:
            raise ValueError("history_noise_temporal_rho 必须位于 [0,1)。")
        if self.region_ratio < 0.0 or self.joint_ratio < 0.0:
            raise ValueError("历史噪声混合比例不能为负。")
        if self.region_ratio + self.joint_ratio <= 0.0:
            raise ValueError("历史噪声至少需要一个非零分量。")
        return self


def corrupt_history_pose_observation(
    history_pose_clean: torch.Tensor,
    history_region_confidence: torch.Tensor,
    history_valid_mask: torch.Tensor,
    pose_mean: torch.Tensor | None,
    pose_scale: torch.Tensor | None,
    config: HistoryPoseNoiseConfig,
) -> torch.Tensor:
    """在反归一化后的 SO(3) 空间添加按区域和时间相关的部署误差。"""

    config = config.validate()
    batch_size = history_pose_clean.shape[0]
    expected = (
        batch_size,
        REALTIME_POSE_HISTORY_ANCHOR_COUNT,
        REALTIME_POSE_TARGET_DIM,
    )
    if tuple(history_pose_clean.shape) != expected:
        raise ValueError("history_pose_clean 必须为 [B,10,144]。")
    if tuple(history_region_confidence.shape) != (
        batch_size,
        REALTIME_POSE_HISTORY_ANCHOR_COUNT,
        5,
    ):
        raise ValueError("history_region_confidence 必须为 [B,10,5]。")
    if tuple(history_valid_mask.shape) != (
        batch_size,
        REALTIME_POSE_HISTORY_ANCHOR_COUNT,
    ):
        raise ValueError("history_valid_mask 必须为 [B,10]。")
    if config.probability <= 0.0 or batch_size == 0:
        return history_pose_clean.clone()

    device = history_pose_clean.device
    dtype = history_pose_clean.dtype
    if pose_mean is None or pose_scale is None:
        mean = torch.zeros(REALTIME_POSE_TARGET_DIM, device=device, dtype=dtype)
        scale = torch.ones(REALTIME_POSE_TARGET_DIM, device=device, dtype=dtype)
    else:
        mean = pose_mean.to(device=device, dtype=dtype).reshape(REALTIME_POSE_TARGET_DIM)
        scale = pose_scale.to(device=device, dtype=dtype).reshape(REALTIME_POSE_TARGET_DIM)
    raw = history_pose_clean * scale + mean
    rotations = rotation_6d_to_matrix_torch(
        raw.reshape(
            batch_size,
            REALTIME_POSE_HISTORY_ANCHOR_COUNT,
            SMPL_JOINT_COUNT,
            6,
        )
    )

    joint_regions = torch.tensor(
        TARGET_JOINT_REGIONS.copy(), device=device, dtype=torch.long
    )
    joint_confidence = history_region_confidence.index_select(2, joint_regions)
    sigma_degrees = config.min_degrees + (1.0 - joint_confidence.clamp(0.0, 1.0)) * (
        config.max_degrees - config.min_degrees
    )
    sigma_radians = sigma_degrees * (math.pi / 180.0)

    region_noise = _correlated_standard_noise(
        batch_size=batch_size,
        item_count=5,
        device=device,
        dtype=dtype,
        temporal_rho=config.temporal_rho,
    )
    joint_noise = _correlated_standard_noise(
        batch_size=batch_size,
        item_count=SMPL_JOINT_COUNT,
        device=device,
        dtype=dtype,
        temporal_rho=config.temporal_rho,
    )
    region_noise = region_noise.index_select(2, joint_regions)
    ratio_sum = config.region_ratio + config.joint_ratio
    region_scale = math.sqrt(config.region_ratio / ratio_sum)
    joint_scale = math.sqrt(config.joint_ratio / ratio_sum)
    tangent = sigma_radians[..., None] * (
        region_scale * region_noise + joint_scale * joint_noise
    )

    apply_sample = torch.rand(batch_size, device=device) < config.probability
    apply_mask = apply_sample[:, None] & history_valid_mask.bool()
    tangent = tangent * apply_mask[..., None, None].to(dtype)
    perturbation = _axis_angle_to_matrix(tangent)
    perturbed_rotations = rotations @ perturbation
    perturbed_raw = rotation_6d_forward_up_torch(perturbed_rotations).reshape(expected)
    perturbed_normalized = (perturbed_raw - mean) / scale
    result = torch.where(apply_sample[:, None, None], perturbed_normalized, history_pose_clean)
    return result.masked_fill(~history_valid_mask.bool()[..., None], 0.0)


def _correlated_standard_noise(
    batch_size: int,
    item_count: int,
    device: torch.device,
    dtype: torch.dtype,
    temporal_rho: float,
) -> torch.Tensor:
    offsets = REALTIME_POSE_HISTORY_FRAME_OFFSETS
    values = [torch.randn(batch_size, item_count, 3, device=device, dtype=dtype)]
    for index in range(1, REALTIME_POSE_HISTORY_ANCHOR_COUNT):
        delta_frames = int(offsets[index] - offsets[index - 1])
        coefficient = float(temporal_rho) ** delta_frames
        innovation_scale = math.sqrt(max(1.0 - coefficient * coefficient, 0.0))
        innovation = torch.randn(
            batch_size, item_count, 3, device=device, dtype=dtype
        )
        values.append(coefficient * values[-1] + innovation_scale * innovation)
    return torch.stack(values, dim=1)


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    theta_squared = theta.square()
    small = theta < 1e-4
    coefficient_a = torch.where(
        small,
        1.0 - theta_squared / 6.0,
        torch.sin(theta) / theta.clamp_min(1e-8),
    )
    coefficient_b = torch.where(
        small,
        0.5 - theta_squared / 24.0,
        (1.0 - torch.cos(theta)) / theta_squared.clamp_min(1e-8),
    )
    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    identity = identity.expand(*axis_angle.shape[:-1], 3, 3)
    return identity + coefficient_a[..., None] * skew + coefficient_b[..., None] * (skew @ skew)
