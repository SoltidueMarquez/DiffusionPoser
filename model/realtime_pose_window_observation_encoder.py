from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data_loaders.realtime_pose_config import TrackerReliabilityConfig
from data_loaders.sensor_masking import (
    NON_HEAD_TRACKER_INDICES,
    REALTIME_POSE_CONDITION_WINDOW_LENGTH,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_COUNT,
    TRACKER_D_ON_OFFSET,
    TRACKER_FEATURE_DIM,
    TRACKER_MEASURED_VALID_OFFSET,
)
from data_loaders.tracker_reliability import compute_tracker_reliability_torch


@dataclass
class WindowObservationEncoding:
    state_tokens: torch.Tensor
    position_tokens: torch.Tensor
    rotation_tokens: torch.Tensor
    kappa_position: torch.Tensor
    kappa_rotation: torch.Tensor


class WindowObservationEncoder(nn.Module):
    """逐帧编码 Tracker 窗口，保留时间轴，不把历史压成单个 summary。"""

    def __init__(
        self,
        latent_dim: int,
        reliability_config: TrackerReliabilityConfig | None = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reliability_config = (
            reliability_config or TrackerReliabilityConfig()
        ).validate()
        self.identity_embedding = nn.Embedding(TRACKER_COUNT, self.latent_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(4, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.rotation_encoder = nn.Sequential(
            nn.Linear(6, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def forward(
        self,
        tracker_window: torch.Tensor,
        window_valid_mask: torch.Tensor,
    ) -> WindowObservationEncoding:
        batch_size = tracker_window.shape[0]
        expected = (
            batch_size,
            REALTIME_POSE_CONDITION_WINDOW_LENGTH,
            TRACKER_COUNT,
            TRACKER_FEATURE_DIM,
        )
        if tuple(tracker_window.shape) != expected:
            raise ValueError(
                f"tracker_window 应为 [B,11,6,13]，实际为 {tuple(tracker_window.shape)}"
            )
        if tuple(window_valid_mask.shape) != (
            batch_size,
            REALTIME_POSE_CONDITION_WINDOW_LENGTH,
        ):
            raise ValueError("window_valid_mask 必须为 [B,11]。")

        tracker_ids = torch.arange(TRACKER_COUNT, device=tracker_window.device)
        identity = self.identity_embedding(tracker_ids)[None, None]
        state = tracker_window[..., TRACKER_CONFIGURED_OFFSET:TRACKER_FEATURE_DIM]
        state_tokens = self.state_encoder(state) + identity
        position_tokens = self.position_encoder(
            tracker_window[:, :, NON_HEAD_TRACKER_INDICES, :3]
        ) + identity[:, :, NON_HEAD_TRACKER_INDICES]
        rotation_tokens = self.rotation_encoder(tracker_window[..., 3:9]) + identity

        frame_valid = window_valid_mask.to(tracker_window.dtype)[..., None, None]
        state_tokens = state_tokens * frame_valid
        position_tokens = position_tokens * frame_valid
        rotation_tokens = rotation_tokens * frame_valid

        configured = tracker_window[..., TRACKER_CONFIGURED_OFFSET] > 0.5
        measured = tracker_window[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5
        d_on = (
            tracker_window[..., TRACKER_D_ON_OFFSET]
            * float(self.reliability_config.duration_cap)
        )
        kappa_position, kappa_rotation = compute_tracker_reliability_torch(
            configured,
            measured,
            d_on,
            config=self.reliability_config,
        )
        valid_scalar = window_valid_mask.to(kappa_position.dtype)[..., None]
        kappa_position = kappa_position * valid_scalar
        kappa_rotation = kappa_rotation * valid_scalar
        return WindowObservationEncoding(
            state_tokens=state_tokens,
            position_tokens=position_tokens,
            rotation_tokens=rotation_tokens,
            kappa_position=kappa_position,
            kappa_rotation=kappa_rotation,
        )
