from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data_loaders.realtime_pose_config import TrackerReliabilityConfig
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    NON_HEAD_TRACKER_INDICES,
    REALTIME_POSE_HISTORY_LENGTH,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_COUNT,
    TRACKER_D_OFF_OFFSET,
    TRACKER_D_ON_OFFSET,
    TRACKER_FEATURE_DIM,
    TRACKER_MEASURED_VALID_OFFSET,
)
from data_loaders.tracker_reliability import (
    compute_region_coverage_torch,
    compute_tracker_reliability_torch,
)


@dataclass
class ObservationEncoding:
    state_tokens: torch.Tensor
    position_tokens: torch.Tensor
    rotation_tokens: torch.Tensor
    history_summary: torch.Tensor
    kappa_position: torch.Tensor
    kappa_rotation: torch.Tensor
    rho_position: torch.Tensor
    rho_rotation: torch.Tensor


class DynamicObservationEncoder(nn.Module):
    """将当前状态、位置、旋转和过去 Tracker 历史编码为彼此独立的 token。"""

    def __init__(
        self,
        latent_dim: int,
        reliability_config: TrackerReliabilityConfig | None = None,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.reliability_config = (reliability_config or TrackerReliabilityConfig()).validate()
        self.identity_embedding = nn.Embedding(TRACKER_COUNT, self.latent_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(4, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.latent_dim), nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.rotation_encoder = nn.Sequential(
            nn.Linear(6, self.latent_dim), nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.history_measurement_encoder = nn.Linear(9, self.latent_dim)
        self.history_state_encoder = nn.Linear(4, self.latent_dim)
        self.history_gru = nn.GRU(self.latent_dim, self.latent_dim, batch_first=True)

    def forward(
        self,
        tracker_history: torch.Tensor,
        current_tracker: torch.Tensor,
        valid_frame_mask: torch.Tensor,
    ) -> ObservationEncoding:
        batch_size = current_tracker.shape[0]
        if tuple(tracker_history.shape) != (
            batch_size,
            REALTIME_POSE_HISTORY_LENGTH,
            TRACKER_COUNT,
            TRACKER_FEATURE_DIM,
        ):
            raise ValueError(f"tracker_history 应为 [B,60,6,13]，实际为 {tuple(tracker_history.shape)}")
        if tuple(current_tracker.shape) != (batch_size, TRACKER_COUNT, TRACKER_FEATURE_DIM):
            raise ValueError(f"current_tracker 应为 [B,6,13]，实际为 {tuple(current_tracker.shape)}")
        if tuple(valid_frame_mask.shape) != (batch_size, REALTIME_POSE_HISTORY_LENGTH):
            raise ValueError("valid_frame_mask 必须为 [B,60]。")
        tracker_ids = torch.arange(TRACKER_COUNT, device=current_tracker.device)
        identity = self.identity_embedding(tracker_ids)[None]

        current_state = current_tracker[..., TRACKER_CONFIGURED_OFFSET:TRACKER_FEATURE_DIM]
        state_tokens = self.state_encoder(current_state) + identity
        position_tokens = self.position_encoder(current_tracker[:, NON_HEAD_TRACKER_INDICES, :3])
        position_tokens = position_tokens + identity[:, NON_HEAD_TRACKER_INDICES]
        rotation_tokens = self.rotation_encoder(current_tracker[..., 3:9]) + identity

        history_state = tracker_history[..., TRACKER_CONFIGURED_OFFSET:TRACKER_FEATURE_DIM]
        history_tokens = (
            self.history_measurement_encoder(tracker_history[..., :9])
            + self.history_state_encoder(history_state)
            + identity[:, None]
        )
        history_tokens = history_tokens.permute(0, 2, 1, 3).reshape(
            batch_size * TRACKER_COUNT,
            REALTIME_POSE_HISTORY_LENGTH,
            self.latent_dim,
        )
        expanded_valid = valid_frame_mask[:, None].expand(-1, TRACKER_COUNT, -1).reshape(
            batch_size * TRACKER_COUNT,
            REALTIME_POSE_HISTORY_LENGTH,
        )
        # runtime 使用左补零；先把有效帧稳定压到序列前部，避免 GRU 的 bias 让 padding 改变 hidden state。
        order = torch.argsort((~expanded_valid).long(), dim=1, stable=True)
        compacted = torch.gather(
            history_tokens,
            1,
            order[..., None].expand(-1, -1, self.latent_dim),
        )
        lengths = expanded_valid.long().sum(dim=1)
        safe_lengths = lengths.clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            compacted,
            safe_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.history_gru(packed)
        summary = hidden[-1]
        summary = summary * (lengths > 0)[:, None].to(summary.dtype)
        history_summary = summary.reshape(batch_size, TRACKER_COUNT, self.latent_dim)

        configured = current_tracker[..., TRACKER_CONFIGURED_OFFSET] > 0.5
        measured = current_tracker[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5
        d_on = current_tracker[..., TRACKER_D_ON_OFFSET] * float(self.reliability_config.duration_cap)
        kappa_position, kappa_rotation = compute_tracker_reliability_torch(
            configured,
            measured,
            d_on,
            config=self.reliability_config,
        )
        rho_position, rho_rotation = compute_region_coverage_torch(kappa_position, kappa_rotation)
        return ObservationEncoding(
            state_tokens=state_tokens,
            position_tokens=position_tokens,
            rotation_tokens=rotation_tokens,
            history_summary=history_summary,
            kappa_position=kappa_position,
            kappa_rotation=kappa_rotation,
            rho_position=rho_position,
            rho_rotation=rho_rotation,
        )
