from __future__ import annotations

import math

import torch
from torch import nn

from data_loaders.sensor_masking import (
    REALTIME_POSE_HISTORY_FRAME_OFFSETS,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH,
    PREDICTOR_CORE_TRACKER_FRAME_OFFSETS,
    PREDICTOR_POSE_HORIZON_FRAME_OFFSETS,
    PREDICTOR_POSE_HORIZON_LENGTH,
    PREDICTOR_SPARSE_DIM,
)


class SinusoidalFrameEmbedding(nn.Module):
    """为带负数的真实 frame offset 生成确定性正余弦 embedding。"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        half = self.latent_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=offsets.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = offsets.float()[..., None] * frequencies
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.latent_dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class RealtimePosePredictor(nn.Module):
    """Head/双手驱动的 rolling motion predictor。

    结构参考 Meta Motion Rolling Prediction 官方仓库
    `facebookresearch/motion_rolling_prediction` 的 `RollingMDM`：motion query
    先对 sparse tracker 做一次 cross-attention，再由 TransformerEncoder 联合
    解码。本实现针对本仓库的 24 关节、144D 当前 Head-yaw 全局旋转重新实现。
    """

    def __init__(
        self,
        pose_dim: int = REALTIME_POSE_TARGET_DIM,
        sparse_dim: int = PREDICTOR_SPARSE_DIM,
        latent_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if int(pose_dim) != REALTIME_POSE_TARGET_DIM:
            raise ValueError("Predictor pose_dim 必须为 144。")
        if int(sparse_dim) != PREDICTOR_SPARSE_DIM:
            raise ValueError("Predictor sparse_dim 必须为 54。")
        self.pose_dim = int(pose_dim)
        self.sparse_dim = int(sparse_dim)
        self.latent_dim = int(latent_dim)
        self.motion_input = nn.Linear(self.pose_dim, self.latent_dim)
        self.sparse_input = nn.Linear(self.sparse_dim, self.latent_dim)
        self.frame_embedding = SinusoidalFrameEmbedding(self.latent_dim)
        self.motion_role = nn.Parameter(torch.zeros(self.latent_dim))
        self.prediction_role = nn.Parameter(torch.zeros(self.latent_dim))
        self.sparse_role = nn.Parameter(torch.zeros(self.latent_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(num_heads),
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.latent_dim),
        )
        self.output = nn.Linear(self.latent_dim, self.pose_dim)

        motion_offsets = (
            *REALTIME_POSE_HISTORY_FRAME_OFFSETS,
            *PREDICTOR_POSE_HORIZON_FRAME_OFFSETS,
        )
        self.register_buffer(
            "motion_frame_offsets",
            torch.tensor(motion_offsets, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "sparse_frame_offsets",
            torch.tensor(PREDICTOR_CORE_TRACKER_FRAME_OFFSETS, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        motion_context: torch.Tensor,
        core_tracker_context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = motion_context.shape[0]
        if tuple(motion_context.shape) != (
            batch_size,
            REALTIME_POSE_HISTORY_LENGTH,
            self.pose_dim,
        ):
            raise ValueError("motion_context 必须为 [B,10,144]。")
        if tuple(core_tracker_context.shape) != (
            batch_size,
            PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH,
            self.sparse_dim,
        ):
            raise ValueError("core_tracker_context 必须为 [B,11,54]。")
        prediction_slots = motion_context.new_zeros(
            batch_size, PREDICTOR_POSE_HORIZON_LENGTH, self.pose_dim
        )
        motion_values = torch.cat([motion_context, prediction_slots], dim=1)
        motion_tokens = self.motion_input(motion_values)
        motion_tokens = motion_tokens + self.frame_embedding(
            self.motion_frame_offsets.to(motion_tokens.device)
        )[None]
        motion_tokens[:, :REALTIME_POSE_HISTORY_LENGTH] += self.motion_role
        motion_tokens[:, REALTIME_POSE_HISTORY_LENGTH:] += self.prediction_role

        sparse_tokens = self.sparse_input(core_tracker_context)
        sparse_tokens = (
            sparse_tokens
            + self.frame_embedding(
                self.sparse_frame_offsets.to(sparse_tokens.device)
            )[None]
            + self.sparse_role
        )
        cross = self.cross_attention(
            motion_tokens, sparse_tokens, sparse_tokens, need_weights=False
        )[0]
        encoded = self.encoder(self.cross_norm(motion_tokens + cross))
        prediction = encoded[:, -PREDICTOR_POSE_HORIZON_LENGTH:]
        return self.output(prediction)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
