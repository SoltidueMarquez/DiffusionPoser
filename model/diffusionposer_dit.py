from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from data_loaders.sensor_masking import REALTIME_POSE_TARGET_START, STATIONARY_PROB_DIM
from model.causal_attention import build_frame_causal_mask


class SinusoidalTimestepEmbedding(nn.Module):
    """把扩散时间步编码成连续向量，表示当前样本的噪声强度。"""

    def __init__(self, embedding_dim: int, max_period: int = 10_000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=torch.float32, device=timestep.device)
            / half_dim
        )
        args = timestep.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.proj(embedding)


class StationaryHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, STATIONARY_PROB_DIM),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class DiffusionPoserDiT(nn.Module):
    """
    realtime_pose_v2 条件扩散去噪网络。

    输入和输出均为 `[B, C, T]`，默认 `C=211, T<=61`。`inpaint_cond=True`
    的位置由扩散生成，False 的位置保持为观测条件。frame positional embedding
    用来告诉 Transformer token 在 61 帧窗口中的位置。
    """

    def __init__(
        self,
        input_feats: int = 211,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        zero_init: bool = False,
        max_seq_len: int = 61,
        use_stationary_head: bool = False,
    ):
        super().__init__()
        self.input_feats = int(input_feats)
        self.output_feats = int(input_feats)
        self.latent_dim = int(latent_dim)
        self.max_seq_len = int(max_seq_len)
        self.use_stationary_head = bool(use_stationary_head)

        self.input_proj = nn.Linear(self.input_feats * 2, self.latent_dim)
        self.time_embed = SinusoidalTimestepEmbedding(self.latent_dim)
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=self.latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(self.latent_dim, self.input_feats)
        self.stationary_head = StationaryHead(self.latent_dim) if self.use_stationary_head else None

        if zero_init:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def num_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        inpaint_cond: Optional[torch.Tensor] = None,
        valid_frame_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_stationary_head: bool = False,
        **kwargs,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states 应为 [B, C, T]，实际为 {tuple(hidden_states.shape)}")
        batch_size, channels, seq_len = hidden_states.shape
        if channels != self.input_feats:
            raise ValueError(f"输入特征维应为 {self.input_feats}，实际为 {channels}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")

        if inpaint_cond is None:
            inpaint_cond = torch.ones_like(hidden_states, dtype=torch.bool)
        if inpaint_cond.shape != hidden_states.shape:
            raise ValueError("inpaint_cond 必须与 hidden_states 同形状，均为 [B, C, T]")

        if valid_frame_mask is None:
            valid_frame_mask = attention_mask
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        valid_frame_mask = valid_frame_mask.bool()
        if valid_frame_mask.shape != (batch_size, seq_len):
            raise ValueError(f"valid_frame_mask 应为 [B, T]，实际为 {tuple(valid_frame_mask.shape)}")

        motion_tokens = hidden_states.transpose(1, 2)
        mask_tokens = inpaint_cond.float().transpose(1, 2)
        tokens = torch.cat([motion_tokens, mask_tokens], dim=-1)

        hidden = self.input_proj(tokens)
        hidden = hidden + self.time_embed(timestep).unsqueeze(1)
        hidden = hidden + self.frame_pos_embed[:, :seq_len]

        key_padding_mask = ~valid_frame_mask
        causal_mask = build_frame_causal_mask(seq_len, device=hidden_states.device)
        hidden = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        motion = self.output_proj(hidden).transpose(1, 2)
        if not return_stationary_head:
            return motion
        if self.stationary_head is None:
            return {"motion": motion}
        if seq_len <= REALTIME_POSE_TARGET_START:
            raise ValueError(f"stationary_head requires frame {REALTIME_POSE_TARGET_START}, got seq_len={seq_len}")
        stationary_logits = self.stationary_head(hidden[:, REALTIME_POSE_TARGET_START])
        return {"motion": motion, "stationary_logits": stationary_logits}
