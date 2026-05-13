import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """把扩散时间步编码成连续向量，供每一层 Transformer 作为去噪强度条件。"""

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


class DiffusionPoserDiT(nn.Module):
    """
    面向 DiffusionPoser 稀疏传感器重建的轻量 DiT 骨架。

    输入和输出都使用 `[B, C, T]`：
    - `C` 是动作特征维度，默认 190；
    - `T` 是时间长度；
    - `inpaint_cond` 使用同形状布尔张量标记待补全位置，模型会把它作为显式条件通道拼接进去。
    """

    def __init__(
        self,
        input_feats: int = 190,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        zero_init: bool = False,
    ):
        super().__init__()
        self.input_feats = input_feats
        self.output_feats = input_feats
        self.latent_dim = latent_dim

        # 把“当前 noisy motion”和“哪些位置需要补全”一起投影成 token。
        self.input_proj = nn.Linear(input_feats * 2, latent_dim)
        self.time_embed = SinusoidalTimestepEmbedding(latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, input_feats)

        if zero_init:
            # 复现早期训练时更稳定：初始模型近似输出 0，逐步学习 x0 或 epsilon。
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
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states 应为 [B, C, T]，实际得到 {tuple(hidden_states.shape)}")
        batch_size, channels, seq_len = hidden_states.shape
        if channels != self.input_feats:
            raise ValueError(f"输入特征维度应为 {self.input_feats}，实际得到 {channels}")

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
            raise ValueError("valid_frame_mask 必须为 [B, T]")

        # Transformer 使用 [B, T, C]；mask 通道保留为 0/1，让模型知道哪些 token/特征是待生成目标。
        motion_tokens = hidden_states.transpose(1, 2)
        mask_tokens = inpaint_cond.float().transpose(1, 2)
        tokens = torch.cat([motion_tokens, mask_tokens], dim=-1)

        hidden = self.input_proj(tokens)
        hidden = hidden + self.time_embed(timestep).unsqueeze(1)

        # PyTorch 的 src_key_padding_mask 中 True 表示忽略该 token，因此这里取反。
        key_padding_mask = ~valid_frame_mask
        hidden = self.transformer(hidden, src_key_padding_mask=key_padding_mask)
        output = self.output_proj(hidden).transpose(1, 2)
        return output
