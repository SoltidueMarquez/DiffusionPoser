from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """把扩散时间步编码成连续向量，供不同去噪网络共享。"""

    def __init__(self, embedding_dim: int, max_period: int = 10_000):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.max_period = int(max_period)
        self.proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=torch.float32, device=timestep.device)
            / max(half_dim, 1)
        )
        args = timestep.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.proj(embedding)
