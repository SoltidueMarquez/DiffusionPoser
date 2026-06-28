from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from data_loaders.sensor_masking import REALTIME_POSE_TARGET_START, get_schema_spec
from model.causal_attention import build_target_dit_causal_mask
from model.diffusionposer_dit import SinusoidalTimestepEmbedding, StationaryHead


class SensorTokenEncoder(nn.Module):
    """把 6 个 tracker 条件编码成 token，供 target token 通过 attention 读取。"""

    def __init__(self, input_dim: int, latent_dim: int, tracker_count: int = 6):
        super().__init__()
        self.tracker_count = int(tracker_count)
        self.type_embed = nn.Embedding(self.tracker_count, latent_dim)
        self.value_proj = nn.Linear(input_dim, latent_dim)

    def forward(self, sensor_values: torch.Tensor) -> torch.Tensor:
        # sensor_values: [B, 6, 10] = pos_ref(3) + rot_ref_6d(6) + valid(1)
        batch_size, tracker_count, _ = sensor_values.shape
        if tracker_count != self.tracker_count:
            raise ValueError(f"tracker token 数量应为 {self.tracker_count}，实际为 {tracker_count}")
        tracker_ids = torch.arange(tracker_count, device=sensor_values.device)
        return self.value_proj(sensor_values) + self.type_embed(tracker_ids)[None]


class RealtimePoseTargetDiT(nn.Module):
    """
    target-only denoiser。

    外部 API 仍然是 `[B,C,T] -> [B,C,T]`，但模型只学习第 61 帧 target slice。
    这样不会把 capacity 浪费在 tracker/sensor_valid 这些条件通道上。
    """

    def __init__(
        self,
        input_feats: int,
        schema_name: str,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        zero_init: bool = False,
        max_seq_len: int = 61,
        use_stationary_head: bool = False,
    ):
        super().__init__()
        self.schema = get_schema_spec(schema_name)
        self.input_feats = int(input_feats)
        self.output_feats = int(input_feats)
        self.latent_dim = int(latent_dim)
        self.max_seq_len = int(max_seq_len)
        self.use_stationary_head = bool(use_stationary_head)
        if self.input_feats != self.schema.feature_dim:
            raise ValueError(f"{self.schema.name} 需要 input_feats={self.schema.feature_dim}，实际为 {self.input_feats}")

        self.frame_proj = nn.Linear(self.input_feats, self.latent_dim)
        self.target_proj = nn.Linear(self.schema.target_dim, self.latent_dim)
        self.time_embed = SinusoidalTimestepEmbedding(self.latent_dim)
        self.frame_pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, self.latent_dim))
        self.sensor_encoder = SensorTokenEncoder(input_dim=10, latent_dim=self.latent_dim)
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
        self.output_proj = nn.Linear(self.latent_dim, self.schema.target_dim)
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
            raise ValueError(f"hidden_states 应为 [B,C,T]，实际为 {tuple(hidden_states.shape)}")
        batch_size, channels, seq_len = hidden_states.shape
        if channels != self.input_feats:
            raise ValueError(f"输入特征维应为 {self.input_feats}，实际为 {channels}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}")
        if seq_len <= REALTIME_POSE_TARGET_START:
            raise ValueError(f"target_dit 固定读取第 {REALTIME_POSE_TARGET_START + 1} 帧，实际 seq_len={seq_len}")
        if inpaint_cond is None:
            inpaint_cond = torch.ones_like(hidden_states, dtype=torch.bool)
        if inpaint_cond.shape != hidden_states.shape:
            raise ValueError("inpaint_cond 必须与 hidden_states 同形状，均为 [B, C, T]")

        if valid_frame_mask is None:
            valid_frame_mask = attention_mask
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        valid_frame_mask = valid_frame_mask.bool()

        frame_tokens = self.frame_proj(hidden_states.transpose(1, 2))
        frame_tokens = frame_tokens + self.frame_pos_embed[:, :seq_len]

        target_values = hidden_states[:, self.schema.target_slice(), REALTIME_POSE_TARGET_START]
        target_token = self.target_proj(target_values) + self.time_embed(timestep)
        target_token = target_token.unsqueeze(1)

        sensor_values = self._extract_sensor_values(hidden_states)
        sensor_tokens = self.sensor_encoder(sensor_values)
        tokens = torch.cat([target_token, sensor_tokens, frame_tokens], dim=1)

        # token 顺序：[target, 6 sensor, T frame]。causal mask 负责时间可见性，padding mask 只表示有效帧。
        token_mask = torch.zeros(batch_size, 1 + sensor_tokens.shape[1] + seq_len, dtype=torch.bool, device=hidden_states.device)
        token_mask[:, 1 + sensor_tokens.shape[1]:] = ~valid_frame_mask
        causal_mask = build_target_dit_causal_mask(
            seq_len=seq_len,
            tracker_count=sensor_tokens.shape[1],
            target_frame=REALTIME_POSE_TARGET_START,
            device=hidden_states.device,
        )
        hidden = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=token_mask)
        target_hidden = hidden[:, 0]
        pred_target = self.output_proj(target_hidden)
        target_mask = inpaint_cond[:, self.schema.target_slice(), REALTIME_POSE_TARGET_START].to(dtype=hidden_states.dtype)
        input_target = hidden_states[:, self.schema.target_slice(), REALTIME_POSE_TARGET_START]
        pred_target = pred_target * target_mask + input_target * (1.0 - target_mask)

        output = hidden_states.clone()
        output[:, self.schema.target_slice(), REALTIME_POSE_TARGET_START] = pred_target
        if not return_stationary_head:
            return output
        if self.stationary_head is None:
            return {"motion": output}
        stationary_logits = self.stationary_head(target_hidden)
        return {"motion": output, "stationary_logits": stationary_logits}

    def _extract_sensor_values(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = []
        frame = REALTIME_POSE_TARGET_START
        for tracker_index in range(6):
            pos = hidden_states[:, self.schema.tracker_pos_slice(tracker_index), frame]
            rot = hidden_states[:, self.schema.tracker_rot_slice(tracker_index), frame]
            valid = hidden_states[:, self.schema.sensor_valid_start + tracker_index: self.schema.sensor_valid_start + tracker_index + 1, frame]
            values.append(torch.cat([pos, rot, valid], dim=1))
        return torch.stack(values, dim=1)
