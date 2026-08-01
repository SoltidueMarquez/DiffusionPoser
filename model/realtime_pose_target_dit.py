from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from data_loaders.sensor_masking import (
    NON_PELVIS_JOINT_COUNT,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    ROOT_YAW_RELATIVE_START,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_MEASURED_VALID_OFFSET,
    TRACKER_MISSING_AGE_OFFSET,
)
from model.diffusionposer_dit import SinusoidalTimestepEmbedding


class TrackerTokenEncoder(nn.Module):
    """显式编码 Tracker 身份、三态、掉线时长以及各自的 60 帧历史。"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.continuous_proj = nn.Linear(9, latent_dim)
        self.identity_embed = nn.Embedding(TRACKER_COUNT, latent_dim)
        self.state_embed = nn.Embedding(3, latent_dim)
        self.age_mlp = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.temporal_embed = nn.Parameter(torch.zeros(1, 61, 1, latent_dim))
        self.history_gru = nn.GRU(latent_dim, latent_dim, batch_first=True)

    @staticmethod
    def state_index(tracker_window: torch.Tensor) -> torch.Tensor:
        configured = tracker_window[..., TRACKER_CONFIGURED_OFFSET] > 0.5
        measured = tracker_window[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5
        if torch.any(measured & ~configured):
            raise ValueError("measured_valid 必须是 configured 的子集。")
        # 0=unconfigured, 1=configured_valid, 2=configured_missing
        return torch.where(configured, torch.where(measured, 1, 2), 0).long()

    def embed_frames(self, tracker_window: torch.Tensor) -> torch.Tensor:
        batch_size, frames, tracker_count, feature_dim = tracker_window.shape
        if tracker_count != TRACKER_COUNT or feature_dim != TRACKER_FEATURE_DIM or frames > 61:
            raise ValueError(
                f"tracker_window 应为 [B,T,{TRACKER_COUNT},{TRACKER_FEATURE_DIM}] 且 T<=61，"
                f"实际为 {tuple(tracker_window.shape)}"
            )
        tracker_ids = torch.arange(TRACKER_COUNT, device=tracker_window.device)
        age = tracker_window[..., TRACKER_MISSING_AGE_OFFSET : TRACKER_MISSING_AGE_OFFSET + 1]
        tokens = (
            self.continuous_proj(tracker_window[..., :9])
            + self.identity_embed(tracker_ids)[None, None]
            + self.state_embed(self.state_index(tracker_window))
            + self.age_mlp(age)
            + self.temporal_embed[:, :frames]
        )
        return tokens

    def forward(
        self,
        tracker_window: torch.Tensor,
        valid_frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tracker_window.shape[1] != 61:
            raise ValueError(f"tracker_window 固定为 61 帧，实际为 {tracker_window.shape[1]}")
        embedded = self.embed_frames(tracker_window)
        history = embedded[:, :REALTIME_POSE_HISTORY_LENGTH]  # [B,60,6,D]
        batch_size, history_len, tracker_count, latent_dim = history.shape
        history = history.permute(0, 2, 1, 3).reshape(batch_size * tracker_count, history_len, latent_dim)
        mask = valid_frame_mask[:, None, :, None].expand(-1, tracker_count, -1, latent_dim)
        history = history * mask.reshape(batch_size * tracker_count, history_len, latent_dim).to(history.dtype)
        output, _ = self.history_gru(history)
        lengths = valid_frame_mask.long().sum(dim=1).clamp_min(1) - 1
        gather_index = lengths[:, None].expand(-1, tracker_count).reshape(-1)
        summary = output[torch.arange(batch_size * tracker_count, device=output.device), gather_index]
        summary = summary.reshape(batch_size, tracker_count, latent_dim)
        current = embedded[:, REALTIME_POSE_HISTORY_LENGTH]
        return summary, current


class TargetDiTBlock(nn.Module):
    """目标 self-attention + 条件 cross-attention + timestep AdaLN/gate。"""

    def __init__(self, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.cross_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, latent_dim * 9))

    @staticmethod
    def _modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return value * (1.0 + scale[:, None]) + shift[:, None]

    def forward(
        self,
        target: torch.Tensor,
        condition: torch.Tensor,
        timestep_embedding: torch.Tensor,
        condition_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        values = self.modulation(timestep_embedding).chunk(9, dim=-1)
        self_shift, self_scale, self_gate, cross_shift, cross_scale, cross_gate, mlp_shift, mlp_scale, mlp_gate = values
        query = self._modulate(self.self_norm(target), self_shift, self_scale)
        self_value = self.self_attention(query, query, query, need_weights=False)[0]
        target = target + torch.tanh(self_gate)[:, None] * self_value

        query = self._modulate(self.cross_norm(target), cross_shift, cross_scale)
        cross_value = self.cross_attention(
            query,
            condition,
            condition,
            key_padding_mask=condition_padding_mask,
            need_weights=False,
        )[0]
        target = target + torch.tanh(cross_gate)[:, None] * cross_value

        mlp_value = self.mlp(self._modulate(self.mlp_norm(target), mlp_shift, mlp_scale))
        return target + torch.tanh(mlp_gate)[:, None] * mlp_value


class RealtimePoseTargetDiT(nn.Module):
    """对 23 个全局旋转 Token 与 1 个 Root 相对 yaw Token 直接去噪。"""

    def __init__(
        self,
        input_feats: int = REALTIME_POSE_TARGET_DIM,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        zero_init: bool = False,
        max_seq_len: int = 61,
    ):
        super().__init__()
        self.input_feats = int(input_feats)
        self.output_feats = int(input_feats)
        self.latent_dim = int(latent_dim)
        if self.input_feats != REALTIME_POSE_TARGET_DIM:
            raise ValueError(f"TargetDiT 仅接受 {REALTIME_POSE_TARGET_DIM} 维目标，实际为 {input_feats}。")
        if int(max_seq_len) != 61:
            raise ValueError("TargetDiT 固定使用 60 帧历史和 1 帧当前 Tracker。")

        self.joint_input = nn.Linear(6, latent_dim)
        self.root_input = nn.Linear(2, latent_dim)
        self.target_identity = nn.Embedding(NON_PELVIS_JOINT_COUNT + 1, latent_dim)
        self.known_state_embed = nn.Embedding(2, latent_dim)
        self.pose_history_proj = nn.Linear(REALTIME_POSE_TARGET_DIM, latent_dim)
        self.pose_temporal_embed = nn.Parameter(torch.zeros(1, REALTIME_POSE_HISTORY_LENGTH, latent_dim))
        self.tracker_encoder = TrackerTokenEncoder(latent_dim)
        self.time_embed = SinusoidalTimestepEmbedding(latent_dim)
        self.blocks = nn.ModuleList(
            [TargetDiTBlock(latent_dim, num_heads, dropout) for _ in range(int(num_layers))]
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.joint_output = nn.Linear(latent_dim, 6)
        self.root_output = nn.Linear(latent_dim, 2)
        if zero_init:
            nn.init.zeros_(self.joint_output.weight)
            nn.init.zeros_(self.joint_output.bias)
            nn.init.zeros_(self.root_output.weight)
            nn.init.zeros_(self.root_output.bias)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        pose_history: Optional[torch.Tensor] = None,
        tracker_window: Optional[torch.Tensor] = None,
        known_mask: Optional[torch.Tensor] = None,
        inpaint_cond: Optional[torch.Tensor] = None,
        valid_frame_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        y: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if hidden_states.ndim != 2 or hidden_states.shape[1] != REALTIME_POSE_TARGET_DIM:
            raise ValueError(
                f"hidden_states 应为 [B,{REALTIME_POSE_TARGET_DIM}]，实际为 {tuple(hidden_states.shape)}"
            )
        y = y or {}
        pose_history = pose_history if pose_history is not None else y.get("pose_history")
        tracker_window = tracker_window if tracker_window is not None else y.get("tracker_window")
        known_mask = known_mask if known_mask is not None else y.get("known_mask")
        if known_mask is None and inpaint_cond is not None:
            known_mask = ~inpaint_cond.bool()
        if pose_history is None or tracker_window is None or known_mask is None:
            raise ValueError("TargetDiT 需要 pose_history、tracker_window 和 known_mask。")

        batch_size = hidden_states.shape[0]
        if tuple(pose_history.shape) != (batch_size, REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM):
            raise ValueError(f"pose_history 形状错误：{tuple(pose_history.shape)}")
        if tuple(tracker_window.shape) != (batch_size, 61, TRACKER_COUNT, TRACKER_FEATURE_DIM):
            raise ValueError(f"tracker_window 形状错误：{tuple(tracker_window.shape)}")
        if tuple(known_mask.shape) != tuple(hidden_states.shape):
            raise ValueError("known_mask 必须与当前 140 维扩散状态同形。")
        valid_frame_mask = valid_frame_mask if valid_frame_mask is not None else attention_mask
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(
                batch_size,
                REALTIME_POSE_HISTORY_LENGTH,
                dtype=torch.bool,
                device=hidden_states.device,
            )
        valid_frame_mask = valid_frame_mask.bool()
        if tuple(valid_frame_mask.shape) != (batch_size, REALTIME_POSE_HISTORY_LENGTH):
            raise ValueError("valid_frame_mask 必须为 [B,60]。")

        joint_values = hidden_states[:, :ROOT_YAW_RELATIVE_START].reshape(
            batch_size, NON_PELVIS_JOINT_COUNT, 6
        )
        root_value = hidden_states[:, ROOT_YAW_RELATIVE_START:].unsqueeze(1)
        target = torch.cat([self.joint_input(joint_values), self.root_input(root_value)], dim=1)
        target_ids = torch.arange(NON_PELVIS_JOINT_COUNT + 1, device=hidden_states.device)
        atomic_known = torch.cat(
            [
                known_mask[:, :ROOT_YAW_RELATIVE_START].reshape(batch_size, NON_PELVIS_JOINT_COUNT, 6).all(-1),
                known_mask[:, ROOT_YAW_RELATIVE_START:].all(-1, keepdim=True),
            ],
            dim=1,
        )
        target = (
            target
            + self.target_identity(target_ids)[None]
            + self.known_state_embed(atomic_known.long())
        )

        pose_tokens = self.pose_history_proj(pose_history) + self.pose_temporal_embed
        tracker_summary, current_tracker = self.tracker_encoder(tracker_window, valid_frame_mask)
        condition = torch.cat([pose_tokens, tracker_summary, current_tracker], dim=1)
        condition_padding_mask = torch.cat(
            [
                ~valid_frame_mask,
                torch.zeros(batch_size, TRACKER_COUNT * 2, dtype=torch.bool, device=hidden_states.device),
            ],
            dim=1,
        )
        time_embedding = self.time_embed(timestep)
        for block in self.blocks:
            target = block(target, condition, time_embedding, condition_padding_mask)

        target = self.output_norm(target)
        joint_output = self.joint_output(target[:, :NON_PELVIS_JOINT_COUNT]).reshape(
            batch_size, ROOT_YAW_RELATIVE_START
        )
        root_output = self.root_output(target[:, NON_PELVIS_JOINT_COUNT])
        return torch.cat([joint_output, root_output], dim=-1)
