from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from data_loaders.realtime_pose_config import TARGET_JOINT_REGIONS
from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from data_loaders.sensor_masking import (
    CURRENT_JOINT_CONDITION_DIM,
    REALTIME_POSE_HISTORY_FRAME_OFFSETS,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    ROTATION_6D_DIM,
    PREDICTOR_FUTURE_FRAME_COUNT,
    PREDICTOR_POSE_HORIZON_LENGTH,
    SMPL_JOINT_COUNT,
)
from model.timestep_embedding import SinusoidalTimestepEmbedding


CURRENT_DIT_CONTEXT_OFFSETS = (
    *REALTIME_POSE_HISTORY_FRAME_OFFSETS,
    *tuple(range(1, PREDICTOR_FUTURE_FRAME_COUNT + 1)),
)


@dataclass(frozen=True)
class PreparedCurrentConditioning:
    """可在 DDIM 轨迹内复用的单帧 DiT 条件 token。"""

    current_tokens: torch.Tensor  # [B,24,D]
    temporal_context: torch.Tensor  # [B,24,20,D]


def _modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class CurrentDiTBlock(nn.Module):
    """当前关节空间注意力 + 同关节 20 帧上下文 cross-attention。"""

    def __init__(self, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.temporal_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.temporal_context_norm = nn.LayerNorm(latent_dim)
        self.mlp_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.spatial_attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(latent_dim, latent_dim * 9)
        )
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)

    def forward(
        self,
        current: torch.Tensor,
        temporal_context: torch.Tensor,
        diffusion_time: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, joint_count, latent_dim = current.shape
        if tuple(temporal_context.shape[:2]) != (batch_size, joint_count):
            raise ValueError("temporal_context 必须与 current 的 batch/joint 维一致。")
        (
            spatial_shift,
            spatial_scale,
            spatial_gate,
            temporal_shift,
            temporal_scale,
            temporal_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.adaln_modulation(diffusion_time).chunk(9, dim=-1)

        spatial_query = _modulate(
            self.spatial_norm(current), spatial_shift, spatial_scale
        )
        spatial_value = self.spatial_attention(
            spatial_query, spatial_query, spatial_query, need_weights=False
        )[0]
        current = current + spatial_gate[:, None] * spatial_value

        temporal_query = current.reshape(batch_size * joint_count, 1, latent_dim)
        temporal_shift_bj = temporal_shift[:, None].expand(-1, joint_count, -1).reshape(
            batch_size * joint_count, latent_dim
        )
        temporal_scale_bj = temporal_scale[:, None].expand(-1, joint_count, -1).reshape(
            batch_size * joint_count, latent_dim
        )
        temporal_gate_bj = temporal_gate[:, None].expand(-1, joint_count, -1).reshape(
            batch_size * joint_count, latent_dim
        )
        temporal_query = _modulate(
            self.temporal_norm(temporal_query),
            temporal_shift_bj,
            temporal_scale_bj,
        )
        temporal_keys = self.temporal_context_norm(
            temporal_context.reshape(batch_size * joint_count, -1, latent_dim)
        )
        temporal_value = self.temporal_attention(
            temporal_query, temporal_keys, temporal_keys, need_weights=False
        )[0]
        current = current + (
            temporal_gate_bj[:, None] * temporal_value
        ).reshape(batch_size, joint_count, latent_dim)

        mlp_query = _modulate(self.mlp_norm(current), mlp_shift, mlp_scale)
        return current + mlp_gate[:, None] * self.mlp(mlp_query)


class RealtimePoseCurrentDiT(nn.Module):
    """仅扩散当前 `[B,144]`，并读取过去 10 帧与 Predictor 未来 10 帧。"""

    def __init__(
        self,
        input_feats: int = REALTIME_POSE_TARGET_DIM,
        latent_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
        zero_init: bool = False,
        max_seq_len: int = 21,
    ):
        super().__init__()
        if int(input_feats) != REALTIME_POSE_TARGET_DIM:
            raise ValueError("单帧 DiT 的 Pose 必须为 144D。")
        if int(max_seq_len) != 21:
            raise ValueError("max_seq_len 固定为 21（过去 10、当前 1、Predictor 未来 10）。")
        self.input_feats = int(input_feats)
        self.output_feats = int(input_feats)
        self.latent_dim = int(latent_dim)
        self.joint_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.history_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.predictor_current_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.predictor_future_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.current_joint_condition_input = nn.Linear(
            CURRENT_JOINT_CONDITION_DIM, latent_dim
        )
        self.joint_identity = nn.Embedding(SMPL_JOINT_COUNT, latent_dim)
        self.region_identity = nn.Embedding(5, latent_dim)
        self.history_role = nn.Parameter(torch.zeros(latent_dim))
        self.predictor_current_role = nn.Parameter(torch.zeros(latent_dim))
        self.predictor_future_role = nn.Parameter(torch.zeros(latent_dim))
        self.diffusion_time_embedding = SinusoidalTimestepEmbedding(latent_dim)
        self.frame_offset_embedding = SinusoidalTimestepEmbedding(latent_dim)
        self.blocks = nn.ModuleList(
            [
                CurrentDiTBlock(latent_dim, num_heads, dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.joint_output = nn.Linear(latent_dim, ROTATION_6D_DIM)
        self.contact_head = nn.Linear(latent_dim * 2, 2)
        self.register_buffer(
            "joint_regions",
            torch.tensor(TARGET_JOINT_REGIONS.copy(), dtype=torch.long),
        )
        self.register_buffer(
            "context_frame_offsets",
            torch.tensor(CURRENT_DIT_CONTEXT_OFFSETS, dtype=torch.float32),
            persistent=False,
        )
        if zero_init:
            nn.init.zeros_(self.joint_output.weight)
            nn.init.zeros_(self.joint_output.bias)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def prepare_conditioning(
        self,
        motion_context: torch.Tensor,
        predictor_pose_horizon: torch.Tensor,
        current_joint_condition: torch.Tensor,
    ) -> PreparedCurrentConditioning:
        batch_size = motion_context.shape[0]
        if tuple(motion_context.shape) != (
            batch_size,
            REALTIME_POSE_HISTORY_LENGTH,
            REALTIME_POSE_TARGET_DIM,
        ):
            raise ValueError("motion_context 必须为 [B,10,144]。")
        if tuple(predictor_pose_horizon.shape) != (
            batch_size,
            PREDICTOR_POSE_HORIZON_LENGTH,
            REALTIME_POSE_TARGET_DIM,
        ):
            raise ValueError("predictor_pose_horizon 必须为 [B,11,144]。")
        if tuple(current_joint_condition.shape) != (
            batch_size,
            SMPL_JOINT_COUNT,
            CURRENT_JOINT_CONDITION_DIM,
        ):
            raise ValueError("current_joint_condition 必须为 [B,24,10]。")
        values = (motion_context, predictor_pose_horizon, current_joint_condition)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("单帧 DiT 条件中包含 NaN 或 Inf。")

        history = motion_context.reshape(
            batch_size,
            REALTIME_POSE_HISTORY_LENGTH,
            SMPL_JOINT_COUNT,
            ROTATION_6D_DIM,
        )
        predictor = predictor_pose_horizon.reshape(
            batch_size,
            PREDICTOR_POSE_HORIZON_LENGTH,
            SMPL_JOINT_COUNT,
            ROTATION_6D_DIM,
        )
        history_tokens = self.history_input(history) + self.history_role
        future_tokens = self.predictor_future_input(predictor[:, 1:]) + self.predictor_future_role
        context = torch.cat([history_tokens, future_tokens], dim=1)
        frame_tokens = self.frame_offset_embedding(
            self.context_frame_offsets.to(context.device)
        )
        context = context + frame_tokens[None, :, None]
        identity = (
            self.joint_identity.weight + self.region_identity(self.joint_regions)
        )
        context = context + identity[None, None]
        temporal_context = context.permute(0, 2, 1, 3).contiguous()

        current_tokens = (
            self.predictor_current_input(predictor[:, 0])
            + self.predictor_current_role
            + self.current_joint_condition_input(current_joint_condition)
            + identity[None]
        )
        return PreparedCurrentConditioning(
            current_tokens=current_tokens,
            temporal_context=temporal_context,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        motion_context: torch.Tensor | None = None,
        predictor_pose_horizon: torch.Tensor | None = None,
        current_joint_condition: torch.Tensor | None = None,
        prepared_conditioning: PreparedCurrentConditioning | None = None,
        y: dict | None = None,
        return_aux_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = hidden_states.shape[0]
        if tuple(hidden_states.shape) != (batch_size, REALTIME_POSE_TARGET_DIM):
            raise ValueError("hidden_states 必须为单帧 diffusion state [B,144]。")
        values = y or {}
        if prepared_conditioning is None:
            motion_context = (
                motion_context
                if motion_context is not None
                else values.get("motion_context")
            )
            predictor_pose_horizon = (
                predictor_pose_horizon
                if predictor_pose_horizon is not None
                else values.get("predictor_pose_horizon")
            )
            current_joint_condition = (
                current_joint_condition
                if current_joint_condition is not None
                else values.get("current_joint_condition")
            )
            if any(
                value is None
                for value in (
                    motion_context,
                    predictor_pose_horizon,
                    current_joint_condition,
                )
            ):
                raise ValueError("单帧 DiT 缺少 motion/Predictor/joint condition。")
            prepared_conditioning = self.prepare_conditioning(
                motion_context, predictor_pose_horizon, current_joint_condition
            )

        joint_values = hidden_states.reshape(
            batch_size, SMPL_JOINT_COUNT, ROTATION_6D_DIM
        )
        current = self.joint_input(joint_values) + prepared_conditioning.current_tokens
        diffusion_time = self.diffusion_time_embedding(timestep)
        for block in self.blocks:
            current = block(
                current,
                prepared_conditioning.temporal_context,
                diffusion_time,
            )
        current = self.output_norm(current)
        output = self.joint_output(current).reshape(
            batch_size, REALTIME_POSE_TARGET_DIM
        )
        if not return_aux_outputs:
            return output
        feet = torch.as_tensor(
            [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]],
            device=current.device,
        )
        auxiliary = {
            "contact_logits": self.contact_head(
                current.index_select(1, feet).flatten(1)
            )
        }
        return output, auxiliary
