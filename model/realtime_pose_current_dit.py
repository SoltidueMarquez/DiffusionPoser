from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from data_loaders.realtime_pose_config import TARGET_JOINT_REGIONS
from data_loaders.sensor_masking import (
    CURRENT_JOINT_CONSTRAINT_TYPE_COUNT,
    REALTIME_POSE_HISTORY_FRAME_OFFSETS,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    ROTATION_6D_DIM,
    PREDICTOR_FUTURE_FRAME_COUNT,
    PREDICTOR_POSE_HORIZON_LENGTH,
    SMPL_JOINT_COUNT,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
)
from model.timestep_embedding import SinusoidalTimestepEmbedding


CURRENT_DIT_CONTEXT_OFFSETS = (
    *REALTIME_POSE_HISTORY_FRAME_OFFSETS,
    *tuple(range(1, PREDICTOR_FUTURE_FRAME_COUNT + 1)),
)
IK_SCALAR_CONDITION_DIM = 3 + CURRENT_JOINT_CONSTRAINT_TYPE_COUNT


@dataclass(frozen=True)
class PreparedCurrentConditioning:
    """可在完整 DDIM 轨迹中复用的静态条件。"""

    joint_condition_tokens: torch.Tensor  # [B,24,D]
    temporal_context: torch.Tensor  # [B,24,20,D]
    tracker_tokens: torch.Tensor  # [B,6,D]
    tracker_available: torch.Tensor  # [B,6]
    denoise_strength: torch.Tensor  # [B,24]


def _modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class CurrentDiTBlock(nn.Module):
    """24 关节空间注意力 + 同关节过去/未来 20 帧 cross-attention。"""

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
    """以 Predictor 为 prior，去噪当前 144D 门控 residual。"""

    def __init__(
        self,
        input_feats: int = REALTIME_POSE_TARGET_DIM,
        latent_dim: int = 192,
        num_layers: int = 4,
        num_heads: int = 6,
        dropout: float = 0.0,
        max_seq_len: int = 21,
    ):
        super().__init__()
        if int(input_feats) != REALTIME_POSE_TARGET_DIM:
            raise ValueError("单帧 DiT residual 必须为 144D。")
        if int(max_seq_len) != 21:
            raise ValueError("max_seq_len 固定为 21（过去 10、当前 1、未来 10）。")
        if int(latent_dim) % int(num_heads) != 0:
            raise ValueError("latent_dim 必须能被 num_heads 整除。")
        self.input_feats = int(input_feats)
        self.output_feats = int(input_feats)
        self.latent_dim = int(latent_dim)

        self.residual_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.history_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.predictor_current_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.predictor_future_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.ik_residual_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.ik_scalar_input = nn.Linear(IK_SCALAR_CONDITION_DIM, latent_dim)
        self.tracker_input = nn.Linear(TRACKER_CONTINUOUS_DIM, latent_dim)

        self.joint_identity = nn.Embedding(SMPL_JOINT_COUNT, latent_dim)
        self.region_identity = nn.Embedding(5, latent_dim)
        self.tracker_identity = nn.Embedding(TRACKER_COUNT, latent_dim)
        self.history_role = nn.Parameter(torch.zeros(latent_dim))
        self.predictor_current_role = nn.Parameter(torch.zeros(latent_dim))
        self.predictor_future_role = nn.Parameter(torch.zeros(latent_dim))
        self.tracker_role = nn.Parameter(torch.zeros(latent_dim))

        self.tracker_query_norm = nn.LayerNorm(latent_dim)
        self.tracker_context_norm = nn.LayerNorm(latent_dim)
        self.tracker_cross_attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.tracker_output_norm = nn.LayerNorm(latent_dim)
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
        self.register_buffer(
            "joint_regions",
            torch.tensor(TARGET_JOINT_REGIONS.copy(), dtype=torch.long),
        )
        self.register_buffer(
            "context_frame_offsets",
            torch.tensor(CURRENT_DIT_CONTEXT_OFFSETS, dtype=torch.float32),
            persistent=False,
        )
        # 门控 residual 的零输出严格等于 Predictor prior，因此输出头始终零初始化。
        nn.init.zeros_(self.joint_output.weight)
        nn.init.zeros_(self.joint_output.bias)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def prepare_conditioning(
        self,
        motion_context: torch.Tensor,
        predictor_pose_horizon: torch.Tensor,
        tracker_geometry: torch.Tensor,
        tracker_available: torch.Tensor,
        ik_residual: torch.Tensor,
        ik_gap: torch.Tensor,
        ik_confidence: torch.Tensor,
        denoise_strength: torch.Tensor,
        constraint_type: torch.Tensor,
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
        if tuple(tracker_geometry.shape) != (
            batch_size,
            TRACKER_COUNT,
            TRACKER_CONTINUOUS_DIM,
        ):
            raise ValueError("tracker_geometry 必须为 [B,6,9]。")
        if tuple(tracker_available.shape) != (batch_size, TRACKER_COUNT):
            raise ValueError("tracker_available 必须为 [B,6]。")
        if tracker_available.dtype != torch.bool:
            raise ValueError("tracker_available 必须为 bool。")
        if tuple(ik_residual.shape) != (
            batch_size,
            SMPL_JOINT_COUNT,
            ROTATION_6D_DIM,
        ):
            raise ValueError("ik_residual 必须为 [B,24,6]。")
        joint_shape = (batch_size, SMPL_JOINT_COUNT)
        if any(
            tuple(value.shape) != joint_shape
            for value in (ik_gap, ik_confidence, denoise_strength, constraint_type)
        ):
            raise ValueError("IK gap/confidence/strength/type 必须同为 [B,24]。")
        if constraint_type.dtype != torch.long:
            raise ValueError("constraint_type 必须为 torch.long。")

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
        identity = self.joint_identity.weight + self.region_identity(self.joint_regions)
        history_tokens = self.history_input(history) + self.history_role
        future_tokens = (
            self.predictor_future_input(predictor[:, 1:])
            + self.predictor_future_role
        )
        context = torch.cat([history_tokens, future_tokens], dim=1)
        context = context + self.frame_offset_embedding(
            self.context_frame_offsets.to(context.device)
        )[None, :, None]
        context = context + identity[None, None]
        temporal_context = context.permute(0, 2, 1, 3).contiguous()

        one_hot = F.one_hot(
            constraint_type,
            num_classes=CURRENT_JOINT_CONSTRAINT_TYPE_COUNT,
        ).to(predictor.dtype)
        ik_scalars = torch.cat(
            [
                ik_gap[..., None],
                ik_confidence[..., None],
                denoise_strength[..., None],
                one_hot,
            ],
            dim=-1,
        )
        joint_condition_tokens = (
            self.predictor_current_input(predictor[:, 0])
            + self.predictor_current_role
            + self.ik_residual_input(ik_residual)
            + self.ik_scalar_input(ik_scalars)
            + identity[None]
        )
        tracker_tokens = (
            self.tracker_input(tracker_geometry)
            + self.tracker_identity.weight[None]
            + self.tracker_role
        )
        return PreparedCurrentConditioning(
            joint_condition_tokens=joint_condition_tokens,
            temporal_context=temporal_context,
            tracker_tokens=tracker_tokens,
            tracker_available=tracker_available,
            denoise_strength=denoise_strength,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        motion_context: torch.Tensor | None = None,
        predictor_pose_horizon: torch.Tensor | None = None,
        tracker_geometry: torch.Tensor | None = None,
        tracker_available: torch.Tensor | None = None,
        ik_residual: torch.Tensor | None = None,
        ik_gap: torch.Tensor | None = None,
        ik_confidence: torch.Tensor | None = None,
        denoise_strength: torch.Tensor | None = None,
        constraint_type: torch.Tensor | None = None,
        prepared_conditioning: PreparedCurrentConditioning | None = None,
        y: dict | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        if tuple(hidden_states.shape) != (batch_size, REALTIME_POSE_TARGET_DIM):
            raise ValueError("hidden_states 必须为 residual diffusion state [B,144]。")
        if prepared_conditioning is None:
            values = y or {}
            arguments = {
                "motion_context": motion_context,
                "predictor_pose_horizon": predictor_pose_horizon,
                "tracker_geometry": tracker_geometry,
                "tracker_available": tracker_available,
                "ik_residual": ik_residual,
                "ik_gap": ik_gap,
                "ik_confidence": ik_confidence,
                "denoise_strength": denoise_strength,
                "constraint_type": constraint_type,
            }
            arguments = {
                name: value if value is not None else values.get(name)
                for name, value in arguments.items()
            }
            if any(value is None for value in arguments.values()):
                missing = [name for name, value in arguments.items() if value is None]
                raise ValueError(f"单帧 DiT 缺少条件：{missing}")
            prepared_conditioning = self.prepare_conditioning(**arguments)

        current = self.residual_input(
            hidden_states.reshape(
                batch_size, SMPL_JOINT_COUNT, ROTATION_6D_DIM
            )
        ) + prepared_conditioning.joint_condition_tokens
        tracker_query = self.tracker_query_norm(current)
        tracker_context = self.tracker_context_norm(
            prepared_conditioning.tracker_tokens
        )
        tracker_value = self.tracker_cross_attention(
            tracker_query,
            tracker_context,
            tracker_context,
            key_padding_mask=~prepared_conditioning.tracker_available,
            need_weights=False,
        )[0]
        current = self.tracker_output_norm(current + tracker_value)

        diffusion_time = self.diffusion_time_embedding(timestep)
        for block in self.blocks:
            current = block(
                current,
                prepared_conditioning.temporal_context,
                diffusion_time,
            )
        current = self.output_norm(current)
        raw_output = self.joint_output(current)
        return (
            raw_output * prepared_conditioning.denoise_strength.to(raw_output)[..., None]
        ).reshape(batch_size, REALTIME_POSE_TARGET_DIM)
