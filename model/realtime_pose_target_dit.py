from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from data_loaders.realtime_pose_config import (
    POSITION_COVERAGE,
    ROTATION_COVERAGE,
    TARGET_JOINT_REGIONS,
    TRAJECTORY_REGION_MULTIPLIERS,
    TrackerReliabilityConfig,
)
from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from data_loaders.sensor_masking import (
    NON_HEAD_TRACKER_INDICES,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
)
from model.timestep_embedding import SinusoidalTimestepEmbedding
from model.realtime_pose_motion_encoder import MotionEncoding, RegionalMotionEncoder
from model.realtime_pose_observation_encoder import DynamicObservationEncoder, ObservationEncoding


@dataclass
class PreparedRealtimeConditioning:
    observation: ObservationEncoding
    motion: MotionEncoding
    trajectory_token: torch.Tensor


def apply_adaln_modulation(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """用 `[B,D]` 的时间参数调制 `[B,J,D]` token。"""

    return value * (1.0 + scale[:, None]) + shift[:, None]


class RegionAdaptiveDiTBlock(nn.Module):
    """融合区域条件，并用标准六参数 AdaLN-Zero 更新 self-attention 和 MLP。"""

    def __init__(self, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = int(num_heads)
        self.conditioning_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.self_attention_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.context_attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.position_attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.rotation_attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.prior_attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attention = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        # 扩散时间一次生成 self-attention 和 MLP 各自的 shift/scale/gate，共 6 组 D 维参数。
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim * 6),
        )
        nn.init.zeros_(self.adaln_modulation[-1].weight)
        nn.init.zeros_(self.adaln_modulation[-1].bias)
        self.trajectory_modulation = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, latent_dim * 2))

    def forward(
        self,
        target: torch.Tensor,
        time_embedding: torch.Tensor,
        prepared: PreparedRealtimeConditioning,
        position_bias: torch.Tensor,
        rotation_bias: torch.Tensor,
        prior_bias: torch.Tensor,
        joint_regions: torch.Tensor,
        trajectory_multipliers: torch.Tensor,
    ) -> torch.Tensor:
        observation = prepared.observation
        context = torch.cat([observation.state_tokens, observation.history_summary], dim=1)
        query = self.conditioning_norm(target)
        target = target + self.context_attention(query, context, context, need_weights=False)[0]

        pos_value = self.position_attention(
            self.conditioning_norm(target),
            observation.position_tokens,
            observation.position_tokens,
            attn_mask=position_bias,
            need_weights=False,
        )[0]
        rot_value = self.rotation_attention(
            self.conditioning_norm(target),
            observation.rotation_tokens,
            observation.rotation_tokens,
            attn_mask=rotation_bias,
            need_weights=False,
        )[0]
        rho_pos = observation.rho_position.index_select(1, joint_regions)
        rho_rot = observation.rho_rotation.index_select(1, joint_regions)
        target = target + rho_pos[..., None] * pos_value + rho_rot[..., None] * rot_value

        # 每个 region 同时提供 60 个 past-only temporal token 和 1 个汇总 latent。
        prior_keys = torch.cat(
            [prepared.motion.temporal_tokens, prepared.motion.latents[:, :, None]],
            dim=2,
        ).flatten(1, 2)
        prior_value = self.prior_attention(
            self.conditioning_norm(target),
            prior_keys,
            prior_keys,
            attn_mask=prior_bias,
            need_weights=False,
        )[0]
        prior_gate = torch.clamp(1.0 - 0.5 * (rho_pos + rho_rot), min=0.1)
        target = target + prior_gate[..., None] * prior_value

        trajectory_shift, trajectory_scale = self.trajectory_modulation(
            prepared.trajectory_token[:, 0]
        ).chunk(2, dim=-1)
        multiplier = trajectory_multipliers.index_select(0, joint_regions)[None, :, None]
        target = target + multiplier * (
            trajectory_shift[:, None] + trajectory_scale[:, None] * self.conditioning_norm(target)
        )

        (
            shift_self_attention,
            scale_self_attention,
            gate_self_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_modulation(time_embedding).chunk(6, dim=-1)
        self_query = apply_adaln_modulation(
            self.self_attention_norm(target),
            shift_self_attention,
            scale_self_attention,
        )
        self_value = self.self_attention(self_query, self_query, self_query, need_weights=False)[0]
        target = target + gate_self_attention[:, None] * self_value
        mlp_input = apply_adaln_modulation(
            self.mlp_norm(target),
            shift_mlp,
            scale_mlp,
        )
        return target + gate_mlp[:, None] * self.mlp(mlp_input)


class RealtimePoseTargetDiT(nn.Module):
    """可靠性感知的 24-joint TargetDiT；模型只预测 raw 144D x0。"""

    def __init__(
        self,
        input_feats: int = REALTIME_POSE_TARGET_DIM,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        zero_init: bool = False,
        max_seq_len: int = 61,
        motion_layers: int = 4,
        reliability_config: TrackerReliabilityConfig | None = None,
    ):
        super().__init__()
        if int(input_feats) != REALTIME_POSE_TARGET_DIM or int(max_seq_len) != 61:
            raise ValueError("TargetDiT 固定使用 144D 目标、60 帧历史和 1 帧当前观测。")
        self.input_feats = int(input_feats)
        self.output_feats = int(input_feats)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.observation_encoder = DynamicObservationEncoder(latent_dim, reliability_config)
        self.motion_encoder = RegionalMotionEncoder(latent_dim, motion_layers, num_heads, dropout)
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(5, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.joint_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.joint_identity = nn.Embedding(SMPL_JOINT_COUNT, latent_dim)
        self.region_identity = nn.Embedding(5, latent_dim)
        self.time_embedding = SinusoidalTimestepEmbedding(latent_dim)
        self.blocks = nn.ModuleList(
            [RegionAdaptiveDiTBlock(latent_dim, num_heads, dropout) for _ in range(int(num_layers))]
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.joint_output = nn.Linear(latent_dim, ROTATION_6D_DIM)
        self.future_leg_head = nn.Sequential(
            nn.Linear(latent_dim * 8, latent_dim), nn.SiLU(), nn.Linear(latent_dim, 3 * 8 * 6)
        )
        self.contact_head = nn.Linear(latent_dim * 2, 2)
        self.register_buffer("joint_regions", torch.tensor(TARGET_JOINT_REGIONS.copy(), dtype=torch.long))
        self.register_buffer(
            "trajectory_multipliers", torch.tensor(TRAJECTORY_REGION_MULTIPLIERS.copy(), dtype=torch.float32)
        )
        self.register_buffer(
            "position_coverage", torch.tensor(POSITION_COVERAGE.copy(), dtype=torch.float32)
        )
        self.register_buffer(
            "rotation_coverage", torch.tensor(ROTATION_COVERAGE.copy(), dtype=torch.float32)
        )
        if zero_init:
            nn.init.zeros_(self.joint_output.weight)
            nn.init.zeros_(self.joint_output.bias)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def prepare_conditioning(
        self,
        pose_history: torch.Tensor,
        tracker_history: torch.Tensor,
        current_tracker: torch.Tensor,
        trajectory_history: torch.Tensor,
        current_trajectory: torch.Tensor,
        valid_frame_mask: torch.Tensor,
    ) -> PreparedRealtimeConditioning:
        observation = self.observation_encoder(tracker_history, current_tracker, valid_frame_mask)
        motion = self.motion_encoder(
            pose_history,
            observation.history_summary,
            trajectory_history,
            valid_frame_mask,
        )
        if tuple(current_trajectory.shape) != (pose_history.shape[0], 1, 5):
            raise ValueError("current_trajectory 必须为 [B,1,5]。")
        return PreparedRealtimeConditioning(
            observation=observation,
            motion=motion,
            trajectory_token=self.trajectory_encoder(current_trajectory),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        pose_history: Optional[torch.Tensor] = None,
        tracker_history: Optional[torch.Tensor] = None,
        current_tracker: Optional[torch.Tensor] = None,
        trajectory_history: Optional[torch.Tensor] = None,
        current_trajectory: Optional[torch.Tensor] = None,
        valid_frame_mask: Optional[torch.Tensor] = None,
        prepared_conditioning: Optional[PreparedRealtimeConditioning] = None,
        y: Optional[dict] = None,
        return_aux_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del kwargs
        if hidden_states.ndim != 2 or hidden_states.shape[1] != REALTIME_POSE_TARGET_DIM:
            raise ValueError("hidden_states 必须为 [B,144]。")
        values = y or {}
        if prepared_conditioning is None:
            pose_history = pose_history if pose_history is not None else values.get("pose_history")
            tracker_history = tracker_history if tracker_history is not None else values.get("tracker_history")
            current_tracker = current_tracker if current_tracker is not None else values.get("current_tracker")
            trajectory_history = trajectory_history if trajectory_history is not None else values.get("trajectory_history")
            current_trajectory = current_trajectory if current_trajectory is not None else values.get("current_trajectory")
            valid_frame_mask = valid_frame_mask if valid_frame_mask is not None else values.get("valid_frame_mask")
            required = (pose_history, tracker_history, current_tracker, trajectory_history, current_trajectory, valid_frame_mask)
            if any(value is None for value in required):
                raise ValueError("TargetDiT 缺少新动态观测契约字段。")
            prepared_conditioning = self.prepare_conditioning(*required)

        batch_size = hidden_states.shape[0]
        joint_values = hidden_states.reshape(batch_size, SMPL_JOINT_COUNT, ROTATION_6D_DIM)
        joint_ids = torch.arange(SMPL_JOINT_COUNT, device=hidden_states.device)
        target = (
            self.joint_input(joint_values)
            + self.joint_identity(joint_ids)[None]
            + self.region_identity(self.joint_regions)[None]
        )
        position_bias = self._measurement_bias(
            prepared_conditioning.observation.kappa_position[:, NON_HEAD_TRACKER_INDICES],
            self.position_coverage[:, NON_HEAD_TRACKER_INDICES],
        )
        rotation_bias = self._measurement_bias(
            prepared_conditioning.observation.kappa_rotation,
            self.rotation_coverage,
        )
        prior_bias = self._prior_bias(
            batch_size,
            target.dtype,
            target.device,
            prepared_conditioning.motion.valid_frame_mask,
        )
        time_embedding = self.time_embedding(timestep)
        for block in self.blocks:
            target = block(
                target,
                time_embedding,
                prepared_conditioning,
                position_bias,
                rotation_bias,
                prior_bias,
                self.joint_regions,
                self.trajectory_multipliers.to(target.dtype),
            )
        target = self.output_norm(target)
        raw_xstart = self.joint_output(target).reshape(batch_size, REALTIME_POSE_TARGET_DIM)
        if not return_aux_outputs:
            return raw_xstart
        leg_indices = torch.as_tensor([1, 4, 7, 10, 2, 5, 8, 11], device=target.device)
        feet_indices = torch.as_tensor([JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]], device=target.device)
        auxiliary = {
            "future_leg": self.future_leg_head(target.index_select(1, leg_indices).flatten(1)).reshape(
                batch_size, 3, 8, 6
            ),
            "contact_logits": self.contact_head(target.index_select(1, feet_indices).flatten(1)),
        }
        return raw_xstart, auxiliary

    def _measurement_bias(self, kappa: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        """返回 `[B*H,24,K]`，零可靠性或区域不覆盖的 key 使用 -inf。"""

        batch_size, key_count = kappa.shape
        coverage = coverage.to(dtype=kappa.dtype)
        allowed = coverage.index_select(0, self.joint_regions).bool()  # [24,K]
        log_kappa = torch.log(kappa.clamp_min(1e-6))[:, None, :].expand(-1, SMPL_JOINT_COUNT, -1)
        bias = log_kappa.masked_fill(~allowed[None], float("-inf"))
        bias = bias.masked_fill(kappa[:, None, :] <= 0.0, float("-inf"))
        # MultiheadAttention 不接受一整行全 -inf；该行临时开放零 token，输出随后被 rho gate 清零。
        empty = ~torch.isfinite(bias).any(dim=-1)
        if torch.any(empty):
            bias = bias.clone()
            batch_index, joint_index = torch.nonzero(empty, as_tuple=True)
            bias[batch_index, joint_index, 0] = 0.0
        return bias[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            batch_size * self.num_heads, SMPL_JOINT_COUNT, key_count
        )

    def _prior_bias(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        valid_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        routes = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 0, 1],
            ],
            dtype=torch.bool,
            device=device,
        )
        allowed_regions = routes.index_select(0, self.joint_regions)
        allowed = allowed_regions[:, :, None].expand(-1, -1, REALTIME_POSE_HISTORY_LENGTH + 1).flatten(1)
        temporal_valid = valid_frame_mask[:, None].expand(-1, 4, -1)
        key_valid = torch.cat(
            [
                temporal_valid,
                torch.ones(batch_size, 4, 1, dtype=torch.bool, device=device),
            ],
            dim=2,
        ).flatten(1)
        bias = torch.zeros(batch_size, SMPL_JOINT_COUNT, allowed.shape[1], dtype=dtype, device=device)
        bias.masked_fill_(~allowed[None], float("-inf"))
        bias.masked_fill_(~key_valid[:, None], float("-inf"))
        return bias[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            batch_size * self.num_heads, SMPL_JOINT_COUNT, allowed.shape[1]
        )
