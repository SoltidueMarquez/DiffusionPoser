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
    REALTIME_POSE_HISTORY_ANCHOR_COUNT,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_WINDOW_LENGTH,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
)
from model.realtime_pose_window_observation_encoder import (
    WindowObservationEncoder,
    WindowObservationEncoding,
)
from model.timestep_embedding import SinusoidalTimestepEmbedding


@dataclass
class PreparedSpatioTemporalConditioning:
    observation: WindowObservationEncoding
    static_pose_condition: torch.Tensor
    window_valid_mask: torch.Tensor
    position_attention_bias: torch.Tensor
    rotation_attention_bias: torch.Tensor
    temporal_attention_mask: torch.Tensor
    prior_gate_joint: torch.Tensor


def _modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class SpatioTemporalDiTBlock(nn.Module):
    """按 Tracker 条件、空间注意力、时间注意力的顺序更新窗口 token。"""

    def __init__(self, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = int(num_heads)
        self.conditioning_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.spatial_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.temporal_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.state_attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.position_attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.rotation_attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
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
        target: torch.Tensor,
        diffusion_time: torch.Tensor,
        prepared: PreparedSpatioTemporalConditioning,
        joint_regions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, time_count, joint_count, latent_dim = target.shape
        observation = prepared.observation
        frame_valid = prepared.window_valid_mask[..., None, None].to(target.dtype)

        query = self.conditioning_norm(target).reshape(
            batch_size * time_count, joint_count, latent_dim
        )
        state_keys = observation.state_tokens.reshape(
            batch_size * time_count, observation.state_tokens.shape[2], latent_dim
        )
        position_keys = observation.position_tokens.reshape(
            batch_size * time_count, observation.position_tokens.shape[2], latent_dim
        )
        rotation_keys = observation.rotation_tokens.reshape(
            batch_size * time_count, observation.rotation_tokens.shape[2], latent_dim
        )
        state_value = self.state_attention(
            query, state_keys, state_keys, need_weights=False
        )[0]
        position_value = self.position_attention(
            query,
            position_keys,
            position_keys,
            attn_mask=prepared.position_attention_bias,
            need_weights=False,
        )[0]
        rotation_value = self.rotation_attention(
            query,
            rotation_keys,
            rotation_keys,
            attn_mask=prepared.rotation_attention_bias,
            need_weights=False,
        )[0]
        rho_position = observation.rho_position.index_select(2, joint_regions).reshape(
            batch_size * time_count, joint_count
        )
        rho_rotation = observation.rho_rotation.index_select(2, joint_regions).reshape(
            batch_size * time_count, joint_count
        )
        target = target + (
            state_value
            + rho_position[..., None] * position_value
            + rho_rotation[..., None] * rotation_value
        ).reshape(batch_size, time_count, joint_count, latent_dim)
        target = target * frame_valid

        modulation = self.adaln_modulation(diffusion_time)
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
        ) = modulation.chunk(9, dim=-1)

        spatial = target.reshape(batch_size * time_count, joint_count, latent_dim)
        spatial_shift_bt = spatial_shift[:, None].expand(-1, time_count, -1).reshape(
            batch_size * time_count, latent_dim
        )
        spatial_scale_bt = spatial_scale[:, None].expand(-1, time_count, -1).reshape(
            batch_size * time_count, latent_dim
        )
        spatial_gate_bt = spatial_gate[:, None].expand(-1, time_count, -1).reshape(
            batch_size * time_count, latent_dim
        )
        spatial_query = _modulate(
            self.spatial_norm(spatial), spatial_shift_bt, spatial_scale_bt
        )
        spatial_value = self.spatial_attention(
            spatial_query, spatial_query, spatial_query, need_weights=False
        )[0]
        spatial = spatial + spatial_gate_bt[:, None] * spatial_value
        target = spatial.reshape(batch_size, time_count, joint_count, latent_dim) * frame_valid

        temporal = target.permute(0, 2, 1, 3).reshape(
            batch_size * joint_count, time_count, latent_dim
        )
        temporal_shift_bj = temporal_shift[:, None].expand(-1, joint_count, -1).reshape(
            batch_size * joint_count, latent_dim
        )
        temporal_scale_bj = temporal_scale[:, None].expand(-1, joint_count, -1).reshape(
            batch_size * joint_count, latent_dim
        )
        temporal_gate_bj = temporal_gate[:, None].expand(-1, joint_count, -1).reshape(
            batch_size * joint_count, latent_dim
        )
        prior_gate_bjt = prepared.prior_gate_joint.permute(0, 2, 1).reshape(
            batch_size * joint_count, time_count, 1
        )
        temporal_query = _modulate(
            self.temporal_norm(temporal), temporal_shift_bj, temporal_scale_bj
        )
        temporal_value = self.temporal_attention(
            temporal_query,
            temporal_query,
            temporal_query,
            attn_mask=prepared.temporal_attention_mask,
            need_weights=False,
        )[0]
        # 当前区域 Tracker 越可靠，历史时序残差越弱；最低保留 0.1 维持全身协调。
        temporal = temporal + (
            prior_gate_bjt * temporal_gate_bj[:, None] * temporal_value
        )
        target = temporal.reshape(
            batch_size, joint_count, time_count, latent_dim
        ).permute(0, 2, 1, 3)
        target = target * frame_valid

        mlp_input = target.reshape(batch_size, time_count * joint_count, latent_dim)
        mlp_query = _modulate(self.mlp_norm(mlp_input), mlp_shift, mlp_scale)
        target = mlp_input + mlp_gate[:, None] * self.mlp(mlp_query)
        return target.reshape(batch_size, time_count, joint_count, latent_dim) * frame_valid


class RealtimePoseSpatioTemporalDiT(nn.Module):
    """对 10 个历史锚点和当前帧联合扩散的因子化时空 DiT。"""

    def __init__(
        self,
        input_feats: int = REALTIME_POSE_TARGET_DIM,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        zero_init: bool = False,
        max_seq_len: int = REALTIME_POSE_WINDOW_LENGTH,
        reliability_config: TrackerReliabilityConfig | None = None,
    ):
        super().__init__()
        if int(input_feats) != REALTIME_POSE_TARGET_DIM:
            raise ValueError("时空 DiT 的每帧 Pose 必须为 144D。")
        if int(max_seq_len) != REALTIME_POSE_WINDOW_LENGTH:
            raise ValueError("时空 DiT 固定读取 11 个时间锚点。")
        self.input_feats = int(input_feats)
        self.output_feats = int(input_feats)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        # Runtime 直接读取模型上的同一个配置，避免 duration 与 kappa 在部署时被另一套参数解释。
        self.reliability_config = (
            reliability_config or TrackerReliabilityConfig()
        ).validate()
        self.observation_encoder = WindowObservationEncoder(
            latent_dim, self.reliability_config
        )
        self.joint_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.history_pose_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.head_path_encoder = nn.Sequential(
            nn.Linear(5, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.joint_identity = nn.Embedding(SMPL_JOINT_COUNT, latent_dim)
        self.region_identity = nn.Embedding(5, latent_dim)
        self.diffusion_time_embedding = SinusoidalTimestepEmbedding(latent_dim)
        self.frame_offset_embedding = SinusoidalTimestepEmbedding(latent_dim)
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(latent_dim, num_heads, dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.joint_output = nn.Linear(latent_dim, ROTATION_6D_DIM)
        self.future_leg_head = nn.Sequential(
            nn.Linear(latent_dim * 8, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 3 * 8 * 6),
        )
        self.contact_head = nn.Linear(latent_dim * 2, 2)
        self.register_buffer(
            "joint_regions", torch.tensor(TARGET_JOINT_REGIONS.copy(), dtype=torch.long)
        )
        self.register_buffer(
            "head_path_multipliers",
            torch.tensor(TRAJECTORY_REGION_MULTIPLIERS.copy(), dtype=torch.float32),
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
        history_pose_observation: torch.Tensor,
        tracker_window: torch.Tensor,
        head_path_window: torch.Tensor,
        history_region_confidence: torch.Tensor,
        window_valid_mask: torch.Tensor,
        frame_offsets: torch.Tensor,
    ) -> PreparedSpatioTemporalConditioning:
        batch_size = history_pose_observation.shape[0]
        if tuple(history_pose_observation.shape) != (
            batch_size,
            REALTIME_POSE_HISTORY_ANCHOR_COUNT,
            REALTIME_POSE_TARGET_DIM,
        ):
            raise ValueError("history_pose_observation 必须为 [B,10,144]。")
        if tuple(head_path_window.shape) != (
            batch_size,
            REALTIME_POSE_WINDOW_LENGTH,
            5,
        ):
            raise ValueError("head_path_window 必须为 [B,11,5]。")
        if tuple(history_region_confidence.shape) != (
            batch_size,
            REALTIME_POSE_HISTORY_ANCHOR_COUNT,
            5,
        ):
            raise ValueError("history_region_confidence 必须为 [B,10,5]。")
        if tuple(window_valid_mask.shape) != (
            batch_size,
            REALTIME_POSE_WINDOW_LENGTH,
        ):
            raise ValueError("window_valid_mask 必须为 [B,11]。")
        if not torch.all(window_valid_mask[:, -1]):
            raise ValueError("当前帧必须始终有效。")

        observation = self.observation_encoder(tracker_window, window_valid_mask)
        history = history_pose_observation.reshape(
            batch_size,
            REALTIME_POSE_HISTORY_ANCHOR_COUNT,
            SMPL_JOINT_COUNT,
            ROTATION_6D_DIM,
        )
        confidence = history_region_confidence.index_select(2, self.joint_regions)
        history_tokens = self.history_pose_input(history) * confidence[..., None]
        history_tokens = torch.cat(
            [
                history_tokens,
                history_tokens.new_zeros(
                    (batch_size, 1, SMPL_JOINT_COUNT, self.latent_dim)
                ),
            ],
            dim=1,
        )
        head_tokens = self.head_path_encoder(head_path_window)
        head_multiplier = self.head_path_multipliers.index_select(
            0, self.joint_regions
        ).to(head_tokens.dtype)
        head_tokens = head_tokens[:, :, None] * head_multiplier[None, None, :, None]

        if frame_offsets.ndim == 1:
            frame_offsets = frame_offsets[None].expand(batch_size, -1)
        if tuple(frame_offsets.shape) != (batch_size, REALTIME_POSE_WINDOW_LENGTH):
            raise ValueError("frame_offsets 必须为 [11] 或 [B,11]。")
        offset_tokens = self.frame_offset_embedding(frame_offsets.reshape(-1)).reshape(
            batch_size, REALTIME_POSE_WINDOW_LENGTH, self.latent_dim
        )
        static_condition = history_tokens + head_tokens + offset_tokens[:, :, None]
        static_condition = static_condition * window_valid_mask[..., None, None].to(
            static_condition.dtype
        )
        # 这些 mask 只依赖输入窗口，在同一轮 DDIM 的所有去噪步之间保持不变。
        position_attention_bias = self._measurement_bias(
            observation.kappa_position[:, :, NON_HEAD_TRACKER_INDICES],
            self.position_coverage[:, NON_HEAD_TRACKER_INDICES],
        )
        rotation_attention_bias = self._measurement_bias(
            observation.kappa_rotation,
            self.rotation_coverage,
        )
        temporal_attention_mask = self._temporal_mask(
            window_valid_mask.bool(),
            joint_count=SMPL_JOINT_COUNT,
        )
        prior_gate_region = torch.clamp(
            1.0 - 0.5 * (observation.rho_position + observation.rho_rotation),
            min=0.1,
        )
        prior_gate_joint = prior_gate_region.index_select(2, self.joint_regions)
        return PreparedSpatioTemporalConditioning(
            observation=observation,
            static_pose_condition=static_condition,
            window_valid_mask=window_valid_mask.bool(),
            position_attention_bias=position_attention_bias,
            rotation_attention_bias=rotation_attention_bias,
            temporal_attention_mask=temporal_attention_mask,
            prior_gate_joint=prior_gate_joint,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        history_pose_observation: Optional[torch.Tensor] = None,
        tracker_window: Optional[torch.Tensor] = None,
        head_path_window: Optional[torch.Tensor] = None,
        history_region_confidence: Optional[torch.Tensor] = None,
        window_valid_mask: Optional[torch.Tensor] = None,
        frame_offsets: Optional[torch.Tensor] = None,
        prepared_conditioning: Optional[PreparedSpatioTemporalConditioning] = None,
        y: Optional[dict] = None,
        return_aux_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = hidden_states.shape[0]
        if tuple(hidden_states.shape) != (
            batch_size,
            REALTIME_POSE_WINDOW_LENGTH,
            REALTIME_POSE_TARGET_DIM,
        ):
            raise ValueError("hidden_states 必须为 [B,11,144]。")
        values = y or {}
        if prepared_conditioning is None:
            history_pose_observation = (
                history_pose_observation
                if history_pose_observation is not None
                else values.get("history_pose_observation")
            )
            tracker_window = (
                tracker_window if tracker_window is not None else values.get("tracker_window")
            )
            head_path_window = (
                head_path_window
                if head_path_window is not None
                else values.get("head_path_window")
            )
            history_region_confidence = (
                history_region_confidence
                if history_region_confidence is not None
                else values.get("history_region_confidence")
            )
            window_valid_mask = (
                window_valid_mask
                if window_valid_mask is not None
                else values.get("window_valid_mask")
            )
            frame_offsets = (
                frame_offsets if frame_offsets is not None else values.get("frame_offsets")
            )
            required = (
                history_pose_observation,
                tracker_window,
                head_path_window,
                history_region_confidence,
                window_valid_mask,
                frame_offsets,
            )
            if any(value is None for value in required):
                raise ValueError("时空 DiT 缺少窗口条件字段。")
            prepared_conditioning = self.prepare_conditioning(*required)

        joint_values = hidden_states.reshape(
            batch_size,
            REALTIME_POSE_WINDOW_LENGTH,
            SMPL_JOINT_COUNT,
            ROTATION_6D_DIM,
        )
        joint_ids = torch.arange(SMPL_JOINT_COUNT, device=hidden_states.device)
        target = (
            self.joint_input(joint_values)
            + self.joint_identity(joint_ids)[None, None]
            + self.region_identity(self.joint_regions)[None, None]
            + prepared_conditioning.static_pose_condition
        )
        target = target * prepared_conditioning.window_valid_mask[..., None, None].to(
            target.dtype
        )
        diffusion_time = self.diffusion_time_embedding(timestep)
        for block in self.blocks:
            target = block(
                target,
                diffusion_time,
                prepared_conditioning,
                self.joint_regions,
            )
        target = self.output_norm(target)
        raw_xstart = self.joint_output(target).reshape(
            batch_size, REALTIME_POSE_WINDOW_LENGTH, REALTIME_POSE_TARGET_DIM
        )
        raw_xstart = raw_xstart * prepared_conditioning.window_valid_mask[..., None].to(
            raw_xstart.dtype
        )
        if not return_aux_outputs:
            return raw_xstart

        current_tokens = target[:, -1]
        leg_indices = torch.as_tensor(
            [1, 4, 7, 10, 2, 5, 8, 11], device=target.device
        )
        feet_indices = torch.as_tensor(
            [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]], device=target.device
        )
        auxiliary = {
            "future_leg": self.future_leg_head(
                current_tokens.index_select(1, leg_indices).flatten(1)
            ).reshape(batch_size, 3, 8, 6),
            "contact_logits": self.contact_head(
                current_tokens.index_select(1, feet_indices).flatten(1)
            ),
        }
        return raw_xstart, auxiliary

    def _measurement_bias(
        self, kappa: torch.Tensor, coverage: torch.Tensor
    ) -> torch.Tensor:
        batch_size, time_count, key_count = kappa.shape
        flat = kappa.reshape(batch_size * time_count, key_count)
        allowed = coverage.to(dtype=torch.bool).index_select(0, self.joint_regions)
        bias = torch.log(flat.clamp_min(1e-6))[:, None].expand(
            -1, SMPL_JOINT_COUNT, -1
        )
        bias = bias.masked_fill(~allowed[None], float("-inf"))
        bias = bias.masked_fill(flat[:, None] <= 0.0, float("-inf"))
        empty = ~torch.isfinite(bias).any(dim=-1)
        # 空行回退到第一个 key；保持为纯张量操作，避免 CUDA 到 CPU 的条件同步。
        bias[..., 0] = torch.where(empty, torch.zeros_like(bias[..., 0]), bias[..., 0])
        return bias[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            batch_size * time_count * self.num_heads,
            SMPL_JOINT_COUNT,
            key_count,
        )

    def _temporal_mask(
        self, window_valid_mask: torch.Tensor, joint_count: int
    ) -> torch.Tensor:
        batch_size, time_count = window_valid_mask.shape
        causal = torch.ones(
            time_count, time_count, device=window_valid_mask.device, dtype=torch.bool
        ).tril()
        allowed = causal[None] & window_valid_mask[:, None, :].bool()
        empty = ~allowed.any(dim=-1)
        # 左侧 padding query 没有可读 key 时只允许读取自身，避免全屏蔽产生 NaN。
        identity = torch.eye(
            time_count, device=window_valid_mask.device, dtype=torch.bool
        )
        allowed = allowed | (empty[..., None] & identity[None])
        blocked = ~allowed
        return blocked[:, None, None].expand(
            -1, joint_count, self.num_heads, -1, -1
        ).reshape(
            batch_size * joint_count * self.num_heads, time_count, time_count
        )
