from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from data_loaders.realtime_pose_config import (
    TARGET_JOINT_REGIONS,
    TRAJECTORY_REGION_MULTIPLIERS,
)
from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from data_loaders.sensor_masking import (
    CURRENT_JOINT_CONDITION_DIM,
    REALTIME_POSE_CONDITION_WINDOW_LENGTH,
    REALTIME_POSE_FPS,
    REALTIME_POSE_HISTORY_ANCHOR_COUNT,
    REALTIME_POSE_HISTORY_FRAME_OFFSETS,
    REALTIME_POSE_MODEL_TOKEN_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_LENGTH,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
)
from model.timestep_embedding import SinusoidalTimestepEmbedding


HEAD_MOTION_FEATURE_DIM = 14
HEAD_MOTION_ANCHOR_OFFSETS = (-1, -8, -60)
HEAD_MOTION_ANCHOR_INDICES = tuple(
    REALTIME_POSE_HISTORY_FRAME_OFFSETS.index(offset)
    for offset in HEAD_MOTION_ANCHOR_OFFSETS
)


@dataclass
class PreparedSpatioTemporalConditioning:
    static_pose_condition: torch.Tensor
    token_valid_mask: torch.Tensor
    temporal_attention_mask: torch.Tensor


def _modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return value * (1.0 + scale[:, None]) + shift[:, None]


def build_head_motion_summary(
    head_path_window: torch.Tensor,
    window_valid_mask: torch.Tensor,
    frame_offsets: torch.Tensor,
) -> torch.Tensor:
    """从稀疏 Head 路径构造 `[B,14]` 多尺度运动摘要。

    `head_path_window` 的布局为 `[x,z,height,sin(yaw),cos(yaw)]`。XZ 与高度
    已做仿射归一化，因此差分会自动消除均值；再除以真实秒数后，仍保留
    速度方向与各尺度间的物理比例，只差一个由原 normalizer 决定的固定尺度。
    """

    if head_path_window.ndim != 3 or tuple(head_path_window.shape[1:]) != (
        REALTIME_POSE_CONDITION_WINDOW_LENGTH,
        5,
    ):
        raise ValueError("head_path_window 必须为 [B,11,5]。")
    batch_size = head_path_window.shape[0]
    if tuple(window_valid_mask.shape) != (
        batch_size,
        REALTIME_POSE_CONDITION_WINDOW_LENGTH,
    ):
        raise ValueError("window_valid_mask 必须为 [B,11]。")
    if frame_offsets.ndim == 1:
        frame_offsets = frame_offsets[None].expand(batch_size, -1)
    if tuple(frame_offsets.shape) != (batch_size, REALTIME_POSE_MODEL_TOKEN_LENGTH):
        raise ValueError("frame_offsets 必须为 [21] 或 [B,21]。")

    condition_offsets = frame_offsets[:, :REALTIME_POSE_CONDITION_WINDOW_LENGTH]
    current_index = REALTIME_POSE_CONDITION_WINDOW_LENGTH - 1
    expected_offsets = (*HEAD_MOTION_ANCHOR_OFFSETS, 0)
    actual_offsets = torch.stack(
        [
            condition_offsets[:, anchor_index]
            for anchor_index in HEAD_MOTION_ANCHOR_INDICES
        ]
        + [condition_offsets[:, current_index]],
        dim=-1,
    )
    expected = actual_offsets.new_tensor(expected_offsets)[None].expand_as(actual_offsets)
    if not bool(torch.equal(actual_offsets, expected)):
        raise ValueError("Head 运动摘要要求 frame_offsets 包含固定的 -1/-8/-60/0 锚点。")

    current = head_path_window[:, current_index]
    anchors = head_path_window[:, HEAD_MOTION_ANCHOR_INDICES]
    scale_valid = (
        window_valid_mask[:, current_index, None]
        & window_valid_mask[:, HEAD_MOTION_ANCHOR_INDICES]
    )
    scale_weight = scale_valid.to(dtype=head_path_window.dtype)
    delta_seconds = (
        condition_offsets[:, current_index, None]
        - condition_offsets[:, HEAD_MOTION_ANCHOR_INDICES]
    ).to(dtype=head_path_window.dtype) / float(REALTIME_POSE_FPS)

    # `[B,3,2]` 依次对应 1/8/60 帧尺度；无效历史必须在进入 adapter 前清零。
    horizontal_velocity = (
        current[:, None, :2] - anchors[:, :, :2]
    ) / delta_seconds[..., None]
    vertical_velocity = (current[:, None, 2] - anchors[:, :, 2]) / delta_seconds

    current_sin = current[:, None, 3]
    current_cos = current[:, None, 4]
    anchor_sin = anchors[:, :, 3]
    anchor_cos = anchors[:, :, 4]
    sin_delta = current_sin * anchor_cos - current_cos * anchor_sin
    cos_delta = current_cos * anchor_cos + current_sin * anchor_sin
    yaw_rate = torch.atan2(sin_delta, cos_delta) / delta_seconds / math.pi

    horizontal_velocity = horizontal_velocity * scale_weight[..., None]
    vertical_velocity = vertical_velocity * scale_weight
    yaw_rate = yaw_rate * scale_weight
    return torch.cat(
        [
            horizontal_velocity[:, 0],
            horizontal_velocity[:, 1],
            horizontal_velocity[:, 2],
            yaw_rate[:, 1:3],
            current[:, 2:3],
            vertical_velocity[:, 1:3],
            scale_weight,
        ],
        dim=-1,
    )


class SpatioTemporalDiTBlock(nn.Module):
    """按空间注意力、时间注意力的顺序更新窗口 token。"""

    def __init__(self, latent_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = int(num_heads)
        self.spatial_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.temporal_norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
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
        target: torch.Tensor,
        diffusion_time: torch.Tensor,
        prepared: PreparedSpatioTemporalConditioning,
    ) -> torch.Tensor:
        batch_size, time_count, joint_count, latent_dim = target.shape
        frame_valid = prepared.token_valid_mask[..., None, None].to(target.dtype)
        # 这里只用二值有效性清除为防止 softmax NaN 而临时开放的 padding 输出。
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
        # 时序残差强度只由可学习 gate 决定。
        temporal = temporal + temporal_gate_bj[:, None] * temporal_value
        target = temporal.reshape(
            batch_size, joint_count, time_count, latent_dim
        ).permute(0, 2, 1, 3)
        target = target * frame_valid

        mlp_input = target.reshape(batch_size, time_count * joint_count, latent_dim)
        mlp_query = _modulate(self.mlp_norm(mlp_input), mlp_shift, mlp_scale)
        target = mlp_input + mlp_gate[:, None] * self.mlp(mlp_query)
        return target.reshape(batch_size, time_count, joint_count, latent_dim) * frame_valid


class RealtimePoseSpatioTemporalDiT(nn.Module):
    """以 10 个历史条件为前缀，联合恢复当前帧和未来 10 帧的时空 DiT。"""

    def __init__(
        self,
        input_feats: int = REALTIME_POSE_TARGET_DIM,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.0,
        zero_init: bool = False,
        max_seq_len: int = REALTIME_POSE_MODEL_TOKEN_LENGTH,
    ):
        super().__init__()
        if int(input_feats) != REALTIME_POSE_TARGET_DIM:
            raise ValueError("时空 DiT 的每帧 Pose 必须为 144D。")
        if int(max_seq_len) != REALTIME_POSE_MODEL_TOKEN_LENGTH:
            raise ValueError(
                "max_seq_len 必须为 21；单帧 checkpoint 与联合 11 帧模型不兼容。"
            )
        self.input_feats = int(input_feats)
        self.output_feats = int(input_feats)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.joint_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        self.history_pose_input = nn.Linear(ROTATION_6D_DIM, latent_dim)
        # 当前 Tracker 条件已与 24 个关节对齐，只需一次线性投影到 token 空间。
        # 它不新增 token 或 attention，也不会重复编码已由 IK 注入的 rotation。
        self.current_joint_condition_input = nn.Linear(
            CURRENT_JOINT_CONDITION_DIM,
            latent_dim,
        )
        self.head_path_encoder = nn.Sequential(
            nn.Linear(5, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim)
        )
        # Head 的完整轨迹仍走原路径；这里只把可解释的速度/高度摘要直达 pelvis。
        self.pelvis_head_motion_input = nn.Linear(
            HEAD_MOTION_FEATURE_DIM,
            latent_dim,
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
        self.contact_head = nn.Linear(latent_dim * 2, 2)
        self.register_buffer(
            "joint_regions", torch.tensor(TARGET_JOINT_REGIONS.copy(), dtype=torch.long)
        )
        self.register_buffer(
            "head_path_multipliers",
            torch.tensor(TRAJECTORY_REGION_MULTIPLIERS.copy(), dtype=torch.float32),
        )
        # 持久化结构标记，使旧单帧权重不能被 strict=False 静默接受。
        self.register_buffer(
            "joint_diffusion_horizon_length",
            torch.tensor(REALTIME_POSE_TARGET_LENGTH, dtype=torch.long),
        )
        if zero_init:
            nn.init.zeros_(self.joint_output.weight)
            nn.init.zeros_(self.joint_output.bias)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def prepare_conditioning(
        self,
        history_pose_observation: torch.Tensor,
        head_path_window: torch.Tensor,
        history_region_confidence: torch.Tensor,
        window_valid_mask: torch.Tensor,
        frame_offsets: torch.Tensor,
        current_joint_condition: torch.Tensor,
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
            REALTIME_POSE_CONDITION_WINDOW_LENGTH,
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
            REALTIME_POSE_CONDITION_WINDOW_LENGTH,
        ):
            raise ValueError("window_valid_mask 必须为 [B,11]。")
        if not torch.all(window_valid_mask[:, -1]):
            raise ValueError("当前帧必须始终有效。")
        if tuple(current_joint_condition.shape) != (
            batch_size,
            SMPL_JOINT_COUNT,
            CURRENT_JOINT_CONDITION_DIM,
        ):
            raise ValueError("current_joint_condition 必须为 [B,24,10]。")
        if not bool(torch.isfinite(current_joint_condition).all()):
            raise ValueError("current_joint_condition 必须为有限数值。")

        history = history_pose_observation.reshape(
            batch_size,
            REALTIME_POSE_HISTORY_ANCHOR_COUNT,
            SMPL_JOINT_COUNT,
            ROTATION_6D_DIM,
        )
        # 历史槽位是否存在由 window_valid_mask 统一控制，避免冷启动 padding 泄漏。
        history_tokens = self.history_pose_input(history)
        history_tokens = torch.cat(
            [
                history_tokens,
                history_tokens.new_zeros(
                    (
                        batch_size,
                        REALTIME_POSE_TARGET_LENGTH,
                        SMPL_JOINT_COUNT,
                        self.latent_dim,
                    )
                ),
            ],
            dim=1,
        )
        if frame_offsets.ndim == 1:
            frame_offsets = frame_offsets[None].expand(batch_size, -1)
        if tuple(frame_offsets.shape) != (
            batch_size,
            REALTIME_POSE_MODEL_TOKEN_LENGTH,
        ):
            raise ValueError("frame_offsets 必须为 [21] 或 [B,21]。")
        head_motion_summary = build_head_motion_summary(
            head_path_window,
            window_valid_mask,
            frame_offsets,
        )
        head_tokens = self.head_path_encoder(head_path_window)
        head_multiplier = self.head_path_multipliers.index_select(
            0, self.joint_regions
        ).to(head_tokens.dtype)
        head_tokens = head_tokens[:, :, None] * head_multiplier[None, None, :, None]
        head_tokens = torch.cat(
            [
                head_tokens,
                head_tokens.new_zeros(
                    (
                        batch_size,
                        REALTIME_POSE_MODEL_TOKEN_LENGTH
                        - REALTIME_POSE_CONDITION_WINDOW_LENGTH,
                        SMPL_JOINT_COUNT,
                        self.latent_dim,
                    )
                ),
            ],
            dim=1,
        )

        offset_tokens = self.frame_offset_embedding(frame_offsets.reshape(-1)).reshape(
            batch_size, REALTIME_POSE_MODEL_TOKEN_LENGTH, self.latent_dim
        )
        static_condition = history_tokens + head_tokens + offset_tokens[:, :, None]
        # 10 个历史槽之后的第一个目标槽就是 current。未来条件不直接复制，
        # 后续 10 帧只能通过 temporal self-attention 读取当前 Tracker 影响。
        current_index = REALTIME_POSE_HISTORY_ANCHOR_COUNT
        static_condition[:, current_index] = (
            static_condition[:, current_index]
            + self.current_joint_condition_input(current_joint_condition)
        )
        pelvis_index = JOINT_INDEX["pelvis"]
        static_condition[:, current_index, pelvis_index] = (
            static_condition[:, current_index, pelvis_index]
            + self.pelvis_head_motion_input(head_motion_summary)
        )
        target_valid = torch.ones(
            batch_size,
            REALTIME_POSE_TARGET_LENGTH,
            device=window_valid_mask.device,
            dtype=torch.bool,
        )
        token_valid_mask = torch.cat(
            [window_valid_mask[:, :-1].bool(), target_valid], dim=1
        )
        static_condition = static_condition * token_valid_mask[..., None, None].to(
            static_condition.dtype
        )
        temporal_attention_mask = self._temporal_mask(
            token_valid_mask,
            joint_count=SMPL_JOINT_COUNT,
        )
        return PreparedSpatioTemporalConditioning(
            static_pose_condition=static_condition,
            token_valid_mask=token_valid_mask,
            temporal_attention_mask=temporal_attention_mask,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        history_pose_observation: Optional[torch.Tensor] = None,
        head_path_window: Optional[torch.Tensor] = None,
        history_region_confidence: Optional[torch.Tensor] = None,
        window_valid_mask: Optional[torch.Tensor] = None,
        frame_offsets: Optional[torch.Tensor] = None,
        current_joint_condition: Optional[torch.Tensor] = None,
        prepared_conditioning: Optional[PreparedSpatioTemporalConditioning] = None,
        y: Optional[dict] = None,
        return_aux_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = hidden_states.shape[0]
        if tuple(hidden_states.shape) != (
            batch_size,
            REALTIME_POSE_TARGET_LENGTH,
            REALTIME_POSE_TARGET_DIM,
        ):
            raise ValueError("hidden_states 必须为联合扩散状态 [B,11,144]。")
        values = y or {}
        if prepared_conditioning is None:
            history_pose_observation = (
                history_pose_observation
                if history_pose_observation is not None
                else values.get("history_pose_observation")
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
            current_joint_condition = (
                current_joint_condition
                if current_joint_condition is not None
                else values.get("current_joint_condition")
            )
            required = (
                history_pose_observation,
                head_path_window,
                history_region_confidence,
                window_valid_mask,
                frame_offsets,
                current_joint_condition,
            )
            if any(value is None for value in required):
                raise ValueError("时空 DiT 缺少窗口条件字段。")
            prepared_conditioning = self.prepare_conditioning(*required)

        joint_values = hidden_states.reshape(
            batch_size,
            REALTIME_POSE_TARGET_LENGTH,
            SMPL_JOINT_COUNT,
            ROTATION_6D_DIM,
        )
        joint_ids = torch.arange(SMPL_JOINT_COUNT, device=hidden_states.device)
        target_tokens = self.joint_input(joint_values)
        # 历史槽位只读取独立历史条件；最后 11 个槽位分别读取当前和未来带噪状态。
        diffusion_tokens = torch.cat(
            [
                target_tokens.new_zeros(
                    (
                        batch_size,
                        REALTIME_POSE_HISTORY_ANCHOR_COUNT,
                        SMPL_JOINT_COUNT,
                        self.latent_dim,
                    )
                ),
                target_tokens,
            ],
            dim=1,
        )
        target = (
            diffusion_tokens
            + self.joint_identity(joint_ids)[None, None]
            + self.region_identity(self.joint_regions)[None, None]
            + prepared_conditioning.static_pose_condition
        )
        target = target * prepared_conditioning.token_valid_mask[..., None, None].to(
            target.dtype
        )
        diffusion_time = self.diffusion_time_embedding(timestep)
        for block in self.blocks:
            target = block(
                target,
                diffusion_time,
                prepared_conditioning,
            )
        target = self.output_norm(target)
        target_tokens = target[:, REALTIME_POSE_HISTORY_ANCHOR_COUNT :]
        raw_xstart = self.joint_output(target_tokens).reshape(
            batch_size, REALTIME_POSE_TARGET_LENGTH, REALTIME_POSE_TARGET_DIM
        )
        if not return_aux_outputs:
            return raw_xstart

        feet_indices = torch.as_tensor(
            [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]], device=target.device
        )
        auxiliary = {
            "contact_logits": self.contact_head(
                target_tokens[:, 0].index_select(1, feet_indices).flatten(1)
            ),
        }
        return raw_xstart, auxiliary

    def _temporal_mask(
        self, token_valid_mask: torch.Tensor, joint_count: int
    ) -> torch.Tensor:
        batch_size, time_count = token_valid_mask.shape
        if time_count != REALTIME_POSE_MODEL_TOKEN_LENGTH:
            raise ValueError("Temporal Attention 固定使用 10 个历史槽和 11 个目标槽。")
        # 行为 query、列为 key：历史内部双向可见但不能读取任何带噪目标；
        # 所有目标 query 可读取有效历史和完整目标窗口，从而实现联合去噪。
        role_allowed = torch.ones(
            time_count, time_count, device=token_valid_mask.device, dtype=torch.bool
        )
        role_allowed[
            :REALTIME_POSE_HISTORY_ANCHOR_COUNT,
            REALTIME_POSE_HISTORY_ANCHOR_COUNT:,
        ] = False
        query_valid = token_valid_mask[:, :, None].bool()
        key_valid = token_valid_mask[:, None, :].bool()
        allowed = role_allowed[None] & query_valid & key_valid
        empty = ~allowed.any(dim=-1)
        # padding query 没有合法 key，只临时开放自身防止 Softmax 全屏蔽；
        # 其 residual state 会在 block 边界再次清零，不会写入任何有效 token。
        identity = torch.eye(
            time_count, device=token_valid_mask.device, dtype=torch.bool
        )
        allowed = allowed | (empty[..., None] & identity[None])
        blocked = ~allowed
        return blocked[:, None, None].expand(
            -1, joint_count, self.num_heads, -1, -1
        ).reshape(
            batch_size * joint_count * self.num_heads, time_count, time_count
        )
