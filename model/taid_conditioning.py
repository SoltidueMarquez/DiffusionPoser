from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from data_loaders.realtime_pose_config import (
    TAID_ABLATIONS,
    TAID_PRIOR_TRACKER_AGGREGATIONS,
    TARGET_JOINT_REGIONS,
    TaIDConfig,
)
from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_torch,
    reexpress_previous_position_residual_torch,
    resolve_root_head_reference_torch,
    so3_log_map_torch,
)
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    HIP_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    RIGHT_FOOT_TRACKER_INDEX,
    RIGHT_HAND_TRACKER_INDEX,
    ROTATION_6D_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_COUNT,
    TRACKER_DURATION_CAP,
    TRACKER_TO_JOINT,
)
from data_loaders.tracker_roles import (
    ANCHOR_REGION_COVERAGE,
    TrackerRoleStateTorch,
    compute_tracker_roles_torch,
)
from model.realtime_pose_observation_encoder import ObservationEncoding


@dataclass
class AnchorPriorOutput:
    pose_model: torch.Tensor
    pose_raw: torch.Tensor
    root_head: torch.Tensor
    contact_logits: torch.Tensor
    contact_probability: torch.Tensor
    joint_velocity_head: torch.Tensor
    region_coverage: torch.Tensor
    tracker_tokens: torch.Tensor
    joints_head: torch.Tensor
    tracker_positions_head: torch.Tensor
    tracker_rotations_head: torch.Tensor


@dataclass
class PreparedTaIDConditioning:
    role_state: TrackerRoleStateTorch
    prior: AnchorPriorOutput
    observation_weight: torch.Tensor
    innovation_residual: torch.Tensor
    innovation_delta: torch.Tensor
    innovation_tokens: torch.Tensor
    region_injection: torch.Tensor
    joint_condition: torch.Tensor


class FixedSlotAnchorProjection(nn.Module):
    """按固定 Tracker 顺序把 `[B,6,D]` 槽位投影回 `[B,D]`。"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        # 不调用随机初始化：六个单位矩阵使初始输出严格等价于槽位求和，
        # 也不会改变同 seed 下其余 Prior 参数的初始化随机序列。
        initial_weight = torch.eye(self.latent_dim).repeat(1, TRACKER_COUNT)
        self.weight = nn.Parameter(initial_weight)

    def forward(self, normalized_anchor_slots: torch.Tensor) -> torch.Tensor:
        expected = (TRACKER_COUNT, self.latent_dim)
        if tuple(normalized_anchor_slots.shape[1:]) != expected:
            raise ValueError(
                "normalized_anchor_slots 应为 [B,6,D]，"
                f"实际为 {tuple(normalized_anchor_slots.shape)}"
            )
        flattened = normalized_anchor_slots.reshape(normalized_anchor_slots.shape[0], -1)
        return torch.matmul(flattened, self.weight.t())


class AnchorPriorRegressor(nn.Module):
    """只融合 deployed pose history、Head 与乘过 alpha 的独立 Tracker token。"""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.history_frame_encoder = nn.Linear(REALTIME_POSE_TARGET_DIM, self.latent_dim)
        self.history_gru = nn.GRU(self.latent_dim, self.latent_dim, batch_first=True)
        self.tracker_fusion = nn.Sequential(
            nn.Linear(self.latent_dim * 4, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.coverage_encoder = nn.Sequential(
            nn.Linear(5, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.latent_dim * 4, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, REALTIME_POSE_TARGET_DIM),
        )
        self.root_head = nn.Sequential(
            nn.Linear(self.latent_dim, max(32, self.latent_dim // 2)),
            nn.SiLU(),
            nn.Linear(max(32, self.latent_dim // 2), 3),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(self.latent_dim, max(32, self.latent_dim // 4)),
            nn.SiLU(),
            nn.Linear(max(32, self.latent_dim // 4), 2),
        )
        self.joint_velocity_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, SMPL_JOINT_COUNT * 3),
        )
        self.anchor_slot_projection = FixedSlotAnchorProjection(self.latent_dim)
        # Prior 从上一 deployed pose 做残差预测；零初始化保证接入初期不会随机
        # 破坏历史中心，Root/contact 则从中性输出开始学习。
        nn.init.zeros_(self.pose_head[-1].weight)
        nn.init.zeros_(self.pose_head[-1].bias)
        nn.init.zeros_(self.root_head[-1].weight)
        nn.init.zeros_(self.root_head[-1].bias)
        nn.init.zeros_(self.contact_head[-1].weight)
        nn.init.zeros_(self.contact_head[-1].bias)
        nn.init.zeros_(self.joint_velocity_head[-1].weight)
        nn.init.zeros_(self.joint_velocity_head[-1].bias)

    def forward(
        self,
        pose_history: torch.Tensor,
        valid_frame_mask: torch.Tensor,
        observation: ObservationEncoding,
        trajectory_token: torch.Tensor,
        role_state: TrackerRoleStateTorch,
        current_tracker_raw: torch.Tensor,
        joint_offsets_parent: torch.Tensor,
        pose_mean: torch.Tensor | None,
        pose_std: torch.Tensor | None,
    ) -> AnchorPriorOutput:
        batch_size = pose_history.shape[0]
        history_summary, last_pose_model = self._encode_history(
            pose_history,
            valid_frame_mask,
            pose_mean,
            pose_std,
        )
        position_tokens = torch.zeros(
            batch_size,
            TRACKER_COUNT,
            self.latent_dim,
            device=pose_history.device,
            dtype=pose_history.dtype,
        )
        position_tokens[:, 1:] = observation.position_tokens
        tracker_tokens = self.tracker_fusion(
            torch.cat(
                [
                    observation.state_tokens,
                    position_tokens,
                    observation.rotation_tokens,
                    observation.history_summary,
                ],
                dim=-1,
            )
        )
        # 每个 Tracker 先独立融合当前观测与其60帧历史，再用连续 alpha 门控。
        # 因此 U/M/未配置 Tracker 的 current/history 均不会泄漏进跨 Tracker 聚合。
        anchor_tokens = tracker_tokens * role_state.alpha[..., None].to(tracker_tokens.dtype)
        anchor_denominator = role_state.alpha.sum(dim=1, keepdim=True).clamp_min(1.0)
        # 固定顺序为 Head/LHand/RHand/Hip/LFoot/RFoot。先按旧公式归一化，
        # 再保留六个槽位进入可训练投影；初始化时与原加权平均逐元素相同。
        normalized_anchor_slots = anchor_tokens / anchor_denominator[..., None]
        anchor_summary = self.anchor_slot_projection(normalized_anchor_slots)
        coverage_token = self.coverage_encoder(role_state.region_coverage.to(pose_history.dtype))
        fused = self.fusion(
            torch.cat(
                [history_summary, anchor_summary, trajectory_token[:, 0], coverage_token],
                dim=-1,
            )
        )
        pose_model = last_pose_model + self.pose_head(fused)
        pose_raw = _inverse_pose(pose_model, pose_mean, pose_std)
        root_xyz_head = self.root_head(fused)
        contact_logits = self.contact_head(fused)
        joint_velocity_head = self.joint_velocity_head(fused).reshape(
            batch_size, SMPL_JOINT_COUNT, 3
        )
        rotations, pose_root_yaw_head = decode_target_head_rotations_torch(pose_raw)
        # 144D pose 是 runtime 最终消费的唯一姿态输出，因此 Root yaw 必须由
        # 其中的 Pelvis forward heading 派生。Root MLP 只预测 xyz，避免训练时
        # 另一个独立 yaw head 看似准确、部署时 pose heading 却落入相反 π 模态。
        root_head = torch.cat([root_xyz_head, pose_root_yaw_head[:, None]], dim=-1)
        base_root, _, joints = resolve_root_head_reference_torch(
            rotations,
            root_head[:, 3],
            joint_offsets_parent,
            observed_head_height=current_tracker_raw[:, HEAD_TRACKER_INDEX, 1],
        )
        # Root head 是 Actor Root 的 C_n 内部状态；把解析式 FK 平移到该预测 Root，
        # 使 Hip/Foot position innovation 确实对 Root head 有梯度。
        joints = joints + (root_head[:, :3] - base_root)[:, None]
        tracker_joints = torch.as_tensor(TRACKER_TO_JOINT, device=pose_history.device)
        return AnchorPriorOutput(
            pose_model=pose_model,
            pose_raw=pose_raw,
            root_head=root_head,
            contact_logits=contact_logits,
            contact_probability=torch.sigmoid(contact_logits),
            joint_velocity_head=joint_velocity_head,
            region_coverage=role_state.region_coverage,
            tracker_tokens=tracker_tokens,
            joints_head=joints,
            tracker_positions_head=joints.index_select(1, tracker_joints),
            tracker_rotations_head=rotations.index_select(1, tracker_joints),
        )

    def _encode_history(
        self,
        pose_history: torch.Tensor,
        valid_frame_mask: torch.Tensor,
        pose_mean: torch.Tensor | None,
        pose_std: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = pose_history.shape[0]
        tokens = self.history_frame_encoder(pose_history)
        lengths = valid_frame_mask.long().sum(dim=1)
        # 左 padding 已保证所有有效帧连续位于尾部；压紧后再 pack，避免 GRU bias
        # 把 padding 当作真实历史。
        order = torch.argsort((~valid_frame_mask.bool()).long(), dim=1, stable=True)
        compacted = torch.gather(
            tokens,
            1,
            order[..., None].expand(-1, -1, self.latent_dim),
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            compacted,
            lengths.clamp_min(1).detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.history_gru(packed)
        summary = hidden[-1] * (lengths > 0)[:, None].to(tokens.dtype)
        last_index = torch.where(
            valid_frame_mask.bool(),
            torch.arange(REALTIME_POSE_HISTORY_LENGTH, device=pose_history.device)[None],
            -1,
        ).max(dim=1).values.clamp_min(0)
        last_pose = pose_history[torch.arange(batch_size, device=pose_history.device), last_index]
        identity_raw = _identity_pose_raw(batch_size, pose_history.device, pose_history.dtype)
        identity_model = _normalize_pose(identity_raw, pose_mean, pose_std)
        last_pose = torch.where((lengths > 0)[:, None], last_pose, identity_model)
        return summary, last_pose


class TaIDConditioner(nn.Module):
    """准备每目标帧固定的 Prior、FK innovation 与直接区域条件。"""

    def __init__(self, latent_dim: int, config: TaIDConfig):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.config = config.validate()
        self.register_buffer(
            "ablation_code",
            torch.tensor(TAID_ABLATIONS.index(self.config.ablation), dtype=torch.long),
        )
        # 同构的 B1～B6 仅靠 state-dict key 无法发现配置串用；把所有影响角色、
        # innovation 和固定路由语义的值写入 checkpoint，由加载前审计逐项比较。
        self.register_buffer(
            "config_contract",
            torch.tensor(
                [
                    self.config.role.anchor_ramp_start,
                    self.config.role.anchor_ramp_end,
                    self.config.role.innovation_ramp_frames,
                    self.config.innovation_dim,
                    self.config.innovation_clip,
                    self.config.hip_leg_secondary_weight,
                    self.config.hand_torso_weight,
                    self.config.foot_root_contact_weight,
                    TAID_PRIOR_TRACKER_AGGREGATIONS.index(
                        self.config.prior_tracker_aggregation
                    ),
                ],
                dtype=torch.float64,
            ),
        )
        self.prior = AnchorPriorRegressor(self.latent_dim)
        self.innovation_residual_encoder = nn.Sequential(
            nn.Linear(12, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, self.config.innovation_dim, bias=False),
        )
        self.innovation_output_adapter = nn.Linear(
            self.config.innovation_dim, self.latent_dim, bias=False
        )
        self.tracker_type_embedding = nn.Embedding(TRACKER_COUNT, self.latent_dim)
        self.innovation_context_gate = nn.Sequential(
            nn.Linear(self.latent_dim + 6, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Sigmoid(),
        )
        self.absolute_adapter = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.prior_pose_adapter = nn.Linear(ROTATION_6D_DIM, self.latent_dim)
        self.prior_root_adapter = nn.Sequential(
            nn.Linear(4, self.latent_dim), nn.SiLU(), nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.coverage_adapter = nn.Linear(1, self.latent_dim, bias=False)
        self.role_embedding = nn.Embedding(4, self.latent_dim)
        self.register_buffer(
            "joint_regions",
            torch.as_tensor(TARGET_JOINT_REGIONS.copy(), dtype=torch.long),
        )
        self.register_buffer(
            "anchor_routes",
            torch.as_tensor(ANCHOR_REGION_COVERAGE.T.copy(), dtype=torch.float32),
        )
        self.register_buffer(
            "tracker_joint_indices",
            torch.as_tensor(TRACKER_TO_JOINT, dtype=torch.long),
        )
        self.register_buffer(
            "position_scales",
            torch.tensor(self.config.position_scales, dtype=torch.float32),
        )
        self.register_buffer(
            "rotation_scales",
            torch.tensor(self.config.rotation_scales, dtype=torch.float32),
        )

    def forward(
        self,
        pose_history: torch.Tensor,
        tracker_history: torch.Tensor,
        current_tracker: torch.Tensor,
        current_tracker_raw: torch.Tensor,
        current_trajectory: torch.Tensor,
        trajectory_token: torch.Tensor,
        valid_frame_mask: torch.Tensor,
        observation: ObservationEncoding,
        joint_offsets_parent: torch.Tensor,
        pose_mean: torch.Tensor | None = None,
        pose_std: torch.Tensor | None = None,
        tracker_mean: torch.Tensor | None = None,
        tracker_std: torch.Tensor | None = None,
    ) -> PreparedTaIDConditioning:
        configured = current_tracker[..., 9] > 0.5
        measured = current_tracker[..., 10] > 0.5
        d_on = current_tracker[..., 12] * float(TRACKER_DURATION_CAP)
        role_state = compute_tracker_roles_torch(
            configured,
            measured,
            d_on,
            config=self.config.role,
        )
        prior = self.prior(
            pose_history,
            valid_frame_mask,
            observation,
            trajectory_token,
            role_state,
            current_tracker_raw,
            joint_offsets_parent,
            pose_mean,
            pose_std,
        )
        conditioning_prior = _detach_prior(prior) if not self.config.prior_only else prior
        residual, delta = self._compute_innovation(
            conditioning_prior,
            pose_history,
            tracker_history,
            current_tracker_raw,
            current_trajectory,
            valid_frame_mask,
            joint_offsets_parent,
            pose_mean,
            pose_std,
            tracker_mean,
            tracker_std,
        )
        if self.config.uses_continuous_transition:
            beta = role_state.beta
        elif self.config.uses_uncertain_condition:
            beta = role_state.beta_hard
        else:
            # B1/B2 不消费 Uncertain 当前观测；其 consistency 也只能监督已进入
            # Prior 的 alpha 部分，避免用不可见测量改变 B2 的训练目标。
            beta = torch.zeros_like(role_state.beta)
        observation_weight = role_state.alpha + beta
        innovation_tokens = self._encode_innovation(
            residual,
            delta,
            current_tracker,
            conditioning_prior.contact_probability,
            beta,
        )
        region_injection = self._posterior_region_injection(
            conditioning_prior,
            innovation_tokens,
            beta,
        )
        joint_condition = self._joint_condition(
            conditioning_prior,
            role_state,
            region_injection,
        )
        return PreparedTaIDConditioning(
            role_state=role_state,
            prior=prior,
            observation_weight=observation_weight,
            innovation_residual=residual,
            innovation_delta=delta,
            innovation_tokens=innovation_tokens,
            region_injection=region_injection,
            joint_condition=joint_condition,
        )

    def _compute_innovation(
        self,
        prior: AnchorPriorOutput,
        pose_history: torch.Tensor,
        tracker_history: torch.Tensor,
        current_tracker_raw: torch.Tensor,
        current_trajectory: torch.Tensor,
        valid_frame_mask: torch.Tensor,
        joint_offsets_parent: torch.Tensor,
        pose_mean: torch.Tensor | None,
        pose_std: torch.Tensor | None,
        tracker_mean: torch.Tensor | None,
        tracker_std: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = pose_history.shape[0]
        measured = current_tracker_raw[..., 10] > 0.5
        observed_rotation = _safe_tracker_rotations(current_tracker_raw[..., 3:9], measured)
        current_pos = current_tracker_raw[..., :3] - prior.tracker_positions_head
        current_rot = so3_log_map_torch(
            prior.tracker_rotations_head.transpose(-1, -2) @ observed_rotation
        )
        current = torch.cat([current_pos, current_rot], dim=-1)
        current = current * measured[..., None].to(current.dtype)

        previous_pose_model, previous_tracker_model, has_history = _last_history_frame(
            pose_history,
            tracker_history,
            valid_frame_mask,
        )
        identity_model = _normalize_pose(
            _identity_pose_raw(batch_size, pose_history.device, pose_history.dtype),
            pose_mean,
            pose_std,
        )
        previous_pose_model = torch.where(
            has_history[:, None],
            previous_pose_model,
            identity_model,
        )
        previous_pose_raw = _inverse_pose(previous_pose_model, pose_mean, pose_std)
        previous_tracker_raw = _inverse_tracker(
            previous_tracker_model,
            tracker_mean,
            tracker_std,
        )
        previous_rotations, previous_root_yaw = decode_target_head_rotations_torch(previous_pose_raw)
        _, _, previous_joints = resolve_root_head_reference_torch(
            previous_rotations,
            previous_root_yaw,
            joint_offsets_parent,
            observed_head_height=previous_tracker_raw[:, HEAD_TRACKER_INDEX, 1],
        )
        previous_tracker_positions = previous_joints.index_select(1, self.tracker_joint_indices)
        previous_tracker_rotations = previous_rotations.index_select(1, self.tracker_joint_indices)
        previous_measured = (previous_tracker_raw[..., 10] > 0.5) & has_history[:, None]
        previous_observed_rotation = _safe_tracker_rotations(
            previous_tracker_raw[..., 3:9],
            previous_measured,
        )
        previous_pos = previous_tracker_raw[..., :3] - previous_tracker_positions
        # previous position residual 位于 C_(n-1)，先转入当前 C_n 再与 e_t 相减。
        # trajectory 的 sin/cos 按契约不做归一化，可恢复 C_(n-1) -> C_n
        # 的 Head yaw 变化，先统一参考系后才允许计算 delta innovation。
        previous_pos = reexpress_previous_position_residual_torch(
            previous_pos,
            current_trajectory,
        )
        previous_rot = so3_log_map_torch(
            previous_tracker_rotations.transpose(-1, -2) @ previous_observed_rotation
        )
        previous = torch.cat([previous_pos, previous_rot], dim=-1)
        previous = previous * previous_measured[..., None].to(previous.dtype)

        position_scale = self.position_scales.to(current.dtype)[None, :, None]
        rotation_scale = self.rotation_scales.to(current.dtype)[None, :, None]
        scale = torch.cat(
            [
                position_scale.expand(-1, -1, 3),
                rotation_scale.expand(-1, -1, 3),
            ],
            dim=-1,
        )
        clip = float(self.config.innovation_clip)
        current_normalized = clip * torch.tanh(current / scale / clip)
        previous_normalized = clip * torch.tanh(previous / scale / clip)
        delta = current_normalized - previous_normalized
        current_normalized = current_normalized * measured[..., None].to(current.dtype)
        delta = delta * measured[..., None].to(delta.dtype)
        current_normalized[:, HEAD_TRACKER_INDEX] = 0.0
        delta[:, HEAD_TRACKER_INDEX] = 0.0
        return current_normalized, delta

    def _encode_innovation(
        self,
        residual: torch.Tensor,
        delta: torch.Tensor,
        current_tracker: torch.Tensor,
        prior_contact: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = residual.shape[0]
        residual_token = self.innovation_output_adapter(
            self.innovation_residual_encoder(torch.cat([residual, delta], dim=-1))
        )
        tracker_ids = torch.arange(TRACKER_COUNT, device=residual.device)
        type_token = self.tracker_type_embedding(tracker_ids)[None].expand(batch_size, -1, -1)
        context = torch.cat(
            [
                type_token,
                current_tracker[..., 9:13],
                prior_contact[:, None].expand(-1, TRACKER_COUNT, -1),
            ],
            dim=-1,
        )
        token = residual_token * self.innovation_context_gate(context)
        return token * beta[..., None].to(token.dtype)

    def _posterior_region_injection(
        self,
        prior: AnchorPriorOutput,
        innovation_tokens: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = innovation_tokens.shape[0]
        zero = innovation_tokens.new_zeros((batch_size, 5, self.latent_dim))
        if self.config.uses_absolute_uncertain:
            tokens = self.absolute_adapter(prior.tracker_tokens) * beta[..., None]
        elif self.config.uses_innovation:
            tokens = innovation_tokens
        else:
            return zero
        if not self.config.uses_fixed_routing:
            denominator = beta.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = tokens.sum(dim=1) / denominator
            return pooled[:, None].expand(-1, 5, -1)
        routes = self.fixed_route_weights(prior.contact_probability).to(tokens.dtype)
        return torch.einsum("btr,btd->brd", routes, tokens)

    def fixed_route_weights(self, prior_contact: torch.Tensor) -> torch.Tensor:
        batch_size = prior_contact.shape[0]
        routes = prior_contact.new_zeros((batch_size, TRACKER_COUNT, 5))
        routes[:, HIP_TRACKER_INDEX, 0] = 1.0
        routes[:, HIP_TRACKER_INDEX, 3:5] = float(self.config.hip_leg_secondary_weight)
        routes[:, LEFT_HAND_TRACKER_INDEX, 1] = 1.0
        routes[:, LEFT_HAND_TRACKER_INDEX, 0] = float(self.config.hand_torso_weight)
        routes[:, RIGHT_HAND_TRACKER_INDEX, 2] = 1.0
        routes[:, RIGHT_HAND_TRACKER_INDEX, 0] = float(self.config.hand_torso_weight)
        routes[:, LEFT_FOOT_TRACKER_INDEX, 3] = 1.0
        routes[:, LEFT_FOOT_TRACKER_INDEX, 0] = (
            float(self.config.foot_root_contact_weight) * prior_contact[:, 0]
        )
        routes[:, RIGHT_FOOT_TRACKER_INDEX, 4] = 1.0
        routes[:, RIGHT_FOOT_TRACKER_INDEX, 0] = (
            float(self.config.foot_root_contact_weight) * prior_contact[:, 1]
        )
        # Head 只进入 Prior，整个 route row 保持严格为零。
        return routes

    def _joint_condition(
        self,
        prior: AnchorPriorOutput,
        role_state: TrackerRoleStateTorch,
        region_injection: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prior.pose_model.shape[0]
        if not self.config.uses_prior_condition:
            return prior.pose_model.new_zeros((batch_size, SMPL_JOINT_COUNT, self.latent_dim))
        conditioning_prior = _detach_prior(prior)
        prior_pose = self.prior_pose_adapter(
            conditioning_prior.pose_model.reshape(batch_size, SMPL_JOINT_COUNT, ROTATION_6D_DIM)
        )
        coverage = conditioning_prior.region_coverage.index_select(1, self.joint_regions)
        prior_pose = prior_pose + self.coverage_adapter(coverage[..., None])
        root_token = self.prior_root_adapter(conditioning_prior.root_head)
        root_mask = (self.joint_regions == 0).to(prior_pose.dtype)[None, :, None]
        prior_pose = prior_pose + root_mask * root_token[:, None]

        role_tokens = self.role_embedding(role_state.roles)
        role_routes = self.anchor_routes.to(role_tokens.dtype)[None]
        role_regions = torch.einsum("btd,btr->brd", role_tokens, role_routes.expand(batch_size, -1, -1))
        role_denominator = role_routes.sum(dim=1).clamp_min(1.0)
        role_regions = role_regions / role_denominator[..., None]
        region_total = role_regions + region_injection
        return prior_pose + region_total.index_select(1, self.joint_regions)


def _detach_prior(prior: AnchorPriorOutput) -> AnchorPriorOutput:
    return replace(
        prior,
        pose_model=prior.pose_model.detach(),
        pose_raw=prior.pose_raw.detach(),
        root_head=prior.root_head.detach(),
        contact_logits=prior.contact_logits.detach(),
        contact_probability=prior.contact_probability.detach(),
        joint_velocity_head=prior.joint_velocity_head.detach(),
        region_coverage=prior.region_coverage.detach(),
        tracker_tokens=prior.tracker_tokens.detach(),
        joints_head=prior.joints_head.detach(),
        tracker_positions_head=prior.tracker_positions_head.detach(),
        tracker_rotations_head=prior.tracker_rotations_head.detach(),
    )


def _identity_pose_raw(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    identity = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], device=device, dtype=dtype)
    return identity.repeat(batch_size, SMPL_JOINT_COUNT)


def _normalize_pose(
    pose_raw: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return pose_raw
    return (pose_raw - mean.to(device=pose_raw.device, dtype=pose_raw.dtype)) / std.to(
        device=pose_raw.device,
        dtype=pose_raw.dtype,
    ).clamp_min(1e-8)


def _inverse_pose(
    pose_model: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or std is None:
        return pose_model
    return pose_model * std.to(device=pose_model.device, dtype=pose_model.dtype) + mean.to(
        device=pose_model.device,
        dtype=pose_model.dtype,
    )


def _inverse_tracker(
    tracker_model: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    result = tracker_model.clone()
    if mean is not None and std is not None:
        result[..., :9] = (
            result[..., :9] * std.to(device=result.device, dtype=result.dtype)
            + mean.to(device=result.device, dtype=result.dtype)
        )
    measured = result[..., 10] > 0.5
    result[..., :9] = torch.where(
        measured[..., None],
        result[..., :9],
        torch.zeros_like(result[..., :9]),
    )
    return result


def _last_history_frame(
    pose_history: torch.Tensor,
    tracker_history: torch.Tensor,
    valid_frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = pose_history.shape[0]
    frame_indices = torch.arange(REALTIME_POSE_HISTORY_LENGTH, device=pose_history.device)
    last = torch.where(valid_frame_mask.bool(), frame_indices[None], -1).max(dim=1).values
    safe_last = last.clamp_min(0)
    batch_index = torch.arange(batch_size, device=pose_history.device)
    return (
        pose_history[batch_index, safe_last],
        tracker_history[batch_index, safe_last],
        last >= 0,
    )


def _safe_tracker_rotations(rotation_6d: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    identity = torch.tensor(
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        device=rotation_6d.device,
        dtype=rotation_6d.dtype,
    )
    safe = torch.where(measured[..., None], rotation_6d, identity)
    return rotation_6d_to_matrix_torch(safe)
