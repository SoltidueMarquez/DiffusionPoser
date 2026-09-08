"""只用于部署的 Tensor 接口；复用原模型和最终旋转投影。"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from diffusion.realtime_pose_inpainting import build_tracker_geometry_condition
from diffusion.realtime_pose_projection import project_realtime_pose_xstart
from model.realtime_pose_current_dit import _modulate

SAMPLER_INPUTS = ['motion_context', 'predictor_pose_horizon', 'current_tracker_raw',
                 'ik_residual', 'ik_gap', 'ik_confidence', 'denoise_strength',
                 'constraint_type', 'noise']


def example_inputs():
    """静态形状示例；零输入只用于追踪，不代表真实初始化姿态。"""
    return (torch.zeros(1, 10, 144), torch.zeros(1, 11, 144),
            torch.ones(1, 6, 10), torch.zeros(1, 24, 6),
            torch.zeros(1, 24), torch.ones(1, 24), torch.ones(1, 24),
            torch.zeros(1, 24, dtype=torch.long), torch.zeros(1, 144))


class SamplerTemporalAttention(nn.Module):
    """仅用于部署的时间 Attention；K/V 由调用者在同一 tick 内显式复用。"""

    def __init__(self, attention):
        super().__init__()
        # attention 已属于 PoseSampler 的深拷贝，不触碰训练模型的参数。
        self.in_proj_weight = attention.in_proj_weight
        self.in_proj_bias = attention.in_proj_bias
        self.out_proj = attention.out_proj
        self.num_heads = attention.num_heads

    def _split_heads(self, tensor):
        """原 MHA 的 [T,24,192] → [24,6,T,32] 拆装，不改变元素顺序。"""
        length, batch, width = tensor.shape
        head_width = width // self.num_heads
        return tensor.view(length, batch * self.num_heads, head_width).transpose(0, 1).view(
            batch, self.num_heads, length, head_width)

    def prepare_kv(self, context):
        """输入 [24,20,192]；返回缩放后的 K 和已转置 V，仅在本次调用内使用。"""
        width = self.in_proj_weight.shape[1]
        # 与 _in_projection_packed 的 cross-attention 分支保持同一联合投影和拆分顺序。
        projected = F.linear(context.transpose(0, 1), self.in_proj_weight[width:],
                             self.in_proj_bias[width:])
        projected = projected.unflatten(-1, (2, width)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        key_heads, value_heads = self._split_heads(projected[0]), self._split_heads(projected[1])
        scale_root = key_heads.new_tensor(float(width // self.num_heads)).sqrt().reciprocal().sqrt()
        return key_heads * scale_root, value_heads.transpose(-2, -1)

    def forward(self, query, temporal_kv):
        """query [24,1,192] 每步变化；K [24,6,20,32]、Vᵀ [24,6,32,20] 显式传入。"""
        query = query.transpose(0, 1)
        target_length, batch, width = query.shape
        projected = F.linear(query, self.in_proj_weight[:width], self.in_proj_bias[:width])
        query_heads = self._split_heads(projected)
        head_width = width // self.num_heads
        # 与原 SDPA 的 ONNX 表达相同，在 Q/K 两侧各乘 sqrt(1/sqrt(D))。
        scale_root = query_heads.new_tensor(float(head_width)).sqrt().reciprocal().sqrt()
        scaled_query = query_heads * scale_root
        scaled_key, transposed_value = temporal_kv
        # [24,6,1,32] × [24,6,20,32]：反向乘法避免 M=1 的向量批乘路径。
        scores = (scaled_key @ scaled_query.transpose(-2, -1)).transpose(-2, -1)
        probabilities = scores.softmax(dim=-1)
        attended = (transposed_value @ probabilities.transpose(-2, -1)).transpose(-2, -1)
        output = attended.permute(2, 0, 1, 3).contiguous().view(batch * target_length, width)
        output = F.linear(output, self.out_proj.weight, self.out_proj.bias)
        return output.view(target_length, batch, width).transpose(0, 1)


def prepare_temporal_kv(dit, temporal_context):
    """为四个 block 各准备一组 K/V；返回局部 tuple，不在模块内保存跨 tick 状态。"""
    batch, joints, length, width = temporal_context.shape
    context = temporal_context.reshape(batch * joints, length, width)
    return tuple(block.temporal_attention.prepare_kv(block.temporal_context_norm(context))
                 for block in dit.blocks)


def forward_sampler_block(block, current, temporal_kv, diffusion_time):
    """复用原 block 层与计算顺序，唯一差别是时间 K/V 已由循环外准备。"""
    batch, joints, width = current.shape
    (spatial_shift, spatial_scale, spatial_gate, temporal_shift, temporal_scale,
     temporal_gate, mlp_shift, mlp_scale, mlp_gate) = block.adaln_modulation(diffusion_time).chunk(9, dim=-1)
    spatial_query = _modulate(block.spatial_norm(current), spatial_shift, spatial_scale)
    spatial_value = block.spatial_attention(
        spatial_query, spatial_query, spatial_query, need_weights=False)[0]
    current = current + spatial_gate[:, None] * spatial_value
    temporal_query = current.reshape(batch * joints, 1, width)
    temporal_shift_bj = temporal_shift[:, None].expand(-1, joints, -1).reshape(batch * joints, width)
    temporal_scale_bj = temporal_scale[:, None].expand(-1, joints, -1).reshape(batch * joints, width)
    temporal_gate_bj = temporal_gate[:, None].expand(-1, joints, -1).reshape(batch * joints, width)
    temporal_query = _modulate(block.temporal_norm(temporal_query), temporal_shift_bj, temporal_scale_bj)
    temporal_value = block.temporal_attention(temporal_query, temporal_kv)
    current = current + (temporal_gate_bj[:, None] * temporal_value).reshape(batch, joints, width)
    mlp_query = _modulate(block.mlp_norm(current), mlp_shift, mlp_scale)
    return current + mlp_gate[:, None] * block.mlp(mlp_query)


def forward_sampler_dit(dit, hidden_states, timestep, condition, temporal_kv):
    """部署单步：Tracker Attention、时间嵌入和输出仍按原模型逐步执行。"""
    batch = hidden_states.shape[0]
    current = dit.residual_input(hidden_states.reshape(batch, 24, 6)) + condition.joint_condition_tokens
    tracker_query = dit.tracker_query_norm(current)
    tracker_context = dit.tracker_context_norm(condition.tracker_tokens)
    tracker_value = dit.tracker_cross_attention(
        tracker_query, tracker_context, tracker_context,
        key_padding_mask=~condition.tracker_available, need_weights=False)[0]
    current = dit.tracker_output_norm(current + tracker_value)
    diffusion_time = dit.diffusion_time_embedding(timestep)
    for block, block_kv in zip(dit.blocks, temporal_kv):
        current = forward_sampler_block(block, current, block_kv, diffusion_time)
    current = dit.output_norm(current)
    return dit.joint_output(current).reshape(batch, 144)


class SamplerLayerNorm(nn.Module):
    """只用于 DiT 部署副本的无仿射 LayerNorm，沿最后一维归一化。"""

    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # 输入为 [B,24,D] 或 [B*24,1,D]；均值、总体方差和 eps 与原层相同。
        # 用 centered * rsqrt 表达标准化，避免 Unity 2.6.1 将它与后续
        # 三维 AdaLN scale/bias 错融为只接受一维仿射参数的 LayerNormalization。
        centered = value - value.mean(dim=-1, keepdim=True)
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(variance + self.eps)


class PoseSampler(nn.Module):
    def __init__(self, dit, diffusion, normalizer):
        super().__init__()
        # 不改传入的原模型；数值对照仍必须以未经部署替换的 PyTorch 为参考。
        self.dit = deepcopy(dit)
        for block in self.dit.blocks:
            block.temporal_attention = SamplerTemporalAttention(block.temporal_attention)
            for name in ('spatial_norm', 'temporal_norm', 'mlp_norm'):
                setattr(block, name, SamplerLayerNorm(getattr(block, name).eps))
        self.steps = diffusion.num_timesteps
        for name in ('pose_mean', 'pose_scale', 'tracker_mean'):
            self.register_buffer(name, getattr(normalizer, name).clone())
        self.register_buffer('tracker_scale', normalizer.tracker_std + normalizer.eps)
        self.register_buffer('timesteps', torch.tensor(diffusion.timestep_map, dtype=torch.long))
        # 先按原 sampler 的顺序转 float32 再 sqrt，保持舍入行为一致。
        self.register_buffer('recip', torch.tensor(diffusion.sqrt_recip_alphas_cumprod, dtype=torch.float32))
        self.register_buffer('recipm1', torch.tensor(diffusion.sqrt_recipm1_alphas_cumprod, dtype=torch.float32))
        self.register_buffer('alpha_prev', torch.tensor(diffusion.alphas_cumprod_prev, dtype=torch.float32))

    def forward(self, motion_context, predictor_pose_horizon, current_tracker_raw,
                ik_residual, ik_gap, ik_confidence, denoise_strength, constraint_type, noise):
        """输入 batch=1；输出归一化的 [1,144]，历史与 IK 留在调用端。"""
        geometry = build_tracker_geometry_condition(
            current_tracker_raw, self.tracker_mean, self.tracker_scale)
        condition = self.dit.prepare_conditioning(
            motion_context, predictor_pose_horizon, geometry,
            current_tracker_raw[..., 9] > 0.5, ik_residual, ik_gap,
            ik_confidence, denoise_strength, constraint_type)
        prior = predictor_pose_horizon[:, 0]
        # 每次 forward 都从本次条件重新准备；10 步共享，同 block 之外不共用。
        temporal_kv = prepare_temporal_kv(self.dit, condition.temporal_context)
        state = noise
        for index in reversed(range(self.steps)):
            residual = forward_sampler_dit(
                self.dit, state, self.timesteps[index:index + 1], condition, temporal_kv)
            # 原实现只在最终步投影；eta=0 不需要逐步生成随机噪声。
            if index == 0:
                return project_realtime_pose_xstart(
                    prior + residual, current_tracker_raw, self.pose_mean, self.pose_scale)
            eps = (self.recip[index] * state - residual) / self.recipm1[index]
            state = (residual * torch.sqrt(self.alpha_prev[index])
                     + torch.sqrt(1 - self.alpha_prev[index]) * eps)
        raise RuntimeError('采样步数必须为正数。')
