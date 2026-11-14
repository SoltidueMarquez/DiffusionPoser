"""固定运动模型下的场景引导对照；不改变训练和 Unity 部署接口。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import torch
from torch import nn

from data_loaders.sensor_masking import TRACKER_TO_JOINT
from diffusion.realtime_pose_inpainting import build_tracker_geometry_condition
from diffusion.realtime_pose_projection import project_realtime_pose_xstart
from eval.scene_guidance_cache import SceneObservationCache, prepare_scene_frame
from export.residual_collision_postprocess import qrotate, rotations_from_6d, segment_box_sdf
from export.unity_onnx_models import forward_sampler_dit, prepare_temporal_kv


@dataclass(frozen=True)
class GuidanceConfig:
    method: str = "none"
    iterations: int = 1
    step_size: float = 1e-4
    damping: float = 1e-2
    strength: float = 1.0
    covariance_scale: float = 1.0

    @property
    def name(self):
        if self.method == "postedit":
            return f"postedit_n{self.iterations}_h{self.step_size:g}"
        if self.method == "lm":
            return f"lm_n{self.iterations}_d{self.damping:g}"
        if self.method == "pigdm":
            return f"pigdm_s{self.strength:g}_c{self.covariance_scale:g}"
        return self.method

    def to_dict(self):
        return dict(name=self.name, **asdict(self))


class SceneGeometry(nn.Module):
    """最终姿态的胶囊—OBB 距离；不包含显示插值、顶面吸附或部署辅助损失。"""

    def __init__(self, rest, profile, pose_mean, pose_scale, margin=.002):
        super().__init__()
        self.margin = float(margin)
        self.parents = tuple(rest["parents"])
        self.register_buffer("pose_mean", pose_mean.clone())
        self.register_buffer("pose_scale", pose_scale.clone())
        self.register_buffer("offsets", torch.tensor(rest["restLocalPositions"], dtype=torch.float32))
        self.register_buffer("parent_indices", torch.tensor(self.parents[1:]))
        # 全局旋转已知时，每条骨向量仅需旋转一次；路径矩阵代替逐关节 Python 循环。
        paths = torch.zeros(24, 23)
        for joint in range(1, 24):
            child = joint
            while child > 0:
                paths[joint, child - 1] = 1
                child = self.parents[child]
        self.register_buffer("paths", paths)
        capsules = [capsule for arm in profile["capsules"] for capsule in arm]
        self.register_buffer("bones", torch.tensor([c["bone"] for c in capsules]))
        self.register_buffer("starts", torch.tensor([c["start"] for c in capsules], dtype=torch.float32))
        self.register_buffer("ends", torch.tensor([c["end"] for c in capsules], dtype=torch.float32))
        self.register_buffer("radii", torch.tensor([c["radius"] for c in capsules], dtype=torch.float32))
        self.register_buffer("arms", torch.tensor([0] * len(profile["capsules"][0]) + [1] * len(profile["capsules"][1])))
        self.register_buffer("tracker_indices", torch.tensor(TRACKER_TO_JOINT))
        self.register_buffer("cross_yzx", torch.tensor([1, 2, 0]))
        self.register_buffer("cross_zxy", torch.tensor([2, 0, 1]))
        pairs = torch.triu_indices(6, 6, offset=1)
        self.register_buffer("pair_i", pairs[0].clone())
        self.register_buffer("pair_j", pairs[1].clone())
        self.register_buffer("observation_basis", torch.eye(len(capsules) * 2))
        self.execution = None

    def prepare_execution(self, feed):
        """固定设备/形状的执行图在实验初始化时创建；每帧仍独立准备条件。"""
        if self.execution is None:
            self.execution = SceneObservationCache(self, prepare_scene_frame(self, feed))
        return self.execution

    def prepare_observation(self, feed):
        self.execution.bind(prepare_scene_frame(self, feed))
        return self.execution

    def positions_and_rotations(self, pose, tracker):
        """pose [1,144]，tracker [1,6,10]；返回 [1,24,3] / [1,24,4]。"""
        projected = project_realtime_pose_xstart(pose, tracker, self.pose_mean, self.pose_scale)
        raw = (projected * self.pose_scale + self.pose_mean).reshape(1, 24, 6)
        rotations = rotations_from_6d(raw)
        bones = qrotate(rotations[:, self.parent_indices], self.offsets[None, 1:])
        relative = self.paths @ bones
        positions = relative + tracker[:, :1, :3] - relative[:, 15:16]
        return positions, rotations

    def gaps(self, pose, feed):
        """返回 [胶囊数×盒体数] 的米制距离以及有效约束 mask。"""
        positions, rotations = self.positions_and_rotations(pose, feed["current_tracker_raw"])
        origins, quaternions = positions[:, self.bones], rotations[:, self.bones]
        start = origins + qrotate(quaternions, self.starts)
        end = origins + qrotate(quaternions, self.ends)
        boxes = feed["armrest_geometry"]
        axes = boxes[..., 3:12].reshape(1, 2, 3, 3)

        def local(point):
            delta = point[:, :, None] - boxes[:, None, :, :3]
            return (axes[:, None] * delta.unsqueeze(-2)).sum(-1)

        gap = segment_box_sdf(local(start), local(end), boxes[:, None, :, 12:15])
        gap = gap - self.radii[None, :, None]
        # 原回放 mode=2 的额外顶面规则不属于论文 SDF 目标；这里只用它表示约束存在。
        active = feed["contact_mode"][:, self.arms] > 0
        return gap.flatten(), active.flatten()

    def observation(self, pose, feed):
        gap, active = self.gaps(pose, feed)
        return torch.where(active, (self.margin - gap).clamp_min(0), torch.zeros_like(gap))

    @torch.no_grad()
    def metrics(self, pose, baseline, feed):
        gap, active = self.gaps(pose, feed)
        depth = (-gap[active]).clamp_min(0)
        violation = (self.margin - gap[active]).clamp_min(0)
        positions, _ = self.positions_and_rotations(pose, feed["current_tracker_raw"])
        base_positions, _ = self.positions_and_rotations(baseline, feed["current_tracker_raw"])
        tracker = feed["current_tracker_raw"]
        errors = (positions[:, [15, 20, 21, 0, 10, 11]] - tracker[..., :3]).norm(dim=-1)
        available = tracker[..., 9] > .5
        return {
            "penetration_mean_mm": float(depth.mean() * 1000) if depth.numel() else 0.,
            "penetration_max_mm": float(depth.max() * 1000) if depth.numel() else 0.,
            "penetrating": bool((depth > 1e-6).any()),
            "clearance_mean_mm": float(violation.mean() * 1000) if violation.numel() else 0.,
            "clearance_max_mm": float(violation.max() * 1000) if violation.numel() else 0.,
            "tracker_error_mm": float(errors[available].mean() * 1000),
            "pose_displacement_mm": float((positions - base_positions).norm(dim=-1).mean() * 1000),
        }


# region 三种引导；共同的观测残差以米计，先验残差在模型的归一化空间中
def correct_clean_residual(reference, observation, gate, variance, scale, config):
    """PostEdit/LM 均最小化 ||r-r_hat||²/(2σ²) + ||h(r)||²/(2γ²)。"""
    current = reference.detach()
    reference = current
    inverse_variance, inverse_scale_squared = variance.reciprocal(), 1 / scale**2
    if config.method == "postedit":
        for iteration in range(config.iterations):
            if iteration:
                observation.evaluate(current)
            value = observation.value.detach()
            # 几何项仍由自动微分求导；二次保真项使用相同导数，复用本步权重。
            gradient = (observation.vjp(value * inverse_scale_squared)
                        + (current - reference) * inverse_variance)
            # 与论文的逐关节门控和部署实现的步长衰减一致；没有另加回拉或噪声。
            step = config.step_size * (1 - .99 * iteration / config.iterations)
            current = (current - step * gate * gradient).detach()
        return current
    if config.method != "lm":
        raise ValueError(f"未知干净残差优化器：{config.method}")

    root_gate = gate.flatten().sqrt()
    prior_diagonal = torch.diag(gate.flatten() / variance)
    damping = config.damping
    for _ in range(config.iterations):
        # 初始检测及上一轮接受的候选已有前向结果，直接在该点求 Jacobian。
        value, jacobian = observation.value.detach(), observation.jacobian()
        # 令增量 dr=sqrt(m)*du；共同目标不变，门控只预条件化求解方向。
        jacobian = jacobian * root_gate[None] / scale
        offset = (current - reference).flatten()
        hessian = jacobian.T @ jacobian + prior_diagonal
        gradient = jacobian.T @ (value / scale) + root_gate * offset / variance
        energy = .5 * (offset.square().sum() / variance + value.square().sum() / scale**2)
        if bool(gradient.abs().max() < 1e-6):
            break
        diagonal = hessian.diagonal().clamp_min(1e-6)
        accepted = False
        for _ in range(5):
            update = torch.linalg.solve(hessian + torch.diag(damping * diagonal), -gradient)
            trial = current + (root_gate * update).reshape_as(current)
            trial_value = observation.evaluate(trial)
            trial_energy = ((trial - reference).square().sum() / (2 * variance)
                            + trial_value.square().sum() / (2 * scale**2))
            if bool(torch.isfinite(trial_energy) & (trial_energy < energy)):
                current = trial.detach()
                damping = max(damping / 10, 1e-8)
                accepted = True
                break
            damping = min(damping * 10, 1e8)
        # 无下降方向或已收敛时停止，不重复计算同一线性化问题。
        if not accepted or bool((energy - trial_energy).abs() < 1e-6 * energy.abs().clamp_min(1)):
            break
    return current.detach()


def pseudoinverse_guidance(state, clean, observation, gate, variance, scale, config):
    """SDF 的局部线性 ΠGDM 适配，而非 InPose 的关节位置闭式协方差。

    h(r)=穿透残差、y=0，r_t²=cσ²，Σ=γ²I+r_t²JJᵀ。
    ΠGDM 的采样增量使用 -r_t²(d r_hat / d z_t)ᵀ JᵀΣ⁻¹h；
    等价于 InPose 的 (JJᵀ+γ²/r_t² I)⁻¹ 写法，不能直接把未缩放似然分数加到 DDIM。
    Σ/J 在当前步固定，只做一次 DiT VJP。
    """
    with torch.enable_grad():
        value, jacobian = observation.value.detach(), observation.jacobian()
        conditional_variance = config.covariance_scale * variance
        covariance = (conditional_variance * (jacobian @ jacobian.T)
                      + scale**2 * observation.scene.observation_basis)
        score_clean = -jacobian.T @ torch.linalg.solve(covariance, value)
        score_state = torch.autograd.grad(clean, state, score_clean.reshape_as(clean))[0]
    return config.strength * gate * conditional_variance * score_state.detach()
# endregion


def sample_scene_guidance(sampler, scene, feed, config, measurement_scale=.01, measure=False):
    """单帧五步采样；条件、网络前向、引导和最终投影均计入 sampler 时间。"""
    if config.method not in {"none", "postedit", "lm", "pigdm"}:
        raise ValueError(config.method)
    device = feed["noise"].device
    gpu_timing = measure and device.type == "cuda"
    # 图捕获属于一次性初始化；benchmark 显式提前调用，并单独记录初始化成本。
    if config.method != "none":
        scene.prepare_execution(feed)
    events, cpu_guidance, prepare_ms = [], 0., 0.
    if gpu_timing:
        torch.cuda.synchronize(device)
        initial_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    begin = time.perf_counter()
    with torch.no_grad():
        tracker = feed["current_tracker_raw"]
        geometry = build_tracker_geometry_condition(tracker, sampler.tracker_mean, sampler.tracker_scale)
        condition = sampler.dit.prepare_conditioning(
            feed["motion_context"], feed["predictor_pose_horizon"], geometry, tracker[..., 9] > .5,
            feed["ik_residual"], feed["ik_gap"], feed["ik_confidence"],
            feed["denoise_strength"], feed["constraint_type"])
        temporal_kv = prepare_temporal_kv(sampler.dit, condition.temporal_context)
        prior = feed["predictor_pose_horizon"][:, 0]
        gate = feed["denoise_strength"].clamp(0, 1).repeat_interleave(6, dim=-1)
        state = feed["noise"]
    if config.method != "none":
        if gpu_timing:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        prepare_begin = time.perf_counter()
        observation = scene.prepare_observation(feed)
        prepare_ms = (time.perf_counter() - prepare_begin) * 1000
        cpu_guidance += prepare_ms
        if gpu_timing:
            end.record()
            events.append((start, end))
    for index in reversed(range(sampler.steps)):
        with torch.set_grad_enabled(config.method == "pigdm"):
            state = state.detach().requires_grad_(config.method == "pigdm")
            residual = forward_sampler_dit(sampler.dit, state, sampler.timesteps[index:index + 1],
                                           condition, temporal_kv)
        alpha = sampler.recip[index].square().reciprocal()
        variance = (1 - alpha).clamp_min(1e-6)
        if config.method != "none":
            if gpu_timing:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            step_begin = time.perf_counter()
            # 三组共享检测、损失与导数的前向；当前状态每次更新后重新计算。
            active = bool((observation.evaluate(residual) > 0).any())
            if not active:
                guidance = torch.zeros_like(state)
            elif config.method == "pigdm":
                guidance = pseudoinverse_guidance(state, residual, observation, gate, variance,
                                                  measurement_scale, config)
            else:
                residual = correct_clean_residual(residual, observation, gate, variance,
                                                   measurement_scale, config)
            cpu_guidance += (time.perf_counter() - step_begin) * 1000
            if gpu_timing:
                end.record()
                events.append((start, end))
        with torch.no_grad():
            residual = residual.detach()
            # ΠGDM 的似然分数作用于 DDIM 状态，包括最后一步；其他方法先编辑 r_hat。
            if index == 0:
                result = residual
            else:
                eps = (sampler.recip[index] * state - residual) / sampler.recipm1[index]
                result = (residual * sampler.alpha_prev[index].sqrt()
                          + (1 - sampler.alpha_prev[index]).sqrt() * eps)
            if config.method == "pigdm":
                result = result + alpha.sqrt() * guidance
            state = result
    with torch.no_grad():
        pose = project_realtime_pose_xstart(prior + state, tracker, sampler.pose_mean, sampler.pose_scale)
    if gpu_timing:
        torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - begin) * 1000
    if gpu_timing and events:
        prepare_ms = events[0][0].elapsed_time(events[0][1])
    stats = {"sampler_ms": elapsed, "guidance_ms": sum(s.elapsed_time(e) for s, e in events) if gpu_timing else cpu_guidance,
             "cache_prepare_ms": prepare_ms,
             "geometry_evaluations": observation.forward_calls if config.method != "none" else 0,
             "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if gpu_timing else None,
             "extra_peak_mb": (torch.cuda.max_memory_allocated(device) - initial_memory) / 2**20 if gpu_timing else None}
    if not bool(torch.isfinite(pose).all() & torch.isfinite(state).all()):
        raise FloatingPointError(f"{config.name} 产生非有限姿态/采样状态")
    return pose.detach(), stats
