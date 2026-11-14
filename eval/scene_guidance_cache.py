"""三种修正器共用的帧常量与几何自动微分执行缓存。"""
from __future__ import annotations

from dataclasses import dataclass, fields
import time

import torch

from diffusion.realtime_pose_projection import project_rotation_6d_to_so3
from export.residual_collision_postprocess import box_sdf, matrix_quaternion, safe_divide


@dataclass
class SceneFrame:
    """仅包含本帧不变量；任何随残差变化的姿态、距离和导数均不放在这里。"""

    prior: torch.Tensor
    tracker_rotations: torch.Tensor
    tracker_available: torch.Tensor
    head_position: torch.Tensor
    box_centers: torch.Tensor
    box_axes: torch.Tensor
    box_half: torch.Tensor
    active: torch.Tensor

    def tensors(self):
        return tuple(getattr(self, field.name) for field in fields(self))


@torch.no_grad()
def prepare_scene_frame(scene, feed):
    """tracker [1,6,10]、盒体 [1,2,15]；每次完整采样重新准备并计时。"""
    tracker, boxes = feed["current_tracker_raw"], feed["armrest_geometry"]
    return SceneFrame(
        prior=feed["predictor_pose_horizon"][:, 0],
        tracker_rotations=project_rotation_6d_to_so3(tracker[..., 3:9]),
        tracker_available=(tracker[..., 9] > .5)[..., None],
        head_position=tracker[:, :1, :3],
        box_centers=boxes[:, None, :, :3],
        box_axes=boxes[..., 3:12].reshape(1, 1, 2, 3, 3),
        box_half=boxes[:, None, :, 12:15],
        active=(feed["contact_mode"].index_select(1, scene.arms) > 0).flatten(),
    )


def cached_segment_box_sdf(a, b, half, pair_i, pair_j):
    """沿用 segment_box_sdf 的候选点与运算顺序，仅复用六平面对的设备索引。"""
    direction = b - a
    breaks = torch.cat([torch.zeros_like(a[..., :1]), torch.ones_like(a[..., :1]),
                        safe_divide(half - a, direction), safe_divide(-half - a, direction)], -1)
    breaks = breaks.clamp(0, 1).topk(8, dim=-1, largest=False, sorted=True).values
    lo, hi = breaks[..., :-1], breaks[..., 1:]
    middle = a.unsqueeze(-2) + ((lo + hi) * .5).unsqueeze(-1) * direction.unsqueeze(-2)
    outside = middle.abs() > half.unsqueeze(-2)
    boundary = torch.where(middle >= 0, half.unsqueeze(-2), -half.unsqueeze(-2))
    numerator = ((a.unsqueeze(-2) - boundary) * direction.unsqueeze(-2) * outside).sum(-1)
    denominator = (direction.unsqueeze(-2).square() * outside).sum(-1)
    stationary = safe_divide(-numerator, denominator).maximum(lo).minimum(hi)
    intercept = torch.cat([a - half, -a - half], -1)
    slope = torch.cat([direction, -direction], -1)
    crossing = safe_divide(intercept.index_select(-1, pair_j) - intercept.index_select(-1, pair_i),
                           slope.index_select(-1, pair_i) - slope.index_select(-1, pair_j)).clamp(0, 1)
    candidates = torch.cat([breaks[..., :1], breaks[..., -1:], stationary, crossing], -1)
    points = a.unsqueeze(-2) + candidates.unsqueeze(-1) * direction.unsqueeze(-2)
    return box_sdf(points, half.unsqueeze(-2)).amin(-1)


def scene_observation(scene, residual, frame):
    """残差 [1,144] → 观测 [16]；保留原投影、四元数、胶囊和 SDF 计算。"""
    raw = ((frame.prior + residual) * scene.pose_scale + scene.pose_mean).reshape(1, 24, 6)
    rotations = project_rotation_6d_to_so3(raw)
    replacement = torch.where(frame.tracker_available, frame.tracker_rotations,
                              rotations.index_select(1, scene.tracker_indices))
    rotations = rotations.index_copy(1, scene.tracker_indices, replacement)
    # 保持原 project_realtime_pose_xstart 后再 inverse normalize 的 FP32 运算顺序。
    projected = (rotations.reshape(1, 144) - scene.pose_mean) / scene.pose_scale
    raw = (projected * scene.pose_scale + scene.pose_mean).reshape(1, 24, 6)
    def cross(a, b):
        # 与原 cross3 同一运算，仅把每次构造的 Python 索引移至设备常量。
        return (a.index_select(-1, scene.cross_yzx) * b.index_select(-1, scene.cross_zxy)
                - a.index_select(-1, scene.cross_zxy) * b.index_select(-1, scene.cross_yzx))

    def rotate(q, vector):
        u = 2 * cross(q[..., :3], vector)
        return vector + q[..., 3:] * u + cross(q[..., :3], u)

    forward, up = raw[..., :3], raw[..., 3:]
    rotations = matrix_quaternion(torch.stack([cross(up, forward), up, forward], -1))
    bones = rotate(rotations.index_select(1, scene.parent_indices), scene.offsets[None, 1:])
    relative = scene.paths @ bones
    positions = relative + frame.head_position - relative[:, 15:16]
    origins = positions.index_select(1, scene.bones)
    quaternions = rotations.index_select(1, scene.bones)
    start = origins + rotate(quaternions, scene.starts)
    end = origins + rotate(quaternions, scene.ends)

    def local(point):
        delta = point[:, :, None] - frame.box_centers
        return (frame.box_axes * delta.unsqueeze(-2)).sum(-1)

    gap = cached_segment_box_sdf(local(start), local(end), frame.box_half, scene.pair_i, scene.pair_j)
    gap = (gap - scene.radii[None, :, None]).flatten()
    return torch.where(frame.active, (scene.margin - gap).clamp_min(0), torch.zeros_like(gap))


class SceneObservationCache:
    """一套前向与两种自动微分请求，供所有修正器共用。

    CUDA 捕获的是执行流程；每次 evaluate 都写入当前残差并重新求值。
    同一点的穿透判断、损失与求导共享 value 和前向保存的张量。
    输出缓冲区在下一次同类请求时覆盖，调用者应先完成本轮计算。
    """

    def __init__(self, scene, frame):
        self.scene = scene
        self.frame = frame
        self.is_cuda = frame.prior.is_cuda
        self.initialization_ms = 0.
        self.forward_calls = 0
        if self.is_cuda:
            self._capture(frame)

    def _capture(self, frame):
        device = frame.prior.device
        torch.cuda.synchronize(device)
        begin = time.perf_counter()
        self.frame = SceneFrame(*(tensor.detach().clone() for tensor in frame.tensors()))
        self.point = torch.zeros_like(frame.prior, requires_grad=True)
        self.weights = torch.zeros(self.scene.observation_basis.shape[0], device=device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        # 在旁路 stream 上完成自动微分内核预热；所有方法使用同一套捕获结果。
        with torch.cuda.stream(stream), torch.enable_grad():
            for _ in range(3):
                value = scene_observation(self.scene, self.point, self.frame)
                torch.autograd.grad(value, self.point, self.weights, retain_graph=True)
                torch.autograd.grad(value, self.point, self.scene.observation_basis, is_grads_batched=True)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.forward_graph = torch.cuda.CUDAGraph()
        self.vjp_graph = torch.cuda.CUDAGraph()
        self.jacobian_graph = torch.cuda.CUDAGraph()
        # 分开的内存池保留前向中间量；后续反向重放读到本次前向更新后的数据。
        with torch.cuda.graph(self.forward_graph, stream=stream), torch.enable_grad():
            self.value = scene_observation(self.scene, self.point, self.frame)
        with torch.cuda.graph(self.vjp_graph, stream=stream), torch.enable_grad():
            self.gradient = torch.autograd.grad(self.value, self.point, self.weights, retain_graph=True)[0]
        with torch.cuda.graph(self.jacobian_graph, stream=stream), torch.enable_grad():
            self.derivatives = torch.autograd.grad(
                self.value, self.point, self.scene.observation_basis,
                is_grads_batched=True, retain_graph=True)[0].reshape(self.value.numel(), -1)
        torch.cuda.synchronize(device)
        self.initialization_ms = (time.perf_counter() - begin) * 1000

    @torch.no_grad()
    def bind(self, frame):
        """帧缓存只在本次采样内复用；复制和准备开销均在计时范围内。"""
        self.forward_calls = 0
        if self.is_cuda:
            for target, source in zip(self.frame.tensors(), frame.tensors()):
                target.copy_(source)
        else:
            self.frame = frame

    def evaluate(self, residual):
        self.forward_calls += 1
        if self.is_cuda:
            with torch.no_grad():
                self.point.copy_(residual)
            self.forward_graph.replay()
        else:
            with torch.enable_grad():
                self.point = residual.detach().requires_grad_(True)
                self.value = scene_observation(self.scene, self.point, self.frame)
        return self.value.detach()

    def vjp(self, weights):
        if self.is_cuda:
            self.weights.copy_(weights.detach())
            self.vjp_graph.replay()
            return self.gradient.detach()
        return torch.autograd.grad(self.value, self.point, weights.detach())[0].detach()

    def jacobian(self):
        if self.is_cuda:
            self.jacobian_graph.replay()
            return self.derivatives.detach()
        return torch.autograd.grad(self.value, self.point, self.scene.observation_basis,
                                   is_grads_batched=True)[0].reshape(self.value.numel(), -1).detach()
