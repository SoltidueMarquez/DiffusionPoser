"""Unity 部署专用的确定性残差编辑；不接入训练和 Python 常规采样。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn

from data_loaders.realtime_pose_kinematics import SMPL_PARENTS
from diffusion.realtime_pose_projection import project_realtime_pose_xstart, project_rotation_6d_to_so3


ACTIVE_JOINTS = (16, 17, 18, 19, 3, 6, 9, 12, 1, 2, 4, 5)
COLLISION_INPUTS = ['armrest_geometry', 'contact_mode', 'seated_context', 'display_from_pose', 'head_height_context', 'correction_context']


@dataclass(frozen=True)
class CollisionSettings:
    steps: int = 5
    h: float = 1e-4
    m: float = .01
    difference_step: float = 1e-3
    margin: float = .002
    tracking_tolerance: float = .02
    spine_bend_scale: float = .45
    spine_bend_tolerance: float = .15
    body_tracking_weight: float = .15
    torso_curve_scale: float = .2
    torso_curve_limit: float = .7
    chest_head_scale: float = .15
    chest_head_limit: float = .35
    head_forward_limit: float = 0.
    head_forward_weight: float = 1.
    head_forward_smooth_weight: float = .5
    head_forward_speed: float = .05
    head_y_limit: float = 0.
    head_y_weight: float = 1.
    head_y_smooth_weight: float = .5
    head_y_speed: float = .15
    seat_collision_weight: float = 1.
    correction_smooth_weight: float = 0.
    pelvis_forward_smooth_weight: float = 0.
    pelvis_forward_update_cost: float = 0.

    def validate(self):
        if not all(torch.isfinite(torch.tensor(v)) and v >= 0 for v in
                   (self.head_forward_limit, self.head_forward_weight, self.head_forward_smooth_weight, self.head_forward_speed, self.head_y_limit, self.head_y_weight, self.head_y_smooth_weight, self.head_y_speed, self.seat_collision_weight, self.correction_smooth_weight, self.pelvis_forward_smooth_weight, self.pelvis_forward_update_cost)):
            raise ValueError('头部高度边界、权重和速度必须为有限非负数。')
        if self.head_forward_limit > 0 and (self.head_forward_weight <= 0 or self.head_forward_speed <= 0):
            raise ValueError("前后自由度需要正回零权重和速度上限。")
        if self.head_y_limit > 0 and (self.head_y_weight <= 0 or self.head_y_speed <= 0):
            raise ValueError('启用高度自由度时必须保留回零权重和正速度上限。')
        if self.steps < 1:
            raise ValueError('碰撞迭代次数必须为正。')
        if not all(torch.isfinite(torch.tensor(v)) and v > 0 for v in
                   (self.h, self.m, self.difference_step)):
            raise ValueError('PostEdit 步长、测量尺度和差分步幅必须为有限正数。')
        if not all(torch.isfinite(torch.tensor(v)) and v >= 0 for v in
                   (self.margin, self.tracking_tolerance, self.spine_bend_scale, self.spine_bend_tolerance, self.body_tracking_weight, self.torso_curve_scale, self.torso_curve_limit, self.chest_head_scale, self.chest_head_limit)):
            raise ValueError('碰撞间隙和跟踪容差必须为有限非负数。')


def load_collision_profile(path: Path, rest: dict) -> dict:
    profile = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    capsules = profile.get('capsules', [])
    if len(capsules) != 2 or any(len(arm) != 4 for arm in capsules):
        raise ValueError('capsules 必须为 [左右臂, 肘侧/中段/腕侧/手掌]。')
    for h, arm in enumerate(capsules):
        for k, capsule in enumerate(arm):
            if capsule['bone'] != 18 + h + (0 if k < 3 else 2):
                raise ValueError('计算胶囊只支持 SMPL24 小臂和腕骨骼。')
            endpoints = torch.tensor([capsule['start'], capsule['end']], dtype=torch.float32)
            radius = torch.tensor(capsule['radius'])
            if endpoints.shape != (2, 3) or not torch.isfinite(endpoints).all() or not torch.isfinite(radius) or radius <= 0:
                raise ValueError('胶囊端点必须有限，半径必须为有限正数。')
    seat = profile.get('seatCapsules', [])
    if len(seat) != 3 or [c['bone'] for c in seat] != [0, 1, 2]:
        raise ValueError('seatCapsules 必须为臀部、左大腿、右大腿。')
    for c in seat:
        endpoints = torch.tensor([c['start'], c['end']])
        if endpoints.shape != (2, 3) or not torch.isfinite(endpoints).all() or not 0 < c['radius'] < float('inf'):
            raise ValueError('座面接触胶囊参数无效。')
    back=profile.get('backCapsules', [])
    for c in back:
        endpoints=torch.tensor([c['start'],c['end']])
        if c['bone'] not in (3,6,9) or endpoints.shape!=(2,3) or not torch.isfinite(endpoints).all() or not 0<c['radius']<float('inf'):
            raise ValueError('背部胶囊参数无效。')
    if back:
        box=torch.tensor(profile.get('backrest') or [])
        if box.shape!=(15,) or not torch.isfinite(box).all() or not (box[12:]>0).all():
            raise ValueError('靠背 OBB 参数无效。')
    # 标定必须绑定同一套骨架，避免把另一个角色的几何静默带入图中。
    offsets = torch.tensor(profile['restLocalPositions'], dtype=torch.float32)
    expected = torch.tensor(rest['restLocalPositions'], dtype=torch.float32)
    if offsets.shape != (24, 3) or expected.shape != (24, 3) or not torch.isfinite(offsets).all() or not torch.allclose(
            offsets, expected, atol=1e-6, rtol=0):
        raise ValueError('碰撞代理配置与导出角色的骨架不一致，请重新标定。')
    return profile


def collision_manifest(settings, profile, rest, sampler_name):
    return dict(settings=asdict(settings), profile=profile, parents=rest['parents'],
                restLocalPositions=rest['restLocalPositions'], sampler=sampler_name,
                inputs=COLLISION_INPUTS, interpolation=[.25, .5, .75, 1.],
                langevin_noise=False, reference='https://arxiv.org/html/2410.04844v2')


# region 可导出的旋转与骨架计算
def norm(value, eps=1e-12):
    return value.square().sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()


def unit(value):
    return value / norm(value)


def qnormalize(q):
    identity = torch.cat([torch.zeros_like(q[..., :3]), torch.ones_like(q[..., 3:])], -1)
    return torch.where(q.square().sum(-1, keepdim=True) > 1e-12, unit(q), identity)


def qconjugate(q):
    return torch.cat([-q[..., :3], q[..., 3:]], -1)


def cross3(a, b):
    """固定三维叉积；四次 Gather 避免 torch.cross 导出的大量 Slice/Concat。"""
    return a[..., [1, 2, 0]] * b[..., [2, 0, 1]] - a[..., [2, 0, 1]] * b[..., [1, 2, 0]]


def qmul(a, b):
    av, bv = a[..., :3], b[..., :3]
    return torch.cat([a[..., 3:] * bv + b[..., 3:] * av + cross3(av, bv),
                      a[..., 3:] * b[..., 3:] - (av * bv).sum(-1, keepdim=True)], -1)


def qrotate(q, v):
    u = 2 * cross3(q[..., :3], v)
    return v + q[..., 3:] * u + cross3(q[..., :3], u)


def qslerp(a, b, t):
    a, b = qnormalize(a), qnormalize(b)
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    cosine = dot.abs().clamp(0, 1)
    angle = cosine.minimum(torch.full_like(cosine, 1 - 1e-7)).acos()
    spherical = (torch.sin((1 - t) * angle) * a + torch.sin(t * angle) * b) / angle.sin()
    return qnormalize(torch.where(cosine > .9995, a + t * (b - a), spherical))


def from_to(a, b):
    av, bv = unit(a), unit(b)
    dot = (av * bv).sum(-1, keepdim=True).clamp(-1, 1)
    xaxis = torch.zeros_like(av) + av.new_tensor([1., 0., 0.])
    yaxis = torch.zeros_like(av) + av.new_tensor([0., 1., 0.])
    axis = unit(cross3(av, torch.where(av[..., :1].abs() < .9, xaxis, yaxis)))
    opposite = torch.cat([axis, torch.zeros_like(dot)], -1)
    normal = qnormalize(torch.cat([cross3(av, bv), 1 + dot], -1))
    result = torch.where(dot < -1 + 1e-6, opposite, normal)
    identity = torch.zeros_like(result) + result.new_tensor([0., 0., 0., 1.])
    valid = (a.square().sum(-1, keepdim=True) > 1e-12) & (b.square().sum(-1, keepdim=True) > 1e-12)
    return torch.where(valid, result, identity)


def matrix_quaternion(matrix):
    """按最大对角分支恢复 xyzw，避免接近 180° 时除以很小的 w。"""
    m00, m11, m22 = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    score = torch.stack([1 + m00 - m11 - m22, 1 - m00 + m11 - m22,
                         1 - m00 - m11 + m22, 1 + m00 + m11 + m22], -1)
    xy = matrix[..., 0, 1] + matrix[..., 1, 0]
    xz = matrix[..., 0, 2] + matrix[..., 2, 0]
    yz = matrix[..., 1, 2] + matrix[..., 2, 1]
    wx = matrix[..., 2, 1] - matrix[..., 1, 2]
    wy = matrix[..., 0, 2] - matrix[..., 2, 0]
    wz = matrix[..., 1, 0] - matrix[..., 0, 1]
    candidates = torch.stack([torch.stack([score[..., 0], xy, xz, wx], -1),
                              torch.stack([xy, score[..., 1], yz, wy], -1),
                              torch.stack([xz, yz, score[..., 2], wz], -1),
                              torch.stack([wx, wy, wz, score[..., 3]], -1)], -2)
    selected = score.argmax(-1, keepdim=True).unsqueeze(-1).expand(*score.shape[:-1], 1, 4)
    return qnormalize(candidates.gather(-2, selected).squeeze(-2))


def rotations_from_6d(pose):
    forward, up = pose[..., :3], pose[..., 3:]
    right = cross3(up, forward)
    return matrix_quaternion(torch.stack([right, up, forward], -1))


def forward_kinematics(root, root_q, pelvis, local, offsets, parents):
    positions, rotations = [], []
    for joint, parent in enumerate(parents):
        if parent < 0:
            rotations.append(qmul(root_q, local[:, joint]))
            positions.append(root + qrotate(root_q, pelvis))
        else:
            rotations.append(qmul(rotations[parent], local[:, joint]))
            positions.append(positions[parent] + qrotate(rotations[parent], offsets[joint].expand_as(root)))
    return torch.stack(positions, 1), torch.stack(rotations, 1)
# endregion


# region 胶囊与盒体：整条线段的最小有符号距离
def box_sdf(point, half):
    q = point.abs() - half
    outside = q.clamp_min(0)
    # sqrt(0) 的自动求导不稳定；减去 eps 后零距离仍精确为零。
    length = (outside.square().sum(-1) + 1e-16).sqrt() - 1e-8
    inside = q.amax(-1)
    # 显式 Min 避免 Unity GPU 路径中仅有上界的 ONNX Clip 未生效。
    return length + inside.minimum(torch.zeros_like(inside))


def safe_divide(numerator, denominator):
    return numerator / torch.where(denominator.abs() > 1e-9, denominator, torch.ones_like(denominator))


def segment_box_sdf(a, b, half):
    """分段二次距离的驻点，加盒内六平面交点；不以离散采样代替胶囊。"""
    direction = b - a
    breaks = torch.cat([torch.zeros_like(a[..., :1]), torch.ones_like(a[..., :1]),
                        safe_divide(half - a, direction), safe_divide(-half - a, direction)], -1)
    # 长度固定为八；显式 K 避免旧 ONNX exporter 为 sort 生成标量 K。
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
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    i, j = [p[0] for p in pairs], [p[1] for p in pairs]
    crossing = safe_divide(intercept[..., j] - intercept[..., i], slope[..., i] - slope[..., j]).clamp(0, 1)
    candidates = torch.cat([breaks[..., :1], breaks[..., -1:], stationary, crossing], -1)
    points = a.unsqueeze(-2) + candidates.unsqueeze(-1) * direction.unsqueeze(-2)
    return box_sdf(points, half.unsqueeze(-2)).amin(-1)


def top_gap(a, b, half, radius):
    """把轴线裁到有限扶手足迹，再求最低高度；半径扩张保守覆盖边缘。"""
    direction = b - a
    start, delta = a[..., [0, 2]], direction[..., [0, 2]]
    extent = half[..., [0, 2]] + radius.unsqueeze(-1)
    parallel = delta.abs() <= 1e-9
    t0 = safe_divide(-extent - start, delta)
    t1 = safe_divide(extent - start, delta)
    lower = torch.where(parallel, torch.zeros_like(t0), t0.minimum(t1)).amax(-1).clamp_min(0)
    upper = torch.where(parallel, torch.ones_like(t1), t0.maximum(t1)).amin(-1)
    upper = upper.minimum(torch.ones_like(upper))
    valid = (lower <= upper) & (~(parallel & (start.abs() > extent))).all(-1)
    y = (a[..., 1] + lower * direction[..., 1]).minimum(a[..., 1] + upper * direction[..., 1])
    return torch.where(valid, y - half[..., 1] - radius, torch.full_like(y, 1e3))


def primitive_gaps(start, end, radii, boxes, modes):
    """start/end [N,2,2,3]；box [1,2,15]；返回 [N,2,2,2] 间隙。"""
    axes = boxes[..., 3:12].reshape(-1, 2, 3, 3)
    def local(point):
        delta = point.unsqueeze(-2) - boxes[:, None, None, :, :3]
        return (axes[:, None, None] * delta.unsqueeze(-2)).sum(-1)
    a, b = local(start), local(end)
    half = boxes[:, None, None, :, 12:15]
    radius = radii[None, :, :, None]
    gap = segment_box_sdf(a, b, half) - radius
    top = top_gap(a, b, half, radius)
    gap = torch.where(modes[:, :, None, :] == 2, gap.minimum(top), gap)
    return torch.where(modes[:, :, None, :] > 0, gap, torch.full_like(gap, 1e3))
# endregion


class ResidualCollisionPostprocess(nn.Module):
    """batch=1 的部署模块；差分候选批次仅进入几何分支。"""

    def __init__(self, rest, profile, pose_mean, pose_scale, settings=None):
        super().__init__()
        self.settings = settings or CollisionSettings()
        self.settings.validate()
        self.parents = tuple(int(p) for p in rest['parents'])
        if self.parents != tuple(SMPL_PARENTS.tolist()):
            raise ValueError('碰撞几何固定使用 SMPL24 的肩、肘、腕层级。')
        self.register_buffer('offsets', torch.tensor(rest['restLocalPositions'], dtype=torch.float32))
        self.register_buffer('pose_mean', pose_mean.clone())
        self.register_buffer('pose_scale', pose_scale.clone())
        capsules = profile['capsules']
        self.register_buffer('capsule_start', torch.tensor([[c['start'] for c in arm] for arm in capsules], dtype=torch.float32))
        self.register_buffer('capsule_end', torch.tensor([[c['end'] for c in arm] for arm in capsules], dtype=torch.float32))
        self.register_buffer('radii', torch.tensor([[c['radius'] for c in arm] for arm in capsules], dtype=torch.float32))
        self.register_buffer('fractions', torch.tensor([.25, .5, .75, 1.]))
        active = [j * 6 + k for j in ACTIVE_JOINTS for k in range(6)]
        self.register_buffer('directions', torch.eye(144)[active])
        self.register_buffer('active_indices', torch.tensor(active))
        self.register_buffer('difference_offsets', torch.cat([torch.zeros(1, len(active)), torch.eye(len(active)), -torch.eye(len(active))]) * self.settings.difference_step)
        seat = profile['seatCapsules']
        self.register_buffer('seat_start', torch.tensor([c['start'] for c in seat], dtype=torch.float32))
        self.register_buffer('seat_end', torch.tensor([c['end'] for c in seat], dtype=torch.float32))
        self.register_buffer('seat_radii', torch.tensor([c['radius'] for c in seat], dtype=torch.float32))
        back = profile.get('backCapsules', [])
        self.back_count = len(back)
        self.register_buffer('back_bones', torch.tensor([c['bone'] for c in back], dtype=torch.long))
        self.register_buffer('back_start', torch.tensor([c['start'] for c in back], dtype=torch.float32).reshape(-1,3))
        self.register_buffer('back_end', torch.tensor([c['end'] for c in back], dtype=torch.float32).reshape(-1,3))
        self.register_buffer('back_radii', torch.tensor([c['radius'] for c in back], dtype=torch.float32))
        self.register_buffer('backrest', torch.tensor(profile.get('backrest') or [0.]*15, dtype=torch.float32))
        self.register_buffer('parent_indices', torch.tensor(self.parents[1:], dtype=torch.long))
    def target_components(self, normalized, tracker, display=None, height=None, forward=None):
        """一次投影后同时准备显示局部旋转和几何所需的全局旋转。"""
        count = normalized.shape[0]
        projected = project_realtime_pose_xstart(normalized, tracker.expand(count, -1, -1), self.pose_mean, self.pose_scale)
        raw = (projected * self.pose_scale + self.pose_mean).reshape(count, 24, 6)
        world = rotations_from_6d(raw)
        basis_forward = raw[:, 0, :3]
        right = cross3(raw[:, 0, 3:], basis_forward)
        planar = torch.stack([basis_forward[:, 0] - right[:, 2], torch.zeros_like(basis_forward[:, 0]),
                              right[:, 0] + basis_forward[:, 2]], -1)
        fallback = torch.zeros_like(planar) + planar.new_tensor([0., 0., 1.])
        if display is not None:
            fallback = qrotate(qnormalize(display[:, 3:7]).expand(count, -1), fallback)
        planar = torch.where(planar.square().sum(-1, keepdim=True) > 1e-8, unit(planar), fallback)
        up = torch.zeros_like(planar) + planar.new_tensor([0., 1., 0.])
        root_q = matrix_quaternion(torch.stack([cross3(up, planar), up, planar], -1))
        parent_world = torch.cat([root_q[:, None], world[:, self.parent_indices]], 1)
        local = qmul(qconjugate(parent_world), world)
        # 根锚点只依赖骨盆到头的链路，无需先构造其余关节的位置。
        head_relative = qrotate(world[:, [0, 3, 6, 9, 12]], self.offsets[None, [3, 6, 9, 12, 15]]).sum(1)
        hip = tracker[:, 0, 1] - head_relative[:, 1]
        if height is not None:
            hip = hip + height.reshape(-1)
        pelvis = torch.stack([torch.zeros_like(hip) + self.offsets[0, 0], hip,
                              torch.zeros_like(hip) + self.offsets[0, 2]], -1)
        head = head_relative + qrotate(root_q, pelvis)
        root = torch.stack([-head[:, 0], torch.zeros_like(hip), -head[:, 2]], -1)
        if forward is not None: root=root+forward
        return root, root_q, pelvis, local, world

    def target_frame(self, normalized, tracker, display=None, height=None, forward=None):
        root, root_q, pelvis, local, _ = self.target_components(normalized, tracker, display, height, forward)
        return torch.cat([root, root_q, pelvis, local.reshape(-1, 96)], -1)

    def seated_geometry(self, frame, context):
        # 座面由 posterior correction 处理；显示 FK 不再施加独立托举。
        return forward_kinematics(frame[:, :3], qnormalize(frame[:, 3:7]), frame[:, 7:10],
                           qnormalize(frame[:, 10:].reshape(-1, 24, 4)), self.offsets, self.parents)

    def interpolated_frames(self, normalized, tracker, display_from_pose, height=None, forward=None):
        target = self.target_frame(normalized, tracker, display_from_pose, height, forward)
        count = target.shape[0]
        t = self.fractions[None, :, None]
        begin, end = display_from_pose[:, None], target[:, None]
        root = begin[..., :3] + t * (end[..., :3] - begin[..., :3])
        root_q = qslerp(begin[..., 3:7], end[..., 3:7], t)
        pelvis = begin[..., 7:10] + t * (end[..., 7:10] - begin[..., 7:10])
        local = qslerp(begin[..., 10:].reshape(1, 1, 24, 4), end[..., 10:].reshape(count, 1, 24, 4), t[..., None])
        return torch.cat([root, root_q, pelvis, local.reshape(count, 4, 96)], -1).reshape(-1, 106)

    def display_geometry(self, normalized, tracker, seated_context, display_from_pose, height=None, forward=None):
        count = normalized.shape[0]
        frame = self.interpolated_frames(normalized, tracker, display_from_pose, height, forward)
        context = seated_context
        positions, rotations = self.seated_geometry(frame, context)
        return positions.reshape(count, 4, 24, 3), rotations.reshape(count, 4, 24, 4)

    def pelvis_reference(self, base_pose, tracker, context, display, height, height_context, history):
        # 原始基准不包含头部前后热启动值，历史覆盖头部平移与姿态调整的总修正。
        axis=context[:,9:12]*context.new_tensor([1.,0.,1.])
        axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        root,rotation,pelvis,_,_=self.target_components(base_pose,tracker,display,height)
        target=((root+qrotate(rotation,pelvis))*axis).sum(-1)+history[:,73]
        before=display[:,:3]+qrotate(qnormalize(display[:,3:7]),display[:,7:10])
        speed=(target-(before*axis).sum(-1)).abs()/height_context[:,1].clamp_min(.001)
        weight=self.settings.pelvis_forward_smooth_weight*history[:,72]*(context[:,15]>1.5)*((.6-speed)/.45).clamp(0,1)
        return axis,target,weight

    def measurement(self, normalized, tracker, boxes, modes, context, display, height=None, previous_height=None, pelvis_reference=None, forward=None, previous_forward=None):
        positions, rotations = self.display_geometry(normalized, tracker, context, display, height, forward)
        e, w, q = positions[:, :, [18, 19]], positions[:, :, [20, 21]], rotations[:, :, [20, 21]]
        loss, penetration = self.geometry_loss(e, w, q, rotations[:, :, [18, 19]], tracker, boxes, modes)
        origins, quat = positions[:, :, :3], rotations[:, :, :3]
        start = origins + qrotate(quat, self.seat_start)
        end = origins + qrotate(quat, self.seat_end)
        axes = context[:, 3:12].reshape(-1, 3, 3)
        def local(point):
            return ((point - context[:, None, None, :3]).unsqueeze(-2) * axes[:, None, None]).sum(-1)
        a, b = local(start), local(end)
        half = context[:, None, None, 12:15]
        gap = segment_box_sdf(a, b, half) - self.seat_radii
        gap = torch.where(context[:, 15:16, None] > 1.5, gap.minimum(top_gap(a, b, half, self.seat_radii)), gap)
        violation = torch.where(context[:, 15:16, None] > 0, (self.settings.margin - gap).clamp_min(0), torch.zeros_like(gap))
        # 三个独立接触部位分别求和，避免最深的一个遮蔽另一侧大腿的梯度。
        seat_loss = self.settings.seat_collision_weight * violation.square().sum((1, 2))
        if self.back_count:
            origins, quat = positions[:, :, self.back_bones], rotations[:, :, self.back_bones]
            a, b = local(origins + qrotate(quat, self.back_start)), local(origins + qrotate(quat, self.back_end))
            back_axes = self.backrest[3:12].reshape(3,3)
            a = ((a-self.backrest[:3]).unsqueeze(-2)*back_axes).sum(-1)
            b = ((b-self.backrest[:3]).unsqueeze(-2)*back_axes).sum(-1)
            half = self.backrest[12:15]
            gap = (segment_box_sdf(a,b,half)-self.back_radii).minimum(top_gap(a[..., [0,2,1]],b[..., [0,2,1]],half[[0,2,1]],self.back_radii))
            back_v = torch.where(context[:,15:16,None]>0,(self.settings.margin-gap).clamp_min(0),torch.zeros_like(gap))
            seat_loss = seat_loss + back_v.square().sum((1,2))
            penetration = penetration.maximum(back_v.flatten(1).amax(-1))
        error = norm(positions[:, -1, [0, 10, 11]] - tracker[:, [3, 4, 5], :3]).squeeze(-1)
        error = (error - self.settings.tracking_tolerance).clamp_min(0) * tracker[:, [3, 4, 5], 9]
        world = rotations[:, -1]
        # 头部仍使用跟踪旋转；末端父子差使颈部不能靠相对头部后仰逃避分配约束。
        local = qnormalize(qmul(qconjugate(world[:, [0,3,6,9,12]]), world[:, [3,6,9,12,15]]))
        bend = 2*local[..., 0]*torch.where(local[..., 3] < 0, -1., 1.)
        excess = (abs(bend[:, 1:]-bend[:, :-1])-self.settings.spine_bend_tolerance).clamp_min(0)
        spine_loss = self.settings.spine_bend_scale**2*excess.square().sum(-1)
        penetration = penetration.maximum(self.settings.spine_bend_scale*excess.amax(-1))
        curve = (bend[:, :4].abs().sum(-1)-self.settings.torso_curve_limit).clamp_min(0)*self.settings.torso_curve_scale
        chest_head = qnormalize(qmul(qconjugate(world[:,9]),world[:,15]))
        neck = (abs(2*chest_head[:,0])-self.settings.chest_head_limit).clamp_min(0)*self.settings.chest_head_scale
        spine_loss = spine_loss + curve.square() + neck.square()
        penetration = penetration.maximum(curve).maximum(neck)
        total = loss.sum(-1) + (seat_loss + spine_loss + self.settings.body_tracking_weight*error.square().sum(-1)) / (2 * self.settings.m ** 2)
        if height is not None:
            y = height.reshape(-1)
            previous = 0. if previous_height is None else previous_height.reshape(-1)
            total = total + (self.settings.head_y_weight*y.square()+self.settings.head_y_smooth_weight*(y-previous).square())/(2*self.settings.m**2)
            penetration = penetration.maximum(y.abs())
        if forward is not None:
            axis=context[:,9:12]*context.new_tensor([1.,0.,1.]);axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            f=(forward*axis).sum(-1);previous_f=0. if previous_forward is None else previous_forward.reshape(-1)
            total=total+(self.settings.head_forward_weight*f.square()+self.settings.head_forward_smooth_weight*(f-previous_f).square())/(2*self.settings.m**2)
            penetration=penetration.maximum(f.abs())
        if pelvis_reference is not None:
            axis,target,weight=pelvis_reference
            forward=(positions[:,-1,0]*axis).sum(-1)
            error=forward-target
            total=total+weight*(error.square())/(2*self.settings.m**2)
        return total, penetration.maximum(violation.flatten(1).amax(-1))

    def geometry_loss(self, e, w, q, eq, tracker, boxes, modes):
        """几何 [N,4,2,...]，返回独立的左右臂损失 [N,2]；不合并跨臂项。"""
        count = e.shape[0]
        # [候选×插值, 左右臂, 肘侧/中段/腕侧/手掌, xyz/xyzw]
        origins = torch.stack([e, e, e, w], dim=-2).reshape(-1, 2, 4, 3)
        rotations = torch.stack([eq, eq, eq, q], dim=-2).reshape(-1, 2, 4, 4)
        start = origins + qrotate(rotations, self.capsule_start[None].expand_as(origins))
        end = origins + qrotate(rotations, self.capsule_end[None].expand_as(origins))
        gaps = primitive_gaps(start, end, self.radii, boxes, modes)
        penetration = (self.settings.margin - gaps).clamp_min(0).reshape(count, 4, 2, 4, 2)
        # 只约束最终候选的手腕；过渡起点的跟踪延迟不能通过让终点过冲补偿。
        error = norm(w[:, -1] - tracker[:, [1, 2], :3]).squeeze(-1)
        error = (error - self.settings.tracking_tolerance).clamp_min(0) * tracker[:, [1, 2], 9]
        # 延续远端三段小臂的聚合方式，重叠胶囊不重复累加惩罚。
        squared = penetration.square()
        contact = squared[..., :3, :].amax(dim=3) + squared[..., 3, :]
        loss = (contact.sum((1, 3)) + error.square()) / (2 * self.settings.m ** 2)
        return loss, penetration.reshape(count, -1).amax(-1)

    def forward(self, prior, residual, tracker, boxes, modes, context, display, variance):
        if self.settings.head_forward_limit > 0 or self.settings.head_y_limit > 0 or self.settings.correction_smooth_weight > 0 or self.settings.pelvis_forward_smooth_weight > 0:
            raise ValueError('启用高度或修正量时序约束必须使用 forward_with_height 并传递相应状态。')
        edited,_ = self.forward_with_height(prior,residual,tracker,boxes,modes,context,display,variance,
                                          prior.new_tensor([[0.,1/30]]))
        return edited

    def forward_with_height(self, prior, residual, tracker, boxes, modes, context, display, variance,
                            height_context, initial_height=None, trace=None, correction_context=None, forward_context=None, initial_forward=None, forward_result=None):
        """高度上下文 [1,2] = 上次接受的偏移(m)、时间差(s)；返回残差及高度；forward_result["offset"] 保存前后偏移(m)。"""
        reference = residual[:, self.active_indices]
        # 当前采样阶段的历史 [1,74]：72 个归一化修正量及有效权重，只约束后48个脊柱/腿部坐标。
        history = prior.new_zeros(1,74) if correction_context is None else correction_context
        def temporal(current):
            error = current[:,24:72]-reference[:,24:]-history[:,24:72]
            weight = self.settings.correction_smooth_weight*history[:,72:73]
            return .5*weight[:,0]*error.square().sum(-1), torch.nn.functional.pad(weight*error,(24,2))
        previous = height_context[:, :1]
        height = previous if initial_height is None else initial_height.reshape(1,1)
        height = height.clamp(-self.settings.head_y_limit, self.settings.head_y_limit)
        previous_f=prior.new_zeros(1,1) if forward_context is None else forward_context.reshape(1,1)
        step_f=self.settings.head_forward_speed*height_context[:,1:2].clamp(0,.1)
        lo_f=(previous_f-step_f).clamp_min(-self.settings.head_forward_limit)/.1
        hi_f=(previous_f+step_f).clamp_max(self.settings.head_forward_limit)/.1
        release=previous_f.sign()*(previous_f.abs()-step_f).clamp_min(0)/.1
        lo_f=torch.where(context[:,15:16]>1.5,lo_f,release);hi_f=torch.where(context[:,15:16]>1.5,hi_f,release)
        f=previous_f if initial_forward is None else initial_forward.reshape(1,1)
        f=(f/.1).maximum(lo_f).minimum(hi_f)
        axis=context[:,9:12]*context.new_tensor([1.,0.,1.]);axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        current = torch.cat([reference, height/.1,f], -1)
        offsets = torch.cat([torch.nn.functional.pad(self.difference_offsets,(0,2)),
                             torch.cat([torch.zeros(4,72,device=prior.device), prior.new_tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])*self.settings.difference_step],-1)])
        variance = variance.clamp_min(1e-6)
        delta = self.settings.head_y_speed*height_context[:,1:2].clamp(0,.1)
        lo = (previous-delta).clamp_min(-self.settings.head_y_limit)/.1
        hi = (previous+delta).clamp_max(self.settings.head_y_limit)/.1
        warm = current+torch.nn.functional.pad(history[:,24:72]*(history[:,72:73]>0),(24,2))
        pelvis_reference=self.pelvis_reference(prior+residual,tracker,context,display,height,height_context,history)
        def evaluate(candidates):
            pose = prior+residual+(candidates[:,:72]-reference)@self.directions
            options=dict(forward=axis*candidates[:,73:74]*.1,previous_forward=previous_f) if self.settings.head_forward_limit>0 else {}
            return self.measurement(pose,tracker,boxes,modes,context,display,candidates[:,72]*.1,previous,pelvis_reference,**options)
        starts = torch.cat([current,warm])
        start_loss,start_penetration = evaluate(starts)
        start_energy = start_loss+(starts[:,:72]-reference).square().sum(-1)/(2*variance)+temporal(starts)[0]
        use_warm=(start_penetration[:1]>0) & torch.isfinite(start_energy[1:]) & (start_energy[1:]<start_energy[:1])
        current=torch.where(use_warm[:,None],warm,current)
        def forward_position(candidate):
            pose=prior+residual+(candidate[:,:72]-reference)@self.directions
            root,rq,pp,_,_=self.target_components(pose,tracker,display,candidate[:,72:73]*.1,axis*candidate[:,73:74]*.1)
            return ((root+qrotate(rq,pp))*pelvis_reference[0]).sum(-1)
        for k in range(self.settings.steps):
            values, penetration = evaluate(current+offsets)
            temporal_energy,temporal_gradient = temporal(current)
            energy = values[:1]+(current[:,:72]-reference).square().sum(-1)/(2*variance)+temporal_energy
            if k == 0:
                active = penetration[:1,None] > 0
                best, best_energy = current, energy
            else:
                better = (torch.isfinite(energy) & (energy < best_energy))[:,None]
                best = torch.where(better,current,best)
                best_energy = torch.where(better[:,0],energy,best_energy)
            g = torch.cat([values[1:73]-values[73:145],values[145:146]-values[146:147],values[147:148]-values[148:149]])[None]/(2*self.settings.difference_step)
            g = g+torch.nn.functional.pad((current[:,:72]-reference)/variance,(0,2))
            g = g+temporal_gradient
            if trace is not None and k < 5:
                trace[k*146:k*146+72]=current[0,:72]
                trace[k*146+72:k*146+144]=g[0,:72]
                trace[k*146+144]=energy[0];trace[k*146+145]=best_energy[0]
                trace[1000+k*2]=current[0,72];trace[1001+k*2]=g[0,72]
                trace[1030+k*2]=current[0,73];trace[1031+k*2]=g[0,73]
            proposed = current-self.settings.h*(1-.99*k/self.settings.steps)*g
            # 原步长及三次折半，仅接受总能量不增加的更新；不新增跨帧状态。
            direction = proposed-current
            accepted = current
            accepted_energy = energy
            accepted_scale = current.new_zeros(1)
            pending = active & torch.isfinite(direction).all(-1,keepdim=True)
            direction = torch.where(pending,direction,torch.zeros_like(direction))
            base_forward=forward_position(current)
            for attempt in range(4):
                trial = current+direction*(.5**attempt)
                trial = torch.cat([trial[:,:72],trial[:,72:73].maximum(lo).minimum(hi),trial[:,73:74].maximum(lo_f).minimum(hi_f)],-1)
                trial_loss,_ = evaluate(trial)
                trial_energy = trial_loss+(trial[:,:72]-reference).square().sum(-1)/(2*variance)+temporal(trial)[0]
                movement_cost=pelvis_reference[2]*self.settings.pelvis_forward_update_cost*(forward_position(trial)-base_forward).abs()/(2*self.settings.m**2)
                take = pending & (torch.isfinite(trial_energy) & (trial_energy+movement_cost<=energy))[:,None]
                accepted = torch.where(take,trial,accepted)
                accepted_energy = torch.where(take[:,0],trial_energy,accepted_energy)
                accepted_scale = torch.where(take[:,0],.5**attempt,accepted_scale)
                pending = pending & ~take
            if trace is not None and k<20:
                trace[1100+k*3]=energy[0];trace[1101+k*3]=accepted_energy[0];trace[1102+k*3]=accepted_scale[0]
            current = accepted
        measured,_ = evaluate(current)
        energy = measured+(current[:,:72]-reference).square().sum(-1)/(2*variance)+temporal(current)[0]
        best = torch.where((torch.isfinite(energy) & (energy<best_energy))[:,None],current,best)
        edited = torch.where(active,best[:,:72],reference)
        if trace is not None:
            trace[800:872]=current[0,:72];trace[872:944]=edited[0]
            trace[1020]=current[0,73];trace[1021]=best[0,73]*.1 if active.item() else 0
            trace[1010]=current[0,72];trace[1011]=best[0,72]*.1 if active.item() else 0
            base_position=pelvis_reference[1]-history[:,73]
            best_pose=prior+residual+(edited-reference)@self.directions
            root,rq,pp,_,_=self.target_components(best_pose,tracker,display,best[:,72:73]*.1,axis*best[:,73:74]*.1)
            selected=((root+qrotate(rq,pp))*pelvis_reference[0]).sum(-1)
            trace[1012]=base_position[0];trace[1013]=pelvis_reference[1][0];trace[1014]=pelvis_reference[2][0]
            trace[1015]=selected[0]-base_position[0] if active.item() else 0
            pose=prior+residual+(current[:,:72]-reference)@self.directions
            p,q=self.display_geometry(pose,tracker,context,display,current[:,72]*.1,axis*current[:,73:74]*.1)
            trace[2000:]=torch.cat([p,q],-1).flatten()
        if forward_result is not None: forward_result["offset"]=torch.where(active,best[:,73:74]*.1,torch.zeros_like(previous_f))
        return residual.scatter(1,self.active_indices[None],edited),torch.where(active,best[:,72:73]*.1,torch.zeros_like(previous))
