"""Unity 部署专用的确定性残差编辑；不接入训练和 Python 常规采样。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn

from data_loaders.realtime_pose_kinematics import SMPL_PARENTS
from diffusion.realtime_pose_projection import project_realtime_pose_xstart, project_rotation_6d_to_so3


ACTIVE_JOINTS = (16, 17, 18, 19)
COLLISION_INPUTS = ['armrest_geometry', 'contact_mode', 'seated_context', 'display_from_pose']


@dataclass(frozen=True)
class CollisionSettings:
    steps: int = 5
    h: float = 1e-4
    m: float = .01
    difference_step: float = 1e-3
    margin: float = .002
    tracking_tolerance: float = .02

    def validate(self):
        if self.steps < 1:
            raise ValueError('碰撞迭代次数必须为正。')
        if not all(torch.isfinite(torch.tensor(v)) and v > 0 for v in
                   (self.h, self.m, self.difference_step)):
            raise ValueError('PostEdit 步长、测量尺度和差分步幅必须为有限正数。')
        if not all(torch.isfinite(torch.tensor(v)) and v >= 0 for v in
                   (self.margin, self.tracking_tolerance)):
            raise ValueError('碰撞间隙和跟踪容差必须为有限非负数。')


def load_collision_profile(path: Path, rest: dict) -> dict:
    profile = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for name, shape in [('forearmRadii', (2,)), ('palmSpheres', (2, 3, 4))]:
        value = torch.tensor(profile[name], dtype=torch.float32)
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError(f'{name} 必须为有限的 {shape} 数组。')
        radii = value if name == 'forearmRadii' else value[..., 3]
        if (radii <= 0).any():
            raise ValueError(f'{name} 的半径必须为正。')
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
    """start/end [N,2,4,3]；box [1,2,15]；返回 [N,2,4,2] 间隙。"""
    axes = boxes[..., 3:12].reshape(-1, 2, 3, 3)
    def local(point):
        delta = point.unsqueeze(-2) - boxes[:, None, None, :, :3]
        return (axes[:, None, None] * delta.unsqueeze(-2)).sum(-1)
    a, b = local(start), local(end)
    half = boxes[:, None, None, :, 12:15]
    radius = radii[None, :, :, None]
    # 代理轴依次为胶囊、三个球；球心距离与退化线段等价，无需重复分段和排序。
    capsule = segment_box_sdf(a[:, :, :1], b[:, :, :1], half)
    spheres = box_sdf(a[:, :, 1:], half)
    gap = torch.cat([capsule, spheres], dim=2) - radius
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
        self.register_buffer('palms', torch.tensor(profile['palmSpheres'], dtype=torch.float32))
        radii = torch.cat([torch.tensor(profile['forearmRadii'])[:, None], self.palms[..., 3]], -1)
        self.register_buffer('radii', radii)
        self.register_buffer('fractions', torch.tensor([.25, .5, .75, 1.]))
        active = [j * 6 + k for j in ACTIVE_JOINTS for k in range(6)]
        self.register_buffer('directions', torch.eye(144)[active])
        # 肩/肘各六维，同时扰动左右臂的同一分量；损失按手臂分开归约。
        paired = torch.eye(12).reshape(12, 2, 1, 6).expand(-1, -1, 2, -1).reshape(12, 24)
        self.register_buffer('difference_offsets', torch.cat([torch.zeros(1, 24), paired, -paired])
                             * self.settings.difference_step)
        self.register_buffer('parent_indices', torch.tensor(self.parents[1:], dtype=torch.long))
        masks = []
        for joint in range(24):
            row = []
            for child in range(24):
                ancestor = child
                while ancestor >= 0 and ancestor != joint:
                    ancestor = self.parents[ancestor]
                row.append(ancestor == joint)
            masks.append(row)
        self.register_buffer('descendants', torch.tensor(masks, dtype=torch.bool))

    def target_components(self, normalized, tracker, display=None):
        """一次投影后同时准备显示局部旋转和几何所需的全局旋转。"""
        count = normalized.shape[0]
        projected = project_realtime_pose_xstart(normalized, tracker.expand(count, -1, -1), self.pose_mean, self.pose_scale)
        raw = (projected * self.pose_scale + self.pose_mean).reshape(count, 24, 6)
        world = rotations_from_6d(raw)
        forward = raw[:, 0, :3]
        right = cross3(raw[:, 0, 3:], forward)
        planar = torch.stack([forward[:, 0] - right[:, 2], torch.zeros_like(forward[:, 0]),
                              right[:, 0] + forward[:, 2]], -1)
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
        pelvis = torch.stack([torch.zeros_like(hip) + self.offsets[0, 0], hip,
                              torch.zeros_like(hip) + self.offsets[0, 2]], -1)
        head = head_relative + qrotate(root_q, pelvis)
        root = torch.stack([-head[:, 0], torch.zeros_like(hip), -head[:, 2]], -1)
        return root, root_q, pelvis, local, world

    def target_frame(self, normalized, tracker, display=None):
        root, root_q, pelvis, local, _ = self.target_components(normalized, tracker, display)
        return torch.cat([root, root_q, pelvis, local.reshape(-1, 96)], -1)

    def rotate_subtree(self, positions, rotations, joint, delta):
        center = positions[:, joint:joint + 1]
        mask = self.descendants[joint][None, :, None]
        positions = torch.where(mask, center + qrotate(delta[:, None], positions - center), positions)
        rotations = torch.where(mask, qmul(delta[:, None], rotations), rotations)
        return positions, rotations

    def solve_arm(self, positions, rotations, shoulder, goal, wrist_q):
        elbow, wrist = shoulder + 2, shoulder + 4
        a, b, c = positions[:, shoulder], positions[:, elbow], positions[:, wrist]
        ab, bc = norm(b - a), norm(c - b)
        delta = goal - a
        distance = norm(delta).maximum((ab - bc).abs() + .0001).minimum(ab + bc - .0001)
        axis = unit(delta)
        bend = b - a - ((b - a) * axis).sum(-1, keepdim=True) * axis
        forward = qrotate(rotations[:, shoulder], torch.zeros_like(a) + a.new_tensor([0., 0., 1.]))
        fallback = forward - (forward * axis).sum(-1, keepdim=True) * axis
        secondary = cross3(axis, torch.where(axis[:, :1].abs() < .9,
                                torch.zeros_like(a) + a.new_tensor([1., 0., 0.]),
                                torch.zeros_like(a) + a.new_tensor([0., 1., 0.])))
        fallback = torch.where(fallback.square().sum(-1, keepdim=True) < 1e-8, secondary, fallback)
        bend = unit(torch.where(bend.square().sum(-1, keepdim=True) < 1e-8, fallback, bend))
        x = (ab.square() - bc.square() + distance.square()) / (2 * distance.clamp_min(1e-6))
        middle = a + axis * x + bend * (ab.square() - x.square()).clamp_min(0).sqrt()
        valid = (ab > 1e-5) & (bc > 1e-5) & (delta.square().sum(-1, keepdim=True) >= 1e-10)
        original_p, original_q = positions, rotations
        positions, rotations = self.rotate_subtree(positions, rotations, shoulder, from_to(b - a, middle - a))
        positions, rotations = self.rotate_subtree(positions, rotations, elbow,
            from_to(positions[:, wrist] - positions[:, elbow], a + axis * distance - positions[:, elbow]))
        positions, rotations = self.rotate_subtree(positions, rotations, wrist,
                                                   qmul(wrist_q, qconjugate(rotations[:, wrist])))
        return torch.where(valid[:, None], positions, original_p), torch.where(valid[:, None], rotations, original_q)

    def seat_torso(self, positions, rotations, context):
        original_p, original_q = positions, rotations
        head, head_q = positions[:, 15], rotations[:, 15]
        lift = (context[:, 0] + context[:, 2] - positions[:, 0, 1]).clamp_min(0).minimum(context[:, 3].clamp_min(0)) * context[:, 1].clamp(0, 1)
        positions = positions + lift[:, None, None] * positions.new_tensor([0., 1., 0.])
        for _ in range(3):
            for joint in (9, 6, 3):
                positions, rotations = self.rotate_subtree(positions, rotations, joint,
                    from_to(positions[:, 15] - positions[:, joint], head - positions[:, joint]))
        positions, rotations = self.rotate_subtree(positions, rotations, 15, qmul(head_q, qconjugate(rotations[:, 15])))
        return (torch.where(lift[:, None, None] > 0, positions, original_p),
                torch.where(lift[:, None, None] > 0, rotations, original_q), lift)

    def seated_geometry(self, frame, context):
        positions, rotations = forward_kinematics(frame[:, :3], qnormalize(frame[:, 3:7]), frame[:, 7:10],
                           qnormalize(frame[:, 10:].reshape(-1, 24, 4)), self.offsets, self.parents)
        original_p, original_q = positions, rotations
        left, right = positions[:, 20], positions[:, 21]
        left_q, right_q = rotations[:, 20], rotations[:, 21]
        positions, rotations, lift = self.seat_torso(positions, rotations, context)
        positions, rotations = self.solve_arm(positions, rotations, 16, left, left_q)
        positions, rotations = self.solve_arm(positions, rotations, 17, right, right_q)
        return (torch.where(lift[:, None, None] > 0, positions, original_p),
                torch.where(lift[:, None, None] > 0, rotations, original_q))

    def interpolated_frames(self, normalized, tracker, display_from_pose):
        target = self.target_frame(normalized, tracker, display_from_pose)
        count = target.shape[0]
        t = self.fractions[None, :, None]
        begin, end = display_from_pose[:, None], target[:, None]
        root = begin[..., :3] + t * (end[..., :3] - begin[..., :3])
        root_q = qslerp(begin[..., 3:7], end[..., 3:7], t)
        pelvis = begin[..., 7:10] + t * (end[..., 7:10] - begin[..., 7:10])
        local = qslerp(begin[..., 10:].reshape(1, 1, 24, 4), end[..., 10:].reshape(count, 1, 24, 4), t[..., None])
        return torch.cat([root, root_q, pelvis, local.reshape(count, 4, 96)], -1).reshape(-1, 106)

    def display_geometry(self, normalized, tracker, seated_context, display_from_pose):
        count = normalized.shape[0]
        frame = self.interpolated_frames(normalized, tracker, display_from_pose)
        context = seated_context[:, None].expand(count, 4, 4).reshape(-1, 4)
        positions, rotations = self.seated_geometry(frame, context)
        return positions.reshape(count, 4, 24, 3), rotations.reshape(count, 4, 24, 4)

    def prepare_geometry(self, reference, tracker, context, display):
        """只准备头、脊柱和肩部；内层编辑不改变这些量，腿和手指无需进入图。"""
        root, root_q, pelvis, local, global_q = self.target_components(reference, tracker, display)
        t = self.fractions[:, None]
        root = display[:, :3] + t * (root - display[:, :3])
        pelvis = display[:, 7:10] + t * (pelvis - display[:, 7:10])
        root_q = qnormalize(qslerp(display[:, 3:7], root_q, t))
        start_local = display[:, 10:].reshape(1, 24, 4)
        joints = [0, 3, 6, 9, 12, 13, 14]
        interpolated = qnormalize(qslerp(start_local[:, joints], local[:, joints], t[:, None]))
        p = root + qrotate(root_q, pelvis)
        q = qmul(root_q, interpolated[:, 0])
        lift = (context[:, 0] + context[:, 2] - p[:, 1]).clamp_min(0).minimum(context[:, 3].clamp_min(0))
        lift = lift[:, None] * context[:, 1:2].clamp(0, 1)
        spine = []
        for index, joint in enumerate((3, 6, 9), start=1):
            p = p + qrotate(q, self.offsets[joint])
            q = qmul(q, interpolated[:, index])
            spine.append(p)
        # Spine3 下的颈部和双侧锁骨可以一次并行计算。
        children_q = qmul(q[:, None], interpolated[:, 4:])
        children_p = p[:, None] + qrotate(q[:, None], self.offsets[None, [12, 13, 14]])
        head = children_p[:, 0] + qrotate(children_q[:, 0], self.offsets[15])
        before_parent = children_q[:, 1:]
        shoulders = children_p[:, 1:] + qrotate(before_parent, self.offsets[None, [16, 17]])
        # CCD 只需三个支点、头和双肩的位置；锁骨的旋转变化由共同的 delta 累积。
        points = torch.cat([torch.stack(spine, 1), head[:, None], shoulders], 1)
        points = points + lift[:, None] * points.new_tensor([0., 1., 0.])
        identity = torch.zeros_like(root_q) + root_q.new_tensor([0., 0., 0., 1.])
        delta_q = identity
        for _ in range(3):
            for joint in (2, 1, 0):
                center = points[:, joint:joint + 1]
                delta = from_to(points[:, 3] - center[:, 0], head - center[:, 0])
                # 紧凑数组按祖先顺序排列；当前支点后面的点都是后代。
                tail = center + qrotate(delta[:, None], points[:, joint + 1:] - center)
                points = torch.cat([points[:, :joint + 1], tail], 1)
                delta_q = qmul(delta, delta_q)
        supported = torch.where(lift[:, None] > 0, points[:, 4:], shoulders)
        delta_q = torch.where(lift > 0, delta_q, identity)
        return (global_q[:, [13, 14]], global_q[:, [20, 21]], start_local[:, 16:22],
                before_parent, shoulders, supported, delta_q[:, None], lift[:, 0])

    def cached_upper_geometry(self, active_pose, cache):
        """active_pose [N,24]：只解码肩肘，再求六个上肢局部旋转及双臂 IK。"""
        parent, wrist, start_local, before_parent, before_shoulder, after_shoulder, delta_q, lift = cache
        count = active_pose.shape[0]
        raw = (active_pose * self.pose_scale[96:120] + self.pose_mean[96:120]).reshape(count, 4, 6)
        global_q = rotations_from_6d(project_rotation_6d_to_so3(raw))
        shoulder_q, elbow_q = global_q[:, :2], global_q[:, 2:]
        target_local = torch.cat([qmul(qconjugate(parent), shoulder_q), qmul(qconjugate(shoulder_q), elbow_q),
                                  qmul(qconjugate(elbow_q), wrist)], 1)
        local = qslerp(start_local[:, None], target_local[:, None], self.fractions[None, :, None, None])
        shoulder = qmul(before_parent[None], local[:, :, :2])
        elbow = qmul(shoulder, local[:, :, 2:4])
        wrist_q = qmul(elbow, local[:, :, 4:6])
        e = before_shoulder[None] + qrotate(shoulder, self.offsets[None, None, [18, 19]].expand(count, 4, -1, -1))
        w = e + qrotate(elbow, self.offsets[None, None, [20, 21]].expand(count, 4, -1, -1))
        a = after_shoulder[None].expand(count, -1, -1, -1)
        b = a + qrotate(delta_q[None], e - before_shoulder[None])
        c = a + qrotate(delta_q[None], w - before_shoulder[None])
        ab, bc = norm(b - a), norm(c - b)
        goal_delta = w - a
        distance = norm(goal_delta).maximum((ab - bc).abs() + .0001).minimum(ab + bc - .0001)
        axis = unit(goal_delta)
        bend = b - a - ((b - a) * axis).sum(-1, keepdim=True) * axis
        forward = qrotate(qmul(delta_q[None], shoulder), torch.zeros_like(a) + a.new_tensor([0., 0., 1.]))
        fallback = forward - (forward * axis).sum(-1, keepdim=True) * axis
        secondary = cross3(axis, torch.where(axis[..., :1].abs() < .9,
                                torch.zeros_like(a) + a.new_tensor([1., 0., 0.]),
                                torch.zeros_like(a) + a.new_tensor([0., 1., 0.])))
        fallback = torch.where(fallback.square().sum(-1, keepdim=True) < 1e-8, secondary, fallback)
        bend = unit(torch.where(bend.square().sum(-1, keepdim=True) < 1e-8, fallback, bend))
        x = (ab.square() - bc.square() + distance.square()) / (2 * distance.clamp_min(1e-6))
        middle = a + axis * x + bend * (ab.square() - x.square()).clamp_min(0).sqrt()
        valid = (ab > 1e-5) & (bc > 1e-5) & (goal_delta.square().sum(-1, keepdim=True) >= 1e-10)
        supported_e = torch.where(valid, middle, b)
        supported_w = torch.where(valid, a + axis * distance, c)
        supported_q = torch.where(valid, wrist_q, qmul(delta_q[None], wrist_q))
        mask = lift[None, :, None, None] > 0
        return torch.where(mask, supported_e, e), torch.where(mask, supported_w, w), torch.where(mask, supported_q, wrist_q)

    def measurement(self, normalized, tracker, boxes, modes, context, display, cache=None):
        if cache is None:
            positions, rotations = self.display_geometry(normalized, tracker, context, display)
            e, w, q = positions[:, :, [18, 19]], positions[:, :, [20, 21]], rotations[:, :, [20, 21]]
        else:
            e, w, q = self.cached_upper_geometry(normalized[:, 96:120], cache)
        loss, penetration = self.geometry_loss(e, w, q, tracker, boxes, modes)
        return loss.sum(-1), penetration

    def arm_measurement(self, active_pose, tracker, boxes, modes, cache):
        e, w, q = self.cached_upper_geometry(active_pose, cache)
        return self.geometry_loss(e, w, q, tracker, boxes, modes)

    def geometry_loss(self, e, w, q, tracker, boxes, modes):
        """几何 [N,4,2,...]，返回独立的左右臂损失 [N,2]；不合并跨臂项。"""
        count = e.shape[0]
        wrists, elbows, wrist_q = w.reshape(-1, 2, 3), e.reshape(-1, 2, 3), q.reshape(-1, 2, 4)
        centers = wrists[:, :, None] + qrotate(wrist_q[:, :, None], self.palms[None, :, :, :3].expand(wrists.shape[0], -1, -1, -1))
        start = torch.cat([elbows[:, :, None], centers], 2)
        end = torch.cat([wrists[:, :, None], centers], 2)
        gaps = primitive_gaps(start, end, self.radii, boxes, modes)
        penetration = (self.settings.margin - gaps).clamp_min(0).reshape(count, 4, 2, 4, 2)
        # 只约束最终候选的手腕；过渡起点的跟踪延迟不能通过让终点过冲补偿。
        error = norm(w[:, -1] - tracker[:, [1, 2], :3]).squeeze(-1)
        error = (error - self.settings.tracking_tolerance).clamp_min(0) * tracker[:, [1, 2], 9]
        loss = (penetration.square().sum((1, 3, 4)) + error.square()) / (2 * self.settings.m ** 2)
        return loss, penetration.reshape(count, -1).amax(-1)

    def forward(self, prior, residual, tracker, boxes, modes, context, display, variance):
        reference = residual[:, 96:120]
        current = reference
        cache = self.prepare_geometry(prior + residual, tracker, context, display)
        prior_active = prior[:, 96:120]
        variance = variance.clamp_min(1e-6)
        epsilon = self.settings.difference_step
        for k in range(self.settings.steps):
            candidates = current + self.difference_offsets
            values, penetration = self.arm_measurement(prior_active + candidates, tracker, boxes, modes, cache)
            energy = values[:1].sum(-1) + (current - reference).square().sum(-1) / (2 * variance)
            if k == 0:
                active = penetration[:1, None] > 0
                best, best_energy = current, energy
            else:
                better = (torch.isfinite(energy) & (energy < best_energy))[:, None]
                best = torch.where(better, current, best)
                best_energy = torch.where(better[:, 0], energy, best_energy)
            # [12,2] -> [肩/肘,左/右,6]，恢复连续的 24 个活动分量。
            paired_gradient = (values[1:13] - values[13:]) / (2 * epsilon)
            geometry_gradient = paired_gradient.reshape(2, 6, 2).permute(0, 2, 1).reshape(1, 24)
            gradient = geometry_gradient + (current - reference) / variance
            h = self.settings.h * (1 - .99 * k / self.settings.steps)
            # 只保留能量中的先验回拉，更新与最佳候选筛选使用同一个目标。
            proposed = current - h * gradient
            proposed = torch.where(active, proposed, reference)
            valid = torch.isfinite(proposed).all(-1, keepdim=True)
            proposed = torch.where(valid, proposed, current)
            current = proposed
        measured, _ = self.arm_measurement(prior_active + current, tracker, boxes, modes, cache)
        energy = measured.sum(-1) + (current - reference).square().sum(-1) / (2 * variance)
        better = (torch.isfinite(energy) & (energy < best_energy))[:, None]
        best = torch.where(better, current, best)
        edited = torch.where(active, best, reference)
        # 只在输出时写回活动切片；其他 120 维直接复用输入，精确保留。
        return torch.cat([residual[:, :96], edited, residual[:, 120:]], -1)
