"""独立几何参考、残差边界和部署数值；不要求训练产物或 Unity 安装。"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from export.residual_collision_postprocess import (
    CollisionSettings, ResidualCollisionPostprocess, box_sdf, primitive_gaps,
    qmul, qrotate, segment_box_sdf,
)


def collision_fixture(settings=None):
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    offsets = torch.zeros(24, 3)
    offsets[0, 1] = .9
    offsets[[3, 6, 9, 12, 15], 1] = torch.tensor([.15, .15, .15, .15, .15])
    offsets[13] = torch.tensor([-.1, .05, 0.])
    offsets[14] = torch.tensor([.1, .05, 0.])
    offsets[16, 0], offsets[17, 0] = -.1, .1
    offsets[18, 0], offsets[19, 0] = -.3, .3
    offsets[20, 0], offsets[21, 0] = -.25, .25
    offsets[22, 0], offsets[23, 0] = -.08, .08
    rest = dict(parents=parents, restLocalPositions=offsets.tolist())
    profile = dict(forearmRadii=[.035, .035], restLocalPositions=offsets.tolist(),
                   palmSpheres=[[[sign * x, 0., 0., .025] for x in (.02, .05, .08)] for sign in (-1, 1)])
    module = ResidualCollisionPostprocess(rest, profile, torch.zeros(144), torch.ones(144), settings)
    pose = torch.tensor([0., 0., 1., 0., 1., 0.]).repeat(1, 24)
    tracker = torch.zeros(1, 6, 10)
    tracker[:, :, 3:9] = pose.reshape(1, 24, 6)[:, 0:1]
    tracker[:, :, 9] = 1
    tracker[:, 0, 1] = 1.65
    tracker[:, 1, :3] = torch.tensor([-.75, 1.4, 0.])
    tracker[:, 2, :3] = torch.tensor([.75, 1.4, 0.])
    display = module.target_frame(pose, tracker).detach()
    boxes = torch.zeros(1, 2, 15)
    boxes[0, :, :3] = torch.tensor([[-.58, 1.37, 0.], [.58, 1.37, 0.]])
    boxes[:, :, 3:12] = torch.eye(3).flatten()
    boxes[:, :, 12:] = torch.tensor([.12, .0095, .05])
    modes = torch.full((1, 2, 2), 2, dtype=torch.long)
    context = torch.zeros(1, 4)
    return module, pose, tracker, boxes, modes, context, display


def test_segment_box_covers_middle_and_matches_dense_reference():
    generator = torch.Generator().manual_seed(23)
    a = torch.randn(32, 3, generator=generator)
    b = torch.randn(32, 3, generator=generator)
    a[0], b[0] = torch.tensor([-1., 0., 0.]), torch.tensor([1., 0., 0.])
    a[1], b[1] = torch.tensor([0., .2, 0.]), torch.tensor([0., .2, 0.])
    half = torch.tensor([.066, .0095, .265])
    actual = segment_box_sdf(a, b, half)
    t = torch.linspace(0, 1, 10001)
    dense = box_sdf(a[:, None] + t[None, :, None] * (b - a)[:, None], half).amin(-1)
    assert actual[0] < 0
    assert box_sdf(a[0], half) > 0 and box_sdf(b[0], half) > 0
    assert torch.all(actual <= dense + 2e-6)
    torch.testing.assert_close(actual, dense, atol=3e-4, rtol=1e-4)


def test_top_mode_does_not_push_through_thin_armrest_or_use_infinite_plane():
    module, _, _, boxes, modes, _, _ = collision_fixture()
    start = torch.zeros(1, 2, 4, 3)
    start[:, 0, :, :] = torch.tensor([-.58, 1.30, 0.])
    start[:, 1, :, :] = torch.tensor([.58, 1.30, 0.])
    plain = primitive_gaps(start, start, module.radii, boxes, torch.ones_like(modes))
    top = primitive_gaps(start, start, module.radii, boxes, modes)
    assert plain[0, 0, 0, 0] > 0
    assert top[0, 0, 0, 0] < -.1
    shifted = start + torch.tensor([0., 0., 1.])
    assert primitive_gaps(shifted, shifted, module.radii, boxes, modes).min() > 0


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_direct_sphere_gap_matches_segment_reference(mode):
    """用原来的通用线段计算核对球体快路径，覆盖盒内、边缘和有限上表面。"""
    from export.residual_collision_postprocess import top_gap

    module, _, _, boxes, modes, _, _ = collision_fixture()
    generator = torch.Generator().manual_seed(18)
    start = torch.randn(17, 2, 4, 3, generator=generator) * .15
    start += boxes[0, :, None, :3]
    end = start.clone()
    end[:, :, 0, 0] += .4
    modes.fill_(mode)
    axes = boxes[..., 3:12].reshape(1, 2, 3, 3)
    a = ((start.unsqueeze(-2) - boxes[:, None, None, :, :3]).unsqueeze(-2)
         * axes[:, None, None]).sum(-1)
    b = ((end.unsqueeze(-2) - boxes[:, None, None, :, :3]).unsqueeze(-2)
         * axes[:, None, None]).sum(-1)
    half, radius = boxes[:, None, None, :, 12:], module.radii[None, :, :, None]
    expected = segment_box_sdf(a, b, half) - radius
    if mode == 2:
        expected = expected.minimum(top_gap(a, b, half, radius))
    if mode == 0:
        expected = torch.full_like(expected, 1e3)
    torch.testing.assert_close(primitive_gaps(start, end, module.radii, boxes, modes), expected,
                               atol=1e-7, rtol=1e-6)


def test_larger_step_improves_reachable_contact_without_changing_tracking_weight():
    torch.set_num_threads(1)
    module, prior, tracker, boxes, modes, context, display = collision_fixture()
    small = deepcopy(module)
    small.settings = replace(module.settings, h=1e-5)
    args = (prior, torch.zeros_like(prior), tracker, boxes, modes, context, display, torch.tensor(.2))
    edited, conservative = module(*args), small(*args)
    assert edited.norm() > conservative.norm()
    loss, penetration = module.measurement(prior + edited, tracker, boxes, modes, context, display)
    small_loss, small_penetration = module.measurement(prior + conservative, tracker, boxes, modes, context, display)
    assert penetration.item() < small_penetration.item()
    assert loss.item() < small_loss.item()


def test_postprocess_only_changes_active_residual_and_reduces_penetration():
    torch.set_num_threads(1)
    module, prior, tracker, boxes, modes, context, display = collision_fixture()
    residual = torch.zeros_like(prior)
    before, penetration = module.measurement(prior, tracker, boxes, modes, context, display)
    result = module(prior, residual, tracker, boxes, modes, context, display, torch.tensor(.2))
    after, remaining = module.measurement(prior + result, tracker, boxes, modes, context, display)
    assert penetration.item() > 0
    assert remaining.item() < penetration.item()
    assert after.item() < before.item()
    assert torch.isfinite(result).all()
    mask = module.directions.sum(0).bool()
    assert torch.equal(result[:, ~mask], residual[:, ~mask])
    assert torch.count_nonzero(result[:, mask]) > 0
    disabled = module(prior, residual, tracker, boxes, torch.zeros_like(modes), context, display, torch.tensor(0.))
    assert torch.equal(disabled, residual)
    far = boxes.clone(); far[..., 2] += 10
    assert torch.equal(module(prior, residual, tracker, far, modes, context, display, torch.tensor(.2)), residual)


def test_degenerate_pose_and_unreachable_tracker_remain_finite():
    module, prior, tracker, boxes, modes, context, display = collision_fixture(CollisionSettings(steps=1))
    prior.zero_(); tracker[:, 1:3, :3] *= 10
    context[:] = torch.tensor([1.0, 1., .1, .18])
    result = module(prior, torch.zeros_like(prior), tracker, boxes, modes, context, display, torch.tensor(0.))
    assert torch.isfinite(result).all()


def test_finite_difference_matches_autograd_geometry_gradient():
    torch.set_num_threads(1)
    module, prior, tracker, boxes, modes, context, display = collision_fixture()
    generator = torch.Generator().manual_seed(4)
    current = (prior + torch.randn(prior.shape, generator=generator) * .015).requires_grad_()
    # 远离接触开关和盒体棱边的点上，对独立自动求导结果做中心差分检验。
    loss, _ = module.measurement(current, tracker, boxes, modes, context, display)
    expected = torch.autograd.grad(loss.sum(), current)[0] @ module.directions.T
    epsilon = module.settings.difference_step
    values, _ = module.measurement(current.detach() + epsilon * torch.cat([module.directions, -module.directions]),
                                   tracker, boxes, modes, context, display)
    actual = (values[:24] - values[24:]) / (2 * epsilon)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected[0], atol=.035, rtol=.025)


def test_paired_arm_differences_match_independent_full_pose_gradient():
    """独立单分量差分同时覆盖双扶手、跟踪项和坐姿补偿，防止左右臂串扰。"""
    torch.set_num_threads(1)
    module, prior, tracker, boxes, modes, context, display = collision_fixture()
    generator = torch.Generator().manual_seed(17)
    current = prior + torch.randn(prior.shape, generator=generator) * .04
    tracker[:, 1, 1] += .06
    tracker[:, 2, 2] += .08
    context[:] = torch.tensor([.93, .8, .1, .18])
    modes[:] = torch.tensor([[[1, 2], [2, 0]]])
    cache = module.prepare_geometry(current, tracker, context, display)
    paired, _ = module.arm_measurement(current[:, 96:120] + module.difference_offsets,
                                       tracker, boxes, modes, cache)
    epsilon = module.settings.difference_step
    actual = ((paired[1:13] - paired[13:]) / (2 * epsilon)).reshape(2, 6, 2).permute(0, 2, 1).flatten()
    independent, _ = module.measurement(current + epsilon * torch.cat([module.directions, -module.directions]),
                                        tracker, boxes, modes, context, display)
    expected = (independent[:24] - independent[24:]) / (2 * epsilon)
    torch.testing.assert_close(actual, expected, atol=.04, rtol=.015)
    # 肩肘以外的全局旋转也参与本次缓存准备，但不应被内层编辑。
    plus = current.clone(); plus[:, 96:102] += .03
    first = module.arm_measurement(current[:, 96:120], tracker, boxes, modes, cache)[0]
    second = module.arm_measurement(plus[:, 96:120], tracker, boxes, modes, cache)[0]
    torch.testing.assert_close(first[:, 1], second[:, 1], atol=0, rtol=0)


def test_compact_torso_cache_matches_full_fk_with_rotated_spine():
    torch.set_num_threads(1)
    module, prior, tracker, boxes, modes, context, display = collision_fixture()
    generator = torch.Generator().manual_seed(62)
    current = prior + torch.randn(prior.shape, generator=generator) * .18
    display = module.target_frame(prior + torch.randn(prior.shape, generator=generator) * .12, tracker)
    context[:] = torch.tensor([1.03, .8, .1, .18])
    cache = module.prepare_geometry(current, tracker, context, display)
    candidates = current + torch.randn((7, 24), generator=generator) @ module.directions * .04
    expected_p, expected_q = module.display_geometry(candidates, tracker, context, display)
    elbows, wrists, rotations = module.cached_upper_geometry(candidates[:, 96:120], cache)
    torch.testing.assert_close(elbows, expected_p[:, :, [18, 19]], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(wrists, expected_p[:, :, [20, 21]], atol=2e-5, rtol=2e-5)
    assert (rotations * expected_q[:, :, [20, 21]]).sum(-1).abs().min() > .99999


@pytest.mark.parametrize('lift', [0., .1])
def test_cached_arm_geometry_matches_full_seated_fk(lift):
    module, prior, tracker, boxes, modes, context, display = collision_fixture()
    context[:] = torch.tensor([.9 + lift, .7, 0., .18])
    # 候选只改变四个全局旋转，躯干和坐姿支撑可在内层迭代间复用。
    cache = module.prepare_geometry(prior, tracker, context, display)
    generator = torch.Generator().manual_seed(51)
    candidates = prior + torch.randn((9, 24), generator=generator) @ module.directions * .04
    positions, rotations = module.display_geometry(candidates, tracker, context, display)
    elbows, wrists, quaternions = module.cached_upper_geometry(candidates[:, 96:120], cache)
    torch.testing.assert_close(elbows, positions[:, :, [18, 19]], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(wrists, positions[:, :, [20, 21]], atol=1e-4, rtol=1e-4)
    assert (quaternions * rotations[:, :, [20, 21]]).sum(-1).abs().min() > .9999


class GeometryGraph(torch.nn.Module):
    def __init__(self, module):
        super().__init__(); self.module = module

    def forward(self, pose, tracker, boxes, modes, context, display):
        positions, rotations = self.module.display_geometry(pose, tracker, context, display)
        loss, penetration = self.module.measurement(pose, tracker, boxes, modes, context, display)
        return positions, rotations, loss, penetration


def test_geometry_onnx_matches_pytorch(tmp_path):
    ort = pytest.importorskip('onnxruntime')
    onnx = pytest.importorskip('onnx')
    torch.set_num_threads(1)
    module, pose, tracker, boxes, modes, context, display = collision_fixture()
    context[:] = torch.tensor([.92, .7, .1, .18])
    graph = GeometryGraph(module).eval()
    inputs = (pose, tracker, boxes, modes, context, display)
    path = tmp_path / 'collision_geometry.onnx'
    with torch.no_grad():
        expected = graph(*inputs)
        torch.onnx.export(graph, inputs, str(path), opset_version=15, dynamo=False,
                          input_names=['pose', 'tracker', 'boxes', 'modes', 'context', 'display'])
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    # 当前 Unity GPU 后端会漏掉仅有上界的 Clip；导出图必须用显式 Min 表达。
    for node in onnx.load(str(path)).graph.node:
        if node.op_type == 'Clip' and len(node.input) > 2 and node.input[2]:
            assert node.input[1], 'Unity 几何图不能使用缺少下界输入的 Clip。'
    actual = session.run(None, dict(zip(['pose', 'tracker', 'boxes', 'modes', 'context', 'display'],
                                       [x.numpy() for x in inputs])))
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right.numpy(), atol=2e-4, rtol=2e-4)


def test_export_switch_defaults_and_separate_filename():
    from export.export_sentis_denoiser import build_arg_parser, sampler_filename
    args = build_arg_parser().parse_args(['--model_path', 'dit.pt', '--predictor_model_path', 'p.pt',
        '--normalizer_dir', 'norm', '--body_fbx_rest_json', 'rest.json'])
    assert not args.collision_postprocess
    assert args.postedit_steps == 5
    assert args.postedit_h == CollisionSettings().h == 1e-4
    assert not hasattr(args, 'postedit_w')
    assert sampler_filename(args, 5) == 'pose_sampler_5step.onnx'
    args.collision_postprocess = True
    with pytest.raises(ValueError):
        sampler_filename(args, 5)
    args.sampler_filename = 'pose_sampler_5step.onnx'
    with pytest.raises(ValueError, match='三个模型'):
        sampler_filename(args, 5)


def test_two_arms_test_both_boxes_and_head_yaw_equivariance():
    module, pose, tracker, boxes, modes, context, display = collision_fixture()
    expected = module.measurement(pose, tracker, boxes, modes, context, display)
    # 扶手编号不决定手的配对，两条小臂都检查两个盒体。
    swapped = module.measurement(pose, tracker, boxes[:, [1, 0]], modes[:, :, [1, 0]], context, display)
    torch.testing.assert_close(swapped, expected)
    angle = torch.tensor(.73)
    q = torch.tensor([0., torch.sin(angle/2), 0., torch.cos(angle/2)])
    rotated_pose = qrotate(q, pose.reshape(1, 24, 2, 3)).reshape(1, 144)
    rotated_tracker = tracker.clone()
    rotated_tracker[..., :3] = qrotate(q, tracker[..., :3])
    rotated_tracker[..., 3:9] = qrotate(q, tracker[..., 3:9].reshape(1, 6, 2, 3)).reshape(1, 6, 6)
    rotated_boxes = boxes.clone()
    rotated_boxes[..., :3] = qrotate(q, boxes[..., :3])
    rotated_boxes[..., 3:12] = qrotate(q, boxes[..., 3:12].reshape(1, 2, 3, 3)).reshape(1, 2, 9)
    rotated_display = display.clone()
    rotated_display[..., :3] = qrotate(q, display[..., :3])
    rotated_display[..., 3:7] = qmul(q.expand(1, -1), display[..., 3:7])
    actual = module.measurement(rotated_pose, rotated_tracker, rotated_boxes, modes, context, rotated_display)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-4)


def test_uniform_change_of_length_units_preserves_objective():
    module, pose, tracker, boxes, modes, context, display = collision_fixture()
    expected = module.measurement(pose, tracker, boxes, modes, context, display)
    for scale in (.75, 1.3):
        scaled = deepcopy(module)
        scaled.offsets *= scale; scaled.palms *= scale; scaled.radii *= scale
        scaled.settings = replace(module.settings, m=module.settings.m*scale,
            margin=module.settings.margin*scale, tracking_tolerance=module.settings.tracking_tolerance*scale)
        scaled_tracker = tracker.clone(); scaled_tracker[..., :3] *= scale
        scaled_boxes = boxes.clone(); scaled_boxes[..., :3] *= scale; scaled_boxes[..., 12:] *= scale
        scaled_display = display.clone(); scaled_display[..., :3] *= scale; scaled_display[..., 7:10] *= scale
        loss, penetration = scaled.measurement(pose, scaled_tracker, scaled_boxes, modes, context, scaled_display)
        torch.testing.assert_close(loss, expected[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(penetration / scale, expected[1], atol=1e-5, rtol=1e-4)


def test_each_ddim_step_uses_edited_residual_without_extra_dit_calls():
    from export.unity_onnx_models import PoseSampler, example_inputs
    from model.realtime_pose_current_dit import RealtimePoseCurrentDiT
    from utils.model_util import create_gaussian_diffusion

    torch.set_num_threads(1); torch.manual_seed(11)
    post, prior, tracker, boxes, modes, context, display = collision_fixture()
    diffusion = create_gaussian_diffusion(SimpleNamespace(diffusion_steps=50, ts_respace='5',
        predict_xstart=True, noise_schedule='cosine', sigma_small=True))
    normalizer = SimpleNamespace(pose_mean=torch.zeros(144), pose_scale=torch.ones(144),
                                 tracker_mean=torch.zeros(6, 9), tracker_std=torch.ones(6, 9), eps=1e-8)
    dit = RealtimePoseCurrentDiT(latent_dim=24, num_layers=1, num_heads=4).eval()
    normal = PoseSampler(dit, diffusion, normalizer).eval()
    edited = PoseSampler(dit, diffusion, normalizer, post).eval()
    inputs = list(example_inputs()); inputs[0] = prior[:, None].expand(1, 10, 144)
    inputs[1] = prior[:, None].expand(1, 11, 144); inputs[2] = tracker
    inputs[-1] = torch.randn(1, 144)
    observed_normal, observed_edit, residuals = [], [], []
    normal.dit.residual_input.register_forward_pre_hook(lambda _, args: observed_normal.append(args[0].clone()))
    edited.dit.residual_input.register_forward_pre_hook(lambda _, args: observed_edit.append(args[0].clone()))
    post.register_forward_hook(lambda _, args, result: residuals.append((args[1].clone(), result.clone())))
    measurements = []
    original_measurement = post.arm_measurement
    def measured(normalized, *args, **kwargs):
        measurements.append(normalized.shape[0])
        return original_measurement(normalized, *args, **kwargs)
    post.arm_measurement = measured
    with torch.no_grad():
        normal(*inputs)
        actual = edited(*inputs, boxes, modes, context, display)
    assert len(observed_normal) == len(observed_edit) == len(residuals) == 5
    assert measurements.count(25) == 25
    assert any(not torch.equal(a, b) for a, b in residuals)
    assert torch.equal(observed_normal[0], observed_edit[0])
    assert not torch.equal(observed_normal[1], observed_edit[1])
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('edit', [False, True])
def test_split_condition_step_projection_match_complete_sampler(edit):
    from export.unity_onnx_models import (PoseSampler, SamplerCondition, SamplerDenoiseStep,
                                         SamplerProjection, example_inputs)
    from model.realtime_pose_current_dit import RealtimePoseCurrentDiT
    from utils.model_util import create_gaussian_diffusion

    torch.manual_seed(71);torch.set_num_threads(1)
    post, prior, tracker, boxes, modes, context, display=collision_fixture()
    diffusion=create_gaussian_diffusion(SimpleNamespace(diffusion_steps=50, ts_respace='5',
        predict_xstart=True, noise_schedule='cosine', sigma_small=True))
    normalizer=SimpleNamespace(pose_mean=torch.zeros(144),pose_scale=torch.ones(144),
        tracker_mean=torch.zeros(6,9),tracker_std=torch.ones(6,9),eps=1e-8)
    sampler=PoseSampler(RealtimePoseCurrentDiT(latent_dim=24,num_layers=1,num_heads=4).eval(),
        diffusion,normalizer,post if edit else None).eval()
    inputs=list(example_inputs());inputs[1]=prior[:,None].expand(1,11,144);inputs[2]=tracker
    inputs[-1]=torch.randn(1,144)
    prepare=SamplerCondition(sampler);step=SamplerDenoiseStep(sampler);project=SamplerProjection(sampler)
    with torch.no_grad():
        cached=prepare(*inputs[:8]);assert cached[2].dtype==torch.int32
        state=inputs[-1];states=[]
        for index in reversed(range(5)):
            residual=step(state,sampler.timesteps[index:index+1],*cached)
            if edit:residual=post(prior,residual,tracker,boxes,modes,context,display,sampler.collision_variance[index])
            states.append(residual)
            if index:
                eps=(sampler.recip[index]*state-residual)/sampler.recipm1[index]
                state=residual*sampler.alpha_prev[index].sqrt()+(1-sampler.alpha_prev[index]).sqrt()*eps
        actual=project(inputs[1],residual,tracker)
        expected=sampler(*inputs,*((boxes,modes,context,display) if edit else ()))
    assert len(states)==5
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
