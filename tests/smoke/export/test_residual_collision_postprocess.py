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
    profile = dict(restLocalPositions=offsets.tolist(), capsules=[
        [dict(bone=18+h, start=[sign*(.02+.07*k), .005, .003], end=[sign*(.02+.07*(k+1)), .005, .003], radius=.035) for k in range(3)] +
        [dict(bone=20+h, start=[sign*.02, 0., 0.], end=[sign*.08, 0., 0.], radius=.025)]
        for h, sign in enumerate((-1, 1))])
    profile["seatCapsules"] = [dict(bone=j, start=[0.,0.,0.], end=[0.,-.1,0.], radius=.08) for j in range(3)]
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
    context = torch.zeros(1, 16)
    context[:, 3:12] = torch.eye(3).flatten()
    context[:, 12:15] = torch.tensor([.23,.03,.2])
    positions, _ = module.display_geometry(pose, tracker, context, display)
    tracker[:, [3,4,5], :3] = positions[:, -1, [0,10,11]]
    return module, pose, tracker, boxes, modes, context, display


def test_head_height_changes_fk_translation_without_changing_tracked_rotations():
    module, pose, tracker, boxes, modes, context, display = collision_fixture()
    p0,q0 = module.display_geometry(pose,tracker,context,display)
    p1,q1 = module.display_geometry(pose,tracker,context,display,torch.tensor([.03]))
    expected = torch.tensor([0.,.03,0.]).expand(1,24,3)
    torch.testing.assert_close(p1[:,-1]-p0[:,-1],expected,atol=2e-7,rtol=0)
    torch.testing.assert_close(q0,q1)
    torch.testing.assert_close(p1[:,-1,15,1],tracker[:,0,1]+.03)


def test_seat_weight_scales_contact_without_scaling_tracking_terms():
    module,pose,tracker,boxes,modes,context,display=collision_fixture()
    modes.zero_();context[:,15]=2;context[:,1]=.73
    tracker[:,3,0]+=.1  # 保留非零跟踪损失，防止错误地将总目标加倍。
    settings=module.settings
    losses=[]
    for weight in [0.,1.,2.]:
        module.settings=replace(settings,seat_collision_weight=weight)
        losses.append(module.measurement(pose,tracker,boxes,modes,context,display)[0])
    assert float(losses[0])>0
    assert float(losses[1]-losses[0])>0
    torch.testing.assert_close(losses[2]-losses[0],2*(losses[1]-losses[0]))


def test_head_height_optimization_respects_bounds_speed_and_no_contact_zero():
    settings = CollisionSettings(steps=8,h=.0002,m=.0028,head_y_limit=.03,
                                 spine_bend_scale=0,torso_curve_scale=0,chest_head_scale=0)
    module,pose,tracker,boxes,modes,context,display = collision_fixture(settings)
    modes.zero_();context[:,15]=2;context[:,1]=.73
    args=(pose,torch.zeros_like(pose),tracker,boxes,modes,context,display,torch.tensor(1.))
    height_context=torch.tensor([[0.,1/30]])
    edited,height=module.forward_with_height(*args,height_context)
    assert 0<float(height)<=.005001
    before,_=module.measurement(pose,tracker,boxes,modes,context,display,torch.zeros(1),torch.zeros(1))
    after,_=module.measurement(pose+edited,tracker,boxes,modes,context,display,height,torch.zeros(1))
    assert float(after)<float(before)
    height_context=torch.tensor([[.029,1/30]])
    _,height=module.forward_with_height(*args,height_context)
    assert abs(float(height))<=.030001 and abs(float(height)-.029)<=.005001
    context[:,15]=0
    edited,height=module.forward_with_height(*args,torch.tensor([[0.,1/30]]))
    torch.testing.assert_close(edited,torch.zeros_like(edited),atol=0,rtol=0)
    assert float(height)==0
    _,height=module.forward_with_height(*args,torch.tensor([[.02,1/30]]))
    assert .014999<=float(height)<.02


def test_temporal_correction_preserves_base_motion_and_ignores_invalid_history():
    settings=CollisionSettings(steps=10,h=.001,correction_smooth_weight=100)
    module,pose,tracker,boxes,modes,context,display=collision_fixture(settings)
    # 隔离时序项：保持求解激活，但几何目标是常数，不产生方向梯度。
    module.measurement=lambda p,*args: (p.new_ones(len(p)),p.new_ones(len(p)))
    history=torch.full((1,74),.05);history[:,72]=1
    def solve(residual,history):
        return module.forward_with_height(pose,residual,tracker,boxes,modes,context,display,
            torch.tensor(1.),torch.tensor([[0.,1/30]]),correction_context=history)[0]
    zero=torch.zeros_like(pose);corrected=solve(zero,history)
    selected=module.active_indices[24:]
    assert torch.all(corrected[:,selected]>0)
    assert torch.all(corrected[:,selected]<.05)
    torch.testing.assert_close(corrected[:,module.active_indices[:24]],torch.zeros(1,24))
    # 当前基础姿态改变时保留相同修正，不能锁住旧绝对姿态。
    moving=zero.clone();moving[:,selected]=.2
    torch.testing.assert_close(solve(moving,history)-moving,corrected,atol=2e-7,rtol=1e-5)
    history[:,72]=0
    torch.testing.assert_close(solve(zero,history),zero,atol=0,rtol=0)
    history[:,72]=1
    module.measurement=lambda p,*args: (p.new_zeros(len(p)),p.new_zeros(len(p)))
    torch.testing.assert_close(solve(zero,history),zero,atol=0,rtol=0)


@pytest.mark.parametrize('step,accept',[(.001,True),(.01,False)])
def test_backtracking_reduces_energy_or_restores_pose(step,accept):
    module,pose,tracker,boxes,modes,context,display=collision_fixture(CollisionSettings(steps=1,h=step))
    axis=int(module.active_indices[24])
    module.measurement=lambda p,*args: (5000*(p[:,axis]-pose[0,axis]-.02).square(),p.new_ones(len(p)))
    trace=torch.zeros(2672)
    edited,height=module.forward_with_height(pose,torch.zeros_like(pose),tracker,boxes,modes,context,display,
        torch.tensor(1.),torch.tensor([[0.,1/30]]),trace=trace)
    assert trace[1101]<=trace[1100]
    if accept:
        assert 0<float(trace[1102])<1
        assert 0<float(edited[0,axis])<.04
        assert trace[1101]<trace[1100]
    else:
        assert float(trace[1102])==0
        torch.testing.assert_close(edited,torch.zeros_like(edited),atol=0,rtol=0)
    assert float(height)==0


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
def test_capsule_gap_matches_segment_reference(mode):
    """核对胶囊与退化胶囊，覆盖盒内、边缘和有限上表面。"""
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


def test_overlapping_forearm_segments_do_not_triple_penalty():
    module, pose, tracker, boxes, modes, context, display = collision_fixture()
    repeated = deepcopy(module)
    # 三个相同的完整胶囊与三个共线分段表示同一表面，不应产生三倍惩罚。
    repeated.capsule_start[:, :3] = module.capsule_start[:, :1]
    repeated.capsule_end[:, :3] = module.capsule_end[:, 2:3]
    expected = module.measurement(pose, tracker, boxes, modes, context, display)
    actual = repeated.measurement(pose, tracker, boxes, modes, context, display)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)


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
    context[:, :4] = torch.tensor([1.0, 1., .1, .18])
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
    actual = (values[:72] - values[72:]) / (2 * epsilon)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected[0], atol=.035, rtol=.025)


def test_seat_contact_has_torso_gradient_and_no_display_lift():
    module, pose, tracker, boxes, modes, context, display = collision_fixture(CollisionSettings(spine_bend_scale=0))
    torch.set_num_threads(1)
    # 倾斜躯干避开直立对称点，双大腿向前；座面单独激活。
    angle = torch.tensor(.35)
    q = torch.tensor([torch.sin(angle/2), 0., 0., torch.cos(angle/2)])
    shaped = pose.reshape(1,24,2,3).clone()
    shaped[:, [3,6,9,12]] = qrotate(q, shaped[:, [3,6,9,12]])
    shaped = shaped.reshape(1,144)
    display = module.target_frame(shaped, tracker)
    p, _ = module.display_geometry(shaped, tracker, context, display)
    tracker[:, [3,4,5], :3] = p[:, -1, [0,10,11]]
    context[:, 1] = p[0,-1,0,1] - .16
    context[:, 15] = 2
    modes.zero_()
    before, penetration = module.measurement(shaped, tracker, boxes, modes, context, display)
    eps = module.settings.difference_step
    values, _ = module.measurement(shaped + eps*torch.cat([module.directions,-module.directions]), tracker, boxes, modes, context, display)
    gradient = (values[:72]-values[72:])/(2*eps)
    assert gradient[24:48].abs().max() > .01
    assert penetration.item() > 0
    edited = module(shaped, torch.zeros_like(shaped), tracker, boxes, modes, context, display, torch.tensor(.2))
    after, _ = module.measurement(shaped+edited, tracker, boxes, modes, context, display)
    assert after.item() < before.item()
    # 上调座面不会直接平移显示骨架；变化只能来自残差编辑。
    raised = context.clone(); raised[:,1] += .2
    torch.testing.assert_close(module.display_geometry(shaped, tracker, raised, display), (p, module.display_geometry(shaped, tracker, context, display)[1]))
    context[:, 0] = 10
    assert torch.equal(module(shaped, torch.zeros_like(shaped), tracker, boxes, modes, context, display, torch.tensor(.2)), torch.zeros_like(shaped))


@pytest.mark.parametrize('side', [1, 2])
def test_thigh_contact_produces_hip_rotation_gradient(side):
    module, pose, tracker, boxes, modes, context, display = collision_fixture()
    angle = torch.tensor(-.7)
    q = torch.tensor([torch.sin(angle/2),0.,0.,torch.cos(angle/2)])
    raw = pose.reshape(1,24,2,3).clone();raw[:,side] = qrotate(q,raw[:,side]);pose=raw.reshape(1,144)
    # 隔离当前大腿，确认两侧各自都有梯度。
    for j in [j for j in range(3) if j!=side]:
        module.seat_start[j,0]+=10;module.seat_end[j,0]+=10
    display=module.target_frame(pose,tracker)
    positions,_=module.display_geometry(pose,tracker,context,display)
    context[:,1]=positions[0,-1,side,1]-.13;context[:,15]=2;modes.zero_()
    eps=module.settings.difference_step
    index=module.active_indices.tolist().index(side*6)
    directions=module.directions[index:index+6]
    candidates=pose+eps*torch.cat([directions,-directions])
    values,penetration=module.measurement(candidates,tracker,boxes,modes,context,display)
    assert penetration.min()>0
    assert ((values[:6]-values[6:])/(2*eps)).abs().max()>.01


def test_spine_fold_is_corrected_without_contact_but_distributed_lean_is_unchanged():
    torch.set_num_threads(1)
    module, pose, tracker, boxes, modes, context, display = collision_fixture()
    modes.zero_()
    def shaped(angles):
        raw=pose.reshape(1,24,2,3).clone()
        for j,a in zip([3,6,9,12,15],angles):
            a=torch.tensor(a);q=torch.stack([torch.sin(a/2),a*0,a*0,torch.cos(a/2)])
            raw[:,j]=qrotate(q,raw[:,j])
        return raw.reshape(1,144)
    for folded,angles in [(False,[.1,.2,.3,.4,.5]),(True,[.8,.2,.1,.6,.5]),(True,[.1,.2,.3,.9,.5])]:
        prior=shaped(angles);tracker[:,0,3:9]=prior[:,90:96]
        display=module.target_frame(prior,tracker)
        positions,_=module.display_geometry(prior,tracker,context,display)
        tracker[:,1:,:3]=positions[:,-1,[20,21,0,10,11]]
        residual=torch.zeros_like(prior)
        edited=module(prior,residual,tracker,boxes,modes,context,display,torch.tensor(.2))
        # 末端参与损失，但头部仍由跟踪旋转固定，不成为新增求解变量。
        assert torch.equal(edited[:,90:96],residual[:,90:96])
        if not folded:
            assert torch.equal(edited,residual)
        else:
            before,fold=module.measurement(prior,tracker,boxes,modes,context,display)
            after,remaining=module.measurement(prior+edited,tracker,boxes,modes,context,display)
            assert after<before and remaining<fold


def test_total_curve_rejects_evenly_distributed_hunch_but_allows_rigid_lean():
    settings = CollisionSettings(spine_bend_scale=0, chest_head_scale=0, body_tracking_weight=0)
    module, pose, tracker, boxes, modes, context, display = collision_fixture(settings)
    modes.zero_(); context[:,15]=0
    def evaluate(angles, hip):
        raw=pose.reshape(1,24,2,3).clone()
        for j,a in zip([0,3,6,9,12,15],[hip]+angles):
            a=torch.tensor(a);q=torch.stack([torch.sin(a/2),a*0,a*0,torch.cos(a/2)])
            raw[:,j]=qrotate(q,raw[:,j])
        prior=raw.reshape(1,144)
        tracker[:,0,3:9]=prior[:,90:96];tracker[:,3,3:9]=prior[:,:6]
        display=module.target_frame(prior,tracker)
        positions,_=module.display_geometry(prior,tracker,context,display)
        tracker[:,1:,:3]=positions[:,-1,[20,21,0,10,11]]
        return module.measurement(prior,tracker,boxes,modes,context,display)[0]
    assert evaluate([.25,.5,.75,1.,1.25],0).item()>0
    assert evaluate([.5]*5,.5).item()<1e-8


def test_backrest_front_support_has_loss_and_pose_gradient():
    settings=CollisionSettings(spine_bend_scale=0,torso_curve_scale=0,chest_head_scale=0,body_tracking_weight=0)
    module, pose, tracker, boxes, modes, context, display=collision_fixture(settings)
    modes.zero_()
    display=module.target_frame(pose,tracker)
    positions,_=module.display_geometry(pose,tracker,context,display)
    tracker[:,1:,:3]=positions[:,-1,[20,21,0,10,11]]
    module.back_count=1
    module.back_bones=torch.tensor([9]);module.back_start=torch.zeros(1,3);module.back_end=torch.zeros(1,3);module.back_radii=torch.tensor([.05])
    module.backrest=torch.tensor([0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.,.3,.3,.01])
    context[:,:3]=positions[:,-1,9]+torch.tensor([0.,0.,.02])
    context[:,3:12]=torch.eye(3).flatten();context[:,12:15]=torch.tensor([.01,.01,.01]);context[:,15]=1
    candidate=pose.clone().requires_grad_()
    loss,_=module.measurement(candidate,tracker,boxes,modes,context,display)
    loss.sum().backward()
    assert loss.item()>0 and candidate.grad[:,18:78].abs().max().item()>0
    context[:,15]=0
    disabled,_=module.measurement(pose,tracker,boxes,modes,context,display)
    assert disabled.item()<1e-8


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
    context[:, :4] = torch.tensor([.92, .7, .1, .18])
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
        scaled.offsets *= scale; scaled.capsule_start *= scale; scaled.capsule_end *= scale; scaled.radii *= scale
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
    original_measurement = post.measurement
    def measured(normalized, *args, **kwargs):
        measurements.append(normalized.shape[0])
        return original_measurement(normalized, *args, **kwargs)
    post.measurement = measured
    with torch.no_grad():
        normal(*inputs)
        actual = edited(*inputs, boxes, modes, context, display)
    assert len(observed_normal) == len(observed_edit) == len(residuals) == 5
    assert measurements.count(149) == 25
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


def test_pelvis_forward_term_releases_and_ignores_invalid_history():
    module,pose,tracker,boxes,modes,context,display=collision_fixture(CollisionSettings(pelvis_forward_smooth_weight=4))
    context[:,15]=2;history=torch.zeros(1,74);history[:,72]=1
    hc=torch.tensor([[0.,1/30]]);height=torch.zeros(1)
    axis,target,weight=module.pelvis_reference(pose,tracker,context,display,height,hc,history)
    torch.testing.assert_close(weight,torch.tensor([4.]))
    # 仅前后偏移有代价；侧向误差不被新项误罚。
    base=module.measurement(pose,tracker,boxes,modes,context,display)[0]
    loss=module.measurement(pose,tracker,boxes,modes,context,display,pelvis_reference=(axis,target+.01,weight))[0]
    torch.testing.assert_close(loss-base,torch.tensor([4*.01**2/(2*module.settings.m**2)]))
    moving=display.clone();moving[:,2]+=.03
    assert module.pelvis_reference(pose,tracker,context,moving,height,hc,history)[2].item()==0
    context[:,15]=1
    assert module.pelvis_reference(pose,tracker,context,display,height,hc,history)[2].item()==0
    context[:,15]=2;history[:,72]=0
    assert module.pelvis_reference(pose,tracker,context,display,height,hc,history)[2].item()==0


def test_pelvis_forward_target_follows_current_base_pose():
    module,pose,tracker,boxes,modes,context,display=collision_fixture(CollisionSettings(pelvis_forward_smooth_weight=4))
    context[:,15]=2;history=torch.zeros(1,74);history[:,72]=1
    hc=torch.tensor([[0.,1/30]]);height=torch.zeros(1)
    target=module.pelvis_reference(pose,tracker,context,display,height,hc,history)[1]
    leaning=pose.clone();leaning[:,3*6+1]=.2
    moving_display=module.target_frame(leaning,tracker)
    reference=module.pelvis_reference(leaning,tracker,context,moving_display,height,hc,history)
    assert abs(float(reference[1]-target))>.01
    expected=module.measurement(leaning,tracker,boxes,modes,context,moving_display)[0]
    actual=module.measurement(leaning,tracker,boxes,modes,context,moving_display,pelvis_reference=reference)[0]
    torch.testing.assert_close(actual,expected)


def test_actual_pelvis_history_is_metric_and_matches_selected_output():
    module,pose,tracker,boxes,modes,context,display=collision_fixture(CollisionSettings(steps=3,pelvis_forward_smooth_weight=16,pelvis_forward_update_cost=.001))
    context[:,15]=2;history=torch.zeros(1,74);history[:,72]=1;history[:,73]=.002
    height=torch.zeros(1);hc=torch.tensor([[0.,1/30]])
    axis,target,_=module.pelvis_reference(pose,tracker,context,display,height,hc,history)
    raw,_=module.display_geometry(pose,tracker,context,display,height)
    torch.testing.assert_close(target-(raw[:,-1,0]*axis).sum(-1),history[:,73],atol=1e-7,rtol=0)
    # 姿态差历史变化不会改变米制目标。
    changed=history.clone();changed[:,:72]=.3
    torch.testing.assert_close(module.pelvis_reference(pose,tracker,context,display,height,hc,changed)[1],target)
    trace=torch.zeros(2672)
    edited,y=module.forward_with_height(pose,torch.zeros_like(pose),tracker,boxes,modes,context,display,torch.tensor(1.),hc,trace=trace,correction_context=history)
    final,_=module.display_geometry(pose+edited,tracker,context,display,y)
    actual=((final[:,-1,0]-raw[:,-1,0])*axis).sum(-1)
    torch.testing.assert_close(trace[1015:1016],actual,atol=1e-6,rtol=0)


@pytest.mark.parametrize('benefit,should_move',[(.004,False),(.064,True)])
def test_movement_cost_rejects_small_gain_but_allows_useful_update(benefit,should_move):
    settings=CollisionSettings(steps=2,h=.0001,pelvis_forward_smooth_weight=16,pelvis_forward_update_cost=.001)
    module,pose,tracker,boxes,modes,context,display=collision_fixture(settings)
    context[:,15]=2;history=torch.zeros(1,74);history[:,72]=1
    def objective(candidate,*args):
        axis,target,weight=args[-1]
        positions,_=module.display_geometry(candidate,tracker,context,display,args[5])
        error=(positions[:,-1,0]*axis).sum(-1)-target
        value=(-benefit*error+weight*(error.square()))/(2*settings.m**2)
        return value,torch.ones_like(value)
    module.measurement=objective
    trace=torch.zeros(2672)
    edited,_=module.forward_with_height(pose,torch.zeros_like(pose),tracker,boxes,modes,context,display,torch.tensor(1.),torch.tensor([[0.,1/30]]),trace=trace,correction_context=history)
    if should_move:
        assert edited.abs().max()>1e-5
        assert trace[1101]<trace[1100]
    else:
        torch.testing.assert_close(edited,torch.zeros_like(edited),atol=0,rtol=0)


def test_head_forward_translation_preserves_rotation_and_lateral_position():
    module,pose,tracker,boxes,modes,context,display=collision_fixture()
    axis=torch.tensor([[.6,0.,.8]])
    p0,q0=module.display_geometry(pose,tracker,context,display)
    p1,q1=module.display_geometry(pose,tracker,context,display,forward=axis*.02)
    torch.testing.assert_close(p1[:,-1]-p0[:,-1],(axis*.02)[:,None].expand(1,24,3),atol=2e-7,rtol=0)
    torch.testing.assert_close(q1,q0,atol=0,rtol=0)


@pytest.mark.parametrize("previous",[-.0195,.0195])
def test_head_forward_limit_rate_and_release(previous):
    settings=CollisionSettings(steps=3,head_forward_limit=.02,head_forward_speed=.05)
    module,pose,tracker,boxes,modes,context,display=collision_fixture(settings)
    modes.zero_();context[:,15]=2;context[:,1]=.73
    args=(pose,torch.zeros_like(pose),tracker,boxes,modes,context,display,torch.tensor(1.),torch.tensor([[0.,1/30]]))
    result={}
    module.forward_with_height(*args,forward_context=torch.tensor([[previous]]),forward_result=result)
    assert abs(float(result['offset']))<=.020001
    assert abs(float(result['offset'])-previous)<=.05/30+1e-6
    context[:,15]=0
    module.forward_with_height(*args,forward_context=torch.tensor([[previous]]),forward_result=result)
    expected=np.sign(previous)*(abs(previous)-.05/30)
    assert abs(float(result['offset'])-expected)<1e-6


def test_pelvis_history_includes_head_forward_warm_start():
    settings=CollisionSettings(steps=3,head_forward_limit=.02,pelvis_forward_smooth_weight=16)
    module,pose,tracker,boxes,modes,context,display=collision_fixture(settings)
    modes.zero_();context[:,15]=2;context[:,1]=.73
    history=torch.zeros(1,74);history[:,72]=1;history[:,73]=.012
    fc=torch.tensor([[.012]]);hc=torch.tensor([[0.,1/30]])
    trace=torch.zeros(2672);result={}
    edited,height=module.forward_with_height(pose,torch.zeros_like(pose),tracker,boxes,modes,context,display,
        torch.tensor(1.),hc,trace=trace,correction_context=history,forward_context=fc,forward_result=result)
    axis=context[:,9:12]
    root,rq,pp,_,_=module.target_components(pose,tracker,display)
    base=((root+qrotate(rq,pp))*axis).sum()
    root,rq,pp,_,_=module.target_components(pose+edited,tracker,display,height,axis*result['offset'])
    selected=((root+qrotate(rq,pp))*axis).sum()
    torch.testing.assert_close(trace[1012],base,atol=1e-6,rtol=0)
    torch.testing.assert_close(trace[1013],base+.012,atol=1e-6,rtol=0)
    torch.testing.assert_close(trace[1015],selected-base,atol=1e-6,rtol=0)
    # 相比扣除热启动位置的旧定义，总修正恰好多包含这12mm。
    torch.testing.assert_close(trace[1015]-(selected-(base+.012)),torch.tensor(.012),atol=1e-6,rtol=0)
