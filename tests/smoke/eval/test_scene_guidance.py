"""四种引导共用一个小型参数化冒烟测试。"""
from types import SimpleNamespace

import pytest
import torch

from eval.scene_guidance import GuidanceConfig, SceneGeometry, pseudoinverse_guidance, sample_scene_guidance
from export.unity_onnx_models import PoseSampler, SAMPLER_INPUTS, example_inputs
from model.realtime_pose_current_dit import RealtimePoseCurrentDiT
from tests.smoke.export.test_residual_collision_postprocess import collision_fixture
from utils.model_util import create_gaussian_diffusion


@pytest.mark.parametrize("method", ["none", "postedit", "lm", "pigdm"])
def test_scene_guidance_small_sampler(method):
    torch.manual_seed(10)
    post, pose, tracker, boxes, modes, _, _ = collision_fixture()
    rest = dict(parents=post.parents, restLocalPositions=post.offsets.tolist())
    profile = {"capsules": [[dict(bone=(18+h if k < 3 else 20+h), start=post.capsule_start[h,k].tolist(),
                                  end=post.capsule_end[h,k].tolist(), radius=float(post.radii[h,k]))
                             for k in range(4)] for h in range(2)]}
    scene = SceneGeometry(rest, profile, post.pose_mean, post.pose_scale)
    diffusion = create_gaussian_diffusion(SimpleNamespace(diffusion_steps=50, ts_respace="5", noise_schedule="cosine", predict_xstart=1, sigma_small=True))
    normalizer = SimpleNamespace(pose_mean=torch.zeros(144), pose_scale=torch.ones(144), tracker_mean=torch.zeros(6,9), tracker_std=torch.ones(6,9), eps=1e-8)
    dit = RealtimePoseCurrentDiT(latent_dim=24, num_layers=1, num_heads=4).eval()
    # 初始化输出头本来为零；赋小权重以确保 ΠGDM 真正经过非零 DiT Jacobian。
    torch.nn.init.normal_(dit.joint_output.weight, std=.01)
    sampler = PoseSampler(dit, diffusion, normalizer).eval().requires_grad_(False)
    feed = dict(zip(SAMPLER_INPUTS, example_inputs()))
    feed.update(motion_context=pose[:, None].repeat(1,10,1), predictor_pose_horizon=pose[:,None].repeat(1,11,1),
                current_tracker_raw=tracker, noise=torch.randn(1,144), armrest_geometry=boxes, contact_mode=modes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler, scene = sampler.to(device), scene.to(device)
    feed = {key: value.to(device) for key, value in feed.items()}
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        result, _ = sample_scene_guidance(sampler, scene, feed, GuidanceConfig(method, iterations=1))
        assert result.shape == (1,144)
        assert torch.isfinite(result).all()
        if method == "none":
            with torch.no_grad():
                expected = sampler(*(feed[key] for key in SAMPLER_INPUTS))
            torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
        else:
            # 换一帧盒体条件后复用执行缓存，确认没有复用前一帧的距离或导数输入。
            changed = dict(feed, armrest_geometry=feed["armrest_geometry"].clone())
            changed["armrest_geometry"][..., 0] += .02
            observation = scene.prepare_observation(changed)
            residual = result - changed["predictor_pose_horizon"][:, 0]
            cached = observation.evaluate(residual)
            expected = scene.observation(changed["predictor_pose_horizon"][:, 0] + residual, changed)
            torch.testing.assert_close(cached, expected, atol=1e-6, rtol=1e-5)
        if method == "pigdm":
            # 无观测噪声的线性问题应退化到伪逆校正，结果与 r_t² 无关，且保留 DiT 的链式导数。
            state = torch.ones(1, 144, device=device, requires_grad=True)
            jacobian = torch.eye(144, device=device)[:2]
            observation = SimpleNamespace(value=torch.full((2,), 2., device=device),
                jacobian=lambda: jacobian, scene=SimpleNamespace(observation_basis=torch.eye(2, device=device)))
            expected = torch.zeros_like(state)
            expected[:, :2] = -4.
            for variance in (.2, .8):
                actual = pseudoinverse_guidance(state, 2*state, observation, torch.ones_like(state),
                                               variance, 0., GuidanceConfig("pigdm"))
                torch.testing.assert_close(actual, expected)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
