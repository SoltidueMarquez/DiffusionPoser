from __future__ import annotations

import numpy as np
import pytest
import torch

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from diffusion.realtime_pose_inpainting import (
    RealtimePoseInpaintingCondition,
    add_future_rolling_prior_to_condition,
    apply_realtime_pose_inpainting,
    build_realtime_pose_inpainting_condition,
    confidence_to_t_soft,
    validate_realtime_pose_inpainting_condition,
)


def _current_pose_raw() -> torch.Tensor:
    return torch.zeros(1, 144)


def _condition(confidence: torch.Tensor) -> RealtimePoseInpaintingCondition:
    horizon_confidence = torch.zeros(1, 11, 24)
    horizon_confidence[:, 0] = confidence
    return RealtimePoseInpaintingCondition(
        pose=torch.zeros(1, 11, 144),
        confidence=horizon_confidence,
    )


def test_condition_only_populates_current_frame_and_normalizes_it():
    current_confidence = torch.zeros(1, 24)
    current_confidence[0, 0] = 1.0
    current_confidence[0, 1] = 0.5
    current_pose = torch.arange(144, dtype=torch.float32).reshape(1, 144)
    mean = torch.ones(144)
    scale = torch.full((144,), 2.0)
    condition = build_realtime_pose_inpainting_condition(
        current_pose_raw=current_pose,
        current_confidence=current_confidence,
        pose_mean=mean,
        pose_scale=scale,
    )
    torch.testing.assert_close(condition.pose[:, 0], (current_pose - mean) / scale)
    torch.testing.assert_close(condition.pose[:, 1:], torch.zeros(1, 10, 144))
    torch.testing.assert_close(condition.confidence[:, 0], current_confidence)
    torch.testing.assert_close(condition.confidence[:, 1:], torch.zeros(1, 10, 24))


def test_first_runtime_step_has_no_future_inpainting():
    condition = build_realtime_pose_inpainting_condition(
        current_pose_raw=_current_pose_raw(),
        current_confidence=torch.ones(1, 24),
        pose_mean=None,
        pose_scale=None,
    )
    assert torch.all(condition.confidence[:, 0] == 1.0)
    assert not (condition.confidence[:, 1:] > 0.0).any()


def test_future_rolling_prior_only_fills_aligned_horizon_one_to_nine():
    current_confidence = torch.zeros(2, 24)
    current_confidence[:, 0] = 1.0
    current = build_realtime_pose_inpainting_condition(
        current_pose_raw=torch.zeros(2, 144),
        current_confidence=current_confidence,
        pose_mean=torch.ones(144),
        pose_scale=torch.full((144,), 2.0),
    )
    aligned_prior = torch.stack(
        [torch.full((2, 144), float(index)) for index in range(2, 11)],
        dim=1,
    )
    condition = add_future_rolling_prior_to_condition(
        current_condition=current,
        aligned_future_prior_raw=aligned_prior,
        future_prior_valid=torch.tensor([True, False]),
        pose_mean=torch.ones(144),
        pose_scale=torch.full((144,), 2.0),
        confidence_decay=0.9,
    )

    torch.testing.assert_close(
        condition.pose[0, 1:-1], (aligned_prior[0] - 1.0) / 2.0
    )
    torch.testing.assert_close(condition.pose[1, 1:], torch.zeros(10, 144))
    expected_decay = torch.tensor([0.9**index for index in range(1, 10)])
    torch.testing.assert_close(condition.confidence[0, 1:-1, 0], expected_decay)
    torch.testing.assert_close(condition.confidence[1, 1:], torch.zeros(10, 24))
    torch.testing.assert_close(condition.pose[:, -1], torch.zeros(2, 144))
    torch.testing.assert_close(condition.confidence[:, -1], torch.zeros(2, 24))
    x_t = torch.full((2, 11, 144), 7.0)
    injected, _ = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.full((2,), 9, dtype=torch.long),
        condition=condition,
        known_noise=torch.zeros_like(x_t),
        alphas_cumprod=np.linspace(0.99, 0.1, 10),
    )
    assert not torch.equal(injected[0, 1:-1], x_t[0, 1:-1])
    torch.testing.assert_close(injected[:, -1], x_t[:, -1])
    with pytest.raises(ValueError, match="current-only"):
        validate_realtime_pose_inpainting_condition(
            condition,
            require_current_only=True,
        )


def test_t_soft_is_linear_without_rounding_or_square_mapping():
    confidence = torch.tensor([1.0, 0.9, 0.5, 0.0])
    t_soft = confidence_to_t_soft(confidence, max_timestep=9)
    torch.testing.assert_close(t_soft, torch.tensor([0.0, 0.9, 4.5, 9.0]))


def test_c_one_inpaints_until_final_step_c_zero_never_inpaints_and_soft_releases():
    confidence = torch.zeros(24)
    confidence[0] = 1.0
    confidence[1] = 0.5
    condition = _condition(confidence)
    x_t = torch.full((1, 11, 144), 7.0)
    known_noise = torch.ones_like(x_t)
    alphas_cumprod = np.linspace(0.99, 0.1, 10)

    active, t_soft = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.tensor([5]),
        condition=condition,
        known_noise=known_noise,
        alphas_cumprod=alphas_cumprod,
    )
    assert t_soft[0, 0, 0] == 0.0
    assert t_soft[0, 0, 1] == 4.5
    assert t_soft[0, 0, 2] == 9.0
    expected = float(np.sqrt(1.0 - alphas_cumprod[5]))
    torch.testing.assert_close(active[0, 0, :12], torch.full((12,), expected))
    torch.testing.assert_close(active[0, 0, 12:], x_t[0, 0, 12:])

    released, _ = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.tensor([4]),
        condition=condition,
        known_noise=known_noise,
        alphas_cumprod=alphas_cumprod,
    )
    assert not torch.equal(released[0, 0, :6], x_t[0, 0, :6])
    torch.testing.assert_close(released[0, 0, 6:], x_t[0, 0, 6:])

    final, _ = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.tensor([0]),
        condition=condition,
        known_noise=known_noise,
        alphas_cumprod=alphas_cumprod,
    )
    torch.testing.assert_close(final, x_t)


def test_zero_confidence_is_exactly_equivalent_to_plain_x_t():
    x_t = torch.randn(2, 11, 144)
    pose = torch.zeros_like(x_t)
    pose[:, 0] = torch.randn(2, 144)
    condition = RealtimePoseInpaintingCondition(
        pose=pose,
        confidence=torch.zeros(2, 11, 24),
    )
    x_model, _ = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.tensor([1, 7]),
        condition=condition,
        known_noise=torch.randn_like(x_t),
        alphas_cumprod=np.linspace(0.99, 0.1, 10),
    )
    torch.testing.assert_close(x_model, x_t)


def test_projected_ddim_reuses_one_known_noise_for_every_step(monkeypatch):
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02, 0.03], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    condition = _condition(torch.ones(24))
    calls: list[int] = []
    from diffusion import gaussian_diffusion as diffusion_module

    original = diffusion_module.apply_realtime_pose_inpainting

    def recording_apply(*args, **kwargs):
        calls.append(kwargs["known_noise"].data_ptr())
        return original(*args, **kwargs)

    monkeypatch.setattr(diffusion_module, "apply_realtime_pose_inpainting", recording_apply)

    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, value, _timestep, **_kwargs):
            return value + self.anchor

    diffusion.projected_ddim_sample_loop(
        IdentityModel(),
        shape=(1, 11, 144),
        projection_fn=lambda value: value,
        model_kwargs={},
        inpaint_condition=condition,
        known_noise=torch.randn(1, 11, 144),
        device=torch.device("cpu"),
    )
    assert len(calls) == 3
    assert len(set(calls)) == 1


def test_projected_ddim_requires_explicit_known_noise():
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )

    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, value, _timestep, **_kwargs):
            return value + self.anchor

    with pytest.raises(ValueError, match="known_noise"):
        diffusion.projected_ddim_sample_loop(
            IdentityModel(),
            shape=(1, 11, 144),
            projection_fn=lambda value: value,
            inpaint_condition=_condition(torch.full((24,), 0.5)),
            device=torch.device("cpu"),
        )


def test_wrong_ik_pose_at_full_confidence_is_released_for_final_correction():
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    wrong_ik_pose = torch.full((1, 144), 123.0)
    condition = build_realtime_pose_inpainting_condition(
        current_pose_raw=wrong_ik_pose,
        current_confidence=torch.ones(1, 24),
        pose_mean=None,
        pose_scale=None,
    )

    class CorrectThenPreserveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.final_input: torch.Tensor | None = None

        def forward(self, value, timestep, **_kwargs):
            final_step = (timestep == 0).view(-1, 1, 1)
            if bool(final_step.all()):
                self.final_input = value.detach().clone()
            corrected = torch.zeros_like(value) + self.anchor
            return torch.where(final_step, value, corrected)

    model = CorrectThenPreserveModel()
    result = diffusion.projected_ddim_sample_loop(
        model,
        shape=(1, 11, 144),
        projection_fn=lambda value: value,
        noise=torch.zeros(1, 11, 144),
        known_noise=torch.zeros(1, 11, 144),
        inpaint_condition=condition,
        device=torch.device("cpu"),
    )["sample"]

    assert model.final_input is not None
    torch.testing.assert_close(result, model.final_input)
    assert not torch.equal(result[:, 0], wrong_ik_pose)


def test_explicit_initial_and_inpaint_noise_fully_control_sampling():
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02, 0.03], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    condition = _condition(torch.full((24,), 0.5))

    class IdentityModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, value, _timestep, **_kwargs):
            return value + self.anchor

    model = IdentityModel()
    initial_noise = torch.zeros(1, 11, 144)
    known_noise = torch.ones_like(initial_noise)

    def sample(value: torch.Tensor) -> torch.Tensor:
        return diffusion.projected_ddim_sample_loop(
            model,
            shape=(1, 11, 144),
            projection_fn=lambda output: output,
            noise=initial_noise,
            known_noise=value,
            inpaint_condition=condition,
            device=torch.device("cpu"),
        )["sample"]

    first = sample(known_noise)
    second = sample(known_noise.clone())
    changed = sample(torch.zeros_like(known_noise))
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, changed)
