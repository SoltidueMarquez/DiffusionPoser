from __future__ import annotations

import numpy as np
import pytest
import torch

from data_loaders.realtime_pose_ik import INHERITED, RealtimePoseIKResult
from data_loaders.sensor_masking import (
    CURRENT_JOINT_CONSTRAINT_TYPE_START,
    CURRENT_JOINT_IK_CONFIDENCE_INDEX,
    CURRENT_JOINT_IK_VALID_INDEX,
    CURRENT_JOINT_TRACKER_POSITION_VALID_INDEX,
    TRACKER_TO_JOINT,
)
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from diffusion.realtime_pose_inpainting import (
    RealtimePoseInpaintingCondition,
    apply_realtime_pose_inpainting,
    build_current_joint_condition,
    build_realtime_pose_inpainting_condition,
    confidence_to_release_level,
    validate_realtime_pose_inpainting_condition,
)


def _ik_result(confidence: torch.Tensor, updated_mask: torch.Tensor | None = None):
    batch_size = confidence.shape[0]
    if updated_mask is None:
        updated_mask = confidence > 0.0
    constraint_type = torch.full((batch_size, 24), INHERITED, dtype=torch.long)
    constraint_type[updated_mask] = 2
    return RealtimePoseIKResult(
        pose=torch.arange(batch_size * 24 * 6, dtype=torch.float32).reshape(
            batch_size, 24, 6
        ),
        updated_mask=updated_mask,
        direct_rotation_mask=torch.zeros_like(updated_mask),
        constraint_type=constraint_type,
        position_residual=torch.zeros(batch_size, 24),
        confidence=confidence,
    )


def _condition(confidence: torch.Tensor) -> RealtimePoseInpaintingCondition:
    return build_realtime_pose_inpainting_condition(
        ik_result=_ik_result(confidence),
        pose_mean=None,
        pose_scale=None,
    )


def test_condition_only_populates_current_frame_and_normalizes_it():
    confidence = torch.zeros(1, 24)
    confidence[0, :2] = torch.tensor([1.0, 0.5])
    result = _ik_result(confidence)
    mean = torch.ones(144)
    scale = torch.full((144,), 2.0)
    condition = build_realtime_pose_inpainting_condition(result, mean, scale)

    current_pose = result.pose.reshape(1, 144)
    torch.testing.assert_close(condition.pose[:, 0], (current_pose - mean) / scale)
    torch.testing.assert_close(condition.pose[:, 1:], torch.zeros(1, 10, 144))
    torch.testing.assert_close(condition.valid[:, 0], result.updated_mask)
    assert not condition.valid[:, 1:].any()
    torch.testing.assert_close(
        condition.release_level[:, 0], confidence_to_release_level(confidence)
    )


def test_current_joint_condition_has_fixed_layout_and_tracker_joint_scatter():
    confidence = torch.zeros(1, 24)
    confidence[0, :2] = torch.tensor([0.8, 0.4])
    result = _ik_result(confidence)
    tracker = torch.zeros(1, 6, 13)
    tracker[0, :, :3] = torch.arange(18, dtype=torch.float32).reshape(6, 3) + 1.0
    configured = torch.tensor([[True, True, True, False, False, False]])
    measured = configured.clone()
    current = build_current_joint_condition(
        ik_result=result,
        current_tracker_raw=tracker,
        configured=configured,
        measured_valid=measured,
        tracker_position_mean=torch.ones(6, 3),
        tracker_position_scale=torch.full((6, 3), 2.0),
    )

    assert current.shape == (1, 24, 10)
    assert torch.isfinite(current).all()
    mapped = torch.as_tensor(TRACKER_TO_JOINT)
    expected_positions = (tracker[0, :3, :3] - 1.0) / 2.0
    torch.testing.assert_close(current[0, mapped[:3], :3], expected_positions)
    torch.testing.assert_close(
        current[0, mapped[3:], :3],
        torch.zeros(3, 3),
    )
    assert current[0, mapped[:3], CURRENT_JOINT_TRACKER_POSITION_VALID_INDEX].all()
    assert not current[0, mapped[3:], CURRENT_JOINT_TRACKER_POSITION_VALID_INDEX].any()
    torch.testing.assert_close(
        current[0, :, CURRENT_JOINT_IK_VALID_INDEX],
        result.updated_mask[0].float(),
    )
    torch.testing.assert_close(
        current[0, :, CURRENT_JOINT_IK_CONFIDENCE_INDEX],
        result.confidence[0],
    )
    torch.testing.assert_close(
        current[0, :, CURRENT_JOINT_CONSTRAINT_TYPE_START:].sum(dim=-1),
        torch.ones(24),
    )

    six_valid = torch.ones(1, 6, dtype=torch.bool)
    six = build_current_joint_condition(
        ik_result=result,
        current_tracker_raw=tracker,
        configured=six_valid,
        measured_valid=six_valid,
        tracker_position_mean=None,
        tracker_position_scale=None,
    )
    expected_position_valid = torch.zeros(24, dtype=torch.bool)
    expected_position_valid[mapped] = True
    torch.testing.assert_close(
        six[0, :, CURRENT_JOINT_TRACKER_POSITION_VALID_INDEX].bool(),
        expected_position_valid,
    )


def test_current_joint_condition_disabled_normalizer_keeps_head_reference_meters():
    result = _ik_result(torch.zeros(1, 24))
    tracker = torch.zeros(1, 6, 13)
    tracker[0, 0, :3] = torch.tensor([1.0, 2.0, 3.0])
    valid = torch.tensor([[True, False, False, False, False, False]])
    raw = build_current_joint_condition(
        result,
        tracker,
        valid,
        valid,
        tracker_position_mean=None,
        tracker_position_scale=None,
    )
    torch.testing.assert_close(raw[0, TRACKER_TO_JOINT[0], :3], tracker[0, 0, :3])
    inherited = ~result.updated_mask[0]
    assert not raw[0, inherited, CURRENT_JOINT_IK_VALID_INDEX].any()
    assert not raw[0, inherited, CURRENT_JOINT_IK_CONFIDENCE_INDEX].any()


def test_release_level_uses_physical_noise_coordinate_only():
    confidence = torch.tensor([1.0, 0.5, 0.0])
    release = confidence_to_release_level(confidence)
    torch.testing.assert_close(
        release,
        torch.tensor([0.0, np.sqrt(0.5), 1.0], dtype=torch.float32),
    )
    torch.testing.assert_close(release, confidence_to_release_level(confidence.clone()))


def test_same_alpha_bar_produces_same_active_mask_at_different_local_indices():
    confidence = torch.zeros(1, 24)
    confidence[0, 0] = 0.5
    condition = _condition(confidence)
    x_t = torch.randn(1, 11, 144)
    known_noise = torch.randn_like(x_t)

    _, first_active = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.tensor([1]),
        condition=condition,
        known_noise=known_noise,
        alphas_cumprod=np.asarray([0.99, 0.49, 0.1]),
    )
    _, second_active = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.tensor([3]),
        condition=condition,
        known_noise=known_noise,
        alphas_cumprod=np.asarray([0.999, 0.9, 0.7, 0.49, 0.1]),
    )
    torch.testing.assert_close(first_active, second_active)
    assert first_active[0, 0, 0]


def test_ddim_five_ten_twenty_steps_keep_the_same_confidence_semantics():
    confidence = torch.zeros(1, 24)
    confidence[0, 0] = 0.5
    condition = _condition(confidence)
    x_t = torch.zeros(1, 11, 144)
    known_noise = torch.zeros_like(x_t)
    masks = []
    for step_count in (5, 10, 20):
        # 把同一个实际 alpha_bar 放在不同局部索引，模拟不同 DDIM respacing。
        alphas = np.linspace(0.99, 0.1, step_count)
        local_index = step_count // 2
        alphas[local_index] = 0.49
        _, active = apply_realtime_pose_inpainting(
            x_t=x_t,
            t=torch.tensor([local_index]),
            condition=condition,
            known_noise=known_noise,
            alphas_cumprod=alphas,
        )
        masks.append(active)
    torch.testing.assert_close(masks[0], masks[1])
    torch.testing.assert_close(masks[1], masks[2])


def test_invalid_joints_are_bitwise_unchanged_and_mid_confidence_releases():
    confidence = torch.zeros(1, 24)
    confidence[0, :2] = torch.tensor([1.0, 0.5])
    condition = _condition(confidence)
    x_t = torch.full((1, 11, 144), 7.0)
    known_noise = torch.ones_like(x_t)
    alphas = np.asarray([0.99, 0.8, 0.49, 0.1])

    injected, active = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.tensor([2]),
        condition=condition,
        known_noise=known_noise,
        alphas_cumprod=alphas,
    )
    assert active[0, 0, 0] and active[0, 0, 1]
    feature_valid = condition.valid.repeat_interleave(6, dim=-1)
    torch.testing.assert_close(injected[~feature_valid], x_t[~feature_valid])

    released, active = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.tensor([1]),
        condition=condition,
        known_noise=known_noise,
        alphas_cumprod=alphas,
    )
    assert active[0, 0, 0]
    assert not active[0, 0, 1]
    torch.testing.assert_close(released[0, 0, 6:12], x_t[0, 0, 6:12])


def test_full_confidence_remains_active_at_t_zero_without_special_release():
    confidence = torch.zeros(1, 24)
    confidence[0, 0] = 1.0
    condition = _condition(confidence)
    x_t = torch.full((1, 11, 144), 7.0)
    injected, active = apply_realtime_pose_inpainting(
        x_t=x_t,
        t=torch.tensor([0]),
        condition=condition,
        known_noise=torch.zeros_like(x_t),
        alphas_cumprod=np.asarray([0.99, 0.5]),
    )
    assert active[0, 0, 0]
    assert not torch.equal(injected[0, 0, :6], x_t[0, 0, :6])


def test_validation_rejects_future_condition_in_first_round():
    condition = _condition(torch.ones(1, 24))
    future_valid = condition.valid.clone()
    future_valid[:, 1] = True
    with pytest.raises(ValueError, match="未来帧"):
        validate_realtime_pose_inpainting_condition(
            RealtimePoseInpaintingCondition(
                pose=condition.pose,
                valid=future_valid,
                release_level=condition.release_level,
            )
        )


def _diffusion(step_count: int = 3) -> GaussianDiffusion:
    return GaussianDiffusion(
        betas=np.linspace(0.01, 0.03, step_count, dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )


class _IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, value, _timestep, **_kwargs):
        return value + self.anchor


def test_projected_ddim_reuses_one_known_noise_for_every_step(monkeypatch):
    diffusion = _diffusion(3)
    condition = _condition(torch.ones(1, 24))
    calls: list[int] = []
    from diffusion import gaussian_diffusion as diffusion_module

    original = diffusion_module.apply_realtime_pose_inpainting

    def recording_apply(*args, **kwargs):
        calls.append(kwargs["known_noise"].data_ptr())
        return original(*args, **kwargs)

    monkeypatch.setattr(diffusion_module, "apply_realtime_pose_inpainting", recording_apply)
    diffusion.projected_ddim_sample_loop(
        _IdentityModel(),
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
    with pytest.raises(ValueError, match="known_noise"):
        _diffusion(2).projected_ddim_sample_loop(
            _IdentityModel(),
            shape=(1, 11, 144),
            projection_fn=lambda value: value,
            inpaint_condition=_condition(torch.full((1, 24), 0.5)),
            device=torch.device("cpu"),
        )


def test_explicit_initial_and_inpaint_noise_fully_control_sampling():
    diffusion = _diffusion(3)
    condition = _condition(torch.full((1, 24), 0.99))
    model = _IdentityModel()
    initial_noise = torch.zeros(1, 11, 144)

    def sample(known_noise: torch.Tensor) -> torch.Tensor:
        return diffusion.projected_ddim_sample_loop(
            model,
            shape=(1, 11, 144),
            projection_fn=lambda output: output,
            noise=initial_noise,
            known_noise=known_noise,
            inpaint_condition=condition,
            device=torch.device("cpu"),
        )["sample"]

    first_noise = torch.ones_like(initial_noise)
    first = sample(first_noise)
    second = sample(first_noise.clone())
    changed = sample(torch.zeros_like(first_noise))
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, changed)
