from __future__ import annotations

import torch

from data_loaders.realtime_pose_config import TARGET_JOINT_REGIONS
from model.realtime_pose_target_dit import RealtimePoseTargetDiT


def _conditioning(batch_size: int = 2):
    pose_history = torch.randn(batch_size, 60, 144)
    tracker_history = torch.zeros(batch_size, 60, 6, 13)
    current_tracker = torch.zeros(batch_size, 6, 13)
    tracker_history[..., 9] = 1.0
    tracker_history[..., 10] = 1.0
    tracker_history[..., 12] = 1.0
    current_tracker[..., 9] = 1.0
    current_tracker[..., 10] = 1.0
    current_tracker[..., 12] = 1.0
    trajectory_history = torch.randn(batch_size, 60, 5)
    current_trajectory = torch.randn(batch_size, 1, 5)
    valid_frame_mask = torch.ones(batch_size, 60, dtype=torch.bool)
    return (
        pose_history,
        tracker_history,
        current_tracker,
        trajectory_history,
        current_trajectory,
        valid_frame_mask,
    )


def _model() -> RealtimePoseTargetDiT:
    return RealtimePoseTargetDiT(
        latent_dim=64,
        num_layers=1,
        num_heads=8,
        motion_layers=1,
        dropout=0.0,
    ).eval()


def test_target_regions_cover_each_joint_once():
    assert TARGET_JOINT_REGIONS.shape == (24,)
    assert set(TARGET_JOINT_REGIONS.tolist()) == {0, 1, 2, 3, 4}


def test_self_attention_and_mlp_use_independent_six_parameter_adaln():
    model = _model()
    block = model.blocks[0]
    projection = block.adaln_modulation[-1]
    assert projection.out_features == model.latent_dim * 6
    assert not block.self_attention_norm.elementwise_affine
    assert not block.mlp_norm.elementwise_affine
    assert isinstance(block.mlp[0], torch.nn.Linear)
    assert torch.count_nonzero(projection.weight) == 0
    assert torch.count_nonzero(projection.bias) == 0

    # gate 保持零，使两个分支读取同一 residual state；不同 shift 应分别到达 MSA 和 MLP。
    with torch.no_grad():
        projection.bias[: model.latent_dim].fill_(0.25)
        projection.bias[3 * model.latent_dim : 4 * model.latent_dim].fill_(-0.5)
    captured: dict[str, torch.Tensor] = {}

    def capture_self_attention(_module, inputs) -> None:
        captured["self_attention"] = inputs[0].detach()

    def capture_mlp(_module, inputs) -> None:
        captured["mlp"] = inputs[0].detach()

    handles = [
        block.self_attention.register_forward_pre_hook(capture_self_attention),
        block.mlp.register_forward_pre_hook(capture_mlp),
    ]
    values = _conditioning(batch_size=1)
    try:
        with torch.no_grad():
            model(torch.randn(1, 144), torch.ones(1, dtype=torch.long), *values)
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(
        captured["self_attention"].mean(dim=-1),
        torch.full((1, 24), 0.25),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        captured["mlp"].mean(dim=-1),
        torch.full((1, 24), -0.5),
        atol=1e-5,
        rtol=0.0,
    )


def test_observation_encoder_excludes_head_position_branch():
    model = _model()
    values = list(_conditioning())
    first = model.observation_encoder(values[1], values[2], values[5])
    changed_current = values[2].clone()
    changed_current[:, 0, :3] = 1000.0
    second = model.observation_encoder(values[1], changed_current, values[5])
    torch.testing.assert_close(first.position_tokens, second.position_tokens)


def test_prepared_conditioning_matches_direct_forward_and_all_invalid_is_finite():
    model = _model()
    values = list(_conditioning())
    # Head 当前测量保持有效，但 60 帧历史全部处于冷启动 padding 状态。
    values[5].zero_()
    values[2][:, 1:, :9] = 0.0
    values[2][:, 1:, 10] = 0.0
    values[2][:, 1:, 12] = 0.0
    hidden = torch.randn(2, 144)
    timestep = torch.tensor([1, 2])
    prepared = model.prepare_conditioning(*values)
    direct, direct_aux = model(hidden, timestep, *values, return_aux_outputs=True)
    cached, cached_aux = model(
        hidden,
        timestep,
        prepared_conditioning=prepared,
        return_aux_outputs=True,
    )
    torch.testing.assert_close(direct, cached, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(direct_aux["future_leg"], cached_aux["future_leg"])
    assert torch.isfinite(cached).all()
    assert cached.shape == (2, 144)
    assert cached_aux["future_leg"].shape == (2, 3, 8, 6)
    assert cached_aux["contact_logits"].shape == (2, 2)


def test_motion_prior_does_not_read_current_tracker_or_current_trajectory():
    model = _model()
    values = list(_conditioning())
    first = model.prepare_conditioning(*values).motion
    values[2] = values[2].clone()
    values[2][..., :9] = values[2][..., :9] + torch.randn_like(values[2][..., :9]) * 10.0
    values[4] = values[4] + torch.randn_like(values[4]) * 10.0
    second = model.prepare_conditioning(*values).motion
    torch.testing.assert_close(first.temporal_tokens, second.temporal_tokens, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(first.latents, second.latents, atol=1e-6, rtol=1e-6)


def test_zero_length_cold_start_motion_prior_is_finite_and_zero():
    model = _model()
    values = list(_conditioning(batch_size=1))
    values[5].zero_()
    prepared = model.prepare_conditioning(*values)
    assert torch.isfinite(prepared.motion.temporal_tokens).all()
    assert torch.count_nonzero(prepared.motion.temporal_tokens) == 0
    assert torch.count_nonzero(prepared.motion.latents) == 0


def test_left_padding_uses_last_valid_history_frame():
    model = _model()
    values = list(_conditioning(batch_size=1))
    values[5].zero_()
    values[5][:, -2:] = True
    first = model.prepare_conditioning(*values)

    changed_padding = list(values)
    changed_padding[0] = values[0].clone()
    changed_padding[1] = values[1].clone()
    changed_padding[3] = values[3].clone()
    changed_padding[0][:, :-2] = torch.randn_like(changed_padding[0][:, :-2]) * 100.0
    changed_padding[1][:, :-2] = torch.randn_like(changed_padding[1][:, :-2]) * 100.0
    changed_padding[3][:, :-2] = torch.randn_like(changed_padding[3][:, :-2]) * 100.0
    second = model.prepare_conditioning(*changed_padding)
    torch.testing.assert_close(first.observation.history_summary, second.observation.history_summary)
    torch.testing.assert_close(first.motion.latents, second.motion.latents, atol=1e-6, rtol=1e-6)


def test_left_leg_history_does_not_enter_right_leg_specific_tokens():
    model = _model()
    values = list(_conditioning(batch_size=1))
    first = model.prepare_conditioning(*values).motion
    changed = list(values)
    changed[0] = values[0].clone().reshape(1, 60, 24, 6)
    changed[0][:, :, [1, 4, 7, 10]] += 10.0
    changed[0] = changed[0].reshape(1, 60, 144)
    second = model.prepare_conditioning(*changed).motion
    torch.testing.assert_close(first.temporal_tokens[:, 3], second.temporal_tokens[:, 3])
    assert not torch.allclose(first.temporal_tokens[:, 0], second.temporal_tokens[:, 0])
