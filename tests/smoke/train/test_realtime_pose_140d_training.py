from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_kinematics import rotation_6d_forward_up_torch
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from model.realtime_pose_target_dit import RealtimePoseTargetDiT, TrackerTokenEncoder


def _identity_6d(count: int) -> torch.Tensor:
    return rotation_6d_forward_up_torch(torch.eye(3).repeat(count, 1, 1))


def _make_batch(batch_size: int = 2):
    target = torch.zeros(batch_size, 140)
    target[:, :138] = _identity_6d(23).reshape(1, 138)
    target[:, 139] = 1.0
    pose_history = target[:, None].repeat(1, 60, 1)
    tracker = torch.zeros(batch_size, 61, 6, 12)
    tracker[..., 9] = 1.0
    tracker[..., 10] = 1.0
    tracker[..., 3:9] = _identity_6d(6).reshape(1, 1, 6, 6)
    tracker[..., 0, 1] = 1.6
    tracker[..., 3, 1] = 0.9

    known = torch.zeros(batch_size, 140, dtype=torch.bool)
    for start in (84, 114, 120, 54, 60):
        known[:, start : start + 6] = True
    known[:, 138:140] = True
    known_target = torch.where(known, target, torch.zeros_like(target))
    rest = _identity_6d(24).unsqueeze(0).repeat(batch_size, 1, 1)
    offsets = torch.zeros(batch_size, 24, 3)
    offsets[:, 0, 1] = 0.9
    offsets[:, 1:, 1] = 0.05
    zeros_joints = torch.zeros(batch_size, 24, 3)
    y = {
        "mask": ~known,
        "inpainted_motion": known_target,
        "known_mask": known,
        "pose_history": pose_history,
        "tracker_window": tracker,
        "target_joints_head_ref": zeros_joints,
        "prev_joints_head_ref": zeros_joints,
        "current_tracker_pos_head_ref": tracker[:, -1, :, :3],
        "joint_offsets_parent": offsets,
        "joint_rest_local_rotations_6d": rest,
        "configured": tracker[..., 9].bool(),
        "measured_valid": tracker[..., 10].bool(),
        "missing_age": torch.zeros(batch_size, 61, 6, dtype=torch.long),
    }
    kwargs = {
        "inpaint_cond": ~known,
        "known_mask": known,
        "pose_history": pose_history,
        "tracker_window": tracker,
        "valid_frame_mask": torch.ones(batch_size, 60, dtype=torch.bool),
        "y": y,
    }
    return target, known_target, known, kwargs


def test_tracker_encoder_distinguishes_state_age_and_reconnect_history():
    encoder = TrackerTokenEncoder(latent_dim=16)
    valid_mask = torch.ones(1, 60, dtype=torch.bool)
    base = torch.zeros(1, 61, 6, 12)
    base[..., 9] = 1.0
    base[..., 10] = 1.0

    unconfigured = base.clone()
    unconfigured[..., 1, 9:11] = 0.0
    short_missing = base.clone()
    short_missing[..., 1, 10] = 0.0
    short_missing[..., 1, 11] = 1.0 / 60.0
    long_missing = short_missing.clone()
    long_missing[..., 1, 11] = 1.0
    assert TrackerTokenEncoder.state_index(unconfigured)[0, -1, 1].item() == 0
    assert TrackerTokenEncoder.state_index(base)[0, -1, 1].item() == 1
    assert TrackerTokenEncoder.state_index(short_missing)[0, -1, 1].item() == 2
    short_token = encoder.embed_frames(short_missing[:, -1:])[:, 0, 1]
    long_token = encoder.embed_frames(long_missing[:, -1:])[:, 0, 1]
    assert not torch.allclose(short_token, long_token)

    reconnect = base.clone()
    reconnect[:, -2, 1, 10] = 0.0
    reconnect[:, -2, 1, 11] = 1.0
    normal_summary, _ = encoder(base, valid_mask)
    reconnect_summary, reconnect_current = encoder(reconnect, valid_mask)
    _, normal_current = encoder(base, valid_mask)
    assert not torch.allclose(normal_summary[:, 1], reconnect_summary[:, 1])
    # 重连当帧 current token 与普通 valid 相同，差异由独立历史 summary 保留。
    assert torch.allclose(normal_current[:, 1], reconnect_current[:, 1])


def test_140d_forward_backward_and_ddim_hard_inpainting():
    target, known_target, known, kwargs = _make_batch()
    model = RealtimePoseTargetDiT(
        input_feats=140,
        latent_dim=64,
        num_layers=2,
        num_heads=4,
    )
    diffusion = GaussianDiffusion(
        betas=get_named_beta_schedule("cosine", 8),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    losses = diffusion.training_losses(
        model,
        target,
        torch.tensor([2, 5]),
        model_kwargs=kwargs,
    )
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["loss"].mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    for output in diffusion.ddim_sample_loop_progressive(
        model,
        shape=tuple(target.shape),
        model_kwargs=kwargs,
        clip_denoised=False,
    ):
        torch.testing.assert_close(output["sample"][known], known_target[known], atol=1e-6, rtol=0.0)
        torch.testing.assert_close(output["pred_xstart"][known], known_target[known], atol=1e-6, rtol=0.0)
