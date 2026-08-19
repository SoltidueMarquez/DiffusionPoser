from __future__ import annotations

import numpy as np
import torch

from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from model.realtime_pose_current_dit import RealtimePoseCurrentDiT


IDENTITY = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0]).repeat(24)


def _batch(batch_size=2):
    tracker = torch.zeros(batch_size, 6, 10)
    tracker[..., 3:9] = IDENTITY[:6]
    tracker[:, :3, 9] = 1.0
    tracker[:, 0, 1] = 1.6
    offsets = torch.zeros(batch_size, 24, 3)
    offsets[:, 1:, 1] = 0.1
    return {
        "current_tracker_raw": tracker,
        "joint_offsets_parent": offsets,
        "target_joints_head_ref": torch.zeros(batch_size, 24, 3),
        "target_root_position_head_ref": torch.zeros(batch_size, 3),
        "target_root_yaw_world": torch.zeros(batch_size),
        "target_hip_height": torch.zeros(batch_size),
        "current_head_yaw_world": torch.zeros(batch_size),
        "previous_contact_target": torch.zeros(batch_size, 2),
        "contact_target": torch.zeros(batch_size, 2),
        "previous_pose_target": IDENTITY.repeat(batch_size, 1),
        "previous_head_position_current_ref": torch.zeros(batch_size, 3),
    }


def test_diffusion_training_accepts_only_single_frame_and_has_no_gt_future_condition():
    model = RealtimePoseCurrentDiT(latent_dim=32, num_layers=1, num_heads=4)
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    batch = _batch()
    x = IDENTITY.repeat(2, 1)
    model_kwargs = {
        "motion_context": x[:, None].repeat(1, 10, 1),
        "predictor_pose_horizon": x[:, None].repeat(1, 11, 1),
        "current_joint_condition": torch.zeros(2, 24, 10),
        "y": batch,
    }
    assert "future_target" not in model_kwargs
    losses = diffusion.training_losses(
        model,
        x,
        torch.tensor([0, 1]),
        model_kwargs=model_kwargs,
        return_pred_xstart=True,
    )
    assert losses["raw_pred_xstart"].shape == (2, 144)
    assert torch.isfinite(losses["loss"]).all()
    losses["loss"].mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_diffusion_rejects_horizon_state():
    model = RealtimePoseCurrentDiT(latent_dim=32, num_layers=1, num_heads=4)
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    with np.testing.assert_raises(ValueError):
        diffusion.training_losses(
            model,
            torch.zeros(1, 11, 144),
            torch.tensor([0]),
            model_kwargs={},
        )
