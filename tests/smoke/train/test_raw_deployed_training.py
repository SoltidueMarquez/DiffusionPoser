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
        "x": IDENTITY.repeat(batch_size, 1),
        "current_tracker_raw": tracker,
        "joint_offsets_parent": offsets,
        "target_joints_head_ref": torch.zeros(batch_size, 24, 3),
        "target_root_position_head_ref": torch.zeros(batch_size, 3),
        "target_root_yaw_world": torch.zeros(batch_size),
        "target_hip_height": torch.zeros(batch_size),
        "current_head_yaw_world": torch.zeros(batch_size),
        "previous_pose_target": IDENTITY.repeat(batch_size, 1),
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
    predictor_current = IDENTITY.repeat(2, 1)
    diffusion_target = torch.zeros(2, 144)
    model_kwargs = {
        "motion_context": predictor_current[:, None].repeat(1, 10, 1),
        "predictor_pose_horizon": predictor_current[:, None].repeat(1, 11, 1),
        "tracker_geometry": batch["current_tracker_raw"][..., :9],
        "tracker_available": batch["current_tracker_raw"][..., 9].bool(),
        "ik_residual": torch.zeros(2, 24, 6),
        "ik_gap": torch.zeros(2, 24),
        "ik_confidence": torch.ones(2, 24),
        "denoise_strength": torch.full((2, 24), 0.05),
        "constraint_type": torch.full((2, 24), 3, dtype=torch.long),
        "y": batch,
    }
    assert "future_target" not in model_kwargs
    losses = diffusion.training_losses(
        model,
        diffusion_target,
        torch.tensor([0, 1]),
        model_kwargs=model_kwargs,
        predictor_current=predictor_current,
        return_pred_xstart=True,
    )
    assert losses["raw_pred_residual"].shape == (2, 144)
    assert losses["raw_pred_pose"].shape == (2, 144)
    torch.testing.assert_close(losses["raw_pred_pose"], predictor_current)
    assert torch.isfinite(losses["loss"]).all()
    assert not any("contact" in name for name in losses)
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
            predictor_current=torch.zeros(1, 144),
        )
