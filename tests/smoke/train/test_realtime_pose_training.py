from __future__ import annotations

import torch

from data_loaders.realtime_pose_geometry import advance_rollout_pose_history_torch
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from model.realtime_pose_target_dit import RealtimePoseTargetDiT
from tests.smoke.train.test_realtime_pose_140d_training import _make_batch
from train.training_loop import TrainLoop


def _training_batch():
    target, known_target, known, kwargs = _make_batch()
    y = kwargs["y"]
    return {
        "x": target,
        "known_target": known_target,
        "known_mask": known,
        "pose_history": kwargs["pose_history"],
        "tracker_window": kwargs["tracker_window"],
        "valid_frame_mask": kwargs["valid_frame_mask"],
        "target_joints_head_ref": y["target_joints_head_ref"],
        "prev_joints_head_ref": y["prev_joints_head_ref"],
        "current_tracker_pos_head_ref": y["current_tracker_pos_head_ref"],
        "joint_offsets_parent": y["joint_offsets_parent"],
        "joint_rest_local_rotations_6d": y["joint_rest_local_rotations_6d"],
        "configured": y["configured"],
        "measured_valid": y["measured_valid"],
        "missing_age": y["missing_age"],
    }


def test_train_loop_builds_140d_model_conditions_and_finite_loss():
    batch = _training_batch()
    loop = TrainLoop.__new__(TrainLoop)
    loop.normalizer_mean = None
    loop.normalizer_std = None
    kwargs = loop.mask_manager(batch, batch["x"])
    assert kwargs["inpaint_cond"].shape == (2, 140)
    assert torch.equal(kwargs["inpaint_cond"], ~batch["known_mask"])

    model = RealtimePoseTargetDiT(input_feats=140, latent_dim=32, num_layers=1, num_heads=4)
    diffusion = GaussianDiffusion(
        betas=get_named_beta_schedule("cosine", 8),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    losses = diffusion.training_losses(model, batch["x"], torch.tensor([1, 4]), model_kwargs=kwargs)
    assert all(torch.isfinite(value).all() for value in losses.values())


def test_rollout_reexpresses_and_keeps_all_previous_predictions():
    batch = _training_batch()
    history = advance_rollout_pose_history_torch(
        pose_history=torch.zeros(2, 60, 140),
        prediction=batch["x"],
        source_head_yaw_world=torch.zeros(2),
        destination_head_yaw_world=torch.ones(2),
        detach_prediction=False,
    )
    relative = torch.atan2(history[:, -1, 138], history[:, -1, 139])
    torch.testing.assert_close(relative, torch.ones_like(relative), atol=1e-5, rtol=0.0)

    second_prediction = batch["x"].clone()
    second_prediction[:, 138] = 1.0
    second_prediction[:, 139] = 0.0
    next_history = advance_rollout_pose_history_torch(
        pose_history=history,
        prediction=second_prediction,
        source_head_yaw_world=torch.ones(2),
        destination_head_yaw_world=torch.ones(2),
        detach_prediction=False,
    )
    torch.testing.assert_close(next_history[:, -2], history[:, -1])
    torch.testing.assert_close(next_history[:, -1], second_prediction)


def test_four_step_training_uses_rollout_batch_without_gt_history():
    batch = _training_batch()
    batch["current_head_yaw_world"] = torch.zeros(2)
    batch["current_head_position_world"] = torch.tensor([[0.0, 1.6, 0.0]]).repeat(2, 1)
    batch["floor_y"] = torch.zeros(2)
    future_template = {key: value for key, value in batch.items() if key != "pose_history"}
    batch["rollout"] = [dict(future_template) for _ in range(3)]

    loop = TrainLoop.__new__(TrainLoop)
    loop.model = RealtimePoseTargetDiT(input_feats=140, latent_dim=32, num_layers=1, num_heads=4)
    loop.diffusion = GaussianDiffusion(
        betas=get_named_beta_schedule("cosine", 8),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    loop.rollout_steps = 4
    loop.rollout_loss_weight = 1.0
    loop.rollout_joint_vel_loss_weight = 0.05
    loop.rollout_rot_vel_loss_weight = 0.02
    loop.rollout_prob = 1.0
    loop.detach_rollout_history = True
    loop.normalizer_mean = None
    loop.normalizer_std = None
    loop.feature_w = None
    loop.device = torch.device("cpu")
    loop.snr_gamma = 0.0
    loop.use_l1 = False
    losses = loop.compute_losses(batch, torch.tensor([1, 4]))
    assert torch.isfinite(losses["loss"]).all()
    assert all(f"rollout_step_{step}_loss" in losses for step in (1, 2, 3))
    losses["loss"].mean().backward()
