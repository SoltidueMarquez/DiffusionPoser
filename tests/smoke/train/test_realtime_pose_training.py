from __future__ import annotations

import argparse

import numpy as np
import torch
import pytest

from data_loaders.generate_realtime_pose_tasks import main as generate_realtime_pose_tasks_main
from data_loaders.get_data import get_dataset_loader
from data_loaders.sensor_masking import (
    HIP_TRACKER_INDEX,
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_START,
    TRACKER_COUNT,
    get_schema_spec,
)
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.diffusionposer_dit import DiffusionPoserDiT
from train.training_loop import TrainLoop, validate_finite_losses
from utils import dist_util
from tests.smoke.realtime_pose_fixtures import IDENTITY_6D, write_toy_source_dataset


def _make_sensor_reprojection_aux_inputs(target_tracker_pos_ref: torch.Tensor, target_sensor_valid: torch.Tensor):
    """构造最小 realtime_pose batch，让 FK 后的 tracker ref 全为 0，便于单独检查位置重投影 loss。"""

    batch_size = target_tracker_pos_ref.shape[0]
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    pred_xstart = torch.zeros(batch_size, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
    identity_pose = torch.as_tensor(IDENTITY_6D).repeat(24)
    pred_xstart[:, schema.body_pose_slice(), REALTIME_POSE_TARGET_START] = identity_pose
    pred_xstart[:, schema.root_yaw_delta_slice(), REALTIME_POSE_TARGET_START] = torch.tensor([0.0, 1.0])
    x_start = pred_xstart.clone()
    model_kwargs = {
        "y": {
            "schema_name": REALTIME_POSE_SCHEMA_NAME,
            "target_joints_world": torch.zeros(batch_size, 24, 3),
            "prev_joints_world": torch.zeros(batch_size, 24, 3),
            "target_root_pos_world": torch.zeros(batch_size, 3),
            "prev_root_pos_world": torch.zeros(batch_size, 3),
            "prev_root_yaw": torch.zeros(batch_size),
            "joint_offsets_parent": torch.zeros(batch_size, 24, 3),
            "target_stationary_prob_5": torch.zeros(batch_size, 5),
            "target_tracker_pos_ref": target_tracker_pos_ref,
            "target_sensor_valid": target_sensor_valid,
        }
    }
    return pred_xstart, x_start, model_kwargs


def _make_loss_test_diffusion() -> SpacedDiffusion:
    betas = gd.get_named_beta_schedule("cosine", 4, scale_betas=1.0)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(4, [4]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        tracker_pos_huber_beta=0.05,
        tracker_pos_timestep_min_weight=0.1,
        tracker_pos_timestep_gamma=2.0,
    )


def test_model_forward_has_frame_positional_embedding_and_seq_limit():
    model = DiffusionPoserDiT(input_feats=REALTIME_POSE_INPUT_DIM, latent_dim=32, num_layers=1, num_heads=4, max_seq_len=REALTIME_POSE_SEQ_LEN)
    assert tuple(model.frame_pos_embed.shape) == (1, REALTIME_POSE_SEQ_LEN, 32)
    x = torch.zeros(2, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
    mask = torch.zeros_like(x, dtype=torch.bool)
    y = model(x, torch.zeros(2), inpaint_cond=mask)
    assert tuple(y.shape) == tuple(x.shape)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.zeros(1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN + 1), torch.zeros(1))


def test_single_batch_training_loss_contains_realtime_aux_terms(tmp_path):
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    loader = get_dataset_loader(
        data_dir=str(task_dir),
        batch_size=1,
        input_feats=REALTIME_POSE_INPUT_DIM,
        seq_len=REALTIME_POSE_SEQ_LEN,
        split="train",
        normalize_input=False,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
    )
    batch = next(iter(loader))
    dist_util.setup_dist(-1)
    model = DiffusionPoserDiT(input_feats=REALTIME_POSE_INPUT_DIM, latent_dim=32, num_layers=1, num_heads=4, max_seq_len=REALTIME_POSE_SEQ_LEN)
    betas = gd.get_named_beta_schedule("cosine", 4, scale_betas=1.0)
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(4, [4]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )
    args = argparse.Namespace(
        batch_size=1,
        lr=1e-4,
        log_interval=1,
        save_interval=0,
        resume_checkpoint="",
        weight_decay=0.0,
        lr_anneal_steps=0,
        gradient_clip=False,
        snr_gamma=0.0,
        l1_loss=False,
        task_mode="realtime_pose_reconstruction",
        schema=REALTIME_POSE_SCHEMA_NAME,
        checkpoint_max_keep=0,
        save_dir=str(tmp_path / "run"),
        num_steps=1,
        eval_during_training=False,
        eval_num_batches=1,
        weighted_loss=False,
        normalizer_dir="",
        feature_w_file="feature_w.pt",
        model_ema=False,
    )
    loop = TrainLoop(args, train_platform=NoopPlatform(), model=model, diffusion=diffusion, data=loader)
    model_kwargs = loop.mask_manager(batch, batch["x"])
    t = torch.zeros(1, dtype=torch.long)
    losses = diffusion.training_losses(model, batch["x"], t, model_kwargs=model_kwargs, snr_gamma=0.0)
    assert {
        "loss",
        "simple_loss",
        "yaw_loss",
        "fk_loss",
        "joint_vel_loss",
        "foot_lock_loss",
        "aux_loss",
        "sensor_reprojection_pos_loss",
        "sensor_reprojection_rot_loss",
    }.issubset(losses)
    assert torch.allclose(losses["loss"], losses["simple_loss"] + losses["aux_loss"])
    losses["loss"].mean().backward()

    model_kwargs["y"]["target_stationary_prob_5"] = torch.zeros_like(model_kwargs["y"]["target_stationary_prob_5"])
    aux_terms = diffusion._realtime_pose_aux_losses(batch["x"], batch["x"], model_kwargs)
    assert torch.allclose(aux_terms["foot_lock_loss"], torch.zeros_like(aux_terms["foot_lock_loss"]))

    model_kwargs_missing_stationary = loop.mask_manager(batch, batch["x"])
    del model_kwargs_missing_stationary["y"]["target_stationary_prob_5"]
    with pytest.raises(KeyError, match="target_stationary_prob_5"):
        diffusion._realtime_pose_aux_losses(batch["x"], batch["x"], model_kwargs_missing_stationary)


def test_rollout_training_loss_reinjects_predicted_target_and_backprops(tmp_path):
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir, frame_count=REALTIME_POSE_SEQ_LEN + 1)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--rollout_steps",
            "2",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    loader = get_dataset_loader(
        data_dir=str(task_dir),
        batch_size=1,
        input_feats=REALTIME_POSE_INPUT_DIM,
        seq_len=REALTIME_POSE_SEQ_LEN,
        split="train",
        normalize_input=False,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
        enable_rollout=True,
        rollout_steps=2,
    )
    batch = next(iter(loader))
    dist_util.setup_dist(-1)
    model = DiffusionPoserDiT(
        input_feats=REALTIME_POSE_INPUT_DIM,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        max_seq_len=REALTIME_POSE_SEQ_LEN,
    )
    betas = gd.get_named_beta_schedule("cosine", 4, scale_betas=1.0)
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(4, [4]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )
    args = argparse.Namespace(
        batch_size=1,
        lr=1e-4,
        log_interval=1,
        save_interval=0,
        resume_checkpoint="",
        weight_decay=0.0,
        lr_anneal_steps=0,
        gradient_clip=False,
        snr_gamma=0.0,
        l1_loss=False,
        task_mode="realtime_pose_reconstruction",
        schema=REALTIME_POSE_SCHEMA_NAME,
        checkpoint_max_keep=0,
        save_dir=str(tmp_path / "run"),
        num_steps=1,
        eval_during_training=False,
        eval_num_batches=1,
        weighted_loss=False,
        normalizer_dir="",
        feature_w_file="feature_w.pt",
        model_ema=False,
        rollout_steps=2,
        rollout_loss_weight=0.25,
        rollout_prob=1.0,
        detach_rollout_history=True,
    )
    loop = TrainLoop(args, train_platform=NoopPlatform(), model=model, diffusion=diffusion, data=loader)

    rollout_batch = batch["rollout"][0]
    pred_xstart = torch.randn_like(batch["x"], requires_grad=True)
    next_conditioned = loop.build_one_step_rollout_conditioned_x(
        rollout_batch=rollout_batch,
        pred_xstart=pred_xstart,
    )
    original_conditioned = rollout_batch["conditioned_x"]
    np.testing.assert_allclose(
        next_conditioned[:, :REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START - 1].detach().numpy(),
        pred_xstart[:, :REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START].detach().numpy(),
    )
    unchanged_mask = torch.ones_like(original_conditioned, dtype=torch.bool)
    unchanged_mask[:, :REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START - 1] = False
    assert torch.allclose(next_conditioned[unchanged_mask], original_conditioned[unchanged_mask])
    assert torch.allclose(
        next_conditioned[:, REALTIME_POSE_TARGET_DIM:, REALTIME_POSE_TARGET_START - 1],
        original_conditioned[:, REALTIME_POSE_TARGET_DIM:, REALTIME_POSE_TARGET_START - 1],
    )

    t = torch.zeros(1, dtype=torch.long)
    losses = loop.compute_losses(batch=batch, timesteps=t)
    assert {"loss", "base_loss", "rollout_loss", "rollout_loss_weighted"}.issubset(losses)
    assert torch.allclose(losses["loss"], losses["base_loss"] + 0.25 * losses["rollout_loss"])
    losses["loss"].mean().backward()
    assert any(param.grad is not None for param in model.parameters())


def test_sensor_reprojection_pos_loss_ignores_hip_tracker_error():
    diffusion = _make_loss_test_diffusion()
    target_tracker_pos_ref = torch.zeros(1, TRACKER_COUNT, 3)
    target_tracker_pos_ref[:, HIP_TRACKER_INDEX] = torch.tensor([10.0, 0.0, 0.0])
    target_sensor_valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=target_tracker_pos_ref,
        target_sensor_valid=target_sensor_valid,
    )

    losses = diffusion._realtime_pose_aux_losses(
        pred_xstart,
        x_start,
        model_kwargs,
        timesteps=torch.zeros(1, dtype=torch.long),
    )

    assert torch.allclose(losses["sensor_reprojection_pos_loss"], torch.zeros(1))


def test_sensor_reprojection_uses_root_y0_pelvis_height_offset():
    diffusion = _make_loss_test_diffusion()
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    target_tracker_pos_ref = torch.zeros(1, TRACKER_COUNT, 3)
    target_tracker_pos_ref[:, 0] = torch.tensor([0.0, 0.9, 0.0])
    target_sensor_valid = torch.zeros(1, TRACKER_COUNT, dtype=torch.bool)
    target_sensor_valid[:, [HIP_TRACKER_INDEX, 0]] = True
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=target_tracker_pos_ref,
        target_sensor_valid=target_sensor_valid,
    )
    pred_xstart[:, schema.root_height_slice(), REALTIME_POSE_TARGET_START] = 0.9
    x_start[:, schema.root_height_slice(), REALTIME_POSE_TARGET_START] = 0.9
    model_kwargs["y"]["joint_offsets_parent"][:, 0, 1] = 0.2

    losses = diffusion._realtime_pose_aux_losses(
        pred_xstart,
        x_start,
        model_kwargs,
        timesteps=torch.zeros(1, dtype=torch.long),
    )

    assert torch.allclose(losses["sensor_reprojection_pos_loss"], torch.zeros(1), atol=1e-6)


def test_sensor_reprojection_pos_loss_is_larger_at_low_noise_timestep():
    diffusion = _make_loss_test_diffusion()
    target_tracker_pos_ref = torch.zeros(1, TRACKER_COUNT, 3)
    target_tracker_pos_ref[:, 0] = torch.tensor([0.1, 0.0, 0.0])
    target_sensor_valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=target_tracker_pos_ref,
        target_sensor_valid=target_sensor_valid,
    )

    low_noise_losses = diffusion._realtime_pose_aux_losses(
        pred_xstart,
        x_start,
        model_kwargs,
        timesteps=torch.zeros(1, dtype=torch.long),
    )
    high_noise_losses = diffusion._realtime_pose_aux_losses(
        pred_xstart,
        x_start,
        model_kwargs,
        timesteps=torch.full((1,), diffusion.num_timesteps - 1, dtype=torch.long),
    )

    low_noise_loss = low_noise_losses["sensor_reprojection_pos_loss"]
    high_noise_loss = high_noise_losses["sensor_reprojection_pos_loss"]
    assert torch.all(low_noise_loss > high_noise_loss)
    assert torch.allclose(high_noise_loss, low_noise_loss * diffusion.tracker_pos_timestep_min_weight)


def test_train_loop_eval_reports_validation_loss(tmp_path):
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir)
    generate_realtime_pose_tasks_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(task_dir),
            "--splits",
            "train",
            "--samples_per_file",
            "1",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    loader = get_dataset_loader(
        data_dir=str(task_dir),
        batch_size=1,
        input_feats=REALTIME_POSE_INPUT_DIM,
        seq_len=REALTIME_POSE_SEQ_LEN,
        split="train",
        normalize_input=False,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
    )
    dist_util.setup_dist(-1)
    model = DiffusionPoserDiT(input_feats=REALTIME_POSE_INPUT_DIM, latent_dim=32, num_layers=1, num_heads=4, max_seq_len=REALTIME_POSE_SEQ_LEN)
    betas = gd.get_named_beta_schedule("cosine", 4, scale_betas=1.0)
    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(4, [4]),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
    )
    args = argparse.Namespace(
        batch_size=1,
        lr=1e-4,
        log_interval=1,
        save_interval=0,
        resume_checkpoint="",
        weight_decay=0.0,
        lr_anneal_steps=0,
        gradient_clip=False,
        snr_gamma=0.0,
        l1_loss=False,
        task_mode="realtime_pose_reconstruction",
        schema=REALTIME_POSE_SCHEMA_NAME,
        checkpoint_max_keep=0,
        save_dir=str(tmp_path / "run"),
        num_steps=1,
        eval_during_training=True,
        eval_num_batches=1,
        weighted_loss=False,
        normalizer_dir="",
        feature_w_file="feature_w.pt",
        model_ema=False,
    )
    platform = NoopPlatform()
    loop = TrainLoop(args, train_platform=platform, model=model, diffusion=diffusion, data=loader, eval_data=loader)

    loop.evaluate()

    reported_names = {item["name"] for item in platform.scalars}
    assert "eval/loss" in reported_names
    assert "eval/simple_loss" in reported_names
    assert "eval/aux_loss" in reported_names


def test_train_loop_rejects_empty_train_loader(tmp_path):
    args = argparse.Namespace(
        batch_size=4,
        lr=1e-4,
        log_interval=1,
        save_interval=0,
        resume_checkpoint="",
        weight_decay=0.0,
        lr_anneal_steps=0,
        gradient_clip=False,
        snr_gamma=0.0,
        l1_loss=False,
        task_mode="realtime_pose_reconstruction",
        schema=REALTIME_POSE_SCHEMA_NAME,
        checkpoint_max_keep=0,
        save_dir=str(tmp_path / "run"),
        num_steps=1,
        eval_during_training=False,
        eval_num_batches=1,
    )

    with pytest.raises(RuntimeError, match="没有可用 batch"):
        TrainLoop(args, train_platform=NoopPlatform(), model=torch.nn.Linear(1, 1), diffusion=object(), data=[])


def test_training_loss_nan_fails_fast():
    losses = {
        "loss": torch.tensor([float("nan")]),
        "simple_loss": torch.tensor([1.0]),
        "fk_loss": torch.tensor([float("inf")]),
    }
    with pytest.raises(FloatingPointError, match="bad_terms"):
        validate_finite_losses(losses=losses, loss=losses["loss"].mean(), batch={"keyid": ["bad_task"]})


class NoopPlatform:
    def __init__(self):
        self.scalars = []

    def report_scalar(self, *args, **kwargs):
        self.scalars.append(kwargs)
        return None
