from __future__ import annotations

import argparse

import torch
import pytest

from data_loaders.generate_realtime_pose_tasks import main as generate_realtime_pose_tasks_main
from data_loaders.get_data import get_dataset_loader
from data_loaders.sensor_masking import REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_SEQ_LEN
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.diffusionposer_dit import DiffusionPoserDiT
from train.training_loop import TrainLoop, validate_finite_losses
from utils import dist_util
from tests.smoke.realtime_pose_fixtures import write_toy_source_dataset


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

    model_kwargs["y"]["target_foot_contact"] = torch.zeros_like(model_kwargs["y"]["target_foot_contact"])
    aux_terms = diffusion._realtime_pose_aux_losses(batch["x"], batch["x"], model_kwargs)
    assert torch.allclose(aux_terms["foot_lock_loss"], torch.zeros_like(aux_terms["foot_lock_loss"]))

    model_kwargs_missing_contact = loop.mask_manager(batch, batch["x"])
    del model_kwargs_missing_contact["y"]["target_foot_contact"]
    with pytest.raises(KeyError, match="target_foot_contact"):
        diffusion._realtime_pose_aux_losses(batch["x"], batch["x"], model_kwargs_missing_contact)


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
