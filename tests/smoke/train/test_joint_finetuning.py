from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from model.realtime_pose_current_dit import RealtimePoseCurrentDiT
from model.realtime_pose_predictor import RealtimePosePredictor
from train.train_platforms import NoPlatform
from train.training_loop import TrainLoop
from utils.model_util import load_realtime_pose_predictor


IDENTITY_6D = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
IDENTITY_POSE = IDENTITY_6D.repeat(24)


def _save_predictor(tmp_path) -> str:
    model = RealtimePosePredictor(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    )
    checkpoint = tmp_path / "predictor.pt"
    torch.save(model.state_dict(), checkpoint)
    (tmp_path / "args.json").write_text(
        json.dumps(
            {
                "latent_dim": 32,
                "layers": 1,
                "heads": 4,
                "feedforward_dim": 64,
                "dropout": 0.0,
            }
        ),
        encoding="utf-8",
    )
    return str(checkpoint)


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    tracker = torch.zeros(batch_size, 6, 10)
    tracker[..., 3:9] = IDENTITY_6D
    tracker[:, :3, 9] = 1.0
    tracker[:, 0, 1] = 1.6
    offsets = torch.zeros(batch_size, 24, 3)
    offsets[:, 1:, 1] = 0.1
    return {
        "x": IDENTITY_POSE.repeat(batch_size, 1),
        "motion_context": IDENTITY_POSE.repeat(batch_size, 10, 1),
        "core_tracker_context": torch.zeros(batch_size, 11, 54),
        "current_tracker_raw": tracker,
        "tracker_available": tracker[..., 9].bool(),
        "joint_offsets_parent": offsets,
        "target_joints_head_ref": torch.zeros(batch_size, 24, 3),
        "target_root_position_head_ref": torch.zeros(batch_size, 3),
        "target_root_yaw_world": torch.zeros(batch_size),
        "target_hip_height": torch.zeros(batch_size),
        "current_head_yaw_world": torch.zeros(batch_size),
        "previous_pose_target": IDENTITY_POSE.repeat(batch_size, 1),
    }


def test_joint_step_updates_dit_and_predictor_and_saves_paired_checkpoint(tmp_path):
    predictor_path = _save_predictor(tmp_path)
    dit = RealtimePoseCurrentDiT(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    save_dir = tmp_path / "joint"
    args = SimpleNamespace(
        joint_finetune=True,
        precision="fp32",
        save_dir=str(save_dir),
        normalizer_dir=str(tmp_path / "normalizer"),
        normalize_input=False,
        fabrik_iterations=1,
        ik_direction_only_quality=0.8,
        ik_residual_scale=0.5,
        ik_position_solved_quality=None,
        ik_gap_low=0.1,
        ik_gap_high=0.5,
        ik_direction_support=0.35,
        ik_untracked_strength=0.05,
        predictor_model_path=predictor_path,
        lr=1e-4,
        predictor_lr=1e-4,
        # 置零额外 Predictor MSE，单独验证最终姿态辅助损失也能回传到 RPM。
        predictor_loss_weight=0.0,
        weight_decay=0.0,
        model_ema_decay=0.9,
        weighted_loss=False,
        resume_checkpoint="",
        gradient_clip=True,
        snr_gamma=0.0,
        l1_loss=False,
        eval_during_training=False,
        save_interval=0,
        log_interval=0,
        num_steps=1,
    )
    loop = TrainLoop(
        args,
        NoPlatform(save_dir),
        dit,
        diffusion,
        data=[],
    )

    dit_before = dit.joint_output.weight.detach().clone()
    predictor_before = loop.predictor.output.weight.detach().clone()
    losses = loop.run_step(_batch())

    assert "predictor_pose_loss" in losses
    assert any(parameter.grad is not None for parameter in dit.parameters())
    assert any(parameter.grad is not None for parameter in loop.predictor.parameters())
    assert not torch.equal(dit_before, dit.joint_output.weight)
    assert not torch.equal(predictor_before, loop.predictor.output.weight)

    loop.step = 1
    loop.save()
    assert (save_dir / "model000000001.pt").is_file()
    assert (save_dir / "predictor000000001.pt").is_file()
    assert (save_dir / "model_latest.pt").is_file()
    assert (save_dir / "predictor_latest.pt").is_file()

    (save_dir / "args.json").write_text(
        json.dumps(
            {
                "predictor_architecture": {
                    "latent_dim": 32,
                    "layers": 1,
                    "heads": 4,
                    "feedforward_dim": 64,
                    "dropout": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    loaded = load_realtime_pose_predictor(
        save_dir / "predictor_latest.pt",
        torch.device("cpu"),
    )
    assert loaded.latent_dim == 32

    resumed_args = SimpleNamespace(
        **{
            **vars(args),
            "resume_checkpoint": str(save_dir / "model000000001.pt"),
        }
    )
    resumed = TrainLoop(
        resumed_args,
        NoPlatform(save_dir),
        RealtimePoseCurrentDiT(
            latent_dim=32,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        ),
        diffusion,
        data=[],
    )
    assert resumed.step == 1
    torch.testing.assert_close(
        resumed.predictor.output.weight,
        loop.predictor.output.weight,
    )
