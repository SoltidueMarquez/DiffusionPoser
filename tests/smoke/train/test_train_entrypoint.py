from __future__ import annotations

import json

import pytest
import torch

from train.train_realtime_pose_predictor import (
    _prepare_save_dir,
    build_parser as build_predictor_parser,
)
from utils.model_util import create_model_and_diffusion
from utils.parser_util import (
    build_train_arg_parser,
    joint_finetune_args,
    train_args,
)
from utils.training_precision import TrainingPrecision


def _required(tmp_path):
    return [
        "--data_dir", str(tmp_path / "tasks"),
        "--normalizer_dir", str(tmp_path / "normalizer"),
        "--predictor_model_path", str(tmp_path / "predictor.pt"),
        "--save_dir", str(tmp_path / "runs"),
        "--ik_direction_only_quality", "0.8",
        "--ik_residual_scale", "0.5",
        "--ik_gap_low", "0.1",
        "--ik_gap_high", "0.5",
    ]


def test_training_parser_exposes_two_model_boundary_without_cold_start(tmp_path):
    args = train_args(_required(tmp_path))
    assert args.model_arch == "current_dit"
    assert args.predictor_model_path.endswith("predictor.pt")
    assert not hasattr(args, "cold_start_prob")
    assert not hasattr(args, "scenario_weights")
    assert not hasattr(args, "tracker_confidence_warmup")
    assert not hasattr(args, "history_noise_prob")
    assert not hasattr(args, "history_noise_min_deg")
    assert not hasattr(args, "history_noise_max_deg")
    assert not hasattr(args, "history_noise_temporal_rho")
    assert args.latent_dim == 192
    assert args.layers == 4
    assert args.heads == 6
    assert not hasattr(args, "contact_loss_weight")
    assert not hasattr(args, "contact_slide_loss_weight")
    assert args.log_interval == 10
    assert args.precision == "fp32"


def test_training_parsers_accept_bf16(tmp_path):
    dit_args = train_args(_required(tmp_path) + ["--precision", "bf16"])
    predictor_args = build_predictor_parser().parse_args(
        [
            "--source_dir", str(tmp_path / "source"),
            "--normalizer_dir", str(tmp_path / "normalizer"),
            "--save_dir", str(tmp_path / "predictor"),
            "--precision", "bf16",
        ]
    )
    assert dit_args.precision == predictor_args.precision == "bf16"


def test_joint_finetune_parser_uses_small_two_model_learning_rates(tmp_path):
    args = joint_finetune_args(
        _required(tmp_path)
        + ["--dit_model_path", str(tmp_path / "dit.pt")]
    )
    assert args.joint_finetune
    assert args.dit_model_path.endswith("dit.pt")
    assert args.lr == 1e-5
    assert args.predictor_lr == 1e-6
    assert args.predictor_loss_weight == 1.0
    assert args.num_steps == 20_000


def test_predictor_parser_uses_single_official_style_schedule(tmp_path):
    args = build_predictor_parser().parse_args(
        [
            "--source_dir", str(tmp_path / "source"),
            "--normalizer_dir", str(tmp_path / "normalizer"),
            "--save_dir", str(tmp_path / "predictor"),
        ]
    )
    assert args.num_steps == 100_000
    assert args.lr == 3e-4
    assert args.weight_decay == 1e-4
    assert args.lr_drop_step == 50_000
    assert args.lr_drop_factor == 30.0
    assert not hasattr(args, "stage")
    assert not hasattr(args, "stage1_model_path")


def test_predictor_fresh_run_refuses_nonempty_save_dir(tmp_path):
    save_dir = tmp_path / "predictor"
    save_dir.mkdir()
    (save_dir / "model000000001.pt").touch()
    with pytest.raises(FileExistsError, match="save_dir 非空"):
        _prepare_save_dir(save_dir, "")
    _prepare_save_dir(save_dir, "latest")


def test_fp32_precision_forward_keeps_gradients():
    model = torch.nn.Linear(4, 2)
    precision = TrainingPrecision("fp32", torch.device("cpu"))
    output = precision.forward(model, torch.randn(3, 4))
    assert output.dtype == torch.float32
    output.square().mean().backward()
    assert model.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_bf16_precision_forward_returns_fp32():
    model = torch.nn.Linear(4, 2).cuda()
    precision = TrainingPrecision("bf16", torch.device("cuda:0"))
    output = precision.forward(model, torch.randn(3, 4, device="cuda:0"))
    assert output.dtype == torch.float32
    output.square().mean().backward()
    assert model.weight.grad is not None


def test_ik_calibration_fills_missing_values(tmp_path):
    calibration = tmp_path / "ik.json"
    calibration.write_text(
        json.dumps(
            {
                "recommended_parameters": {
                    "ik_direction_only_quality": 0.7,
                    "ik_residual_scale": 0.2,
                    "ik_gap_low": 0.12,
                    "ik_gap_high": 0.62,
                }
            }
        ),
        encoding="utf-8",
    )
    args = train_args(
        _required(tmp_path)[:-8] + ["--ik_calibration_path", str(calibration)]
    )
    assert args.ik_direction_only_quality == 0.7
    assert args.ik_residual_scale == 0.2
    assert args.ik_gap_low == 0.12
    assert args.ik_gap_high == 0.62


def test_model_factory_creates_current_dit(tmp_path):
    args = train_args(_required(tmp_path) + ["--latent_dim", "32", "--layers", "1", "--heads", "4", "--diffusion_steps", "2"])
    model, diffusion = create_model_and_diffusion(args)
    assert model.__class__.__name__ == "RealtimePoseCurrentDiT"
    assert not hasattr(model, "contact_head")
    assert diffusion.num_timesteps == 2
    assert not hasattr(diffusion, "contact_loss_weight")
    assert not hasattr(diffusion, "contact_slide_loss_weight")


def test_parser_model_token_length_is_21():
    parser = build_train_arg_parser()
    assert parser.get_default("max_seq_len") == 21
