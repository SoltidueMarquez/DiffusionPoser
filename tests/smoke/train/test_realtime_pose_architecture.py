from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from data_loaders.realtime_pose_kinematics import (
    make_yaw_rotation_torch,
    rotation_6d_forward_up_torch,
    rotation_6d_to_matrix_torch,
)
from data_loaders.realtime_pose_predictor_features import build_predictor_step_features_torch
from model.realtime_pose_current_dit import RealtimePoseCurrentDiT
from model.realtime_pose_predictor import RealtimePosePredictor
from train.predictor_losses import compute_predictor_losses
from train.predictor_training_loop import (
    PredictorTrainLoop,
    append_predictor_current_prediction,
    resolve_predictor_resume_checkpoint,
)
from utils.normalizer import RealtimePoseNormalizer


def test_predictor_transformer_contract():
    model = RealtimePosePredictor(
        latent_dim=64, num_layers=1, num_heads=4, feedforward_dim=128
    )
    output = model(torch.randn(2, 10, 144), torch.randn(2, 11, 54))
    assert output.shape == (2, 11, 144)
    assert torch.isfinite(output).all()


def test_predictor_rollout_only_uses_horizon_zero():
    identity = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 10, 24, 1, 1)
    prediction_a = torch.zeros(1, 11, 144)
    prediction_a[:, 0] = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0]).repeat(24)
    prediction_b = prediction_a.clone()
    prediction_b[:, 1:] = torch.randn_like(prediction_b[:, 1:]) * 100.0
    kwargs = dict(
        motion_world=identity,
        head_yaw_world=torch.zeros(1),
        pose_mean=torch.zeros(144),
        pose_scale=torch.ones(144),
    )
    first = append_predictor_current_prediction(prediction_normalized=prediction_a, **kwargs)
    second = append_predictor_current_prediction(prediction_normalized=prediction_b, **kwargs)
    torch.testing.assert_close(first, second)
    assert first.shape == (1, 10, 24, 3, 3)


def test_current_dit_contract_and_predictor_future_changes_output():
    torch.manual_seed(2)
    model = RealtimePoseCurrentDiT(
        latent_dim=64, num_layers=1, num_heads=4, dropout=0.0
    ).eval()
    # AdaLN gate 默认零初始化；打开 temporal gate 后验证 future 确实位于 forward 路径。
    with torch.no_grad():
        model.blocks[0].adaln_modulation[-1].bias[5 * 64 : 6 * 64] = 1.0
    x = torch.randn(2, 144)
    timestep = torch.tensor([1, 2])
    motion = torch.randn(2, 10, 144)
    predictor = torch.randn(2, 11, 144)
    joint_condition = torch.randn(2, 24, 10)
    first = model(x, timestep, motion, predictor, joint_condition)
    changed = predictor.clone()
    changed[:, 1:] += 2.0
    second = model(x, timestep, motion, changed, joint_condition)
    assert first.shape == (2, 144)
    assert not torch.allclose(first, second)


def test_dit_backward_does_not_create_predictor_gradients():
    predictor = RealtimePosePredictor(
        latent_dim=32, num_layers=1, num_heads=4, feedforward_dim=64
    ).eval().requires_grad_(False)
    dit = RealtimePoseCurrentDiT(latent_dim=32, num_layers=1, num_heads=4)
    motion = torch.randn(1, 10, 144)
    with torch.no_grad():
        horizon = predictor(motion, torch.randn(1, 11, 54))
    output = dit(
        torch.randn(1, 144),
        torch.tensor([1]),
        motion,
        horizon,
        torch.randn(1, 24, 10),
    )
    output.square().mean().backward()
    assert all(parameter.grad is None for parameter in predictor.parameters())
    assert any(parameter.grad is not None for parameter in dit.parameters())


def test_predictor_rotation_output_projects_to_finite_so3():
    rotations = rotation_6d_to_matrix_torch(torch.randn(2, 11, 24, 6))
    assert torch.isfinite(rotations).all()
    assert torch.all(torch.linalg.det(rotations) > 0.999)


def test_predictor_four_full_horizon_losses_are_finite_and_differentiable():
    prediction = torch.randn(2, 11, 144, requires_grad=True)
    losses = compute_predictor_losses(
        prediction_normalized=prediction,
        target_normalized=torch.randn(2, 11, 144),
        motion_context_normalized=torch.randn(2, 10, 144),
        joint_offsets_parent=torch.randn(2, 24, 3) * 0.05,
        pose_mean=torch.zeros(144),
        pose_scale=torch.ones(144),
    )
    assert set(losses) == {
        "loss",
        "pose_mse",
        "rotation_velocity_loss",
        "fk_loss",
        "fk_velocity_loss",
    }
    assert all(value.shape == (2,) for value in losses.values())
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["loss"].mean().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("rollout_steps", [0, 30])
def test_predictor_training_loop_decodes_resident_rotation6d_batch(
    tmp_path, rollout_steps
):
    normalizer_dir = tmp_path / "normalizer"
    RealtimePoseNormalizer(normalizer_dir, disable=True).save(
        pose_mean=torch.zeros(144),
        pose_scale=torch.ones(144),
        tracker_mean=torch.zeros(6, 9),
        tracker_std=torch.ones(6, 9),
        predictor_sparse_mean=torch.zeros(54),
        predictor_sparse_std=torch.ones(54),
    )
    args = SimpleNamespace(
        save_dir=str(tmp_path / "predictor"),
        normalizer_dir=str(normalizer_dir),
        lr=1e-4,
        lr_drop_step=50_000,
        lr_drop_factor=30.0,
        weight_decay=0.0,
        ema_decay=0.995,
        resume_checkpoint="",
    )
    model = RealtimePosePredictor(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    )
    loop = PredictorTrainLoop(args, model, [], torch.device("cpu"))
    identity_6d = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    batch = {
        "joint_rotations_world_6d": identity_6d.repeat(1, 52, 24, 1),
        "tracker_positions_world": torch.zeros(1, 52, 6, 3),
        "tracker_rotations_world_6d": identity_6d.repeat(1, 52, 6, 1),
        "floor_y": torch.zeros(1, 52),
        "joint_offsets_parent": torch.zeros(1, 24, 3),
    }
    losses = loop._forward_loss(batch, rollout_steps, gradient=True)
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["loss"].mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_predictor_rolling_rebuilds_head_yaw_frame_for_30_steps():
    torch.manual_seed(4)
    model = RealtimePosePredictor(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    ).eval()
    motion_world = torch.eye(3).reshape(1, 1, 1, 3, 3).repeat(1, 10, 24, 1, 1)
    frame_count = 41
    tracker_yaws = torch.linspace(0.0, 0.8, frame_count)
    tracker_rotation = make_yaw_rotation_torch(tracker_yaws)
    tracker_rotation = tracker_rotation[:, None].repeat(1, 6, 1, 1)
    tracker_rotation_6d = rotation_6d_forward_up_torch(tracker_rotation)[None]
    tracker_position = torch.zeros(1, frame_count, 6, 3)
    tracker_position[0, :, :, 1] = torch.tensor(
        [1.7, 1.3, 1.3, 1.0, 0.1, 0.1]
    )
    tracker_position[0, :, :, 0] = torch.linspace(0.0, 0.2, frame_count)[:, None]
    head_yaws = []
    with torch.no_grad():
        for step_index in range(30):
            motion, sparse, _, head_yaw = build_predictor_step_features_torch(
                motion_world,
                tracker_position[:, step_index : step_index + 12],
                tracker_rotation_6d[:, step_index : step_index + 12],
                torch.zeros(1),
            )
            prediction = model(motion, sparse)
            assert torch.isfinite(prediction).all()
            motion_world = append_predictor_current_prediction(
                motion_world=motion_world,
                prediction_normalized=prediction,
                head_yaw_world=head_yaw,
                pose_mean=torch.zeros(144),
                pose_scale=torch.ones(144),
            )
            assert torch.isfinite(motion_world).all()
            head_yaws.append(float(head_yaw[0]))
    assert head_yaws[-1] > head_yaws[0] + 0.4


def test_predictor_checkpoint_save_and_strict_resume(tmp_path):
    normalizer_dir = tmp_path / "normalizer"
    RealtimePoseNormalizer(normalizer_dir, disable=True).save(
        pose_mean=torch.zeros(144),
        pose_scale=torch.ones(144),
        tracker_mean=torch.zeros(6, 9),
        tracker_std=torch.ones(6, 9),
        predictor_sparse_mean=torch.zeros(54),
        predictor_sparse_std=torch.ones(54),
    )
    args = SimpleNamespace(
        save_dir=str(tmp_path / "predictor"),
        normalizer_dir=str(normalizer_dir),
        lr=1e-4,
        lr_drop_step=50_000,
        lr_drop_factor=30.0,
        weight_decay=0.0,
        ema_decay=0.995,
        checkpoint_max_keep=2,
        resume_checkpoint="",
    )
    model = RealtimePosePredictor(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    )
    loop = PredictorTrainLoop(args, model, [], torch.device("cpu"))
    for step in (5, 6, 7):
        loop.step = step
        loop.save()

    checkpoint = tmp_path / "predictor" / "model000000007.pt"
    assert (tmp_path / "predictor" / "model_latest.pt").is_file()
    for prefix in ("model", "ema", "opt"):
        assert not (tmp_path / "predictor" / f"{prefix}000000005.pt").exists()
        assert (tmp_path / "predictor" / f"{prefix}000000006.pt").is_file()
        assert (tmp_path / "predictor" / f"{prefix}000000007.pt").is_file()
    resumed_args = SimpleNamespace(**{**vars(args), "resume_checkpoint": "latest"})
    resumed_model = RealtimePosePredictor(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    )
    resumed = PredictorTrainLoop(resumed_args, resumed_model, [], torch.device("cpu"))
    assert resumed.step == 7
    for expected, actual in zip(model.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_predictor_rollout_sampler_covers_single_stage_range(tmp_path):
    torch.manual_seed(0)
    normalizer_dir = tmp_path / "normalizer"
    RealtimePoseNormalizer(normalizer_dir, disable=True).save(
        pose_mean=torch.zeros(144),
        pose_scale=torch.ones(144),
        tracker_mean=torch.zeros(6, 9),
        tracker_std=torch.ones(6, 9),
        predictor_sparse_mean=torch.zeros(54),
        predictor_sparse_std=torch.ones(54),
    )
    args = SimpleNamespace(
        save_dir=str(tmp_path / "predictor"),
        normalizer_dir=str(normalizer_dir),
        lr=3e-4,
        lr_drop_step=50_000,
        lr_drop_factor=30.0,
        weight_decay=1e-4,
        ema_decay=0.995,
        checkpoint_max_keep=3,
        resume_checkpoint="",
    )
    model = RealtimePosePredictor(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    )
    loop = PredictorTrainLoop(args, model, [], torch.device("cpu"))
    samples = [loop.sample_rollout_steps() for _ in range(512)]
    assert min(samples) == 0
    assert max(samples) == 30
    assert all(0 <= value <= 30 for value in samples)
    loop.step = 50_000
    loop._update_learning_rate()
    assert loop.optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_predictor_resume_rejects_inference_alias(tmp_path):
    latest = tmp_path / "model_latest.pt"
    latest.touch()
    with pytest.raises(ValueError, match="仅用于推理"):
        resolve_predictor_resume_checkpoint(tmp_path, latest)
