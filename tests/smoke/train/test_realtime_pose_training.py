from __future__ import annotations

import argparse

import numpy as np
import torch
import pytest

from data_loaders.generate_realtime_pose_tasks import main as generate_realtime_pose_tasks_main
from data_loaders.get_data import get_dataset_loader
from data_loaders.sensor_masking import (
    HIP_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_START,
    RIGHT_FOOT_TRACKER_INDEX,
    TRACKER_COUNT,
    get_schema_spec,
)
from diffusion import gaussian_diffusion as gd
from diffusion.realtime_pose import REALTIME_POSE_LOSS_TERM_TO_WEIGHT
from diffusion.realtime_pose.loss_terms import (
    _normalize_masked_samples,
    _normalize_weighted_feet,
    _rotation_axis_cosine_loss,
)
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.diffusionposer_dit import DiffusionPoserDiT
from train.training_loop import (
    TrainLoop,
    log_loss_dict,
    validate_finite_losses,
)
from train.realtime_rollout import long_rollout_max_horizon
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
            "gt_prev_joints_world": torch.zeros(batch_size, 24, 3),
            "pred_prev_joints_world": torch.zeros(batch_size, 24, 3),
            "gt_prev_local_pose_6d": identity_pose.repeat(batch_size, 1),
            "pred_prev_local_pose_6d": identity_pose.repeat(batch_size, 1),
            "previous_state_is_predicted": torch.zeros(batch_size, dtype=torch.bool),
            "target_frame_dt_seconds": torch.full((batch_size,), 1.0 / 60.0),
            "target_root_pos_world": torch.zeros(batch_size, 3),
            "target_root_yaw": torch.zeros(batch_size),
            "gt_prev_root_yaw": torch.zeros(batch_size),
            "target_floor_y": torch.zeros(batch_size),
            "prev_root_pos_world": torch.zeros(batch_size, 3),
            "prev_root_yaw": torch.zeros(batch_size),
            "tracker_ref_root_pos_world": torch.zeros(batch_size, 3),
            "tracker_ref_root_yaw": torch.zeros(batch_size),
            "joint_offsets_parent": torch.zeros(batch_size, 24, 3),
            "target_stationary_prob_5": torch.zeros(batch_size, 5),
            "target_tracker_pos_ref": target_tracker_pos_ref,
            "target_tracker_rot_ref_6d": torch.as_tensor(IDENTITY_6D).repeat(batch_size, TRACKER_COUNT, 1),
            "target_sensor_valid": target_sensor_valid,
            "resolver_before_target_root_pos_world": torch.zeros(batch_size, 3),
            "resolver_before_target_root_yaw": torch.zeros(batch_size),
            "resolver_before_target_pelvis_height": torch.zeros(batch_size),
            "resolver_before_target_hip_valid": target_sensor_valid[:, HIP_TRACKER_INDEX].clone(),
            "resolver_before_target_reconnect_start_root_pos_world": torch.zeros(batch_size, 3),
            "resolver_before_target_reconnect_start_root_yaw": torch.zeros(batch_size),
            "resolver_before_target_reconnect_start_pelvis_height": torch.zeros(batch_size),
            "resolver_before_target_reconnect_elapsed_seconds": torch.zeros(batch_size),
            "resolver_before_target_last_timestamp_seconds": torch.zeros(batch_size),
            "resolver_before_target_tracking_origin_revision": torch.zeros(batch_size, dtype=torch.long),
            "target_timestamp_seconds": torch.full((batch_size,), 1.0 / 60.0),
            "target_tracking_origin_revision": torch.zeros(batch_size, dtype=torch.long),
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
        tracker_relative_pos_huber_beta=0.05,
        aux_timestep_min_weight=0.1,
        aux_timestep_gamma=2.0,
    )


def test_rotation_axis_cosine_loss_is_non_negative_under_mixed_precision_drift():
    identity_forward_up = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=torch.bfloat16)
    pred = identity_forward_up.repeat(2, 6, 1)
    target = identity_forward_up.repeat(2, 6, 1)

    # Simulate the small axis-length drift produced by BF16 FK composition.
    pred[..., :3] *= torch.tensor(1.015625, dtype=torch.bfloat16)
    pred[..., 3:] *= torch.tensor(1.0078125, dtype=torch.bfloat16)

    loss = _rotation_axis_cosine_loss(pred, target)

    assert loss.dtype == torch.float32
    assert torch.all(loss >= 0.0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_rotation_axis_cosine_loss_preserves_expected_axis_error():
    target = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    pred = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    loss = _rotation_axis_cosine_loss(pred, target)

    # Forward differs by 90 degrees (loss 1), up is identical (loss 0).
    assert loss.item() == pytest.approx(0.5, abs=1e-7)


def test_realtime_aux_backward_is_finite_for_bfloat16_model_output():
    diffusion = _make_loss_test_diffusion()
    valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    valid[:, HIP_TRACKER_INDEX] = False
    pred, target, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=torch.zeros(1, TRACKER_COUNT, 3),
        target_sensor_valid=valid,
    )
    model_kwargs["y"]["previous_state_is_predicted"][:] = True
    model_kwargs["y"]["target_stationary_prob_5"][:, 1:3] = 1.0
    pred = pred.to(torch.bfloat16).requires_grad_(True)

    losses = diffusion.realtime_pose_loss.compute(pred, target, model_kwargs)
    total = sum(losses[name].mean() for name in REALTIME_POSE_LOSS_TERM_TO_WEIGHT)
    total.backward()

    assert total.dtype == torch.float32
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_default_auxiliary_loss_fixed_input_regression():
    """固定输入锁定当前默认配置下的逐项 loss、timestep 衰减和 rollout 速度项。"""

    torch.manual_seed(20260716)
    diffusion = _make_loss_test_diffusion()
    valid = torch.ones(2, TRACKER_COUNT, dtype=torch.bool)
    valid[1, HIP_TRACKER_INDEX] = False
    valid[1, LEFT_FOOT_TRACKER_INDEX] = False
    valid[1, RIGHT_FOOT_TRACKER_INDEX] = False
    tracker_pos = torch.randn(2, TRACKER_COUNT, 3) * 0.05
    pred, target, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=tracker_pos,
        target_sensor_valid=valid,
    )
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    pred[:, schema.body_pose_slice(), REALTIME_POSE_TARGET_START] += torch.randn(2, 144) * 0.03
    pred[:, schema.root_yaw_delta_slice(), REALTIME_POSE_TARGET_START] += torch.tensor(
        [[0.05, -0.03], [0.02, 0.04]]
    )
    pred[:, schema.root_delta_xz_slice(), REALTIME_POSE_TARGET_START] += torch.tensor(
        [[0.02, -0.01], [-0.03, 0.04]]
    )
    pred[:, schema.root_height_slice(), REALTIME_POSE_TARGET_START] += torch.tensor(
        [[0.04], [-0.02]]
    )
    pred[:, schema.stationary_prob_slice(), REALTIME_POSE_TARGET_START] = torch.tensor(
        [[0.8, 0.6, 0.2, 1.1, -0.1], [0.2, 0.9, 0.75, 0.1, 0.3]]
    )
    y = model_kwargs["y"]
    y["target_joints_world"] = torch.randn(2, 24, 3) * 0.04
    y["gt_prev_joints_world"] = torch.randn(2, 24, 3) * 0.03
    y["pred_prev_joints_world"] = torch.randn(2, 24, 3) * 0.02
    y["gt_prev_local_pose_6d"] += torch.randn(2, 144) * 0.01
    y["pred_prev_local_pose_6d"] += torch.randn(2, 144) * 0.015
    y["previous_state_is_predicted"][:] = True
    y["target_root_yaw"] = torch.tensor([0.15, -0.22])
    y["gt_prev_root_yaw"] = torch.tensor([0.10, -0.20])
    y["prev_root_yaw"] = torch.tensor([0.08, -0.18])
    y["target_stationary_prob_5"] = torch.tensor(
        [[0.95, 0.8, 0.15, 0.9, 0.05], [0.1, 0.85, 0.8, 0.2, 0.4]]
    )
    y["target_tracker_rot_ref_6d"] += torch.randn(2, TRACKER_COUNT, 6) * 0.02

    losses = diffusion.realtime_pose_loss.apply_weights(
        diffusion.realtime_pose_loss.compute(
            pred,
            target,
            model_kwargs,
            timesteps=torch.tensor([0, diffusion.num_timesteps - 1]),
        )
    )
    expected = {
        "local_rotation_loss": (0.0006609472, 0.0008690376),
        "body_geometry_loss": (0.0486824922, 0.0855461881),
        "tracker_relative_pos_loss": (0.0934524313, 0.0521277189),
        "tracker_relative_rot_loss": (0.0034956932, 0.0045674741),
        "nohip_yaw_loss": (0.0, 0.0569578409),
        "nohip_root_xz_loss": (0.0, 0.0010689230),
        "nohip_height_loss": (0.0, 0.2328828424),
        "stationary_margin_loss": (0.0036363641, 0.0005000002),
        "contact_height_loss": (0.0, 0.1162884235),
        "contact_velocity_loss": (0.0, 1.4444473982),
        "joint_velocity_loss": (2.4483196735, 2.7352113724),
        "rotation_velocity_loss": (1.1085032225, 1.1846578121),
        "yaw_velocity_loss": (0.0, 25.3074073792),
        "aux_timestep_weight": (1.0, 0.1000000015),
        "aux_loss": (0.0206602681, 0.0555981025),
    }
    for loss_name, values in expected.items():
        assert torch.allclose(losses[loss_name], torch.tensor(values), rtol=1e-5, atol=1e-6)


def test_model_forward_has_frame_positional_embedding_and_seq_limit():
    model = DiffusionPoserDiT(input_feats=REALTIME_POSE_INPUT_DIM, latent_dim=32, num_layers=1, num_heads=4, max_seq_len=REALTIME_POSE_SEQ_LEN)
    assert tuple(model.frame_pos_embed.shape) == (1, REALTIME_POSE_SEQ_LEN, 32)
    x = torch.zeros(2, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN)
    mask = torch.zeros_like(x, dtype=torch.bool)
    y = model(x, torch.zeros(2), inpaint_cond=mask)
    assert tuple(y.shape) == tuple(x.shape)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(torch.zeros(1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN + 1), torch.zeros(1))


def test_log_loss_dict_accepts_bfloat16_values():
    diffusion = argparse.Namespace(num_timesteps=4)
    timesteps = torch.tensor([0, 3], dtype=torch.long)
    losses = {"loss": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    log_loss_dict(diffusion, timesteps, losses)


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
    batch = loop.prepare_teacher_forced_temporal_state(batch)
    model_kwargs = loop.mask_manager(batch, batch["x"])
    t = torch.zeros(1, dtype=torch.long)
    losses = diffusion.training_losses(model, batch["x"], t, model_kwargs=model_kwargs, snr_gamma=0.0)
    assert {
        "loss",
        "simple_loss",
        "local_rotation_loss",
        "body_geometry_loss",
        "tracker_relative_pos_loss",
        "tracker_relative_rot_loss",
        "nohip_yaw_loss",
        "nohip_root_xz_loss",
        "nohip_height_loss",
        "stationary_margin_loss",
        "contact_velocity_loss",
        "contact_height_loss",
        "joint_velocity_loss",
        "rotation_velocity_loss",
        "yaw_velocity_loss",
        "simple_stationary_channel_weight",
        "simple_stationary_upweight",
        "aux_loss",
    }.issubset(losses)
    assert torch.allclose(losses["loss"], losses["simple_loss"] + losses["aux_loss"])
    assert torch.all(losses["tracker_relative_rot_loss"] >= 0.0)
    assert "head_anchor_loss" not in losses
    assert "root_delta_loss" not in losses
    losses["loss"].mean().backward()

    model_kwargs["y"]["target_stationary_prob_5"] = torch.zeros_like(model_kwargs["y"]["target_stationary_prob_5"])
    aux_terms = diffusion.realtime_pose_loss.compute(batch["x"], batch["x"], model_kwargs)
    assert torch.allclose(
        aux_terms["contact_velocity_loss"],
        torch.zeros_like(aux_terms["contact_velocity_loss"]),
    )
    assert torch.allclose(
        aux_terms["contact_height_loss"],
        torch.zeros_like(aux_terms["contact_height_loss"]),
    )

    model_kwargs_missing_stationary = loop.mask_manager(batch, batch["x"])
    del model_kwargs_missing_stationary["y"]["target_stationary_prob_5"]
    with pytest.raises(KeyError, match="target_stationary_prob_5"):
        diffusion.realtime_pose_loss.compute(batch["x"], batch["x"], model_kwargs_missing_stationary)


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
        short_rollout_loss_weight=0.25,
        short_rollout_prob=1.0,
        long_rollout_loss_weight=0.0,
        long_rollout_prob=0.0,
        diffusion_steps=50,
        noise_schedule="cosine",
        predict_xstart=True,
        sigma_small=True,
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
    original_target = rollout_batch["x"].clone()
    next_batch, next_sample = loop.prepare_one_step_rollout_batch(
        batch=loop.prepare_teacher_forced_temporal_state(batch),
        rollout_batch=rollout_batch,
        pred_xstart=pred_xstart,
    )
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    for feature_slice in (schema.root_yaw_delta_slice(), schema.root_delta_xz_slice()):
        assert torch.allclose(
            next_sample[:, feature_slice, REALTIME_POSE_TARGET_START],
            original_target[:, feature_slice, REALTIME_POSE_TARGET_START],
        )
    assert torch.allclose(
        next_batch["target_root_delta_xz_ref"],
        rollout_batch["target_root_delta_xz_ref"],
    )

    t = torch.zeros(1, dtype=torch.long)
    base_losses = loop.compute_losses(batch=batch, timesteps=t)
    rollout_losses = loop.compute_rollout_terminal_losses(batch=batch, horizon=1)
    total_loss = base_losses["loss"] + 0.25 * rollout_losses["loss"]
    total_loss.mean().backward()
    assert any(param.grad is not None for param in model.parameters())


def test_rollout_h8_reinjects_every_overlapping_prediction_without_prefix_grad(tmp_path):
    source_dir = tmp_path / "sources"
    task_dir = tmp_path / "tasks"
    write_toy_source_dataset(source_dir, frame_count=REALTIME_POSE_SEQ_LEN + 8)
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
            "9",
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
        rollout_steps=9,
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
    diffusion = _make_loss_test_diffusion()
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
        rollout_steps=9,
        short_rollout_loss_weight=0.0,
        short_rollout_prob=0.0,
        long_rollout_loss_weight=0.5,
        long_rollout_prob=1.0,
        rollout_ddim_steps=10,
        diffusion_steps=50,
        noise_schedule="cosine",
        predict_xstart=True,
        sigma_small=True,
    )
    loop = TrainLoop(args, train_platform=NoopPlatform(), model=model, diffusion=diffusion, data=loader)
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    stationary_channel = schema.stationary_prob_slice().start
    tracker_condition_channel = schema.tracker_pos_slice().start
    current_batch = batch
    history = []
    predictions = []
    for index in range(8):
        current_batch["conditioned_x"][:, tracker_condition_channel, REALTIME_POSE_TARGET_START] = float(
            index + 11
        )
        prediction = current_batch["x"].clone().requires_grad_(True)
        prediction = prediction.clone()
        prediction[:, stationary_channel, REALTIME_POSE_TARGET_START] = float(index + 1)
        predictions.append(prediction)
        current_batch, _, history = loop.prepare_rollout_batch(
            batch=current_batch,
            rollout_batch=batch["rollout"][index],
            pred_xstart=prediction,
            predicted_history=history,
        )

    history_start = REALTIME_POSE_TARGET_START - 8
    actual = current_batch["conditioned_x"][
        0, stationary_channel, history_start:REALTIME_POSE_TARGET_START
    ]
    assert torch.allclose(actual, torch.arange(1, 9, dtype=actual.dtype))
    actual_tracker_conditions = current_batch["conditioned_x"][
        0, tracker_condition_channel, history_start:REALTIME_POSE_TARGET_START
    ]
    assert torch.allclose(
        actual_tracker_conditions,
        torch.arange(11, 19, dtype=actual_tracker_conditions.dtype),
    )
    assert len(history) == 8
    assert all(value.shape == (1, REALTIME_POSE_INPUT_DIM) for value in history)
    assert all(not value.requires_grad for value in history)
    assert current_batch["previous_state_is_predicted"].all()


def test_long_rollout_curriculum_expands_from_h2_to_h8():
    values = [
        long_rollout_max_horizon(
            global_step=step,
            rollout_steps=9,
            phase1_steps=500,
            phase2_steps=1500,
            phase1_max_horizon=2,
            phase2_max_horizon=4,
        )
        for step in (0, 499, 500, 1499, 1500, 5000)
    ]

    assert values == [2, 2, 4, 4, 8, 8]


def test_long_rollout_transition_sampling_prefers_reconnect_with_double_weight(monkeypatch):
    loop = object.__new__(TrainLoop)
    loop.device = torch.device("cpu")
    loop.long_rollout_transition_prob = 0.5
    counts = {
        2: (0, 1),
        3: (1, 0),
        4: (0, 0),
    }
    loop.rollout_transition_counts = lambda batch, horizon: counts[horizon]

    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))

    def choose_reconnect(weights, count):
        assert count == 1
        assert torch.equal(weights, torch.tensor([1.0, 2.0, 0.0]))
        return torch.tensor([1])

    monkeypatch.setattr(torch, "multinomial", choose_reconnect)

    horizon, transition_aware = loop.sample_long_rollout_horizon(
        batch={},
        max_horizon=4,
    )

    assert horizon == 3
    assert transition_aware is True


def test_tracker_relative_position_loss_ignores_common_anchor_translation():
    diffusion = _make_loss_test_diffusion()
    target_tracker_pos_ref = torch.zeros(1, TRACKER_COUNT, 3)
    target_tracker_pos_ref[:] = torch.tensor([10.0, 0.0, -4.0])
    target_sensor_valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=target_tracker_pos_ref,
        target_sensor_valid=target_sensor_valid,
    )

    losses = diffusion.realtime_pose_loss.compute(
        pred_xstart,
        x_start,
        model_kwargs,
        timesteps=torch.zeros(1, dtype=torch.long),
    )

    assert torch.allclose(losses["tracker_relative_pos_loss"], torch.zeros(1), atol=1e-7)


def test_nohip_endpoint_losses_are_only_active_without_hip():
    diffusion = _make_loss_test_diffusion()
    target_tracker_pos_ref = torch.zeros(1, TRACKER_COUNT, 3)
    target_sensor_valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=target_tracker_pos_ref,
        target_sensor_valid=target_sensor_valid,
    )
    model_kwargs["y"]["target_root_yaw"][:] = 0.4
    model_kwargs["y"]["target_root_pos_world"][:] = torch.tensor([0.5, 0.0, -0.25])
    model_kwargs["y"]["target_joints_world"][:, 0, 1] = 0.8

    hip_valid_losses = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)
    assert torch.allclose(hip_valid_losses["nohip_yaw_loss"], torch.zeros(1))
    assert torch.allclose(hip_valid_losses["nohip_root_xz_loss"], torch.zeros(1))
    assert torch.allclose(hip_valid_losses["nohip_height_loss"], torch.zeros(1))

    model_kwargs["y"]["target_sensor_valid"][:, HIP_TRACKER_INDEX] = False
    no_hip_losses = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)
    assert torch.all(no_hip_losses["nohip_yaw_loss"] > 0)
    assert torch.all(no_hip_losses["nohip_root_xz_loss"] > 0)
    assert torch.all(no_hip_losses["nohip_height_loss"] > 0)


def test_hip_valid_single_foot_dropout_strengthens_only_missing_side():
    diffusion = _make_loss_test_diffusion()
    valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=torch.zeros(1, TRACKER_COUNT, 3),
        target_sensor_valid=valid,
    )
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    left_hip_start = schema.body_pose_slice().start + 6
    pred_xstart[:, left_hip_start : left_hip_start + 6, REALTIME_POSE_TARGET_START] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    full_six = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)

    model_kwargs["y"]["target_sensor_valid"][:, LEFT_FOOT_TRACKER_INDEX] = False
    model_kwargs["y"]["target_stationary_prob_5"][:, 1] = 1.0
    single_foot = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)

    assert single_foot["left_missing_fraction"].item() == 1.0
    assert single_foot["right_missing_fraction"].item() == 0.0
    assert single_foot["local_rotation_loss"].item() > full_six["local_rotation_loss"].item()
    assert single_foot["contact_active_foot_count"].item() == 1.0


def test_temporal_losses_only_use_predicted_previous_state_on_rollout_step():
    diffusion = _make_loss_test_diffusion()
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=torch.zeros(1, TRACKER_COUNT, 3),
        target_sensor_valid=torch.ones(1, TRACKER_COUNT, dtype=torch.bool),
    )
    model_kwargs["y"]["pred_prev_joints_world"][:] = 0.1

    teacher_forced = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)
    assert torch.allclose(teacher_forced["joint_velocity_loss"], torch.zeros(1))

    model_kwargs["y"]["previous_state_is_predicted"][:] = True
    rollout = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)
    # 当前关节为 0、pred previous 为 0.1m，60 FPS 下速度误差为 6m/s。
    assert rollout["joint_velocity_loss"].item() == pytest.approx(5.95, abs=2e-6)


def test_contact_velocity_loss_uses_real_meters_per_second():
    diffusion = _make_loss_test_diffusion()
    target_sensor_valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    target_sensor_valid[:, HIP_TRACKER_INDEX] = False
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=torch.zeros(1, TRACKER_COUNT, 3),
        target_sensor_valid=target_sensor_valid,
    )
    for foot_index in (10, 11):
        model_kwargs["y"]["pred_prev_joints_world"][:, foot_index, 0] = 0.1
        model_kwargs["y"]["pred_prev_joints_world"][:, foot_index, 2] = 0.1
    model_kwargs["y"]["target_stationary_prob_5"][:, 1:3] = 1.0
    model_kwargs["y"]["previous_state_is_predicted"][:] = True

    losses = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)

    expected = 6.0 - 0.5 * diffusion.realtime_pose_loss.config.contact_velocity_huber_beta
    assert losses["contact_velocity_loss"].item() == pytest.approx(expected, abs=1e-6)


def test_airborne_stationary_feet_do_not_activate_contact_losses():
    diffusion = _make_loss_test_diffusion()
    target_sensor_valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    target_sensor_valid[:, HIP_TRACKER_INDEX] = False
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=torch.zeros(1, TRACKER_COUNT, 3),
        target_sensor_valid=target_sensor_valid,
    )
    model_kwargs["y"]["target_joints_world"][:, [10, 11], 1] = 0.2
    model_kwargs["y"]["target_stationary_prob_5"][:, 1:3] = 1.0
    model_kwargs["y"]["previous_state_is_predicted"][:] = True
    losses = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)

    assert torch.allclose(losses["contact_active_foot_count"], torch.zeros(1))
    assert torch.allclose(losses["contact_velocity_loss"], torch.zeros(1))
    assert torch.allclose(losses["contact_height_loss"], torch.zeros(1))


def test_nonstationary_grounded_feet_do_not_activate_contact_losses():
    diffusion = _make_loss_test_diffusion()
    target_sensor_valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    target_sensor_valid[:, HIP_TRACKER_INDEX] = False
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=torch.zeros(1, TRACKER_COUNT, 3),
        target_sensor_valid=target_sensor_valid,
    )
    model_kwargs["y"]["target_stationary_prob_5"][:, 1:3] = 0.69
    model_kwargs["y"]["previous_state_is_predicted"][:] = True

    losses = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)

    assert torch.allclose(losses["contact_active_foot_count"], torch.zeros(1))
    assert torch.allclose(losses["contact_velocity_loss"], torch.zeros(1))
    assert torch.allclose(losses["contact_height_loss"], torch.zeros(1))


def test_weighted_foot_loss_normalizes_left_and_right_independently():
    loss = torch.tensor([[1.0, 10.0], [3.0, 20.0]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

    normalized = _normalize_weighted_feet(loss, weight)

    # 分母只数 active foot；右脚 confidence=2，因此不会被 soft weight 分母抵消。
    assert normalized.mean().item() == pytest.approx(20.5)


def test_stationary_soft_target_only_uses_runtime_margin_in_auxiliary_losses():
    diffusion = _make_loss_test_diffusion()
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=torch.zeros(1, TRACKER_COUNT, 3),
        target_sensor_valid=torch.ones(1, TRACKER_COUNT, dtype=torch.bool),
    )
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    pred_xstart[:, schema.stationary_prob_slice(), REALTIME_POSE_TARGET_START] = torch.tensor(
        [-0.5, 1.5, 0.0, 0.0, 0.0]
    )

    losses = diffusion.realtime_pose_loss.compute(pred_xstart, x_start, model_kwargs)

    assert losses["stationary_margin_loss"].item() > 0.0
    assert "stationary_regression_loss" not in losses
    assert "stationary_range_loss" not in losses


def test_stationary_regression_is_folded_into_simple_mse_channel_weight():
    diffusion = _make_loss_test_diffusion()
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=torch.zeros(1, TRACKER_COUNT, 3),
        target_sensor_valid=torch.ones(1, TRACKER_COUNT, dtype=torch.bool),
    )
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    pred_xstart[:, schema.stationary_prob_slice(), REALTIME_POSE_TARGET_START] = 0.1
    mask = torch.zeros_like(x_start, dtype=torch.bool)
    mask[:, schema.target_slice(), REALTIME_POSE_TARGET_START] = True
    model_kwargs["y"]["mask"] = mask
    model_kwargs["y"]["inpainted_motion"] = x_start
    model_kwargs["inpaint_cond"] = mask

    class FixedPrediction(torch.nn.Module):
        def forward(self, x, timesteps, **kwargs):
            return pred_xstart

    losses = diffusion.training_losses(
        FixedPrediction(),
        x_start,
        torch.zeros(1, dtype=torch.long),
        model_kwargs=model_kwargs,
        noise=torch.zeros_like(x_start),
        snr_gamma=0.0,
    )

    stationary_mse = torch.tensor([0.01])
    old_regression_weight = 0.020235997785184236
    base_mse = stationary_mse * (5.0 / 154.0)
    expected_upweight = stationary_mse * old_regression_weight
    assert torch.allclose(losses["simple_stationary_loss"], stationary_mse)
    assert torch.allclose(losses["simple_stationary_upweight"], expected_upweight)
    assert torch.allclose(losses["simple_loss"], base_mse + expected_upweight)
    assert losses["simple_stationary_channel_weight"].item() == pytest.approx(1.6232687317836745)
    assert "stationary_regression_loss" not in losses
    assert "stationary_range_loss" not in losses


def test_masked_l2_feature_weight_uses_weighted_denominator():
    diffusion = _make_loss_test_diffusion()
    values = torch.ones(1, 2, 1)
    target = torch.zeros_like(values)
    mask = torch.ones_like(values, dtype=torch.bool)
    feature_weight = torch.tensor([[[1.0], [3.0]]])

    loss = diffusion.masked_l2(values, target, mask, feature_w=feature_weight)

    assert loss.item() == pytest.approx(1.0)


def test_no_hip_sample_normalization_is_not_diluted_by_full_six_samples():
    normalized = _normalize_masked_samples(
        torch.tensor([2.0, 100.0]),
        torch.tensor([True, False]),
    )

    assert normalized.tolist() == [4.0, 0.0]
    assert normalized.mean().item() == pytest.approx(2.0)


def test_tracker_relative_loss_uses_resolved_root_y0_pelvis_height_offset():
    diffusion = _make_loss_test_diffusion()
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    target_tracker_pos_ref = torch.zeros(1, TRACKER_COUNT, 3)
    target_tracker_pos_ref[:, 0] = torch.tensor([0.0, 0.9, 0.0])
    target_tracker_pos_ref[:, HIP_TRACKER_INDEX] = torch.tensor([0.0, 0.9, 0.0])
    target_sensor_valid = torch.zeros(1, TRACKER_COUNT, dtype=torch.bool)
    target_sensor_valid[:, [HIP_TRACKER_INDEX, 0]] = True
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=target_tracker_pos_ref,
        target_sensor_valid=target_sensor_valid,
    )
    pred_xstart[:, schema.root_height_slice(), REALTIME_POSE_TARGET_START] = 0.9
    x_start[:, schema.root_height_slice(), REALTIME_POSE_TARGET_START] = 0.9
    model_kwargs["y"]["joint_offsets_parent"][:, 0, 1] = 0.2

    losses = diffusion.realtime_pose_loss.compute(
        pred_xstart,
        x_start,
        model_kwargs,
        timesteps=torch.zeros(1, dtype=torch.long),
    )

    assert torch.allclose(losses["tracker_relative_pos_loss"], torch.zeros(1), atol=1e-6)


def test_all_auxiliary_losses_share_timestep_attenuation():
    diffusion = _make_loss_test_diffusion()
    target_tracker_pos_ref = torch.zeros(1, TRACKER_COUNT, 3)
    target_tracker_pos_ref[:, 0] = torch.tensor([0.1, 0.0, 0.0])
    target_sensor_valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    pred_xstart, x_start, model_kwargs = _make_sensor_reprojection_aux_inputs(
        target_tracker_pos_ref=target_tracker_pos_ref,
        target_sensor_valid=target_sensor_valid,
    )

    low_noise_losses = diffusion.realtime_pose_loss.compute(
        pred_xstart,
        x_start,
        model_kwargs,
        timesteps=torch.zeros(1, dtype=torch.long),
    )
    high_noise_losses = diffusion.realtime_pose_loss.compute(
        pred_xstart,
        x_start,
        model_kwargs,
        timesteps=torch.full((1,), diffusion.num_timesteps - 1, dtype=torch.long),
    )

    assert torch.allclose(
        low_noise_losses["tracker_relative_pos_loss"],
        high_noise_losses["tracker_relative_pos_loss"],
    )
    low_noise_weight = low_noise_losses["aux_timestep_weight"]
    high_noise_weight = high_noise_losses["aux_timestep_weight"]
    assert torch.all(low_noise_weight > high_noise_weight)
    assert torch.allclose(high_noise_weight, low_noise_weight * diffusion.realtime_pose_loss.config.aux_timestep_min_weight)


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
