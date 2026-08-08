from __future__ import annotations

import numpy as np
import pytest
import torch

from data_loaders.generate_realtime_pose_tasks import (
    build_task_bundle_row,
    compute_source_joint_rotations_world,
)
from data_loaders.realtime_pose_geometry import (
    assemble_tracker_features_np,
    decode_target_head_rotations_np,
    decode_target_head_rotations_torch,
    extract_forward_yaw_np,
    resolve_root_head_reference_np,
    resolve_root_head_reference_torch,
)
from data_loaders.realtime_pose_kinematics import (
    rotation_6d_to_matrix_np,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import TRACKER_TO_JOINT
from data_loaders.tracker_timeline import build_task_config_plan
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from diffusion.realtime_pose_losses import _radial_huber_loss, _rotation_angle
from model.realtime_pose_target_dit import RealtimePoseTargetDiT
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source
from train.training_loop import (
    TrainLoop,
    add_rollout_step_diagnostics,
    aggregate_rollout_frame_losses,
    build_rollout_frame_weights,
    zero_invalid_pose_history,
)


def test_training_rejects_removed_legacy_inpainting_contract() -> None:
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    x_start = torch.zeros(1, 144)

    with pytest.raises(ValueError, match="已不兼容旧 mask/inpainting"):
        diffusion.training_losses(
            model=None,
            x_start=x_start,
            t=torch.ones(1, dtype=torch.long),
            model_kwargs={
                "inpaint_cond": torch.ones_like(x_start, dtype=torch.bool),
                "y": {"mask": torch.ones_like(x_start)},
            },
        )


def test_exact_rotation_match_has_zero_finite_rotation6d_gradient() -> None:
    prediction_6d = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    prediction = rotation_6d_to_matrix_torch(prediction_6d)
    angle = _rotation_angle(prediction, prediction.detach())
    angle.square().sum().backward()

    torch.testing.assert_close(angle, torch.zeros_like(angle), atol=1e-7, rtol=0.0)
    assert prediction_6d.grad is not None
    assert torch.isfinite(prediction_6d.grad).all()
    torch.testing.assert_close(
        prediction_6d.grad,
        torch.zeros_like(prediction_6d.grad),
        atol=1e-7,
        rtol=0.0,
    )


def test_rotation_angle_gradient_is_finite_near_zero_and_pi() -> None:
    theta = torch.tensor([1e-7, torch.pi - 1e-4], requires_grad=True)
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    rotation = torch.stack(
        (
            cosine,
            -sine,
            zero,
            sine,
            cosine,
            zero,
            zero,
            zero,
            one,
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    target = torch.eye(3).expand_as(rotation)
    _rotation_angle(rotation, target).square().sum().backward()

    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()


def test_tracker_position_radial_huber_values_and_gradients() -> None:
    distance = torch.tensor([0.0, 0.05, 0.50], requires_grad=True)

    loss = _radial_huber_loss(distance, beta=0.05)
    loss.sum().backward()

    torch.testing.assert_close(loss, torch.tensor([0.0, 0.025, 0.475]))
    assert distance.grad is not None
    assert torch.isfinite(distance.grad).all()
    torch.testing.assert_close(distance.grad, torch.tensor([0.0, 1.0, 1.0]))


def test_tracker_position_huber_rejects_non_positive_beta() -> None:
    with pytest.raises(ValueError, match="tracker_pos_huber_beta"):
        _radial_huber_loss(torch.tensor([0.1]), beta=0.0)


def _training_batch() -> dict[str, torch.Tensor]:
    source = build_toy_realtime_source(frame_count=70)
    joint_rotations = compute_source_joint_rotations_world(source)
    head_yaws = extract_forward_yaw_np(
        rotation_6d_to_matrix_np(source["tracker_rot_world_6d"])[:, 0]
    )
    row = build_task_bundle_row(
        source,
        joint_rotations,
        head_yaws,
        0,
        0,
        build_task_config_plan("raw-deployed", 10, 4),
        4,
    )
    scenario = 0
    configured = row["configured"][scenario, :61].astype(bool)
    measured = row["measured_valid"][scenario, :61].astype(bool)
    d_off = row["d_off"][scenario, :61]
    d_on = row["d_on"][scenario, :61]
    tracker_history = assemble_tracker_features_np(
        row["tracker_history_continuous"][0],
        configured[:-1],
        measured[:-1],
        d_off[:-1],
        d_on[:-1],
    )
    current_tracker = assemble_tracker_features_np(
        row["current_tracker_continuous"][0][None],
        configured[-1:],
        measured[-1:],
        d_off[-1:],
        d_on[-1:],
    )[0]
    values = {
        "x": row["current_target"][0],
        "pose_history": row["pose_history"],
        "tracker_history": tracker_history,
        "current_tracker": current_tracker,
        "current_tracker_raw": current_tracker,
        "trajectory_history": row["trajectory_history"][0],
        "current_trajectory": row["current_trajectory"][0],
        "valid_frame_mask": np.ones(60, dtype=bool),
        "hard_rotation_state": row["hard_rotation_state"][scenario, 60].astype(bool),
        "target_joints_head_ref": row["target_joints_head_ref"][0],
        "current_tracker_pos_head_ref": current_tracker[:, :3],
        "joint_offsets_parent": source["joint_offsets_parent"],
        "joint_rest_local_rotations_6d": source["joint_rest_local_rotations_6d"],
        "target_root_position_head_ref": row["target_root_position_head_ref"][0],
        "target_root_yaw_world": row["target_root_yaw_world"][0],
        "current_head_yaw_world": row["current_head_yaw_world"][0],
        "future_leg_target": row["future_leg_target"][0],
        "contact_target": row["contact_target"][0],
    }
    return {
        name: torch.as_tensor(value).unsqueeze(0)
        for name, value in values.items()
    }


def test_numpy_runtime_and_torch_training_resolvers_match() -> None:
    batch = _training_batch()
    target = batch["x"][0].numpy()
    offsets = batch["joint_offsets_parent"][0].numpy()
    observed_head_height = float(batch["current_tracker_raw"][0, 0, 1])
    rotations_np, yaw_np = decode_target_head_rotations_np(target)
    root_np, hip_np, joints_np = resolve_root_head_reference_np(
        rotations_np,
        yaw_np,
        offsets,
        observed_head_height=observed_head_height,
    )
    rotations_torch, yaw_torch = decode_target_head_rotations_torch(batch["x"].float())
    root_torch, hip_torch, joints_torch = resolve_root_head_reference_torch(
        rotations_torch,
        yaw_torch,
        batch["joint_offsets_parent"].float(),
        observed_head_height=batch["current_tracker_raw"][:, 0, 1].float(),
    )

    np.testing.assert_allclose(root_torch[0].numpy(), root_np, atol=1e-5, rtol=0.0)
    np.testing.assert_allclose(joints_torch[0].numpy(), joints_np, atol=1e-5, rtol=0.0)
    assert abs(float(hip_torch[0]) - float(hip_np)) <= 1e-5


def test_full_144d_training_keeps_raw_hard_joint_gradient_and_deployed_constraint() -> None:
    torch.manual_seed(7)
    batch = _training_batch()
    model = RealtimePoseTargetDiT(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        motion_layers=1,
        dropout=0.0,
    )
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        tracker_pos_loss_weight=10.0,
    )
    model_kwargs = {
        name: batch[name]
        for name in (
            "pose_history",
            "tracker_history",
            "current_tracker",
            "trajectory_history",
            "current_trajectory",
            "valid_frame_mask",
        )
    }
    model_kwargs["y"] = batch
    terms = diffusion.training_losses(
        model,
        batch["x"].float(),
        torch.ones(1, dtype=torch.long),
        model_kwargs=model_kwargs,
        noise=torch.zeros_like(batch["x"], dtype=torch.float32),
        snr_gamma=0.0,
        return_pred_xstart=True,
    )
    assert all(torch.isfinite(value).all() for value in terms.values() if torch.is_tensor(value))
    assert "head_to_root_xz_loss" in terms
    assert "root_trajectory_loss" not in terms
    raw = terms["raw_pred_xstart"]
    deployed = terms["deployed_pred_xstart"]
    raw.retain_grad()
    terms["loss"].mean().backward()

    hard = batch["hard_rotation_state"][0].bool()
    current_tracker = batch["current_tracker_raw"][0]
    hard_channels: list[int] = []
    for tracker_index in torch.nonzero(hard, as_tuple=False).flatten().tolist():
        joint_index = TRACKER_TO_JOINT[tracker_index]
        start = joint_index * 6
        torch.testing.assert_close(
            deployed[0, start : start + 6],
            current_tracker[tracker_index, 3:9],
            atol=1e-5,
            rtol=1e-5,
        )
        hard_channels.extend(range(start, start + 6))
    assert raw.grad is not None
    assert torch.linalg.norm(raw.grad[0, hard_channels]) > 0.0


def test_dynamic_training_uses_tracker_position_huber_beta() -> None:
    torch.manual_seed(9)
    batch = _training_batch()
    model = RealtimePoseTargetDiT(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        motion_layers=1,
        dropout=0.0,
    )
    model.eval()
    model_kwargs = {
        name: batch[name]
        for name in (
            "pose_history",
            "tracker_history",
            "current_tracker",
            "trajectory_history",
            "current_trajectory",
            "valid_frame_mask",
        )
    }
    model_kwargs["y"] = batch

    losses = []
    for beta in (0.01, 0.20):
        diffusion = GaussianDiffusion(
            betas=np.asarray([0.01, 0.02], dtype=np.float64),
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            tracker_pos_huber_beta=beta,
        )
        with torch.no_grad():
            terms = diffusion.training_losses(
                model,
                batch["x"].float(),
                torch.ones(1, dtype=torch.long),
                model_kwargs=model_kwargs,
                noise=torch.zeros_like(batch["x"], dtype=torch.float32),
            )
        losses.append(terms["tracker_position_loss"])

    assert not torch.allclose(losses[0], losses[1])


def test_dynamic_training_respects_l1_and_feature_weights() -> None:
    torch.manual_seed(11)
    batch = _training_batch()
    model = RealtimePoseTargetDiT(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        motion_layers=1,
        dropout=0.0,
    )
    model.eval()
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    model_kwargs = {
        name: batch[name]
        for name in (
            "pose_history",
            "tracker_history",
            "current_tracker",
            "trajectory_history",
            "current_trajectory",
            "valid_frame_mask",
        )
    }
    model_kwargs["y"] = batch
    common = {
        "model": model,
        "x_start": batch["x"].float(),
        "t": torch.ones(1, dtype=torch.long),
        "model_kwargs": model_kwargs,
        "noise": torch.zeros_like(batch["x"], dtype=torch.float32),
        "snr_gamma": 0.0,
    }
    feature_ones = torch.ones_like(batch["x"], dtype=torch.float32)
    with torch.no_grad():
        mse_terms = diffusion.training_losses(
            **common,
            feature_w=feature_ones,
            use_l1=False,
        )
        l1_terms = diffusion.training_losses(
            **common,
            feature_w=feature_ones,
            use_l1=True,
        )
        zero_weight_terms = diffusion.training_losses(
            **common,
            feature_w=torch.zeros_like(feature_ones),
            use_l1=False,
        )
    assert not torch.allclose(mse_terms["simple_loss"], l1_terms["simple_loss"])
    assert not torch.allclose(mse_terms["loss"], zero_weight_terms["loss"])
    torch.testing.assert_close(
        zero_weight_terms["simple_loss"],
        torch.zeros_like(zero_weight_terms["simple_loss"]),
    )


def test_rollout_history_appends_deployed_prediction_without_gt_leakage() -> None:
    from data_loaders.realtime_pose_geometry import advance_rollout_pose_history_torch

    history = torch.zeros(1, 60, 144)
    deployed = torch.arange(144, dtype=torch.float32).reshape(1, 144)
    next_history = advance_rollout_pose_history_torch(
        history,
        deployed,
        torch.tensor([0.0]),
        torch.tensor([0.25]),
        detach_prediction=True,
    )
    torch.testing.assert_close(next_history[:, -1], deployed)
    assert next_history[:, -1].grad_fn is None


def test_rollout_history_restores_invalid_padding_to_literal_zero() -> None:
    history = torch.randn(2, 60, 144)
    valid = torch.zeros(2, 60, dtype=torch.bool)
    valid[0, -1] = True
    valid[1, -3:] = True

    masked = zero_invalid_pose_history(history, valid)

    assert torch.count_nonzero(masked[~valid]) == 0
    torch.testing.assert_close(masked[valid], history[valid])


def test_fifteen_step_rollout_temporal_losses_are_finite() -> None:
    from diffusion.realtime_pose_temporal_losses import compute_rollout_temporal_losses

    source = build_toy_realtime_source(frame_count=15)
    identity_pose = torch.as_tensor(np.tile([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], 24)).float()[None]
    predictions = [identity_pose + 0.001 * step for step in range(15)]
    batches = []
    for step in range(15):
        batches.append(
            {
                "x": identity_pose.clone(),
                "joint_offsets_parent": torch.from_numpy(source["joint_offsets_parent"])[None].float(),
                "current_tracker_pos_head_ref": torch.from_numpy(
                    source["tracker_pos_world"][step]
                )[None].float(),
                "target_joints_head_ref": torch.from_numpy(source["joints_world"][step])[None].float(),
                "current_head_yaw_world": torch.tensor([0.0]),
                "current_head_position_world": torch.from_numpy(
                    source["tracker_pos_world"][step, 0]
                )[None].float(),
                "floor_y": torch.tensor([0.0]),
                "contact_target": torch.ones(1, 2),
            }
        )
    losses = compute_rollout_temporal_losses(predictions, batches, None, None)
    assert set(losses) == {"joint_vel_loss", "rotation_vel_loss", "foot_slide_loss"}
    assert all(torch.isfinite(value).all() for value in losses.values())


def test_fifteen_step_rollout_exports_each_prior_fk_diagnostic_step() -> None:
    result: dict[str, torch.Tensor] = {}
    for step_index in range(1, 15):
        value = torch.tensor([float(step_index)])
        add_rollout_step_diagnostics(
            result,
            {
                "prior_fk_loss": value,
                "prior_internal_fk_loss": value + 1.0,
                "prior_root_pose_gap_m": value + 2.0,
                "prior_root_pose_gap_xz_m": value + 3.0,
                "prior_joint_resolver_gap_m": value + 4.0,
            },
            step_index,
        )
    assert len(result) == 70
    for step_index in range(1, 15):
        assert f"step_{step_index}_prior_fk_loss" in result
        assert f"step_{step_index}_prior_internal_fk_loss" in result
        assert f"step_{step_index}_prior_root_pose_gap_m" in result
        assert f"step_{step_index}_prior_root_pose_gap_xz_m" in result
        assert f"step_{step_index}_prior_joint_resolver_gap_m" in result


@pytest.mark.parametrize("rollout_steps", [4, 15])
def test_rollout_frame_weights_are_normalized_and_deterministic(rollout_steps: int) -> None:
    uniform = build_rollout_frame_weights(rollout_steps, "uniform", dtype=torch.float64)
    linear_late = build_rollout_frame_weights(
        rollout_steps,
        "linear_late",
        dtype=torch.float64,
    )
    future_steps = rollout_steps - 1
    expected_linear = torch.arange(1, future_steps + 1, dtype=torch.float64)
    expected_linear /= expected_linear.sum()

    torch.testing.assert_close(
        uniform,
        torch.full((future_steps,), 1.0 / future_steps, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(linear_late, expected_linear, rtol=0.0, atol=0.0)
    assert float(uniform.sum()) == 1.0
    assert float(linear_late.sum()) == 1.0
    assert linear_late[-1] > linear_late[0]


def test_rollout_frame_weight_validation_rejects_k1_and_unknown_policy() -> None:
    with pytest.raises(ValueError, match="至少需要两个预测帧"):
        build_rollout_frame_weights(1, "uniform")
    with pytest.raises(ValueError, match="rollout_frame_weighting"):
        build_rollout_frame_weights(15, "late_squared")


def test_k1_and_batch_without_rollout_still_skip_rollout_loss() -> None:
    loop = TrainLoop.__new__(TrainLoop)
    loop.rollout_steps = 1
    loop.rollout_loss_weight = 1.0
    loop.rollout_joint_vel_loss_weight = 0.05
    loop.rollout_rot_vel_loss_weight = 0.02
    loop.diffusion = type("DiffusionStub", (), {"foot_slide_loss_weight": 0.05})()
    loop.model = torch.nn.Linear(1, 1).train()

    assert loop.should_compute_rollout_loss({"rollout": []}) is False
    loop.rollout_steps = 15
    assert loop.should_compute_rollout_loss({}) is False


def test_uniform_rollout_aggregation_matches_old_mean_and_gradient_exactly() -> None:
    losses = [torch.tensor([1.0, 2.0], requires_grad=True) for _ in range(14)]
    terms = [{"loss": loss} for loss in losses]
    result = aggregate_rollout_frame_losses(terms, "uniform")
    old_mean = torch.stack(losses, dim=0).mean(dim=0)

    torch.testing.assert_close(result["loss"], old_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result["uniform_frame_loss"], old_mean, rtol=0.0, atol=0.0)
    old_grad = torch.autograd.grad(old_mean.sum(), losses, retain_graph=True)
    new_grad = torch.autograd.grad(result["loss"].sum(), losses)
    for expected, actual in zip(old_grad, new_grad):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_linear_late_rollout_aggregation_matches_hand_calculation() -> None:
    losses = [torch.tensor([float(step)]) for step in range(1, 15)]
    result = aggregate_rollout_frame_losses(
        [{"loss": loss} for loss in losses],
        "linear_late",
    )
    weights = torch.arange(1, 15, dtype=torch.float32) / 105.0
    expected = (torch.stack(losses, dim=0).squeeze(-1) * weights).sum()

    torch.testing.assert_close(result["loss"].squeeze(), expected)
    torch.testing.assert_close(result["uniform_frame_loss"], torch.tensor([7.5]))
    torch.testing.assert_close(result["step_1_weight"], torch.tensor([1.0 / 105.0]))
    torch.testing.assert_close(result["step_14_weight"], torch.tensor([14.0 / 105.0]))
    torch.testing.assert_close(
        result["step_14_weighted_loss"],
        losses[-1] * (14.0 / 105.0),
    )
