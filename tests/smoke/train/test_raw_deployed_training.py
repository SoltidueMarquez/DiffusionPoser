from __future__ import annotations

import numpy as np
import pytest
import torch

from data_loaders.generate_realtime_pose_tasks import (
    build_task_bundle_row,
    compute_source_joint_rotations_world,
)
from data_loaders.realtime_pose_geometry import assemble_tracker_features_np, extract_forward_yaw_np
from data_loaders.realtime_pose_kinematics import (
    rotation_6d_to_matrix_np,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import (
    REALTIME_POSE_FRAME_OFFSETS,
    REALTIME_POSE_HISTORY_ANCHOR_INDICES,
    TRACKER_TO_JOINT,
)
from data_loaders.tracker_reliability import (
    compute_hard_rotation_state_np,
    compute_region_coverage_np,
    compute_tracker_reliability_np,
)
from data_loaders.tracker_timeline import build_task_config_plan, compute_tracker_durations
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from diffusion.realtime_pose_losses import (
    _adjacent_contact_weight,
    _contact_slide_loss,
    _radial_huber_loss,
    _rotation_angle,
    compute_raw_deployed_losses,
)
from model.realtime_pose_spatiotemporal_dit import RealtimePoseSpatioTemporalDiT
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_training_rejects_removed_legacy_inpainting_contract():
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    with pytest.raises(ValueError):
        diffusion.training_losses(
            model=None,
            x_start=torch.zeros(1, 11, 144),
            t=torch.ones(1, dtype=torch.long),
            model_kwargs={"inpaint_cond": torch.ones(1, 11, 144)},
        )


def test_exact_rotation_match_has_zero_finite_rotation6d_gradient():
    prediction_6d = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]], requires_grad=True
    )
    prediction = rotation_6d_to_matrix_torch(prediction_6d)
    angle = _rotation_angle(prediction, prediction.detach())
    angle.square().sum().backward()
    torch.testing.assert_close(angle, torch.zeros_like(angle), atol=1e-7, rtol=0.0)
    assert torch.isfinite(prediction_6d.grad).all()


def test_rotation_angle_gradient_is_finite_near_zero_and_pi():
    theta = torch.tensor([1e-7, torch.pi - 1e-4], requires_grad=True)
    cosine, sine = torch.cos(theta), torch.sin(theta)
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    rotation = torch.stack(
        (cosine, -sine, zero, sine, cosine, zero, zero, zero, one), dim=-1
    ).reshape(-1, 3, 3)
    _rotation_angle(rotation, torch.eye(3).expand_as(rotation)).square().sum().backward()
    assert torch.isfinite(theta.grad).all()


def test_tracker_position_radial_huber_values_and_gradients():
    distance = torch.tensor([0.0, 0.05, 0.50], requires_grad=True)
    loss = _radial_huber_loss(distance, beta=0.05)
    loss.sum().backward()
    torch.testing.assert_close(loss, torch.tensor([0.0, 0.025, 0.475]))
    torch.testing.assert_close(distance.grad, torch.tensor([0.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        _radial_huber_loss(torch.tensor([0.1]), beta=0.0)


def test_contact_slide_loss_uses_soft_contact_and_previous_valid_mask():
    previous = torch.zeros(1, 2, 3)
    predicted = previous.clone()
    predicted[0, 0, 0] = 0.01
    predicted.requires_grad_()
    contact_weight = torch.tensor([[1.0, 0.0]])

    loss = _contact_slide_loss(
        predicted_feet=predicted,
        previous_target_feet=previous,
        contact_weight=contact_weight,
        previous_frame_valid=torch.tensor([True]),
        fps=60.0,
        huber_beta_mps=0.1,
    )
    # 1 cm/frame 在 60 Hz 下是 0.6 m/s，Huber(beta=0.1) 得到 0.55。
    torch.testing.assert_close(loss, torch.tensor([0.55]))
    loss.sum().backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, 0, 0] > 0.0
    torch.testing.assert_close(predicted.grad[0, 1], torch.zeros(3))

    invalid_loss = _contact_slide_loss(
        predicted_feet=predicted.detach(),
        previous_target_feet=previous,
        contact_weight=torch.ones(1, 2),
        previous_frame_valid=torch.tensor([False]),
        fps=60.0,
        huber_beta_mps=0.1,
    )
    torch.testing.assert_close(invalid_loss, torch.zeros_like(invalid_loss))


def test_adjacent_contact_weight_requires_contact_on_both_frames():
    current = torch.tensor([[1.0, 0.4]])
    previous = torch.tensor([[0.2, 0.8]])
    torch.testing.assert_close(
        _adjacent_contact_weight(current, previous),
        torch.tensor([[0.2, 0.4]]),
    )
    torch.testing.assert_close(
        _adjacent_contact_weight(current, torch.zeros_like(previous)),
        torch.zeros_like(current),
    )
    with pytest.raises(ValueError, match=r"\[B,2\]"):
        _adjacent_contact_weight(current, torch.ones(1, 3))


def _training_batch() -> dict[str, torch.Tensor]:
    source = build_toy_realtime_source(frame_count=71)
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
        build_task_config_plan("raw-deployed", 10, 1),
    )
    configured_dense = row["configured"][0].astype(bool)
    measured_dense = row["measured_valid"][0].astype(bool)
    d_off_dense, d_on_dense = compute_tracker_durations(configured_dense, measured_dense)
    indices = np.asarray((*REALTIME_POSE_HISTORY_ANCHOR_INDICES, 60), dtype=np.int64)
    configured = configured_dense[indices]
    measured = measured_dense[indices]
    d_off = d_off_dense[indices]
    d_on = d_on_dense[indices]
    tracker = assemble_tracker_features_np(
        row["tracker_window_continuous"], configured, measured, d_off, d_on
    )
    kappa_pos, kappa_rot = compute_tracker_reliability_np(
        configured[:-1], measured[:-1], d_on[:-1]
    )
    rho_pos, rho_rot = compute_region_coverage_np(kappa_pos, kappa_rot)
    confidence = 0.5 * (rho_pos + rho_rot)
    values = {
        "x": row["pose_target_horizon_clean"],
        "history_pose_observation": row["history_pose_clean"],
        "tracker_window": tracker,
        "head_path_window": row["head_path_window"],
        "history_region_confidence": confidence,
        "window_valid_mask": np.ones(11, dtype=bool),
        "frame_offsets": np.asarray(REALTIME_POSE_FRAME_OFFSETS),
        "tracker_window_raw": tracker,
        "hard_rotation_state_window": compute_hard_rotation_state_np(
            configured, measured, d_on
        ),
        "target_joints_head_ref": row["target_joints_head_ref"],
        "joint_offsets_parent": source["joint_offsets_parent"],
        "joint_rest_local_rotations_6d": source["joint_rest_local_rotations_6d"],
        "target_root_position_head_ref": row["target_root_position_head_ref"],
        "target_root_yaw_world": row["target_root_yaw_world"],
        "current_head_yaw_world": row["current_head_yaw_world"],
        "previous_contact_target": row["previous_contact_target"],
        "contact_target": row["contact_target"],
    }
    return {name: torch.as_tensor(value).unsqueeze(0) for name, value in values.items()}


def _model_and_kwargs(batch):
    model = RealtimePoseSpatioTemporalDiT(
        latent_dim=32, num_layers=1, num_heads=4, dropout=0.0, max_seq_len=21
    )
    condition_names = (
        "history_pose_observation",
        "tracker_window",
        "head_path_window",
        "history_region_confidence",
        "window_valid_mask",
        "frame_offsets",
    )
    kwargs = {name: batch[name] for name in condition_names}
    kwargs["y"] = {
        **batch,
        "previous_pose_target": batch["history_pose_observation"][:, -1],
    }
    return model, kwargs


def test_current_training_keeps_raw_hard_joint_gradient_and_current_constraint():
    torch.manual_seed(7)
    batch = _training_batch()
    model, model_kwargs = _model_and_kwargs(batch)
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        tracker_pos_loss_weight=10.0,
    )
    terms = diffusion.training_losses(
        model,
        batch["x"].float(),
        torch.ones(1, dtype=torch.long),
        model_kwargs=model_kwargs,
        noise=torch.zeros_like(batch["x"], dtype=torch.float32),
        return_pred_xstart=True,
    )
    assert all(torch.isfinite(value).all() for value in terms.values())
    assert "head_ref_joint_distance_loss" in terms
    assert "contact_slide_loss" in terms
    assert "world_joint_loss" not in terms
    raw = terms["raw_pred_xstart"]
    deployed = terms["deployed_pred_xstart"]
    raw.retain_grad()
    terms["loss"].mean().backward()
    hard_channels = []
    for tracker_index in torch.nonzero(
        batch["hard_rotation_state_window"][0, -1]
    ).flatten().tolist():
        joint_index = TRACKER_TO_JOINT[tracker_index]
        start = joint_index * 6
        torch.testing.assert_close(
            deployed[0, 0, start : start + 6],
            batch["tracker_window_raw"][0, -1, tracker_index, 3:9],
            atol=1e-5,
            rtol=1e-5,
        )
        hard_channels.extend(range(start, start + 6))
    assert torch.linalg.norm(raw.grad[0, 0, hard_channels]) > 0.0


def test_current_training_uses_tracker_position_huber_beta():
    torch.manual_seed(9)
    batch = _training_batch()
    model, model_kwargs = _model_and_kwargs(batch)
    model.eval()
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
            losses.append(
                diffusion.training_losses(
                    model,
                    batch["x"].float(),
                    torch.ones(1, dtype=torch.long),
                    model_kwargs=model_kwargs,
                    noise=torch.zeros_like(batch["x"]),
                )["tracker_position_loss"]
            )
    assert not torch.allclose(losses[0], losses[1])


def test_current_training_respects_l1_and_feature_weights():
    torch.manual_seed(11)
    batch = _training_batch()
    model, model_kwargs = _model_and_kwargs(batch)
    model.eval()
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    common = dict(
        model=model,
        x_start=batch["x"].float(),
        t=torch.ones(1, dtype=torch.long),
        model_kwargs=model_kwargs,
        noise=torch.zeros_like(batch["x"]),
    )
    with torch.no_grad():
        mse = diffusion.training_losses(**common, feature_w=torch.ones(1, 144))
        l1 = diffusion.training_losses(
            **common, feature_w=torch.ones(1, 144), use_l1=True
        )
        zero = diffusion.training_losses(**common, feature_w=torch.zeros(1, 144))
    assert not torch.allclose(mse["simple_loss"], l1["simple_loss"])
    torch.testing.assert_close(zero["simple_loss"], torch.zeros_like(zero["simple_loss"]))


def test_future_only_rotation_changes_horizon_losses_but_not_current_losses():
    batch = _training_batch()
    batch["previous_pose_target"] = batch["history_pose_observation"][:, -1]
    target = batch["x"].float()
    auxiliary = {"contact_logits": torch.zeros(1, 2)}
    exact = compute_raw_deployed_losses(
        target,
        target,
        target,
        batch,
        auxiliary,
        tracker_pos_huber_beta=0.05,
    )
    changed = target.clone()
    changed[:, 1, :6] = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    future_error = compute_raw_deployed_losses(
        changed,
        changed,
        target,
        batch,
        auxiliary,
        tracker_pos_huber_beta=0.05,
    )
    torch.testing.assert_close(
        exact["rotation_velocity_loss"], torch.zeros_like(exact["rotation_velocity_loss"])
    )
    assert future_error["global_rotation_loss"].item() > 0.0
    assert future_error["local_rotation_loss"].item() > 0.0
    assert future_error["rotation_velocity_loss"].item() > 0.0
    for name in (
        "tracker_rotation_loss",
        "tracker_position_loss",
        "fk_loss",
        "head_ref_joint_distance_loss",
        "root_loss",
        "head_to_root_xz_loss",
        "contact_loss",
        "contact_slide_loss",
    ):
        torch.testing.assert_close(future_error[name], exact[name])
