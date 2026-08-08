from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from data_loaders.realtime_pose_config import TaIDConfig
from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_torch,
    reexpress_previous_position_residual_torch,
    so3_log_map_torch,
)
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_torch
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    HIP_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    RIGHT_FOOT_TRACKER_INDEX,
    RIGHT_HAND_TRACKER_INDEX,
    TRACKER_COUNT,
    TRACKER_DURATION_CAP,
    TRACKER_NAMES,
)
from model.taid_conditioning import FixedSlotAnchorProjection
from model.realtime_pose_target_dit import RealtimePoseTargetDiT
from diffusion.realtime_pose_losses import wrapped_angle_difference
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from sample.reconstruct_stream import build_sampling_model_kwargs
from train.training_loop import build_rollout_frame_weights, validate_taid_checkpoint_stage
from utils.model_util import create_model_and_diffusion
from utils.parser_util import build_train_arg_parser


def _identity_pose(batch_size: int, frames: int | None = None) -> torch.Tensor:
    identity = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0]).repeat(24)
    shape = (batch_size, 144) if frames is None else (batch_size, frames, 144)
    return identity.reshape((1,) * (len(shape) - 1) + (144,)).expand(shape).clone()


def _taid_inputs(batch_size: int = 2) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    pose_history = _identity_pose(batch_size, 60)
    tracker_history = torch.zeros(batch_size, 60, 6, 13)
    current_tracker = torch.zeros(batch_size, 6, 13)
    identity_rotation = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    tracker_history[..., 3:9] = identity_rotation
    current_tracker[..., 3:9] = identity_rotation
    tracker_history[..., 9:11] = 1.0
    current_tracker[..., 9:11] = 1.0
    tracker_history[..., 12] = 20.0 / TRACKER_DURATION_CAP
    current_tracker[..., 12] = 20.0 / TRACKER_DURATION_CAP
    trajectory_history = torch.zeros(batch_size, 60, 5)
    trajectory_history[..., 4] = 1.0
    current_trajectory = torch.zeros(batch_size, 1, 5)
    current_trajectory[..., 4] = 1.0
    valid_frame_mask = torch.ones(batch_size, 60, dtype=torch.bool)
    current_tracker_raw = current_tracker.clone()
    joint_offsets_parent = torch.zeros(batch_size, 24, 3)
    positional = (
        pose_history,
        tracker_history,
        current_tracker,
        trajectory_history,
        current_trajectory,
        valid_frame_mask,
    )
    keywords = {
        "current_tracker_raw": current_tracker_raw,
        "joint_offsets_parent": joint_offsets_parent,
    }
    return positional, keywords


def _model(ablation: str) -> RealtimePoseTargetDiT:
    return RealtimePoseTargetDiT(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        motion_layers=1,
        dropout=0.0,
        taid_config=TaIDConfig(ablation=ablation, innovation_dim=16),
    )


def _enable_prior_output_sensitivity(model: RealtimePoseTargetDiT) -> None:
    """新 Prior 的末层默认零初始化；测试输入路径时给末层确定性非零权重。"""

    assert model.taid_conditioner is not None
    prior = model.taid_conditioner.prior
    with torch.no_grad():
        for head in (
            prior.pose_head,
            prior.root_head,
            prior.contact_head,
            prior.joint_velocity_head,
        ):
            head[-1].weight.fill_(1e-3)


def test_b0_keeps_legacy_state_dict_without_taid_parameters() -> None:
    model = _model("B0")
    assert model.taid_conditioner is None
    assert not any(name.startswith("taid_conditioner.") for name in model.state_dict())


def test_fixed_slot_projection_initially_matches_weighted_mean() -> None:
    torch.manual_seed(20)
    latent_dim = 8
    projection = FixedSlotAnchorProjection(latent_dim)
    tracker_tokens = torch.randn(3, TRACKER_COUNT, latent_dim)
    alpha = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 0.0, 1.0, 0.25, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    denominator = alpha.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized_slots = tracker_tokens * alpha[..., None] / denominator[..., None]
    expected = normalized_slots.sum(dim=1)
    torch.testing.assert_close(
        projection(normalized_slots),
        expected,
        rtol=0.0,
        atol=1e-6,
    )
    assert tuple(projection.weight.shape) == (latent_dim, TRACKER_COUNT * latent_dim)
    assert TRACKER_NAMES == (
        "head",
        "left_wrist",
        "right_wrist",
        "hip",
        "left_foot",
        "right_foot",
    )


def test_fixed_slot_projection_can_distinguish_left_and_right_hand() -> None:
    latent_dim = 4
    projection = FixedSlotAnchorProjection(latent_dim)
    with torch.no_grad():
        projection.weight.zero_()
        identity = torch.eye(latent_dim)
        left_start = LEFT_HAND_TRACKER_INDEX * latent_dim
        right_start = RIGHT_HAND_TRACKER_INDEX * latent_dim
        projection.weight[:, left_start : left_start + latent_dim] = identity
        projection.weight[:, right_start : right_start + latent_dim] = 2.0 * identity
    token = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    left_slots = torch.zeros(1, TRACKER_COUNT, latent_dim)
    right_slots = torch.zeros_like(left_slots)
    left_slots[:, LEFT_HAND_TRACKER_INDEX] = token
    right_slots[:, RIGHT_HAND_TRACKER_INDEX] = token
    torch.testing.assert_close(projection(left_slots), token)
    torch.testing.assert_close(projection(right_slots), 2.0 * token)


def test_b1_prior_consumes_per_tracker_history_summary() -> None:
    torch.manual_seed(21)
    model = _model("B1").eval()
    _enable_prior_output_sensitivity(model)
    assert model.taid_conditioner is not None
    assert model.taid_conditioner.prior.tracker_fusion[0].in_features == 4 * model.latent_dim
    values, kwargs = _taid_inputs(batch_size=1)
    first = model.prepare_conditioning(*values, **kwargs)
    assert first.observation.history_summary.shape == (1, 6, model.latent_dim)

    changed_values = list(values)
    changed_values[1] = values[1].clone()
    changed_values[1][:, :, LEFT_HAND_TRACKER_INDEX, :9] += 3.0
    second = model.prepare_conditioning(*changed_values, **kwargs)
    assert first.taid is not None and second.taid is not None
    assert not torch.allclose(
        first.observation.history_summary[:, LEFT_HAND_TRACKER_INDEX],
        second.observation.history_summary[:, LEFT_HAND_TRACKER_INDEX],
    )
    assert not torch.allclose(
        first.taid.prior.tracker_tokens[:, LEFT_HAND_TRACKER_INDEX],
        second.taid.prior.tracker_tokens[:, LEFT_HAND_TRACKER_INDEX],
    )
    assert not torch.allclose(first.taid.prior.pose_model, second.taid.prior.pose_model)


def test_b1_alpha_zero_gates_current_and_history_tracker_inputs() -> None:
    torch.manual_seed(22)
    model = _model("B1").eval()
    _enable_prior_output_sensitivity(model)
    values, kwargs = _taid_inputs(batch_size=1)
    values = list(values)
    values[2] = values[2].clone()
    kwargs = {name: tensor.clone() for name, tensor in kwargs.items()}
    values[2][:, LEFT_HAND_TRACKER_INDEX, 12] = 3.0 / TRACKER_DURATION_CAP
    kwargs["current_tracker_raw"][:, LEFT_HAND_TRACKER_INDEX, 12] = (
        3.0 / TRACKER_DURATION_CAP
    )
    first = model.prepare_conditioning(*values, **kwargs).taid

    changed_values = list(values)
    changed_values[1] = values[1].clone()
    changed_values[2] = values[2].clone()
    changed_values[1][:, :, LEFT_HAND_TRACKER_INDEX, :9] += 100.0
    changed_values[2][:, LEFT_HAND_TRACKER_INDEX, :9] += 100.0
    changed_kwargs = {name: tensor.clone() for name, tensor in kwargs.items()}
    changed_kwargs["current_tracker_raw"][:, LEFT_HAND_TRACKER_INDEX, :9] += 100.0
    second = model.prepare_conditioning(*changed_values, **changed_kwargs).taid
    assert first is not None and second is not None
    assert first.role_state.alpha[0, LEFT_HAND_TRACKER_INDEX] == 0.0
    assert not torch.allclose(
        first.prior.tracker_tokens[:, LEFT_HAND_TRACKER_INDEX],
        second.prior.tracker_tokens[:, LEFT_HAND_TRACKER_INDEX],
    )
    torch.testing.assert_close(first.prior.pose_model, second.prior.pose_model)
    torch.testing.assert_close(first.prior.root_head, second.prior.root_head)
    torch.testing.assert_close(first.prior.contact_logits, second.prior.contact_logits)
    torch.testing.assert_close(
        first.prior.joint_velocity_head, second.prior.joint_velocity_head
    )


def test_b1_tracker_history_contribution_follows_continuous_alpha() -> None:
    model = _model("B1").eval()
    values, kwargs = _taid_inputs(batch_size=1)
    values = list(values)
    values[1] = values[1].clone()
    values[1][:, :, LEFT_HAND_TRACKER_INDEX, :9] += 2.0
    contributions = []
    alphas = []
    outputs = []
    _enable_prior_output_sensitivity(model)
    for d_on in (5.0, 10.0, 15.0):
        current_values = list(values)
        current_values[2] = values[2].clone()
        current_values[2][:, LEFT_HAND_TRACKER_INDEX, 12] = d_on / TRACKER_DURATION_CAP
        current_kwargs = {name: tensor.clone() for name, tensor in kwargs.items()}
        current_kwargs["current_tracker_raw"][:, LEFT_HAND_TRACKER_INDEX, 12] = (
            d_on / TRACKER_DURATION_CAP
        )
        prepared = model.prepare_conditioning(*current_values, **current_kwargs).taid
        assert prepared is not None
        alpha = prepared.role_state.alpha[:, LEFT_HAND_TRACKER_INDEX]
        denominator = prepared.role_state.alpha.sum(dim=1).clamp_min(1.0)
        contribution = (
            prepared.prior.tracker_tokens[:, LEFT_HAND_TRACKER_INDEX]
            * alpha[:, None]
            / denominator[:, None]
        )
        alphas.append(float(alpha.item()))
        contributions.append(float(torch.linalg.norm(contribution).item()))
        outputs.append(prepared.prior.pose_model)
    assert alphas == pytest.approx([0.0, 0.5, 1.0])
    assert contributions[0] == 0.0
    assert 0.0 < contributions[1] < contributions[2]
    assert not torch.allclose(outputs[0], outputs[1])
    assert not torch.allclose(outputs[1], outputs[2])


@pytest.mark.parametrize("active_non_head", [(), (1, 2), (1, 2, 3, 4, 5)])
def test_b1_tracker_history_prior_keeps_144d_for_tracker_patterns(
    active_non_head: tuple[int, ...],
) -> None:
    model = _model("B1").eval()
    values, kwargs = _taid_inputs(batch_size=1)
    values = list(values)
    values[2] = values[2].clone()
    kwargs = {name: tensor.clone() for name, tensor in kwargs.items()}
    for tracker_index in range(1, 6):
        active = tracker_index in active_non_head
        values[2][:, tracker_index, 9] = float(active)
        values[2][:, tracker_index, 10] = float(active)
        values[2][:, tracker_index, 12] = (
            20.0 / TRACKER_DURATION_CAP if active else 0.0
        )
        kwargs["current_tracker_raw"][:, tracker_index, 9:13] = values[2][
            :, tracker_index, 9:13
        ]
    output = model(
        torch.randn(1, 144),
        torch.ones(1, dtype=torch.long),
        *values,
        **kwargs,
    )
    assert output.shape == (1, 144)
    assert torch.isfinite(output).all()


def test_cli_and_model_factory_preserve_all_taid_contract_values() -> None:
    args = build_train_arg_parser().parse_args(
        [
            "--data_dir",
            "unused-task",
            "--save_dir",
            "unused-save",
            "--latent_dim",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--motion_layers",
            "1",
            "--diffusion_steps",
            "2",
            "--taid_ablation",
            "B6",
            "--taid_anchor_ramp_start",
            "4",
            "--taid_anchor_ramp_end",
            "12",
            "--taid_innovation_ramp_frames",
            "10",
            "--taid_hand_torso_weight",
            "0.2",
        ]
    )
    model, diffusion = create_model_and_diffusion(args)
    assert model.taid_config.ablation == "B6"
    assert model.taid_config.prior_tracker_aggregation == "fixed_slots"
    assert model.taid_config.role.anchor_ramp_start == 4
    assert model.taid_config.role.anchor_ramp_end == 12
    assert model.taid_config.role.innovation_ramp_frames == 10
    assert model.taid_config.hand_torso_weight == 0.2
    assert diffusion.taid_prior_velocity_loss_weight == 0.25


def test_b1_prior_does_not_read_uncertain_current_measurement() -> None:
    torch.manual_seed(3)
    model = _model("B1").eval()
    values, kwargs = _taid_inputs(batch_size=1)
    values = list(values)
    values[2] = values[2].clone()
    kwargs["current_tracker_raw"] = kwargs["current_tracker_raw"].clone()
    # LeftHand 只有 3 帧连续有效，属于 Uncertain；alpha=0，当前测量不得进入 Prior。
    values[2][:, LEFT_HAND_TRACKER_INDEX, 12] = 3.0 / TRACKER_DURATION_CAP
    kwargs["current_tracker_raw"][:, LEFT_HAND_TRACKER_INDEX, 12] = 3.0 / TRACKER_DURATION_CAP
    first = model.prepare_conditioning(*values, **kwargs).taid
    changed_values = list(values)
    changed_values[2] = values[2].clone()
    changed_kwargs = {name: tensor.clone() for name, tensor in kwargs.items()}
    changed_values[2][:, LEFT_HAND_TRACKER_INDEX, :9] += 100.0
    changed_kwargs["current_tracker_raw"][:, LEFT_HAND_TRACKER_INDEX, :9] += 100.0
    second = model.prepare_conditioning(*changed_values, **changed_kwargs).taid
    assert first is not None and second is not None
    assert first.role_state.alpha[0, LEFT_HAND_TRACKER_INDEX] == 0.0
    torch.testing.assert_close(first.prior.pose_model, second.prior.pose_model)
    torch.testing.assert_close(first.prior.root_head, second.prior.root_head)
    torch.testing.assert_close(first.prior.contact_logits, second.prior.contact_logits)


def test_b1_only_prior_parameters_receive_gradients() -> None:
    model = _model("B1").train()
    values, kwargs = _taid_inputs(batch_size=1)
    output = model(
        torch.randn(1, 144),
        torch.ones(1, dtype=torch.long),
        *values,
        **kwargs,
    )
    output.square().mean().backward()
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith("taid_conditioner.prior.") for name in trainable)
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith("taid_conditioner.prior.")
    )


def test_fifteen_step_b1_backward_keeps_prior_gradient_boundary() -> None:
    model = _model("B1").train()
    values, kwargs = _taid_inputs(batch_size=1)
    frame_losses = []
    for step in range(15):
        hidden = _identity_pose(1) + float(step) * 1e-3
        output = model(
            hidden,
            torch.ones(1, dtype=torch.long),
            *values,
            **kwargs,
        )
        frame_losses.append(output.square().mean())
    weights = build_rollout_frame_weights(15, "linear_late")
    loss = frame_losses[0] + torch.stack(frame_losses[1:]).mul(weights).sum()
    loss.backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert gradients
    assert all(name.startswith("taid_conditioner.prior.") for name in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients.values())


def _b1_loss_batch() -> dict[str, torch.Tensor]:
    values, kwargs = _taid_inputs(batch_size=1)
    joint_offsets_parent = kwargs["joint_offsets_parent"].clone()
    joint_offsets_parent[:, 1:, 0] = 0.1
    return {
        "pose_history": values[0],
        "tracker_history": values[1],
        "current_tracker": values[2],
        "trajectory_history": values[3],
        "current_trajectory": values[4],
        "valid_frame_mask": values[5],
        "current_tracker_raw": kwargs["current_tracker_raw"],
        "joint_offsets_parent": joint_offsets_parent,
        "hard_rotation_state": torch.zeros(1, 6, dtype=torch.bool),
        "target_joints_head_ref": torch.ones(1, 24, 3),
        "prev_joints_head_ref": torch.zeros(1, 24, 3),
        "target_root_position_head_ref": torch.zeros(1, 3),
        "target_root_yaw_world": torch.zeros(1),
        "current_head_yaw_world": torch.zeros(1),
        "contact_target": torch.zeros(1, 2),
    }


def test_b1_prior_root_yaw_is_pose_derived_and_independent_of_root_xyz_head() -> None:
    torch.manual_seed(5)
    model = _model("B1").eval()
    values, kwargs = _taid_inputs(batch_size=1)
    first = model.prepare_conditioning(*values, **kwargs).taid
    assert first is not None
    _, expected_heading = decode_target_head_rotations_torch(first.prior.pose_raw)
    torch.testing.assert_close(first.prior.root_head[:, 3], expected_heading)
    assert model.taid_conditioner.prior.root_head[-1].out_features == 3

    with torch.no_grad():
        model.taid_conditioner.prior.root_head[-1].bias.add_(
            torch.tensor([0.5, -0.25, 0.75])
        )
    second = model.prepare_conditioning(*values, **kwargs).taid
    assert second is not None
    assert not torch.allclose(first.prior.root_head[:, :3], second.prior.root_head[:, :3])
    torch.testing.assert_close(first.prior.pose_raw, second.prior.pose_raw)
    torch.testing.assert_close(first.prior.root_head[:, 3], second.prior.root_head[:, 3])


def test_b1_pose_derived_root_yaw_loss_backpropagates_to_pose_head() -> None:
    model = _model("B1").train()
    values, kwargs = _taid_inputs(batch_size=1)
    prepared = model.prepare_conditioning(*values, **kwargs).taid
    assert prepared is not None
    target_yaw = prepared.prior.root_head[:, 3].detach() + 0.25
    loss = wrapped_angle_difference(prepared.prior.root_head[:, 3], target_yaw).square().mean()
    loss.backward()
    pose_gradient = model.taid_conditioner.prior.pose_head[-1].weight.grad
    assert pose_gradient is not None
    assert torch.isfinite(pose_gradient).all()
    assert torch.linalg.norm(pose_gradient) > 0.0
    # Root xyz head 不再承担 yaw，因此纯 yaw loss 不应更新该支路。
    root_xyz_gradient = model.taid_conditioner.prior.root_head[-1].weight.grad
    assert root_xyz_gradient is not None
    assert torch.count_nonzero(root_xyz_gradient) == 0


def test_wrapped_root_yaw_loss_is_continuous_across_pi_boundary() -> None:
    prediction = torch.tensor(math.pi - 1e-4, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(-math.pi + 1e-4, dtype=torch.float64)
    error = wrapped_angle_difference(prediction, target)
    torch.testing.assert_close(error.abs(), torch.tensor(2e-4, dtype=torch.float64))
    error.square().backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad)


def test_b2_freezes_and_stop_grads_prior() -> None:
    model = _model("B2").train()
    values, kwargs = _taid_inputs(batch_size=1)
    output = model(
        torch.randn(1, 144),
        torch.ones(1, dtype=torch.long),
        *values,
        **kwargs,
    )
    output.square().mean().backward()
    prior_parameters = list(model.taid_conditioner.prior.parameters())
    assert prior_parameters
    assert all(not parameter.requires_grad for parameter in prior_parameters)
    assert all(parameter.grad is None for parameter in prior_parameters)
    assert model.joint_output.weight.grad is not None


def test_b1_uses_prior_specific_loss_and_backpropagates() -> None:
    model = _model("B1").train()
    batch = _b1_loss_batch()
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    terms = diffusion.training_losses(
        model,
        _identity_pose(1),
        torch.ones(1, dtype=torch.long),
        model_kwargs={"y": batch},
        noise=torch.zeros(1, 144),
    )
    expected = {
        "prior_rotation_loss",
        "prior_fk_loss",
        "prior_internal_fk_loss",
        "prior_root_pose_gap_m",
        "prior_root_pose_gap_xz_m",
        "prior_joint_resolver_gap_m",
        "prior_root_loss",
        "prior_velocity_loss",
        "prior_contact_loss",
        "simple_loss",
        "aux_loss",
        "loss",
    }
    assert set(terms) == expected
    assert all(torch.isfinite(value).all() for value in terms.values())
    expected_aux = (
        diffusion.rotation_loss_weight * terms["prior_rotation_loss"]
        + diffusion.fk_loss_weight * terms["prior_fk_loss"]
        + diffusion.root_loss_weight * terms["prior_root_loss"]
        + diffusion.taid_prior_velocity_loss_weight * terms["prior_velocity_loss"]
        + diffusion.contact_loss_weight * terms["prior_contact_loss"]
    )
    torch.testing.assert_close(terms["aux_loss"], expected_aux)
    terms["loss"].mean().backward()
    assert model.taid_conditioner.prior.contact_head[-1].weight.grad is not None
    assert model.taid_conditioner.prior.joint_velocity_head[-1].weight.grad is not None
    pose_gradient = model.taid_conditioner.prior.pose_head[-1].weight.grad
    assert pose_gradient is not None and torch.isfinite(pose_gradient).all()
    assert torch.linalg.norm(pose_gradient) > 0.0


def test_b1_internal_root_xyz_does_not_change_deployed_fk_loss() -> None:
    torch.manual_seed(17)
    model = _model("B1").eval()
    batch = _b1_loss_batch()
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )

    def evaluate() -> dict[str, torch.Tensor]:
        return diffusion.training_losses(
            model,
            _identity_pose(1),
            torch.ones(1, dtype=torch.long),
            model_kwargs={"y": batch},
            noise=torch.zeros(1, 144),
        )

    first = evaluate()
    with torch.no_grad():
        model.taid_conditioner.prior.root_head[-1].bias.add_(
            torch.tensor([0.5, -0.25, 0.75])
        )
    second = evaluate()

    torch.testing.assert_close(first["prior_fk_loss"], second["prior_fk_loss"])
    assert not torch.allclose(first["prior_internal_fk_loss"], second["prior_internal_fk_loss"])
    assert not torch.allclose(first["prior_root_pose_gap_m"], second["prior_root_pose_gap_m"])
    assert not torch.allclose(
        first["prior_joint_resolver_gap_m"], second["prior_joint_resolver_gap_m"]
    )


def test_b1_deployed_fk_uses_hard_projected_tracker_rotation() -> None:
    model = _model("B1").eval()
    batch = _b1_loss_batch()
    batch["joint_offsets_parent"] = batch["joint_offsets_parent"].clone()
    batch["joint_offsets_parent"][:, 1:, 0] = 0.1
    batch["hard_rotation_state"][:, HIP_TRACKER_INDEX] = True
    batch["current_tracker_raw"] = batch["current_tracker_raw"].clone()
    batch["current_tracker_raw"][:, HIP_TRACKER_INDEX, 3:9] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.01, 0.02], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    terms = diffusion.training_losses(
        model,
        _identity_pose(1),
        torch.ones(1, dtype=torch.long),
        model_kwargs={"y": batch},
        noise=torch.zeros(1, 144),
        return_pred_xstart=True,
    )
    pelvis_start = 0
    torch.testing.assert_close(
        terms["deployed_pred_xstart"][0, pelvis_start : pelvis_start + 6],
        batch["current_tracker_raw"][0, HIP_TRACKER_INDEX, 3:9],
    )
    assert torch.isfinite(terms["prior_fk_loss"]).all()
    terms["prior_fk_loss"].mean().backward()
    pose_gradient = model.taid_conditioner.prior.pose_head[-1].weight.grad
    assert pose_gradient is not None and torch.isfinite(pose_gradient).all()
    assert torch.linalg.norm(pose_gradient) > 0.0


def test_checkpoint_stage_allows_only_declared_init_transitions(tmp_path) -> None:
    b0_state = _model("B0").state_dict()
    b1 = _model("B1")
    assert validate_taid_checkpoint_stage(
        b1, b0_state, allow_stage_transition=True
    ) == ("B0", "B1")
    initialized = b1.load_state_dict(b0_state, strict=False)
    assert not initialized.unexpected_keys
    assert initialized.missing_keys
    assert all(name.startswith("taid_conditioner.") for name in initialized.missing_keys)
    with pytest.raises(RuntimeError, match="resume"):
        validate_taid_checkpoint_stage(b1, b0_state, allow_stage_transition=False)

    b1_state = b1.state_dict()
    b6 = _model("B6")
    assert validate_taid_checkpoint_stage(
        b6, b1_state, allow_stage_transition=True
    ) == ("B1", "B6")
    assert int(b1_state["taid_conditioner.ablation_code"]) == 6
    incompatible = b6.load_state_dict(b1_state, strict=False)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert int(b6.taid_conditioner.ablation_code) == 6
    checkpoint = tmp_path / "b6.pt"
    torch.save(b6.state_dict(), checkpoint)
    restored = _model("B6")
    restored_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_taid_checkpoint_stage(
        restored, restored_state, allow_stage_transition=False
    )
    restored.load_state_dict(restored_state, strict=True)
    with pytest.raises(RuntimeError, match="init"):
        validate_taid_checkpoint_stage(_model("B4"), b0_state, allow_stage_transition=True)
    mismatched = RealtimePoseTargetDiT(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        motion_layers=1,
        dropout=0.0,
        taid_config=replace(TaIDConfig(ablation="B6", innovation_dim=16), hand_torso_weight=0.2),
    )
    with pytest.raises(RuntimeError, match="配置"):
        validate_taid_checkpoint_stage(
            mismatched, b6.state_dict(), allow_stage_transition=False
        )


def test_old_three_token_b1_tracker_fusion_checkpoint_is_not_compatible() -> None:
    model = _model("B1")
    state = model.state_dict()
    key = "taid_conditioner.prior.tracker_fusion.0.weight"
    assert state[key].shape == (model.latent_dim, model.latent_dim * 4)
    legacy_state = dict(state)
    legacy_state[key] = state[key][:, : model.latent_dim * 3].clone()
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(legacy_state, strict=False)


def test_pre_fixed_slot_b1_checkpoint_is_not_compatible() -> None:
    model = _model("B1")
    state = model.state_dict()
    key = "taid_conditioner.prior.anchor_slot_projection.weight"
    assert key in state
    legacy_state = dict(state)
    legacy_state.pop(key)
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(legacy_state, strict=True)


def test_offline_ddim_prepares_taid_conditioning_once() -> None:
    model = _model("B6").eval()
    values, kwargs = _taid_inputs(batch_size=2)
    names = (
        "pose_history",
        "tracker_history",
        "current_tracker",
        "trajectory_history",
        "current_trajectory",
        "valid_frame_mask",
    )
    batch = {name: value for name, value in zip(names, values)}
    batch.update(kwargs)
    call_count = 0

    def count_prior(_module, _inputs, _output) -> None:
        nonlocal call_count
        call_count += 1

    handle = model.taid_conditioner.prior.register_forward_hook(count_prior)
    result = build_sampling_model_kwargs(model, batch, torch.device("cpu"))
    assert set(result) == {"prepared_conditioning"}
    assert result["prepared_conditioning"].taid is not None
    for timestep in range(3):
        with torch.no_grad():
            model(
                torch.randn(2, 144),
                torch.full((2,), timestep, dtype=torch.long),
                **result,
            )
    handle.remove()
    assert call_count == 1


def test_zero_innovation_produces_exact_zero_token() -> None:
    model = _model("B6")
    conditioner = model.taid_conditioner
    assert conditioner is not None
    zero = torch.zeros(2, 6, 6)
    current = torch.zeros(2, 6, 13)
    contact = torch.rand(2, 2)
    beta = torch.ones(2, 6)
    token = conditioner._encode_innovation(zero, zero, current, contact, beta)
    assert torch.count_nonzero(token) == 0
    prior_values, prior_kwargs = _taid_inputs(2)
    prior = model.prepare_conditioning(*prior_values, **prior_kwargs).taid.prior
    region = conditioner._posterior_region_injection(prior, token, beta)
    assert torch.count_nonzero(region) == 0

    invalid_beta = beta.clone()
    invalid_beta[:, LEFT_HAND_TRACKER_INDEX] = 0.0
    nonzero = torch.ones_like(zero)
    invalid_token = conditioner._encode_innovation(
        nonzero, nonzero, current, contact, invalid_beta
    )
    assert torch.count_nonzero(invalid_token[:, LEFT_HAND_TRACKER_INDEX]) == 0


def test_previous_position_innovation_is_reexpressed_in_current_head_frame() -> None:
    previous = torch.tensor([[[1.0, 0.0, 0.0]]])
    delta_yaw = torch.tensor([math.pi / 2.0])
    trajectory = torch.tensor(
        [[[0.0, 0.0, 0.0, math.sin(math.pi / 2.0), math.cos(math.pi / 2.0)]]]
    )
    actual = reexpress_previous_position_residual_torch(previous, trajectory)
    expected = torch.einsum(
        "bij,btj->bti", make_yaw_rotation_torch(-delta_yaw), previous
    )
    torch.testing.assert_close(actual, expected)


def test_training_and_runtime_normalization_produce_same_fk_innovation() -> None:
    model = _model("B4").eval()
    raw_values, raw_kwargs = _taid_inputs(batch_size=1)
    raw_values = list(raw_values)
    pose_mean = torch.linspace(-0.2, 0.2, 144)
    pose_std = torch.full((144,), 2.0)
    tracker_mean = torch.linspace(-0.1, 0.1, 54).reshape(6, 9)
    tracker_std = torch.full((6, 9), 1.5)

    normalized_values = list(raw_values)
    normalized_values[0] = (raw_values[0] - pose_mean) / pose_std
    normalized_values[1] = raw_values[1].clone()
    normalized_values[1][..., :9] = (
        raw_values[1][..., :9] - tracker_mean
    ) / tracker_std
    normalized_values[2] = raw_values[2].clone()
    normalized_values[2][..., :9] = (
        raw_values[2][..., :9] - tracker_mean
    ) / tracker_std

    raw = model.prepare_conditioning(*raw_values, **raw_kwargs).taid
    normalized = model.prepare_conditioning(
        *normalized_values,
        **raw_kwargs,
        pose_mean=pose_mean,
        pose_std=pose_std,
        tracker_mean=tracker_mean,
        tracker_std=tracker_std,
    ).taid
    assert raw is not None and normalized is not None
    torch.testing.assert_close(raw.role_state.alpha, normalized.role_state.alpha)
    torch.testing.assert_close(raw.role_state.beta, normalized.role_state.beta)
    torch.testing.assert_close(
        raw.innovation_residual, normalized.innovation_residual, atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        raw.innovation_delta, normalized.innovation_delta, atol=1e-6, rtol=0.0
    )


def test_fixed_routes_are_anatomical_and_contact_gated() -> None:
    model = _model("B5")
    conditioner = model.taid_conditioner
    assert conditioner is not None
    routes = conditioner.fixed_route_weights(torch.tensor([[1.0, 0.0]]))
    # LeftHand 只进入 left_arm 和弱 torso，绝不进入左右腿。
    torch.testing.assert_close(routes[0, LEFT_HAND_TRACKER_INDEX], torch.tensor([0.1, 1.0, 0.0, 0.0, 0.0]))
    assert torch.count_nonzero(routes[0, HEAD_TRACKER_INDEX]) == 0
    assert routes[0, LEFT_FOOT_TRACKER_INDEX, 0] == 0.25
    assert routes[0, LEFT_FOOT_TRACKER_INDEX, 3] == 1.0
    assert routes[0, LEFT_FOOT_TRACKER_INDEX, 0] > routes[0, RIGHT_FOOT_TRACKER_INDEX, 0]


def test_b6_replaces_hard_uncertain_weight_with_continuous_u_to_a_weight() -> None:
    values, kwargs = _taid_inputs(batch_size=1)
    values = list(values)
    values[2] = values[2].clone()
    kwargs["current_tracker_raw"] = kwargs["current_tracker_raw"].clone()
    values[2][:, LEFT_HAND_TRACKER_INDEX, 12] = 7.0 / TRACKER_DURATION_CAP
    kwargs["current_tracker_raw"][:, LEFT_HAND_TRACKER_INDEX, 12] = 7.0 / TRACKER_DURATION_CAP
    b2 = _model("B2").prepare_conditioning(*values, **kwargs).taid
    b5 = _model("B5").prepare_conditioning(*values, **kwargs).taid
    b6 = _model("B6").prepare_conditioning(*values, **kwargs).taid
    assert b2 is not None and b5 is not None and b6 is not None
    torch.testing.assert_close(
        b2.observation_weight[0, LEFT_HAND_TRACKER_INDEX],
        b2.role_state.alpha[0, LEFT_HAND_TRACKER_INDEX],
    )
    assert b5.observation_weight[0, LEFT_HAND_TRACKER_INDEX] > 1.0
    assert b6.observation_weight[0, LEFT_HAND_TRACKER_INDEX] <= 1.0
    assert b6.observation_weight[0, LEFT_HAND_TRACKER_INDEX] < b5.observation_weight[0, LEFT_HAND_TRACKER_INDEX]


@pytest.mark.parametrize("ablation", ["B2", "B3", "B4", "B5", "B6"])
def test_taid_targetdit_keeps_144d_output(ablation: str) -> None:
    model = _model(ablation).eval()
    values, kwargs = _taid_inputs(batch_size=2)
    prepared = model.prepare_conditioning(*values, **kwargs)
    with torch.no_grad():
        output, auxiliary = model(
            torch.randn(2, 144),
            torch.ones(2, dtype=torch.long),
            prepared_conditioning=prepared,
            return_aux_outputs=True,
        )
    assert output.shape == (2, 144)
    assert auxiliary["taid_region_injection"].shape == (2, 5, 32)
    assert torch.isfinite(output).all()


def test_fk_innovation_is_zero_when_observation_matches_prior_fk() -> None:
    model = _model("B4").eval()
    values, kwargs = _taid_inputs(batch_size=1)
    prepared = model.prepare_conditioning(*values, **kwargs).taid
    assert prepared is not None
    torch.testing.assert_close(
        prepared.innovation_residual,
        torch.zeros_like(prepared.innovation_residual),
        atol=1e-6,
        rtol=0.0,
    )


def test_b4_fk_innovation_path_has_finite_gradients() -> None:
    model = _model("B4").train()
    values, kwargs = _taid_inputs(batch_size=1)
    values = list(values)
    values[2] = values[2].clone()
    kwargs["current_tracker_raw"] = kwargs["current_tracker_raw"].clone()
    values[2][:, LEFT_HAND_TRACKER_INDEX, 12] = 3.0 / TRACKER_DURATION_CAP
    kwargs["current_tracker_raw"][:, LEFT_HAND_TRACKER_INDEX, 12] = 3.0 / TRACKER_DURATION_CAP
    values[2][:, LEFT_HAND_TRACKER_INDEX, 0] = 0.3
    kwargs["current_tracker_raw"][:, LEFT_HAND_TRACKER_INDEX, 0] = 0.3
    output = model(
        torch.randn(1, 144),
        torch.ones(1, dtype=torch.long),
        *values,
        **kwargs,
    )
    output.square().mean().backward()
    gradient = model.taid_conditioner.innovation_residual_encoder[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.linalg.norm(gradient) > 0.0


@pytest.mark.parametrize("angle", [1e-7, math.pi - 1e-4])
def test_so3_log_is_finite_and_differentiable_near_zero_and_pi(angle: float) -> None:
    value = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
    zero = value * 0.0
    skew = torch.stack(
        [
            torch.stack([zero, -value, zero]),
            torch.stack([value, zero, zero]),
            torch.stack([zero, zero, zero]),
        ]
    )
    rotation = torch.matrix_exp(skew)
    log = so3_log_map_torch(rotation)
    loss = log.square().sum()
    loss.backward()
    assert torch.isfinite(log).all()
    assert value.grad is not None and torch.isfinite(value.grad)
