from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from data_loaders.realtime_pose_config import TaIDConfig
from data_loaders.realtime_pose_geometry import (
    reexpress_previous_position_residual_torch,
    so3_log_map_torch,
)
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_torch
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    RIGHT_FOOT_TRACKER_INDEX,
    TRACKER_DURATION_CAP,
)
from model.realtime_pose_target_dit import RealtimePoseTargetDiT
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from sample.reconstruct_stream import build_sampling_model_kwargs
from train.training_loop import validate_taid_checkpoint_stage
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


def test_b0_keeps_legacy_state_dict_without_taid_parameters() -> None:
    model = _model("B0")
    assert model.taid_conditioner is None
    assert not any(name.startswith("taid_conditioner.") for name in model.state_dict())


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
    values, kwargs = _taid_inputs(batch_size=1)
    batch = {
        "pose_history": values[0],
        "tracker_history": values[1],
        "current_tracker": values[2],
        "trajectory_history": values[3],
        "current_trajectory": values[4],
        "valid_frame_mask": values[5],
        "current_tracker_raw": kwargs["current_tracker_raw"],
        "joint_offsets_parent": kwargs["joint_offsets_parent"],
        "hard_rotation_state": torch.zeros(1, 6, dtype=torch.bool),
        "target_joints_head_ref": torch.ones(1, 24, 3),
        "prev_joints_head_ref": torch.zeros(1, 24, 3),
        "target_root_position_head_ref": torch.zeros(1, 3),
        "target_root_yaw_world": torch.zeros(1),
        "current_head_yaw_world": torch.zeros(1),
        "contact_target": torch.zeros(1, 2),
    }
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
        "prior_root_loss",
        "prior_velocity_loss",
        "prior_contact_loss",
        "simple_loss",
        "aux_loss",
        "loss",
    }
    assert set(terms) == expected
    assert all(torch.isfinite(value).all() for value in terms.values())
    terms["loss"].mean().backward()
    assert model.taid_conditioner.prior.contact_head[-1].weight.grad is not None
    assert model.taid_conditioner.prior.joint_velocity_head[-1].weight.grad is not None


def test_checkpoint_stage_allows_only_declared_init_transitions(tmp_path) -> None:
    b0_state = _model("B0").state_dict()
    b1 = _model("B1")
    assert validate_taid_checkpoint_stage(
        b1, b0_state, allow_stage_transition=True
    ) == ("B0", "B1")
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
