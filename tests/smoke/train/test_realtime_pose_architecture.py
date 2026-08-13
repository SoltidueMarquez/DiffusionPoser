from __future__ import annotations

import math

import pytest
import torch

from data_loaders.realtime_pose_config import TARGET_JOINT_REGIONS
from data_loaders.realtime_pose_history_noise import (
    HistoryPoseNoiseConfig,
    corrupt_history_pose_observation,
)
from data_loaders.realtime_pose_kinematics import rotation_6d_forward_up_torch
from model.realtime_pose_spatiotemporal_dit import RealtimePoseSpatioTemporalDiT
from train.training_loop import (
    TRAIN_DEVICE_FIELDS,
    TrainLoop,
    move_training_batch_to_device,
)


FRAME_OFFSETS = torch.tensor(
    [-60, -53, -47, -40, -34, -27, -21, -14, -8, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)


def _model() -> RealtimePoseSpatioTemporalDiT:
    return RealtimePoseSpatioTemporalDiT(
        latent_dim=64,
        num_layers=1,
        num_heads=8,
        dropout=0.0,
        max_seq_len=21,
    ).eval()


def _conditioning(batch_size: int = 2, cold_start: bool = False) -> dict[str, torch.Tensor]:
    valid = torch.ones(batch_size, 11, dtype=torch.bool)
    if cold_start:
        valid[:, :-1] = False
    tracker = torch.zeros(batch_size, 11, 6, 13)
    tracker[..., 9:11] = 1.0
    tracker[..., 12] = 1.0
    tracker[~valid] = 0.0
    history = torch.randn(batch_size, 10, 144)
    history[~valid[:, :-1]] = 0.0
    head_path = torch.randn(batch_size, 11, 5)
    head_path[:, -1, :2] = 0.0
    head_path[:, -1, 3:] = torch.tensor([0.0, 1.0])
    head_path[~valid] = 0.0
    confidence = torch.ones(batch_size, 10, 5)
    confidence[~valid[:, :-1]] = 0.0
    return {
        "history_pose_observation": history,
        "tracker_window": tracker,
        "head_path_window": head_path,
        "history_region_confidence": confidence,
        "window_valid_mask": valid,
        "frame_offsets": FRAME_OFFSETS,
    }


def test_target_regions_cover_each_joint_once():
    assert TARGET_JOINT_REGIONS.shape == (24,)
    assert set(TARGET_JOINT_REGIONS.tolist()) == {0, 1, 2, 3, 4}


def test_spatiotemporal_model_shape_cached_condition_and_cold_start_are_finite():
    model = _model()
    values = _conditioning(cold_start=True)
    hidden = torch.randn(2, 11, 144)
    timestep = torch.tensor([1, 2])
    prepared = model.prepare_conditioning(**values)
    direct, direct_aux = model(hidden, timestep, **values, return_aux_outputs=True)
    cached, cached_aux = model(
        hidden,
        timestep,
        prepared_conditioning=prepared,
        return_aux_outputs=True,
    )
    torch.testing.assert_close(direct, cached)
    torch.testing.assert_close(direct_aux["contact_logits"], cached_aux["contact_logits"])
    assert direct.shape == (2, 11, 144)
    assert set(direct_aux) == {"contact_logits"}
    assert direct_aux["contact_logits"].shape == (2, 2)
    assert torch.isfinite(direct).all()


def test_history_and_current_use_independent_input_projections_once():
    model = _model()
    values = _conditioning(batch_size=1)
    inputs: dict[str, list[torch.Size]] = {"history": [], "current": []}
    handles = [
        model.history_pose_input.register_forward_pre_hook(
            lambda _module, args: inputs["history"].append(args[0].shape)
        ),
        model.joint_input.register_forward_pre_hook(
            lambda _module, args: inputs["current"].append(args[0].shape)
        ),
    ]
    try:
        model(torch.randn(1, 11, 144), torch.ones(1), **values)
    finally:
        for handle in handles:
            handle.remove()
    assert inputs == {
        "history": [torch.Size([1, 10, 24, 6])],
        "current": [torch.Size([1, 11, 24, 6])],
    }


def test_tracker_history_is_encoded_per_frame_without_gru_summary():
    model = _model()
    values = _conditioning(batch_size=1)
    first = model.prepare_conditioning(**values)
    changed = {name: value.clone() for name, value in values.items()}
    changed["tracker_window"][:, 3, 1, :9] += 5.0
    second = model.prepare_conditioning(**changed)
    assert not torch.allclose(
        first.observation.rotation_tokens[:, 3],
        second.observation.rotation_tokens[:, 3],
    )
    torch.testing.assert_close(
        first.observation.rotation_tokens[:, 2],
        second.observation.rotation_tokens[:, 2],
    )
    assert not hasattr(model.observation_encoder, "history_gru")


def test_head_path_and_history_pose_are_independent_conditions():
    model = _model()
    values = _conditioning(batch_size=1)
    first = model.prepare_conditioning(**values)
    changed = {name: value.clone() for name, value in values.items()}
    changed["head_path_window"][:, 4, 0] += 10.0
    second = model.prepare_conditioning(**changed)
    assert not torch.allclose(
        first.static_pose_condition[:, 4], second.static_pose_condition[:, 4]
    )
    torch.testing.assert_close(
        values["history_pose_observation"], changed["history_pose_observation"]
    )


def test_temporal_mask_is_prefix_bidirectional_and_ignores_padding_keys():
    model = _model()
    valid = torch.tensor([[False, False] + [True] * 19])
    mask = model._temporal_mask(valid, joint_count=24).reshape(1, 24, 8, 21, 21)
    # 历史 query 不能读取目标；目标 query 可双向读取整个目标窗口。
    assert mask[0, 0, 0, 5, 10:].all()
    assert not mask[0, 0, 0, 10, 2:].any()
    assert not mask[0, 0, 0, 20, 10:].any()
    assert not mask[0, 0, 0, 5, 6]
    assert not mask[0, 0, 0, 6, 5]
    assert mask[0, 0, 0, 5, 10]
    assert mask[0, 0, 0, 10, :2].all()
    assert mask[0, 0, 0, 5, :2].all()


def test_changing_current_token_does_not_change_history_block_rows():
    model = _model()
    values = _conditioning(batch_size=1)
    with torch.no_grad():
        block = model.blocks[0]
        latent_dim = model.latent_dim
        block.adaln_modulation[-1].bias[5 * latent_dim : 6 * latent_dim].fill_(1.0)
    captured: list[torch.Tensor] = []
    handle = model.blocks[0].register_forward_hook(
        lambda _module, _args, output: captured.append(output.detach().clone())
    )
    try:
        model(torch.zeros(1, 11, 144), torch.ones(1), **values)
        changed = torch.zeros(1, 11, 144)
        changed[:, 0] = 10.0
        model(changed, torch.ones(1), **values)
    finally:
        handle.remove()
    torch.testing.assert_close(captured[0][:, :10], captured[1][:, :10])


def test_current_loss_backpropagates_into_history_input_projection():
    model = _model().train()
    values = _conditioning(batch_size=1)
    with torch.no_grad():
        latent_dim = model.latent_dim
        model.blocks[0].adaln_modulation[-1].bias[
            5 * latent_dim : 6 * latent_dim
        ].fill_(1.0)
    output = model(torch.randn(1, 11, 144), torch.ones(1), **values)
    output[:, 0].square().mean().backward()
    gradient = model.history_pose_input.weight.grad
    assert gradient is not None
    assert torch.linalg.norm(gradient) > 0.0


def test_token_and_tracker_condition_masks_have_independent_semantics():
    model = _model()
    prepared = model.prepare_conditioning(**_conditioning(batch_size=1, cold_start=True))
    assert prepared.token_valid_mask.shape == (1, 21)
    assert not prepared.token_valid_mask[:, :10].any()
    assert prepared.token_valid_mask[:, 10:].all()
    assert prepared.tracker_condition_valid_mask[:, 10]
    assert not prepared.tracker_condition_valid_mask[:, 11:].any()
    assert torch.count_nonzero(prepared.observation.state_tokens[:, 11:]) == 0


def test_future_tokens_and_temporal_attention_receive_finite_nonzero_gradients():
    model = _model().train()
    values = _conditioning(batch_size=1)
    with torch.no_grad():
        latent_dim = model.latent_dim
        model.blocks[0].adaln_modulation[-1].bias[
            5 * latent_dim : 6 * latent_dim
        ].fill_(1.0)
    hidden = torch.randn(1, 11, 144, requires_grad=True)
    output = model(hidden, torch.ones(1), **values)
    output[:, 1:].square().mean().backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.linalg.norm(hidden.grad[:, 1:]) > 0.0
    gradient = model.blocks[0].temporal_attention.in_proj_weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.linalg.norm(gradient) > 0.0


def test_history_noise_only_changes_valid_history_in_so3_space():
    identity = torch.eye(3).expand(2, 10, 24, 3, 3)
    clean = rotation_6d_forward_up_torch(identity).reshape(2, 10, 144)
    valid = torch.tensor(
        [[False, False, True, True, True, True, True, True, True, True], [True] * 10]
    )
    clean[~valid] = 0.0
    torch.manual_seed(0)
    noisy = corrupt_history_pose_observation(
        clean,
        history_region_confidence=torch.full((2, 10, 5), 0.5),
        history_valid_mask=valid,
        pose_mean=None,
        pose_scale=None,
        config=HistoryPoseNoiseConfig(probability=1.0),
    )
    assert torch.isfinite(noisy).all()
    assert torch.count_nonzero(noisy[~valid]) == 0
    assert not torch.allclose(noisy[valid], clean[valid])


def test_clean_history_probability_zero_is_exact_identity():
    clean = torch.randn(2, 10, 144)
    valid = torch.ones(2, 10, dtype=torch.bool)
    result = corrupt_history_pose_observation(
        clean,
        torch.ones(2, 10, 5),
        valid,
        None,
        None,
        HistoryPoseNoiseConfig(probability=0.0),
    )
    torch.testing.assert_close(result, clean)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (field, value)
        for field in (
            "probability",
            "min_degrees",
            "max_degrees",
            "temporal_rho",
            "region_ratio",
            "joint_ratio",
        )
        for value in (math.nan, math.inf, -math.inf)
    ],
)
def test_history_noise_config_rejects_nonfinite_values(field: str, value: float):
    with pytest.raises(ValueError, match=field):
        HistoryPoseNoiseConfig(**{field: value}).validate()


def test_model_rejects_unknown_constructor_and_forward_arguments():
    with pytest.raises(TypeError):
        RealtimePoseSpatioTemporalDiT(latent_dmi=64)

    model = _model()
    with pytest.raises(TypeError):
        model(torch.randn(1, 11, 144), torch.ones(1), obsolete_argument=True)


def test_training_device_transfer_keeps_reconstruction_only_fields_on_cpu():
    assert "configured" not in TRAIN_DEVICE_FIELDS
    assert "joint_rest_local_rotations_6d" not in TRAIN_DEVICE_FIELDS
    batch = {
        "x": torch.zeros(1),
        "configured": torch.ones(1, dtype=torch.bool),
        "joint_rest_local_rotations_6d": torch.zeros(24, 6),
    }

    moved = move_training_batch_to_device(batch, torch.device("meta"))

    assert moved["x"].device.type == "meta"
    assert moved["configured"].device.type == "cpu"
    assert moved["joint_rest_local_rotations_6d"].device.type == "cpu"


def test_training_model_kwargs_use_y_as_the_only_condition_source():
    loop = object.__new__(TrainLoop)
    loop.model = torch.nn.Identity().eval()
    loop.pose_mean = None
    loop.pose_scale = None
    sample = torch.zeros(1, 11, 144)
    batch = {
        "history_pose_observation": torch.zeros(1, 10, 144),
        "tracker_window": torch.zeros(1, 11, 6, 13),
        "head_path_window": torch.zeros(1, 11, 5),
        "history_region_confidence": torch.zeros(1, 10, 5),
        "window_valid_mask": torch.ones(1, 11, dtype=torch.bool),
        "frame_offsets": torch.zeros(1, 21, dtype=torch.long),
        "tracker_window_raw": torch.zeros(1, 11, 6, 13),
        "hard_rotation_state_window": torch.zeros(1, 11, 6, dtype=torch.bool),
        "target_joints_head_ref": torch.zeros(1, 24, 3),
        "joint_offsets_parent": torch.zeros(1, 24, 3),
        "target_root_position_head_ref": torch.zeros(1, 3),
        "target_root_yaw_world": torch.zeros(1),
        "current_head_yaw_world": torch.zeros(1),
        "previous_contact_target": torch.zeros(1, 2),
        "contact_target": torch.zeros(1, 2),
    }

    model_kwargs = loop.mask_manager(batch, sample)

    assert set(model_kwargs) == {"y"}
    assert model_kwargs["y"]["tracker_window"] is batch["tracker_window"]
    assert model_kwargs["y"]["window_valid_mask"] is batch["window_valid_mask"]


def test_training_boundary_passes_entire_joint_horizon_to_diffusion():
    class CapturingDiffusion:
        def __init__(self) -> None:
            self.targets: list[torch.Tensor] = []

        def training_losses(self, _model, x_start, _timesteps, **_kwargs):
            self.targets.append(x_start.detach().clone())
            return {"loss": torch.zeros(x_start.shape[0])}

    loop = object.__new__(TrainLoop)
    loop.model = torch.nn.Identity()
    loop.diffusion = CapturingDiffusion()
    loop.feature_w = None
    loop.snr_gamma = 0.0
    loop.use_l1 = False
    loop.mask_manager = lambda batch, sample: {
        "y": {"history_pose_observation": batch["history_pose_observation"]}
    }
    sample_window = torch.randn(2, 11, 144)
    batch = {
        "x": sample_window.clone(),
        "history_pose_observation": torch.randn(2, 10, 144),
    }

    loop.compute_losses(batch, torch.ones(2, dtype=torch.long))
    batch["history_pose_observation"].add_(100.0)
    loop.compute_losses(batch, torch.ones(2, dtype=torch.long))

    torch.testing.assert_close(loop.diffusion.targets[0], sample_window)
    torch.testing.assert_close(loop.diffusion.targets[1], sample_window)
