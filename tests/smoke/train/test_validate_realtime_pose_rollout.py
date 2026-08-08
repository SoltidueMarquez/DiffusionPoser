from __future__ import annotations

import pytest
import torch

from train.validate_realtime_pose_rollout import (
    build_argument_parser,
    losses_to_scalars,
    normalize_batch_candidates,
    summarize_gradient_boundary,
    summarize_rollout_weight_contract,
)


def test_preflight_fixed_slot_contract_is_opt_in() -> None:
    parser = build_argument_parser()
    required = ["--data_dir", "tasks", "--save_dir", "unused"]
    default_args = parser.parse_args(required)
    assert default_args.require_fixed_slot_prior is False

    required_args = parser.parse_args([*required, "--require_fixed_slot_prior"])
    assert required_args.require_fixed_slot_prior is True


def test_preflight_batch_candidates_require_unique_descending_positive_values() -> None:
    assert normalize_batch_candidates([32, 16, 8, 4]) == (32, 16, 8, 4)
    for invalid in ([], [4, 8], [8, 8], [4, 0]):
        with pytest.raises(ValueError):
            normalize_batch_candidates(invalid)


def test_preflight_gradient_boundary_accepts_only_finite_nonzero_prior_gradients() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.taid_conditioner = torch.nn.Module()
            self.taid_conditioner.prior = torch.nn.Linear(2, 2)
            self.frozen = torch.nn.Linear(2, 2)
            for parameter in self.frozen.parameters():
                parameter.requires_grad_(False)

    model = ToyModel()
    model.taid_conditioner.prior(torch.ones(1, 2)).sum().backward()
    summary = summarize_gradient_boundary(model)
    assert summary["prior_only"] is True
    assert summary["all_gradients_finite"] is True
    assert not summary["trainable_outside_prior"]
    assert not summary["gradients_outside_prior"]

    model.frozen.weight.requires_grad_(True)
    assert summarize_gradient_boundary(model)["prior_only"] is False


def test_preflight_loss_report_detaches_and_averages_tensor_values() -> None:
    losses = {
        "loss": torch.tensor([1.0, 3.0], requires_grad=True),
        "rollout_step_14_loss": torch.tensor([2.0]),
    }
    assert losses_to_scalars(losses) == {
        "loss": 2.0,
        "rollout_step_14_loss": 2.0,
    }


def test_preflight_validates_linear_late_step_weights() -> None:
    scalar_losses = {
        f"rollout_step_{step}_weight": step / 105.0
        for step in range(1, 15)
    }
    summary = summarize_rollout_weight_contract(
        scalar_losses,
        rollout_steps=15,
        weighting="linear_late",
    )
    assert summary["matches"] is True
    assert summary["expected_step_1"] == pytest.approx(1.0 / 105.0)
    assert summary["expected_step_last"] == pytest.approx(14.0 / 105.0)
    assert summary["actual_sum"] == pytest.approx(1.0)

    scalar_losses["rollout_step_14_weight"] = 1.0 / 14.0
    assert summarize_rollout_weight_contract(
        scalar_losses,
        rollout_steps=15,
        weighting="linear_late",
    )["matches"] is False
