from __future__ import annotations

import pytest
import torch

from diffusion.realtime_pose import (
    REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS,
    RealtimePoseLossConfig,
)
from train.realtime_loss_calibration import (
    calibrate_realtime_loss_weights,
    expected_active_loss_terms,
)


def test_loss_weight_calibration_matches_target_gradient_ratios_and_clamp():
    sample = {"simple_loss": 2.0}
    for index, loss_name in enumerate(REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS):
        sample[loss_name] = float(index + 1)

    report = calibrate_realtime_loss_weights([sample, sample])

    for loss_name, target_ratio in REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS.items():
        assert report["measured_ratios"][loss_name] == pytest.approx(target_ratio)
    assert report["sample_count"] == 2


def test_loss_weight_calibration_rejects_expected_term_with_zero_gradient():
    sample = {"simple_loss": 1.0}
    for loss_name in REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS:
        sample[loss_name] = 1.0
    sample["stationary_margin_loss"] = 0.0

    with pytest.raises(RuntimeError, match="stationary_margin_loss"):
        calibrate_realtime_loss_weights([sample])


def test_bounded_stationary_defaults_use_main_mse_channel_weight_without_range_target():
    config = RealtimePoseLossConfig()

    assert config.stationary_runtime_threshold == pytest.approx(0.7)
    assert config.stationary_runtime_margin == pytest.approx(0.1)
    assert config.stationary_simple_loss_channel_weight == pytest.approx(1.6232687317836745)
    assert not hasattr(config, "stationary_regression_loss_weight")
    assert not hasattr(config, "stationary_range_loss_weight")
    assert "stationary_regression_loss" not in REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS
    assert "stationary_range_loss" not in REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS


def test_calibration_activation_tracks_nohip_contact_and_predicted_history():
    active = expected_active_loss_terms(
        {
            "hip_missing_fraction": torch.tensor([0.0, 1.0]),
            "temporal_sample_fraction": torch.tensor([1.0, 1.0]),
            "contact_active_foot_count": torch.tensor([0.0, 1.0]),
            "stationary_margin_loss": torch.tensor([0.0, 0.1]),
        }
    )

    assert active["nohip_yaw_loss"]
    assert active["nohip_root_xz_loss"]
    assert active["contact_height_loss"]
    assert active["contact_velocity_loss"]
    assert active["yaw_velocity_loss"]
    assert active["stationary_margin_loss"]
    assert "stationary_regression_loss" not in active
    assert "stationary_range_loss" not in active
