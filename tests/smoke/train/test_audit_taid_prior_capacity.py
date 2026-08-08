from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES
from train.audit_taid_prior_capacity import (
    _feedback_amplification,
    _regional_metrics,
    compare_prior_values,
)


def _identity_pose(frames: int) -> np.ndarray:
    identity = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return np.tile(identity, (frames, 24))


def test_prior_sensitivity_summary_separates_pose_root_and_deployed_joints() -> None:
    baseline = {
        "pose_raw": torch.zeros(2, 144),
        "deployed_joints": torch.zeros(2, 24, 3),
        "deployed_root": torch.zeros(2, 3),
        "root_xyz": torch.zeros(2, 3),
        "root_yaw": torch.tensor([torch.pi - 0.1, -torch.pi + 0.1]),
        "contact": torch.zeros(2, 2),
        "velocity": torch.zeros(2, 24, 3),
    }
    changed = {name: value.clone() for name, value in baseline.items()}
    changed["pose_raw"] += 0.25
    changed["deployed_joints"][:, :, 0] += 0.1
    changed["root_xyz"] += 0.2
    changed["root_yaw"] = torch.tensor([-torch.pi + 0.1, torch.pi - 0.1])
    summary = compare_prior_values(baseline, changed)
    assert summary["prior_pose_raw_mean_abs"] == 0.25
    assert np.isclose(summary["deployed_joint_mean_gap_m"], 0.1)
    assert np.isclose(summary["pose_root_yaw_max_abs_rad"], 0.2)


def test_regional_horizon_metrics_report_all_regions_and_joints() -> None:
    frames = 4
    reference_joints = np.zeros((1, frames, 24, 3), dtype=np.float32)
    predicted_joints = reference_joints.copy()
    predicted_joints[..., 0] = 0.01
    identity = _identity_pose(frames)
    payload = {
        "reference_joints_world": reference_joints,
        "predicted_joints_world": predicted_joints,
        "reference_body_local_delta_6d": identity[None],
        "predicted_body_local_delta_6d": identity[None],
    }
    summary = _regional_metrics(payload, np.asarray([False, True, True, False]))
    assert summary["samples"] == 2
    assert set(summary["by_region"]) == {
        "torso",
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
    }
    assert set(summary["per_joint"]) == set(SMPL_JOINT_NAMES)
    for values in summary["by_region"].values():
        assert np.isclose(values["mpjpe_cm"], 1.0)
        assert np.isclose(values["mpjre_deg"], 0.0)


def test_feedback_amplification_uses_only_consecutive_evaluated_frames() -> None:
    frames = 5
    reference = np.zeros((1, frames, 24, 3), dtype=np.float32)
    predicted = reference.copy()
    predicted[0, :, :, 0] = np.arange(frames, dtype=np.float32)[:, None] * 0.01
    payload = {
        "reference_joints_world": reference,
        "predicted_joints_world": predicted,
    }
    summary = _feedback_amplification(
        payload,
        np.asarray([False, True, True, True, False]),
    )
    assert summary["samples"] == 2
    assert summary["correlation"] == 1.0
    assert summary["mean_gain"] > 1.0
