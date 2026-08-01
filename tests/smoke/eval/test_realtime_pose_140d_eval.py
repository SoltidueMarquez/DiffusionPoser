from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from data_loaders.realtime_pose_kinematics import rotation_6d_forward_up_np
from eval.evaluate_realtime_pose import evaluate_file


def _identity_6d(shape: tuple[int, ...]) -> np.ndarray:
    identity = rotation_6d_forward_up_np(np.eye(3, dtype=np.float64)).astype(np.float32)
    return np.broadcast_to(identity, (*shape, 6)).copy()


def _payload(sequence_count: int, steps: int) -> dict[str, np.ndarray]:
    local = _identity_6d((sequence_count, steps, 24)).reshape(sequence_count, steps, 144)
    target = np.zeros((sequence_count, steps, 140), dtype=np.float32)
    target[..., :138] = _identity_6d((sequence_count, steps, 23)).reshape(sequence_count, steps, 138)
    target[..., 139] = 1.0
    return {
        "reference_body_local_delta_6d": local.copy(),
        "predicted_body_local_delta_6d": local.copy(),
        "reference_joints_world": np.zeros((sequence_count, steps, 24, 3), dtype=np.float32),
        "predicted_joints_world": np.zeros((sequence_count, steps, 24, 3), dtype=np.float32),
        "reference_root_position_world": np.zeros((sequence_count, steps, 3), dtype=np.float32),
        "predicted_root_position_world": np.zeros((sequence_count, steps, 3), dtype=np.float32),
        "reference_root_yaw_world": np.zeros((sequence_count, steps), dtype=np.float32),
        "predicted_root_yaw_world": np.zeros((sequence_count, steps), dtype=np.float32),
        "reference_hip_height": np.ones((sequence_count, steps), dtype=np.float32),
        "predicted_hip_height": np.ones((sequence_count, steps), dtype=np.float32),
        "reference_target_raw": target.copy(),
        "reconstructed_target_raw": target.copy(),
        "known_mask": np.zeros((sequence_count, steps, 140), dtype=bool),
        "tracker_pos_world": np.zeros((sequence_count, steps, 6, 3), dtype=np.float32),
        "configured": np.ones((sequence_count, steps, 6), dtype=bool),
        "measured_valid": np.ones((sequence_count, steps, 6), dtype=bool),
        "missing_age": np.zeros((sequence_count, steps, 6), dtype=np.int64),
        "scenario": np.full((sequence_count, steps), "fixed_six"),
        "eval_frame_mask": np.ones((sequence_count, steps), dtype=bool),
        "fps": np.float32(60.0),
        "known_rotation_max_error": np.zeros((sequence_count, steps), dtype=np.float32),
    }


def test_paper_metrics_use_expected_units_and_keep_sequence_boundaries(tmp_path):
    payload = _payload(sequence_count=2, steps=2)
    payload["predicted_joints_world"][0, 1, :22, 0] = 0.01
    payload["predicted_joints_world"][1, :, :22, 0] = 0.02

    angle = 0.21
    rotation = Rotation.from_rotvec([angle, 0.0, 0.0]).as_matrix()
    payload["predicted_body_local_delta_6d"][0, 1, 6:12] = rotation_6d_forward_up_np(rotation)

    path = tmp_path / "result.npz"
    np.savez(path, **payload)
    result = evaluate_file(path)

    expected_mpjre = np.degrees(angle) / (2 * 2 * 21 * 3)
    assert result["mpjre_deg"] == pytest.approx(expected_mpjre, abs=1e-5)
    assert result["mpjpe_cm"] == pytest.approx(1.25, abs=1e-6)
    assert result["mpjve_cm_s"] == pytest.approx(30.0, abs=1e-5)
    assert result["mpjae_cm_s2"] is None


def test_acceleration_error_uses_three_continuous_frames(tmp_path):
    payload = _payload(sequence_count=1, steps=3)
    payload["predicted_joints_world"][0, 2, :22, 0] = 0.01
    path = tmp_path / "acceleration.npz"
    np.savez(path, **payload)

    result = evaluate_file(path)
    assert result["acceleration_triplets"] == 1
    assert result["mpjae_cm_s2"] == pytest.approx(3600.0, abs=1e-3)


def test_paper_metrics_exclude_two_hand_end_joints(tmp_path):
    payload = _payload(sequence_count=1, steps=2)
    rotation = Rotation.from_rotvec([0.5, 0.0, 0.0]).as_matrix()
    rotation_6d = rotation_6d_forward_up_np(rotation)
    payload["predicted_body_local_delta_6d"][:, 1, 22 * 6 : 24 * 6] = np.tile(rotation_6d, 2)
    payload["predicted_joints_world"][:, 1, 22:, 0] = 1.0

    path = tmp_path / "end_joints.npz"
    np.savez(path, **payload)
    result = evaluate_file(path)
    assert result["mpjre_deg"] == 0.0
    assert result["mpjpe_cm"] == 0.0
    assert result["mpjve_cm_s"] == 0.0


def test_evaluation_reports_scenarios_and_missing_age_buckets(tmp_path):
    payload = _payload(sequence_count=4, steps=1)
    payload["scenario"][:] = "dropout"
    payload["missing_age"][:, 0, 1] = [1, 10, 20, 40]
    path = tmp_path / "dropout.npz"
    np.savez(path, **payload)

    result = evaluate_file(path)
    assert result["mpjve_cm_s"] is None
    assert result["mpjae_cm_s2"] is None
    assert result["by_scenario"]["dropout"]["samples"] == 4
    assert all(
        result["by_missing_age"][name]["samples"] == 1
        for name in ("1-5", "6-15", "16-30", "31-60")
    )
