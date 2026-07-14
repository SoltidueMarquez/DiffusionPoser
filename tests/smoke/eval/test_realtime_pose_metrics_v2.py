from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_kinematics import TRACKER_JOINT_INDICES
from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from eval.evaluate_realtime_pose_rollout import (
    evaluate_rollout_file,
    foot_slide_m_per_second,
    reconnect_peak_jump,
    recovery_time,
    root_drift_cm_per_minute,
)
from eval.stationary_signal_metrics import compute_stationary_signal_metrics


def test_root_drift_uses_closed_form_slope_and_excludes_reset_frames():
    timestamps = np.arange(10, dtype=np.float64)[None]
    error = (timestamps * 0.01).astype(np.float64)
    mask = np.ones((1, 10), dtype=bool)
    root_source = np.zeros((1, 10), dtype=np.int8)
    root_source[0, 5] = 3
    assert np.isclose(
        root_drift_cm_per_minute(error, mask, timestamps, root_source=root_source),
        60.0,
        atol=1e-10,
    )


def test_reconnect_recovery_and_foot_slide_formulas():
    root = np.zeros((1, 8, 3), dtype=np.float32)
    root[0, :, 0] = np.asarray([0.0, 0.1, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35])
    yaw = np.zeros((1, 8), dtype=np.float32)
    yaw[0, 2] = 0.25
    source = np.asarray([[0, 2, 2, 0, 0, 0, 0, 0]], dtype=np.int8)
    alpha = np.asarray([[0.0, 0.2, 0.7, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    jump_xz, jump_yaw = reconnect_peak_jump(root, yaw, source, alpha)
    assert np.isclose(jump_xz, 0.2, atol=1e-6)
    assert np.isclose(jump_yaw, 0.25, atol=1e-6)

    root_error = np.asarray([[0.2, 0.2, 0.1, 0.04, 0.03, 0.02, 0.01, 0.01]], dtype=np.float32)
    yaw_error = np.zeros_like(root_error)
    seconds, frames = recovery_time(root_error, yaw_error, alpha, np.arange(8, dtype=np.float64))
    assert seconds == 2.0
    assert frames == 2

    foot = np.zeros((1, 3, 3), dtype=np.float32)
    foot[0, :, 0] = [0.0, 1.0, 2.0]
    assert foot_slide_m_per_second(
        foot,
        np.ones((1, 3), dtype=bool),
        np.zeros((1, 3), dtype=np.float32),
        np.arange(3, dtype=np.float64),
    ) == 1.0


def test_stationary_reports_pre_clamp_out_of_bounds_ratio():
    target = np.zeros((2, 5), dtype=np.float32)
    predicted = target.copy()
    predicted[0, 0] = -0.1
    predicted[1, 1] = 1.1
    metrics = compute_stationary_signal_metrics(target, predicted, thresholds=(0.5,))
    assert np.isclose(metrics["clamp_pre_out_of_bounds_ratio"], 0.2)
    assert np.isclose(
        metrics["thresholds"]["0.5"]["aggregate"]["clamp_pre_out_of_bounds_ratio"],
        0.2,
    )


def test_rollout_file_reports_valid_position_and_rotation_reprojection(tmp_path):
    schema = get_schema_spec(DEFAULT_REALTIME_POSE_SCHEMA_NAME)
    frames = 6
    features = np.zeros((1, frames, schema.feature_dim), dtype=np.float32)
    features[..., schema.stationary_prob_slice()] = 1.0
    joints = np.zeros((1, frames, 24, 3), dtype=np.float32)
    joints[..., 1] = 0.01
    tracker_positions = joints[..., np.asarray(TRACKER_JOINT_INDICES), :].copy()
    identity_6d = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    tracker_rotations = np.broadcast_to(identity_6d, (1, frames, 6, 6)).copy()
    joint_rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (1, frames, 24, 3, 3)).copy()
    sensor_valid = np.ones((1, frames, 6), dtype=bool)
    path = tmp_path / "metrics.npz"
    np.savez(
        path,
        reference_features_raw=features,
        predicted_features_raw=features,
        reference_joints_world=joints,
        predicted_joints_world=joints,
        predicted_joint_rot_world=joint_rotations,
        root_yaw_reference=np.zeros((1, frames), dtype=np.float32),
        root_yaw_predicted=np.zeros((1, frames), dtype=np.float32),
        root_pos_world_reference=np.zeros((1, frames, 3), dtype=np.float32),
        root_pos_world_predicted=np.zeros((1, frames, 3), dtype=np.float32),
        tracker_pos_world=tracker_positions,
        tracker_rot_world_6d=tracker_rotations,
        sensor_valid=sensor_valid,
        eval_frame_mask=np.ones((1, frames), dtype=bool),
        timestamp_seconds=np.arange(frames, dtype=np.float64)[None] / 60.0,
        floor_y=np.zeros((1, frames), dtype=np.float32),
    )
    result = evaluate_rollout_file(path)
    assert result["tracker_reprojection_pos_mean_cm"] == 0.0
    assert result["tracker_reprojection_rot_mean_deg"] == 0.0
    assert result["stationary_f1"] == 1.0

