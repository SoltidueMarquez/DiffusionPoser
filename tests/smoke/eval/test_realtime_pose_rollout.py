from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_kinematics import rotation_6d_forward_up_np
from eval.evaluate_realtime_pose_rollout import evaluate_rollout_file


def test_rollout_eval_reports_reconnect_jump(tmp_path):
    steps = 3
    identity = rotation_6d_forward_up_np(np.eye(3, dtype=np.float64)).astype(np.float32)
    target = np.zeros((1, steps, 140), dtype=np.float32)
    target[..., :138] = np.broadcast_to(identity, (1, steps, 23, 6)).reshape(1, steps, 138)
    target[..., 139] = 1.0
    local = np.broadcast_to(identity, (1, steps, 24, 6)).reshape(1, steps, 144).copy()
    predicted_joints = np.zeros((1, steps, 24, 3), dtype=np.float32)
    predicted_joints[:, 2, :, 0] = 0.1
    measured = np.ones((1, steps, 6), dtype=bool)
    measured[:, 1, 1] = False
    path = tmp_path / "rollout_result.npz"
    np.savez(
        path,
        reference_body_local_delta_6d=local,
        predicted_body_local_delta_6d=local,
        reference_joints_world=np.zeros_like(predicted_joints),
        predicted_joints_world=predicted_joints,
        reference_root_position_world=np.zeros((1, steps, 3), dtype=np.float32),
        predicted_root_position_world=np.zeros((1, steps, 3), dtype=np.float32),
        reference_root_yaw_world=np.zeros((1, steps), dtype=np.float32),
        predicted_root_yaw_world=np.zeros((1, steps), dtype=np.float32),
        reference_hip_height=np.ones((1, steps), dtype=np.float32),
        predicted_hip_height=np.ones((1, steps), dtype=np.float32),
        reference_target_raw=target,
        reconstructed_target_raw=target,
        known_mask=np.zeros((1, steps, 140), dtype=bool),
        tracker_pos_world=np.zeros((1, steps, 6, 3), dtype=np.float32),
        configured=np.ones((1, steps, 6), dtype=bool),
        measured_valid=measured,
        missing_age=np.zeros((1, steps, 6), dtype=np.int64),
        scenario=np.asarray([["dropout"] * steps]),
        eval_frame_mask=np.ones((1, steps), dtype=bool),
        fps=np.float32(60.0),
        known_rotation_max_error=np.zeros((1, steps), dtype=np.float32),
    )
    result = evaluate_rollout_file(path)
    assert result["reconnect_frames"] == 1
    assert result["reconnect_velocity_jump_m"] > 0.0
