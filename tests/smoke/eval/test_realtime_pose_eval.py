import numpy as np

from eval.evaluate_realtime_pose import (
    _sequence_macro_stats,
    compute_predicted_joint_jitter,
    evaluate_file,
    summarize,
)
from tests.smoke.realtime_pose_fixtures import IDENTITY_6D


METRIC_KEYS = (
    "mpjre_deg",
    "mpjpe_cm",
    "mpjve_cm_s",
    "mpjae_cm_s2",
    "jitter_m_s3",
    "unknown_rotation_deg",
    "root_yaw_error_deg",
    "root_xz_error_m",
    "hip_height_error_m",
    "tracker_position_error_m",
)


def _result(samples: int, mpjpe_mean: float) -> dict:
    stats = {key: {"sum": 0.0, "count": 1} for key in METRIC_KEYS}
    stats["mpjpe_cm"] = {"sum": mpjpe_mean, "count": 1}
    stats["mpjve_cm_s"] = {"sum": 0.0, "count": 0}
    stats["mpjae_cm_s2"] = {"sum": 0.0, "count": 0}
    stats["jitter_m_s3"] = {"sum": 0.0, "count": 0}
    return {
        "sequences": 1,
        "samples": samples,
        "velocity_pairs": 0,
        "acceleration_triplets": 0,
        "by_scenario": {},
        "by_d_off": {},
        "_metric_stats": stats,
    }


def test_eval_summary_uses_sequence_macro_average():
    summary = summarize(
        [
            _result(samples=1, mpjpe_mean=1.0),
            _result(samples=3, mpjpe_mean=3.0),
        ]
    )
    assert summary["samples"] == 4
    assert summary["aggregation"] == "sequence_macro"
    assert summary["mpjpe_cm"] == 2.0
    assert summary["mpjve_cm_s"] is None
    assert summary["jitter_m_s3"] is None


def test_sequence_macro_stats_does_not_give_long_sequences_more_weight():
    values = np.asarray([[1.0, np.nan, np.nan], [3.0, 3.0, 3.0]])
    mask = np.ones((2, 3), dtype=bool)

    stats = _sequence_macro_stats(values, mask)

    assert stats == {"sum": 4.0, "count": 2}


def test_eval_reads_raw_deployed_and_new_reconnect_buckets(tmp_path):
    steps = 3
    pose = np.tile(IDENTITY_6D, (1, steps, 24)).astype(np.float32)
    tracker = np.zeros((1, steps, 6, 13), dtype=np.float32)
    tracker[..., 3:9] = IDENTITY_6D
    tracker[..., 9:11] = 1.0
    tracker[..., 12] = 1.0
    # 所有 Tracker rotation 都处于 hard，位置指标仍必须统计五个非 Head Tracker。
    hard = np.ones((1, steps, 6), dtype=bool)
    predicted_joints = np.zeros((1, steps, 24, 3), dtype=np.float32)
    predicted_joints[:, :, 20, 0] = 5.0
    predicted_joints[0, :, 10, 0] = np.asarray([0.0, 1.0, 3.0])
    predicted_joints[0, :, 11, 0] = np.asarray([0.0, 1.0, 3.0])
    contact_target = np.asarray([[[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)
    contact_logits = np.where(contact_target > 0.5, 1.0, -1.0).astype(np.float32)
    pose_horizon = np.tile(IDENTITY_6D, (1, steps, 11, 24, 1)).reshape(
        1, steps, 11, 144
    ).astype(np.float32)
    path = tmp_path / "result.npz"
    np.savez(
        path,
        fps=np.float32(60.0),
        reference_target_raw=pose,
        raw_pred_target_raw=pose,
        deployed_pred_target_raw=pose,
        reference_body_local_delta_6d=pose,
        predicted_body_local_delta_6d=pose,
        reference_joints_world=np.zeros((1, steps, 24, 3), dtype=np.float32),
        predicted_joints_world=predicted_joints,
        reference_root_position_world=np.zeros((1, steps, 3), dtype=np.float32),
        predicted_root_position_world=np.zeros((1, steps, 3), dtype=np.float32),
        reference_root_yaw_world=np.zeros((1, steps), dtype=np.float32),
        predicted_root_yaw_world=np.zeros((1, steps), dtype=np.float32),
        reference_hip_height=np.ones((1, steps), dtype=np.float32),
        predicted_hip_height=np.ones((1, steps), dtype=np.float32),
        tracker_pos_world=np.zeros((1, steps, 6, 3), dtype=np.float32),
        current_tracker_raw=tracker,
        configured=np.ones((1, steps, 6), dtype=bool),
        measured_valid=np.ones((1, steps, 6), dtype=bool),
        d_off=np.zeros((1, steps, 6), dtype=np.int64),
        d_on=np.tile(np.arange(1, steps + 1)[None, :, None], (1, 1, 6)),
        hard_rotation_state=hard,
        contact_target=contact_target,
        contact_logits=contact_logits,
        reference_pose_horizon_raw=pose_horizon,
        raw_pred_pose_horizon_raw=pose_horizon,
        deployed_pred_pose_horizon_raw=pose_horizon,
        pose_horizon_valid_mask=np.ones((1, steps, 11), dtype=bool),
        scenario=np.full((1, steps), "two_point_dropout_reconnect"),
        eval_frame_mask=np.ones((1, steps), dtype=bool),
        history_length=np.asarray([[0, 1, 60]], dtype=np.int64),
    )
    result = evaluate_file(path)
    assert result["raw_rotation_deg"] == 0.0
    assert result["deployed_rotation_deg"] == 0.0
    assert result["raw_pose_horizon_0_rotation_deg"] == result["raw_rotation_deg"]
    assert result["deployed_pose_horizon_0_rotation_deg"] == result["deployed_rotation_deg"]
    assert result["raw_pose_horizon_10_rotation_deg"] == 0.0
    assert result["deployed_future_pose_rotation_deg"] == 0.0
    assert result["contact_accuracy"] == 1.0
    assert np.isclose(result["measured_non_head_tracker_position_error_m"], 23.0 / 15.0)
    assert result["foot_slide_m_s"] == 120.0
    assert result["by_reconnect_d_on"]["1"]["samples"] == 1
    assert result["by_history_phase"]["cold_start_0_59"]["samples"] == 2
    assert result["by_history_phase"]["steady_state_60_plus"]["samples"] == 1


def test_predicted_joint_jitter_uses_third_difference_and_marks_warmup_nan():
    fps = 2.0
    positions = np.zeros((1, 5, 22, 3), dtype=np.float64)
    frame = np.arange(5, dtype=np.float64)
    positions[0, :, 0, 0] = frame**3

    jitter = compute_predicted_joint_jitter(positions, fps)

    assert np.isnan(jitter[0, :3]).all()
    np.testing.assert_allclose(jitter[0, 3:], 6.0 * fps**3 / 22.0)
