import numpy as np

from eval.evaluate_realtime_pose import evaluate_file, summarize
from tests.smoke.realtime_pose_fixtures import IDENTITY_6D


METRIC_KEYS = (
    "mpjre_deg",
    "mpjpe_cm",
    "mpjve_cm_s",
    "mpjae_cm_s2",
    "unknown_rotation_deg",
    "root_yaw_error_deg",
    "root_xz_error_m",
    "hip_height_error_m",
    "tracker_position_error_m",
)


def _result(samples: int, mpjpe_sum: float, mpjpe_count: int, known_max: float) -> dict:
    stats = {key: {"sum": 0.0, "count": samples} for key in METRIC_KEYS}
    stats["mpjpe_cm"] = {"sum": mpjpe_sum, "count": mpjpe_count}
    stats["mpjve_cm_s"] = {"sum": 0.0, "count": 0}
    stats["mpjae_cm_s2"] = {"sum": 0.0, "count": 0}
    return {
        "sequences": 1,
        "samples": samples,
        "velocity_pairs": 0,
        "acceleration_triplets": 0,
        "known_tracker_rotation_max_error_deg": known_max,
        "by_scenario": {},
        "by_d_off": {},
        "_metric_stats": stats,
    }


def test_eval_summary_uses_weighted_metric_counts_and_global_max():
    summary = summarize(
        [
            _result(samples=1, mpjpe_sum=1.0, mpjpe_count=1, known_max=0.1),
            _result(samples=3, mpjpe_sum=9.0, mpjpe_count=3, known_max=0.3),
        ]
    )
    assert summary["samples"] == 4
    assert summary["mpjpe_cm"] == 2.5
    assert summary["mpjve_cm_s"] is None
    assert summary["known_tracker_rotation_max_error_deg"] == 0.3


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
        future_leg_target=np.tile(IDENTITY_6D, (1, steps, 3, 8, 1)),
        future_leg_prediction=np.tile(IDENTITY_6D, (1, steps, 3, 8, 1)),
        scenario=np.full((1, steps), "two_point_dropout_reconnect"),
        eval_frame_mask=np.ones((1, steps), dtype=bool),
        history_length=np.asarray([[0, 1, 60]], dtype=np.int64),
        hard_rotation_max_error=np.zeros((1, steps), dtype=np.float32),
    )
    result = evaluate_file(path)
    assert result["raw_rotation_deg"] == 0.0
    assert result["deployed_rotation_deg"] == 0.0
    assert result["contact_accuracy"] == 1.0
    assert np.isclose(result["measured_non_head_tracker_position_error_m"], 23.0 / 15.0)
    assert result["foot_slide_m_s"] == 120.0
    assert result["by_reconnect_d_on"]["1"]["samples"] == 1
    assert result["by_history_phase"]["cold_start_0_59"]["samples"] == 2
    assert result["by_history_phase"]["steady_state_60_plus"]["samples"] == 1
