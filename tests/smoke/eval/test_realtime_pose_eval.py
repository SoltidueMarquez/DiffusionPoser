from eval.evaluate_realtime_pose import summarize


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
        "by_missing_age": {},
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
