from __future__ import annotations

import numpy as np

from eval.stationary_signal_metrics import (
    STATIONARY_JOINT_NAMES,
    compute_stationary_signal_metrics,
)


def test_stationary_metrics_count_false_and_missed_locks_per_joint():
    target = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    pred = np.asarray(
        [
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    report = compute_stationary_signal_metrics(target, pred, thresholds=(0.5,))

    assert report["frames"] == 3
    assert report["joints"] == 5
    assert report["joint_names"] == list(STATIONARY_JOINT_NAMES)
    aggregate = report["thresholds"]["0.5"]["aggregate"]
    assert aggregate["tp"] == 5
    assert aggregate["tn"] == 5
    assert aggregate["fp"] == 3
    assert aggregate["fn"] == 2
    assert aggregate["false_lock_rate"] == 3 / 8
    assert aggregate["missed_lock_rate"] == 2 / 7
    assert round(aggregate["f1"], 6) == round(10 / 15, 6)


def test_stationary_metrics_are_finite_for_all_stationary_or_all_moving():
    all_stationary = np.ones((4, 5), dtype=np.float32)
    all_moving = np.zeros((4, 5), dtype=np.float32)

    stationary_report = compute_stationary_signal_metrics(all_stationary, all_stationary)
    moving_report = compute_stationary_signal_metrics(all_moving, all_moving)

    assert stationary_report["thresholds"]["0.5"]["aggregate"]["false_lock_rate"] == 0.0
    assert stationary_report["thresholds"]["0.5"]["aggregate"]["missed_lock_rate"] == 0.0
    assert moving_report["thresholds"]["0.5"]["aggregate"]["false_lock_rate"] == 0.0
    assert moving_report["thresholds"]["0.5"]["aggregate"]["missed_lock_rate"] == 0.0


def test_stationary_metrics_report_jitter_and_transition_lag():
    target = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    pred = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    report = compute_stationary_signal_metrics(target, pred, thresholds=(0.5,), max_transition_lag=3)

    aggregate = report["thresholds"]["0.5"]["aggregate"]
    assert aggregate["prob_jitter_mean_abs"] > 0.0
    assert aggregate["move_to_static_lag_mean_frames"] == 1.0
    assert aggregate["static_to_move_lag_mean_frames"] == 0.0
