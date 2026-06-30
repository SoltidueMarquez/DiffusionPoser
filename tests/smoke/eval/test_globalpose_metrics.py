from __future__ import annotations

import numpy as np
import pytest

from eval.globalpose_metrics import compute_translation_drift, drift_percent_at_window


def straight_walk(length_m: float = 10.0, frame_count: int = 101) -> np.ndarray:
    tran = np.zeros((frame_count, 3), dtype=np.float32)
    tran[:, 0] = np.linspace(0.0, length_m, frame_count, dtype=np.float32)
    return tran


def test_translation_drift_matches_globalpose_distance_windows():
    target = straight_walk()
    prediction = target * np.asarray([0.9, 1.0, 1.0], dtype=np.float32)

    result = compute_translation_drift(prediction, target, window_sizes=range(1, 8))

    assert set(result.keys()) == set(range(1, 8))
    for window_size, stats in result.items():
        assert stats["count"] > 0
        assert stats["mean_m"] == pytest.approx(window_size * 0.1, abs=1e-5)
    assert drift_percent_at_window(result, 7) == pytest.approx(10.0, abs=1e-5)


def test_translation_drift_reports_empty_windows_as_nan():
    target = straight_walk(length_m=0.5, frame_count=6)
    prediction = target.copy()

    result = compute_translation_drift(prediction, target, window_sizes=[1, 7])

    assert result[1]["count"] == 0
    assert np.isnan(result[1]["mean_m"])
    assert np.isnan(drift_percent_at_window(result, 7))
