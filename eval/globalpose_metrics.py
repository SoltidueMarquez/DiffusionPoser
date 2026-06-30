from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def compute_translation_drift(
    prediction_tran: np.ndarray,
    target_tran: np.ndarray,
    window_sizes: Iterable[int] = range(1, 8),
) -> dict[int, dict[str, float | int]]:
    """计算 GlobalPose `test.py` 同款 travelled-distance 平移误差。"""

    prediction = _as_tran_array(prediction_tran, "prediction_tran")
    target = _as_tran_array(target_tran, "target_tran")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction_tran 和 target_tran 形状必须一致，实际为 {prediction.shape} vs {target.shape}")

    move_distance = cumulative_move_distance(target)
    result: dict[int, dict[str, float | int]] = {}
    for window_size in window_sizes:
        window = int(window_size)
        errors = translation_window_errors(
            prediction_tran=prediction,
            target_tran=target,
            move_distance=move_distance,
            window_size_m=window,
        )
        if errors.size == 0:
            result[window] = {"mean_m": float("nan"), "std_m": float("nan"), "count": 0}
        else:
            result[window] = {
                "mean_m": float(errors.mean()),
                "std_m": float(errors.std()),
                "count": int(errors.size),
            }
    return result


def cumulative_move_distance(target_tran: np.ndarray) -> np.ndarray:
    target = _as_tran_array(target_tran, "target_tran")
    distance = np.zeros((target.shape[0],), dtype=np.float64)
    if target.shape[0] > 1:
        step = np.linalg.norm(target[1:] - target[:-1], axis=1)
        distance[1:] = np.cumsum(step)
    return distance


def translation_window_errors(
    prediction_tran: np.ndarray,
    target_tran: np.ndarray,
    move_distance: np.ndarray,
    window_size_m: int,
) -> np.ndarray:
    prediction = _as_tran_array(prediction_tran, "prediction_tran")
    target = _as_tran_array(target_tran, "target_tran")
    distance = np.asarray(move_distance, dtype=np.float64)
    if distance.shape != (target.shape[0],):
        raise ValueError(f"move_distance 应为 [T]，实际为 {distance.shape}")
    if int(window_size_m) <= 0:
        raise ValueError(f"window_size_m 必须为正整数，实际为 {window_size_m}")

    errors: list[float] = []
    for start, end in iter_distance_window_pairs(distance, int(window_size_m)):
        travelled = distance[end] - distance[start]
        if travelled <= 0.0:
            continue
        pred_delta = prediction[end] - prediction[start]
        target_delta = target[end] - target[start]
        error = np.linalg.norm(target_delta - pred_delta) / travelled * int(window_size_m)
        errors.append(float(error))
    return np.asarray(errors, dtype=np.float64)


def iter_distance_window_pairs(move_distance: np.ndarray, window_size_m: int) -> list[tuple[int, int]]:
    """复刻 GlobalPose `test.py` 中按 GT travelled distance 选帧对的逻辑。"""

    distance = np.asarray(move_distance, dtype=np.float64).reshape(-1)
    pairs: list[tuple[int, int]] = []
    start, end = 0, 1
    while end < len(distance):
        if distance[end] - distance[start] < int(window_size_m):
            end += 1
        else:
            if len(pairs) == 0 or pairs[-1][1] != end:
                pairs.append((start, end))
            start += 1
    return pairs


def drift_percent_at_window(result: dict[int, dict[str, float | int]], window_size_m: int = 7) -> float:
    stats = result.get(int(window_size_m))
    if not stats:
        return float("nan")
    mean_m = float(stats["mean_m"])
    if np.isnan(mean_m):
        return float("nan")
    return mean_m / float(window_size_m) * 100.0


def _as_tran_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} 应为 [T,3]，实际为 {array.shape}")
    return array
