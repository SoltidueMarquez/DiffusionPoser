from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


STATIONARY_JOINT_NAMES = ("pelvis", "left_foot", "right_foot", "left_hand", "right_hand")
STATIONARY_PROB_DIM = 5


def _as_prob_matrix(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != STATIONARY_PROB_DIM:
        raise ValueError(f"{name} must have shape [T,5] or [1,T,5], got {array.shape}")
    return np.clip(array, 0.0, 1.0)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _binary_counts(target: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    target_static = target >= threshold
    pred_static = pred >= threshold
    return {
        "tp": np.sum(pred_static & target_static, axis=0),
        "tn": np.sum(~pred_static & ~target_static, axis=0),
        "fp": np.sum(pred_static & ~target_static, axis=0),
        "fn": np.sum(~pred_static & target_static, axis=0),
    }


def _lag_values(target_binary: np.ndarray, pred_binary: np.ndarray, direction: str, max_lag: int) -> list[int]:
    lags: list[int] = []
    for joint in range(target_binary.shape[1]):
        values = target_binary[:, joint]
        pred = pred_binary[:, joint]
        for frame in range(1, values.shape[0]):
            if direction == "move_to_static":
                changed = not values[frame - 1] and values[frame]
                desired = True
            else:
                changed = values[frame - 1] and not values[frame]
                desired = False
            if not changed:
                continue
            for lag in range(max_lag + 1):
                index = frame + lag
                if index >= pred.shape[0]:
                    break
                if bool(pred[index]) == desired:
                    lags.append(lag)
                    break
    return lags


def _mean_lag(target_binary: np.ndarray, pred_binary: np.ndarray, direction: str, max_lag: int) -> float:
    values = _lag_values(target_binary, pred_binary, direction=direction, max_lag=max_lag)
    return float(np.mean(values)) if values else 0.0


def _threshold_report(
    target: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    joint_names: tuple[str, ...],
    max_transition_lag: int,
) -> dict[str, Any]:
    counts = _binary_counts(target, pred, threshold)
    per_joint: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(joint_names):
        tp = float(counts["tp"][index])
        tn = float(counts["tn"][index])
        fp = float(counts["fp"][index])
        fn = float(counts["fn"][index])
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        per_joint[name] = {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
            "f1": _safe_divide(2.0 * precision * recall, precision + recall),
            "false_lock_rate": _safe_divide(fp, fp + tn),
            "missed_lock_rate": _safe_divide(fn, fn + tp),
        }

    tp = float(np.sum(counts["tp"]))
    tn = float(np.sum(counts["tn"]))
    fp = float(np.sum(counts["fp"]))
    fn = float(np.sum(counts["fn"]))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    target_binary = target >= threshold
    pred_binary = pred >= threshold
    aggregate = {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2.0 * precision * recall, precision + recall),
        "false_lock_rate": _safe_divide(fp, fp + tn),
        "missed_lock_rate": _safe_divide(fn, fn + tp),
        "prob_jitter_mean_abs": float(np.mean(np.abs(np.diff(pred, axis=0)))) if pred.shape[0] > 1 else 0.0,
        "move_to_static_lag_mean_frames": _mean_lag(
            target_binary,
            pred_binary,
            direction="move_to_static",
            max_lag=max_transition_lag,
        ),
        "static_to_move_lag_mean_frames": _mean_lag(
            target_binary,
            pred_binary,
            direction="static_to_move",
            max_lag=max_transition_lag,
        ),
    }
    return {"aggregate": aggregate, "per_joint": per_joint}


def compute_stationary_signal_metrics(
    target_stationary_prob_5: np.ndarray,
    predicted_stationary_prob_5: np.ndarray,
    *,
    thresholds: Iterable[float] = (0.5, 0.7),
    joint_names: tuple[str, ...] = STATIONARY_JOINT_NAMES,
    max_transition_lag: int = 30,
) -> dict[str, Any]:
    pred_raw = np.asarray(predicted_stationary_prob_5, dtype=np.float32)
    if pred_raw.ndim == 3 and pred_raw.shape[0] == 1:
        pred_raw = pred_raw[0]
    if pred_raw.ndim != 2 or pred_raw.shape[1] != STATIONARY_PROB_DIM:
        raise ValueError(
            f"predicted_stationary_prob_5 must have shape [T,5] or [1,T,5], got {pred_raw.shape}"
        )
    out_of_bounds_ratio = float(np.mean((pred_raw < 0.0) | (pred_raw > 1.0)))
    target = _as_prob_matrix("target_stationary_prob_5", target_stationary_prob_5)
    pred = _as_prob_matrix("predicted_stationary_prob_5", pred_raw)
    if target.shape != pred.shape:
        raise ValueError(f"target and predicted stationary probabilities must match: {target.shape} vs {pred.shape}")
    if len(joint_names) != STATIONARY_PROB_DIM:
        raise ValueError(f"joint_names must contain {STATIONARY_PROB_DIM} entries, got {len(joint_names)}")

    threshold_reports = {
        f"{float(threshold):g}": _threshold_report(
            target,
            pred,
            float(threshold),
            tuple(joint_names),
            max_transition_lag=max(0, int(max_transition_lag)),
        )
        for threshold in thresholds
    }
    for report in threshold_reports.values():
        report["aggregate"]["clamp_pre_out_of_bounds_ratio"] = out_of_bounds_ratio
    return {
        "frames": int(target.shape[0]),
        "joints": int(target.shape[1]),
        "joint_names": list(joint_names),
        "clamp_pre_out_of_bounds_ratio": out_of_bounds_ratio,
        "thresholds": threshold_reports,
    }
