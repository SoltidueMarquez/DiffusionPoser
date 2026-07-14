from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from eval.stationary_signal_metrics import STATIONARY_JOINT_NAMES, compute_stationary_signal_metrics


SIGNAL_KEYS = {
    "feature_channel": ("feature_stationary_prob_5",),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare stationary signal sources for Unity contact control.",
        allow_abbrev=False,
    )
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--thresholds", default="0.5,0.7", type=str)
    parser.add_argument("--schema", default=REALTIME_POSE_SCHEMA_NAME, type=str)
    return parser


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not thresholds:
        raise ValueError("--thresholds must contain at least one numeric threshold")
    return thresholds


def _read_first(data, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in data.files:
            return np.asarray(data[key], dtype=np.float32)
    return None


def _ensure_t5(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError(f"{name} must be [T,5] or [1,T,5], got {array.shape}")
    return array


def _slice_realtime_target_window(
    target: np.ndarray,
    signals: dict[str, np.ndarray],
    *,
    target_start: int,
    target_length: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if target.shape[0] <= int(target_start):
        return target, signals
    target_slice = slice(int(target_start), int(target_start) + int(target_length))
    return target[target_slice], {name: values[target_slice] for name, values in signals.items()}


def read_stationary_arrays(path: Path, *, schema_name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    schema = get_schema_spec(schema_name)
    with np.load(path, allow_pickle=False) as data:
        target = _read_first(data, ("reference_stationary_prob_5", "target_stationary_prob_5", "stationary_prob_5"))
        if target is None:
            reference = np.asarray(data["reference_features_raw"], dtype=np.float32)
            if reference.ndim == 3:
                reference = reference[0]
            target = reference[:, schema.stationary_prob_slice()]
        signals: dict[str, np.ndarray] = {}
        feature = _read_first(data, SIGNAL_KEYS["feature_channel"])
        if feature is None and "reconstructed_features_raw" in data.files:
            reconstructed = np.asarray(data["reconstructed_features_raw"], dtype=np.float32)
            if reconstructed.ndim == 3:
                reconstructed = reconstructed[0]
            feature = reconstructed[:, schema.stationary_prob_slice()]
        if feature is not None:
            signals["feature_channel"] = _ensure_t5("feature_channel", feature)
        for signal_name, keys in SIGNAL_KEYS.items():
            if signal_name == "feature_channel":
                continue
            value = _read_first(data, keys)
            if value is not None:
                signals[signal_name] = _ensure_t5(signal_name, value)
    target = _ensure_t5("target", target)
    return _slice_realtime_target_window(
        target,
        signals,
        target_start=schema.target_start,
        target_length=schema.target_length,
    )


def _flatten_aggregate_rows(path: Path, signal_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for threshold, payload in metrics["thresholds"].items():
        row = {"clip": path.name, "signal_source": signal_name, "threshold": threshold}
        row.update(payload["aggregate"])
        rows.append(row)
    return rows


def _flatten_joint_rows(path: Path, signal_name: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for threshold, payload in metrics["thresholds"].items():
        for joint_name, values in payload["per_joint"].items():
            row = {"clip": path.name, "signal_source": signal_name, "threshold": threshold, "joint": joint_name}
            row.update(values)
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_signal(signal_metrics: list[dict[str, Any]], thresholds: tuple[float, ...]) -> dict[str, Any]:
    summary = {"clip_count": len(signal_metrics), "thresholds": {}}
    for threshold in thresholds:
        key = f"{float(threshold):g}"
        aggregates = [item["thresholds"][key]["aggregate"] for item in signal_metrics]
        summary["thresholds"][key] = {
            "aggregate": {
                metric: float(np.mean([float(values[metric]) for values in aggregates]))
                for metric in (
                    "precision",
                    "recall",
                    "f1",
                    "false_lock_rate",
                    "missed_lock_rate",
                    "prob_jitter_mean_abs",
                    "clamp_pre_out_of_bounds_ratio",
                    "move_to_static_lag_mean_frames",
                    "static_to_move_lag_mean_frames",
                )
            }
        }
    return summary


def evaluate_directory(
    *,
    input_dir: Path | str,
    output_dir: Path | str,
    thresholds: tuple[float, ...] = (0.5, 0.7),
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> dict[str, Path]:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_signal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    clip_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.npz")):
        target, signals = read_stationary_arrays(path, schema_name=schema_name)
        for signal_name, predicted in signals.items():
            metrics = compute_stationary_signal_metrics(target, predicted, thresholds=thresholds)
            per_signal[signal_name].append(metrics)
            clip_rows.extend(_flatten_aggregate_rows(path, signal_name, metrics))
            joint_rows.extend(_flatten_joint_rows(path, signal_name, metrics))
    if not per_signal:
        raise RuntimeError(f"No stationary signal prediction npz files found under {input_dir}")

    summary = {
        "schema_name": schema_name,
        "joint_names": list(STATIONARY_JOINT_NAMES),
        "signals": {
            signal_name: _summarize_signal(metrics, thresholds)
            for signal_name, metrics in sorted(per_signal.items())
        },
    }
    summary_path = output_dir / "metrics_summary.json"
    per_clip_path = output_dir / "per_clip_metrics.csv"
    per_joint_path = output_dir / "per_joint_metrics.csv"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(per_clip_path, clip_rows)
    _write_csv(per_joint_path, joint_rows)
    return {"summary_path": summary_path, "per_clip_path": per_clip_path, "per_joint_path": per_joint_path}


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_arg_parser().parse_args(argv)
    result = evaluate_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        thresholds=parse_thresholds(args.thresholds),
        schema_name=args.schema,
    )
    print(f"[evaluate_stationary_signal_source] summary={result['summary_path']}")
    return result


if __name__ == "__main__":
    main()
