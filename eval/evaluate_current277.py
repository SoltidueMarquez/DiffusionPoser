from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_loaders.sensor_masking import MODEL_INPUT_DIM, SENSOR_LABEL_DIM, X277_FEATURE_DIM


FEATURE_GROUPS = {
    "body_rot_l2": (slice(0, 144), 24, 6, "l2"),
    "body_vel_l2": (slice(144, 216), 24, 3, "l2"),
    "root_delta_l2": (slice(270, 272), 1, 2, "l2"),
    "root_yaw_abs_degree": (slice(272, 273), 1, 1, "abs"),
    "contact_l1": (slice(273, 277), 4, 1, "abs"),
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate current277 streaming reconstruction outputs.")
    parser.add_argument("--input_dir", required=True, type=str, help="Directory containing stream_outputs.npz files.")
    parser.add_argument("--output_json", default="", type=str, help="Optional summary JSON path.")
    return parser


def iter_output_files(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir]
    return sorted(input_dir.rglob("stream_outputs.npz"))


def scenario_masks(sensor_missing_labels: np.ndarray, valid_frame_mask: np.ndarray) -> dict[str, np.ndarray]:
    missing_count = sensor_missing_labels.sum(axis=1)
    valid = valid_frame_mask.astype(bool)
    previous_missing = np.zeros_like(missing_count, dtype=bool)
    previous_missing[1:] = missing_count[:-1] > 0
    return {
        "normal_online": valid & (missing_count == 0),
        "partial_dropout": valid & (missing_count > 0) & (missing_count < SENSOR_LABEL_DIM),
        "all_dropout": valid & (missing_count == SENSOR_LABEL_DIM),
        "recovery_transition": valid & (missing_count == 0) & previous_missing,
    }


def target_frame_mask_from_inpaint_mask(inpaint_mask: np.ndarray, valid_frame_mask: np.ndarray) -> np.ndarray:
    """只把真正要求模型重建的帧纳入主指标，避免条件帧稀释误差。"""

    target_frame_mask = inpaint_mask[:, :X277_FEATURE_DIM].any(axis=1)
    return target_frame_mask & valid_frame_mask.astype(bool)


def compute_group_metric(reference: np.ndarray, reconstructed: np.ndarray, frame_mask: np.ndarray, group_spec: tuple) -> float:
    feature_slice, item_count, item_dim, metric = group_spec
    if not frame_mask.any():
        return float("nan")
    diff = reconstructed[frame_mask, feature_slice] - reference[frame_mask, feature_slice]
    diff = diff.reshape(diff.shape[0], item_count, item_dim)
    if metric == "l2":
        values = np.linalg.norm(diff, axis=-1)
    elif metric == "abs":
        values = np.abs(diff)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return float(values.mean())


def compute_contact_accuracy(reference: np.ndarray, reconstructed: np.ndarray, frame_mask: np.ndarray) -> float:
    if not frame_mask.any():
        return float("nan")
    ref_contact = reference[frame_mask, 273:277] >= 0.5
    rec_contact = reconstructed[frame_mask, 273:277] >= 0.5
    return float((ref_contact == rec_contact).mean())


def evaluate_file(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        reference = data["reference_motion"].astype(np.float64)
        reconstructed = data["reconstructed_motion"].astype(np.float64)
        sensor_missing_labels = data["sensor_missing_labels"].astype(bool)
        valid_frame_mask = data["valid_frame_mask"].astype(bool)
        inpaint_mask = data["inpaint_mask"].astype(bool)

    if reference.ndim != 2 or reference.shape[1] != X277_FEATURE_DIM:
        raise ValueError(f"{path} reference_motion must be [T, 277], got {reference.shape}")
    if reconstructed.shape != reference.shape:
        raise ValueError(f"{path} reconstructed_motion shape mismatch: {reconstructed.shape} vs {reference.shape}")
    if sensor_missing_labels.shape != (reference.shape[0], SENSOR_LABEL_DIM):
        raise ValueError(f"{path} sensor_missing_labels must be [T, 6], got {sensor_missing_labels.shape}")
    if inpaint_mask.shape[0] != reference.shape[0] or inpaint_mask.shape[1] not in {X277_FEATURE_DIM, MODEL_INPUT_DIM}:
        raise ValueError(f"{path} inpaint_mask must be [T, 277] or [T, 283], got {inpaint_mask.shape}")

    target_frame_mask = target_frame_mask_from_inpaint_mask(
        inpaint_mask=inpaint_mask,
        valid_frame_mask=valid_frame_mask,
    )
    scenarios = scenario_masks(sensor_missing_labels=sensor_missing_labels, valid_frame_mask=valid_frame_mask)
    result = {
        "path": str(path),
        "valid_frames": int(valid_frame_mask.sum()),
        "target_frames": int(target_frame_mask.sum()),
        "context_frames": int((valid_frame_mask & ~target_frame_mask).sum()),
        "scenarios": {},
    }
    for scenario_name, scenario_frame_mask in scenarios.items():
        frame_mask = scenario_frame_mask & target_frame_mask
        metrics = {
            metric_name: compute_group_metric(reference, reconstructed, frame_mask, group_spec)
            for metric_name, group_spec in FEATURE_GROUPS.items()
        }
        metrics["contact_accuracy"] = compute_contact_accuracy(reference, reconstructed, frame_mask)
        metrics["frames"] = int(frame_mask.sum())
        result["scenarios"][scenario_name] = metrics
    return result


def aggregate_results(results: list[dict]) -> dict:
    scenario_names = ("normal_online", "partial_dropout", "all_dropout", "recovery_transition")
    metric_names = tuple(FEATURE_GROUPS.keys()) + ("contact_accuracy",)
    summary = {
        "sample_count": len(results),
        "valid_frames": int(sum(item["valid_frames"] for item in results)),
        "target_frames": int(sum(item["target_frames"] for item in results)),
        "context_frames": int(sum(item["context_frames"] for item in results)),
        "scenarios": {},
    }
    for scenario_name in scenario_names:
        scenario_summary = {"frames": int(sum(item["scenarios"][scenario_name]["frames"] for item in results))}
        for metric_name in metric_names:
            values = [
                item["scenarios"][scenario_name][metric_name]
                for item in results
                if not np.isnan(item["scenarios"][scenario_name][metric_name])
            ]
            scenario_summary[metric_name] = float(np.mean(values)) if values else float("nan")
        summary["scenarios"][scenario_name] = scenario_summary
    return summary


def to_jsonable(value):
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> dict:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    input_dir = Path(args.input_dir)
    files = iter_output_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No stream_outputs.npz files found under {input_dir}")
    results = [evaluate_file(path) for path in files]
    summary = aggregate_results(results)
    payload = {"summary": summary, "samples": results}
    output_json = Path(args.output_json) if args.output_json else input_dir / "current277_eval_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    json_payload = to_jsonable(payload)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
    print(json.dumps(to_jsonable(summary), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return payload


if __name__ == "__main__":
    main()
