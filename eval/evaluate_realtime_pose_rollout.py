from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate realtime_pose rollout result npz files.")
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_json", default="", type=str)
    return parser


def evaluate_rollout_file(path: Path) -> dict[str, float | int | str | list[float]]:
    with np.load(path, allow_pickle=True) as data:
        reference_features = np.asarray(data["reference_features_raw"], dtype=np.float32)
        predicted_features = np.asarray(data["predicted_features_raw"], dtype=np.float32)
        reference_joints = np.asarray(data["reference_joints_world"], dtype=np.float32)
        predicted_joints = np.asarray(data["predicted_joints_world"], dtype=np.float32)
        root_yaw_reference = np.asarray(data["root_yaw_reference"], dtype=np.float32)
        root_yaw_predicted = np.asarray(data["root_yaw_predicted"], dtype=np.float32)
        eval_frame_mask = read_eval_frame_mask(data, reference_features.shape[:2])
        warmup_frames = int(np.asarray(data["warmup_frames"]).item()) if "warmup_frames" in data.files else 0

    if reference_features.shape != predicted_features.shape:
        raise ValueError(f"{path} feature shape 不匹配：{reference_features.shape} vs {predicted_features.shape}")
    if reference_joints.shape != predicted_joints.shape:
        raise ValueError(f"{path} joints shape 不匹配：{reference_joints.shape} vs {predicted_joints.shape}")
    if not eval_frame_mask.any():
        raise ValueError(f"{path} eval_frame_mask 没有可评估帧。")

    joint_error = np.linalg.norm(predicted_joints - reference_joints, axis=-1)
    mpjpe_by_time = masked_time_mean(joint_error, eval_frame_mask)
    yaw_error = np.abs(wrap_radians(root_yaw_predicted - root_yaw_reference))
    yaw_by_time = masked_time_mean(yaw_error[..., None], eval_frame_mask)
    temporal_jitter = np.diff(predicted_features, n=2, axis=1)
    jitter_mask = eval_frame_mask[:, 2:] & eval_frame_mask[:, 1:-1] & eval_frame_mask[:, :-2]
    temporal_jitter_value = float(np.mean(np.abs(temporal_jitter[jitter_mask]))) if jitter_mask.any() else 0.0
    return {
        "path": str(path),
        "batch_size": int(reference_features.shape[0]),
        "frames": int(reference_features.shape[1]),
        "evaluated_frames": int(eval_frame_mask.sum()),
        "warmup_frames": int(warmup_frames),
        "mpjpe_mean": float(np.mean(joint_error[eval_frame_mask])),
        "mpjpe_final": float(mpjpe_by_time[-1]),
        "mpjpe_by_time": [float(value) for value in mpjpe_by_time.tolist()],
        "yaw_error_mean": float(np.mean(yaw_error[eval_frame_mask])),
        "yaw_error_final": float(yaw_by_time[-1]),
        "yaw_drift_by_time": [float(value) for value in yaw_by_time.tolist()],
        "foot_skating_left": 0.0,
        "foot_skating_right": 0.0,
        "ground_penetration_ratio": 0.0,
        "tracker_reprojection_pos_error": 0.0,
        "temporal_jitter": temporal_jitter_value,
    }


def read_eval_frame_mask(data: np.lib.npyio.NpzFile, shape: tuple[int, int]) -> np.ndarray:
    if "eval_frame_mask" not in data.files:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(data["eval_frame_mask"], dtype=bool)
    if mask.shape == (shape[1],):
        mask = np.repeat(mask[None], shape[0], axis=0)
    if mask.shape != shape:
        raise ValueError(f"eval_frame_mask 应为 {shape} 或 [{shape[1]}]，实际为 {mask.shape}")
    return mask


def masked_time_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    result = []
    for frame_index in range(values.shape[1]):
        frame_mask = mask[:, frame_index]
        if not frame_mask.any():
            continue
        result.append(float(np.mean(values[:, frame_index][frame_mask])))
    return np.asarray(result, dtype=np.float32)


def wrap_radians(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def summarize(results: list[dict[str, float | int | str | list[float]]]) -> dict[str, float | int]:
    if not results:
        raise RuntimeError("没有可评估的 rollout npz 文件。")
    metric_names = (
        "mpjpe_mean",
        "mpjpe_final",
        "yaw_error_mean",
        "yaw_error_final",
        "foot_skating_left",
        "foot_skating_right",
        "ground_penetration_ratio",
        "tracker_reprojection_pos_error",
        "temporal_jitter",
    )
    summary: dict[str, float | int] = {
        "file_count": len(results),
        "frames": int(sum(int(item["frames"]) for item in results)),
        "evaluated_frames": int(sum(int(item.get("evaluated_frames", item["frames"])) for item in results)),
        "warmup_frames": int(sum(int(item.get("warmup_frames", 0)) for item in results)),
    }
    for name in metric_names:
        summary[name] = float(np.mean([float(item[name]) for item in results]))
    return summary


def main(argv: list[str] | None = None) -> dict[str, float | int]:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    results = [evaluate_rollout_file(path) for path in sorted(input_dir.rglob("rollout_result*.npz"))]
    if not results:
        results = [evaluate_rollout_file(path) for path in sorted(input_dir.rglob("*.npz"))]
    summary = summarize(results)
    output_json = Path(args.output_json).resolve() if args.output_json else input_dir / "rollout_eval_summary.json"
    with output_json.open("w", encoding="utf-8") as file:
        json.dump({"summary": summary, "files": results}, file, indent=2, ensure_ascii=False)
    print(f"[evaluate_realtime_pose_rollout] wrote {output_json}")
    return summary


if __name__ == "__main__":
    main()
