from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_kinematics import TRACKER_JOINT_INDICES, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from eval.stationary_signal_metrics import compute_stationary_signal_metrics


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
        root_pos_reference = read_optional_array(data, "root_pos_world_reference")
        root_pos_predicted = read_optional_array(data, "root_pos_world_predicted")
        tracker_pos_world = read_optional_array(data, "tracker_pos_world")
        tracker_rot_world_6d = read_optional_array(data, "tracker_rot_world_6d")
        predicted_joint_rot_world = read_optional_array(data, "predicted_joint_rot_world")
        sensor_valid = read_optional_array(data, "sensor_valid")
        timestamps = read_optional_array(data, "timestamp_seconds")
        floor_y = read_optional_array(data, "floor_y")
        root_source = read_optional_array(data, "root_source")
        reconnect_alpha = read_optional_array(data, "reconnect_alpha")
        timing_arrays = {
            name: read_optional_array(data, name)
            for name in ("codec_fk_elapsed_ms", "ddim_elapsed_ms", "resolver_elapsed_ms", "end_to_end_elapsed_ms")
        }

    if reference_features.shape != predicted_features.shape:
        raise ValueError(f"{path} feature shape 不匹配：{reference_features.shape} vs {predicted_features.shape}")
    if reference_joints.shape != predicted_joints.shape:
        raise ValueError(f"{path} joints shape 不匹配：{reference_joints.shape} vs {predicted_joints.shape}")
    if not eval_frame_mask.any():
        raise ValueError(f"{path} eval_frame_mask 没有可评估帧。")

    joint_error = np.linalg.norm(predicted_joints - reference_joints, axis=-1)
    reference_relative = reference_joints - reference_joints[..., 0:1, :]
    predicted_relative = predicted_joints - predicted_joints[..., 0:1, :]
    root_relative_error = np.linalg.norm(predicted_relative - reference_relative, axis=-1)
    mpjpe_by_time = masked_time_mean(joint_error, eval_frame_mask)
    yaw_error = np.abs(wrap_radians(root_yaw_predicted - root_yaw_reference))
    yaw_by_time = masked_time_mean(yaw_error[..., None], eval_frame_mask)
    temporal_jitter = np.diff(predicted_features, n=2, axis=1)
    jitter_mask = eval_frame_mask[:, 2:] & eval_frame_mask[:, 1:-1] & eval_frame_mask[:, :-2]
    temporal_jitter_value = float(np.mean(np.abs(temporal_jitter[jitter_mask]))) if jitter_mask.any() else 0.0
    result = {
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
        "root_relative_mpjpe_mm": float(np.mean(root_relative_error[eval_frame_mask]) * 1000.0),
        "foot_skating_left": 0.0,
        "foot_skating_right": 0.0,
        "ground_penetration_ratio": 0.0,
        "tracker_reprojection_pos_error": 0.0,
        "tracker_reprojection_pos_mean_cm": 0.0,
        "tracker_reprojection_pos_p95_cm": 0.0,
        "tracker_reprojection_rot_mean_deg": 0.0,
        "tracker_reprojection_rot_p95_deg": 0.0,
        "temporal_jitter": temporal_jitter_value,
    }
    if root_pos_reference is not None and root_pos_predicted is not None:
        root_error = np.linalg.norm(
            np.asarray(root_pos_predicted)[..., [0, 2]] - np.asarray(root_pos_reference)[..., [0, 2]],
            axis=-1,
        )
        result["root_xz_mean_error_cm"] = float(np.mean(root_error[eval_frame_mask]) * 100.0)
        result["root_xz_final_error_cm"] = float(masked_time_mean(root_error[..., None], eval_frame_mask)[-1] * 100.0)
        result["root_drift_cm_per_min"] = root_drift_cm_per_minute(
            root_error=root_error,
            mask=eval_frame_mask,
            timestamps=timestamps,
            root_source=root_source,
        )
        jump_xz, jump_yaw = reconnect_peak_jump(
            root_pos=np.asarray(root_pos_predicted),
            root_yaw=root_yaw_predicted,
            root_source=root_source,
            reconnect_alpha=reconnect_alpha,
        )
        result["reconnect_peak_jump_xz_cm"] = jump_xz * 100.0
        result["reconnect_peak_jump_yaw_deg"] = jump_yaw * 180.0 / np.pi
        recovery_seconds, recovery_frames = recovery_time(
            root_error=root_error,
            yaw_error=yaw_error,
            reconnect_alpha=reconnect_alpha,
            timestamps=timestamps,
        )
        result["recovery_seconds"] = recovery_seconds
        result["recovery_frames"] = recovery_frames
    else:
        result.update(
            {
                "root_xz_mean_error_cm": 0.0,
                "root_xz_final_error_cm": 0.0,
                "root_drift_cm_per_min": 0.0,
                "reconnect_peak_jump_xz_cm": 0.0,
                "reconnect_peak_jump_yaw_deg": 0.0,
                "recovery_seconds": 0.0,
                "recovery_frames": 0,
            }
        )

    if tracker_pos_world is not None and sensor_valid is not None:
        predicted_tracker_pos = predicted_joints[..., np.asarray(TRACKER_JOINT_INDICES), :]
        valid = np.asarray(sensor_valid, dtype=bool) & eval_frame_mask[..., None]
        if valid.shape == predicted_tracker_pos.shape[:-1] and valid.any():
            reprojection = np.linalg.norm(predicted_tracker_pos - np.asarray(tracker_pos_world), axis=-1)
            values = reprojection[valid]
            result["tracker_reprojection_pos_error"] = float(np.mean(values))
            result["tracker_reprojection_pos_mean_cm"] = float(np.mean(values) * 100.0)
            result["tracker_reprojection_pos_p95_cm"] = float(np.percentile(values, 95) * 100.0)
            if tracker_rot_world_6d is not None and predicted_joint_rot_world is not None:
                predicted_tracker_rot = np.asarray(predicted_joint_rot_world)[
                    ..., np.asarray(TRACKER_JOINT_INDICES), :, :
                ]
                observed_tracker_rot = rotation_6d_to_matrix_np(np.asarray(tracker_rot_world_6d))
                relative = np.swapaxes(observed_tracker_rot, -1, -2) @ predicted_tracker_rot
                cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
                rotation_error_deg = np.rad2deg(np.arccos(cosine))[valid]
                result["tracker_reprojection_rot_mean_deg"] = float(np.mean(rotation_error_deg))
                result["tracker_reprojection_rot_p95_deg"] = float(np.percentile(rotation_error_deg, 95))

    schema = get_schema_spec(DEFAULT_REALTIME_POSE_SCHEMA_NAME)
    if reference_features.shape[-1] == schema.feature_dim and schema.supports_stationary_prob:
        reference_stationary = reference_features[..., schema.stationary_prob_slice()]
        predicted_stationary = predicted_features[..., schema.stationary_prob_slice()]
        stationary = reference_stationary >= 0.5
        selected_reference = reference_stationary[eval_frame_mask]
        selected_prediction = predicted_stationary[eval_frame_mask]
        if selected_reference.size and selected_prediction.size:
            stationary_metrics = compute_stationary_signal_metrics(
                selected_reference,
                selected_prediction,
                thresholds=(0.5,),
            )["thresholds"]["0.5"]["aggregate"]
            result["stationary_f1"] = float(stationary_metrics["f1"])
            result["stationary_false_lock_rate"] = float(stationary_metrics["false_lock_rate"])
            result["stationary_missed_lock_rate"] = float(stationary_metrics["missed_lock_rate"])
            result["stationary_jitter"] = float(stationary_metrics["prob_jitter_mean_abs"])
            result["stationary_clamp_pre_out_of_bounds_ratio"] = float(
                stationary_metrics["clamp_pre_out_of_bounds_ratio"]
            )
        floor = np.zeros(reference_features.shape[:2], dtype=np.float32) if floor_y is None else np.asarray(floor_y)
        result["foot_skating_left"] = foot_slide_m_per_second(
            predicted_joints[..., 10, :], stationary[..., 1], floor, timestamps
        )
        result["foot_skating_right"] = foot_slide_m_per_second(
            predicted_joints[..., 11, :], stationary[..., 2], floor, timestamps
        )

    for name, values in timing_arrays.items():
        if values is None:
            continue
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            prefix = name.replace("_elapsed_ms", "")
            result[f"{prefix}_p50_ms"] = float(np.percentile(finite, 50))
            result[f"{prefix}_p95_ms"] = float(np.percentile(finite, 95))
            if name == "end_to_end_elapsed_ms" and float(np.mean(finite)) > 0:
                result["fps"] = float(1000.0 / np.mean(finite))
    return result


def read_optional_array(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray | None:
    return np.asarray(data[name]) if name in data.files else None


def root_drift_cm_per_minute(
    root_error: np.ndarray,
    mask: np.ndarray,
    timestamps: np.ndarray | None,
    root_source: np.ndarray | None = None,
) -> float:
    errors = np.asarray(root_error, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if root_source is not None:
        source = np.asarray(root_source)
        if source.shape == valid.shape:
            valid &= source != 3  # RootSource.RESET is excluded from drift.
    if timestamps is None:
        time = np.arange(errors.shape[1], dtype=np.float64) / 60.0
        time = np.broadcast_to(time[None], errors.shape)
    else:
        time = np.asarray(timestamps, dtype=np.float64)
        if time.ndim == 1:
            time = np.broadcast_to(time[None], errors.shape)
    x = time[valid]
    y = errors[valid]
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return 0.0
    x = x - float(np.mean(x))
    denominator = float(np.dot(x, x))
    if denominator <= 1e-16:
        return 0.0
    # Closed-form one-variable least squares avoids a LAPACK/MKL dependency in
    # the per-sequence evaluation hot path.
    y = y - float(np.mean(y))
    slope_m_per_second = float(np.dot(x, y) / denominator)
    return slope_m_per_second * 100.0 * 60.0


def reconnect_peak_jump(
    root_pos: np.ndarray,
    root_yaw: np.ndarray,
    root_source: np.ndarray | None,
    reconnect_alpha: np.ndarray | None,
) -> tuple[float, float]:
    if root_pos.shape[1] < 2:
        return 0.0, 0.0
    active = np.zeros(root_pos.shape[:2], dtype=bool)
    if root_source is not None:
        active |= np.asarray(root_source) == 2
    if reconnect_alpha is not None:
        alpha = np.asarray(reconnect_alpha)
        active |= (alpha > 0.0) & (alpha < 1.0)
    pair_mask = active[:, 1:] | active[:, :-1]
    if not pair_mask.any():
        return 0.0, 0.0
    position_jump = np.linalg.norm(np.diff(root_pos[..., [0, 2]], axis=1), axis=-1)
    yaw_jump = np.abs(wrap_radians(np.diff(root_yaw, axis=1)))
    return float(np.max(position_jump[pair_mask])), float(np.max(yaw_jump[pair_mask]))


def recovery_time(
    root_error: np.ndarray,
    yaw_error: np.ndarray,
    reconnect_alpha: np.ndarray | None,
    timestamps: np.ndarray | None,
) -> tuple[float, int]:
    if reconnect_alpha is None:
        return 0.0, 0
    alpha = np.asarray(reconnect_alpha)
    recovered = (root_error < 0.05) & (yaw_error < np.deg2rad(5.0))
    durations = []
    frames = []
    for batch_index in range(alpha.shape[0]):
        starts = np.where((alpha[batch_index, 1:] > 0) & (alpha[batch_index, :-1] <= 0))[0] + 1
        for start in starts:
            for frame in range(start, recovered.shape[1] - 4):
                if recovered[batch_index, frame : frame + 5].all():
                    frame_count = int(frame - start)
                    frames.append(frame_count)
                    if timestamps is None:
                        durations.append(frame_count / 60.0)
                    else:
                        time = np.asarray(timestamps)
                        values = time if time.ndim == 1 else time[batch_index]
                        durations.append(float(values[frame] - values[start]))
                    break
    return (float(np.mean(durations)), int(round(float(np.mean(frames))))) if frames else (0.0, 0)


def foot_slide_m_per_second(
    foot_positions: np.ndarray,
    contact: np.ndarray,
    floor_y: np.ndarray,
    timestamps: np.ndarray | None,
) -> float:
    foot = np.asarray(foot_positions, dtype=np.float64)
    active = np.asarray(contact, dtype=bool) & ((foot[..., 1] - np.asarray(floor_y)) <= 0.05)
    if foot.shape[1] < 2:
        return 0.0
    pair = active[:, 1:] & active[:, :-1]
    if not pair.any():
        return 0.0
    distance = np.linalg.norm(np.diff(foot[..., [0, 2]], axis=1), axis=-1)
    if timestamps is None:
        duration = np.full(distance.shape, 1.0 / 60.0, dtype=np.float64)
    else:
        time = np.asarray(timestamps, dtype=np.float64)
        if time.ndim == 1:
            time = np.broadcast_to(time[None], foot.shape[:2])
        duration = np.maximum(np.diff(time, axis=1), 1e-6)
    return float(np.sum(distance[pair]) / np.sum(duration[pair]))


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
        "root_relative_mpjpe_mm",
        "yaw_error_mean",
        "yaw_error_final",
        "foot_skating_left",
        "foot_skating_right",
        "ground_penetration_ratio",
        "tracker_reprojection_pos_error",
        "tracker_reprojection_pos_mean_cm",
        "tracker_reprojection_pos_p95_cm",
        "tracker_reprojection_rot_mean_deg",
        "tracker_reprojection_rot_p95_deg",
        "temporal_jitter",
        "root_xz_mean_error_cm",
        "root_xz_final_error_cm",
        "root_drift_cm_per_min",
        "reconnect_peak_jump_xz_cm",
        "reconnect_peak_jump_yaw_deg",
        "recovery_seconds",
        "recovery_frames",
        "stationary_f1",
        "stationary_false_lock_rate",
        "stationary_missed_lock_rate",
        "stationary_jitter",
        "stationary_clamp_pre_out_of_bounds_ratio",
        "codec_fk_p50_ms",
        "codec_fk_p95_ms",
        "ddim_p50_ms",
        "ddim_p95_ms",
        "resolver_p50_ms",
        "resolver_p95_ms",
        "end_to_end_p50_ms",
        "end_to_end_p95_ms",
        "fps",
    )
    summary: dict[str, float | int] = {
        "file_count": len(results),
        "frames": int(sum(int(item["frames"]) for item in results)),
        "evaluated_frames": int(sum(int(item.get("evaluated_frames", item["frames"])) for item in results)),
        "warmup_frames": int(sum(int(item.get("warmup_frames", 0)) for item in results)),
    }
    for name in metric_names:
        summary[name] = float(np.mean([float(item.get(name, 0.0)) for item in results]))
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
