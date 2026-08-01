from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from data_loaders.sensor_masking import (
    FOOT_TRACKER_INDICES,
    HAND_TRACKER_INDICES,
    JOINT_GLOBAL_ROTATION_DIM,
    ROOT_YAW_RELATIVE_START,
    TRACKER_TO_JOINT,
)


PAPER_JOINT_SLICE = slice(0, 22)
PAPER_BODY_ROTATION_SLICE = slice(1, 22)
MISSING_AGE_BUCKETS = (
    ("1-5", 1, 5),
    ("6-15", 6, 15),
    ("16-30", 16, 30),
    ("31-60", 31, 60),
)
REQUIRED_RESULT_FIELDS = {
    "reference_body_local_delta_6d",
    "predicted_body_local_delta_6d",
    "reference_joints_world",
    "predicted_joints_world",
    "reference_root_position_world",
    "predicted_root_position_world",
    "reference_root_yaw_world",
    "predicted_root_yaw_world",
    "reference_hip_height",
    "predicted_hip_height",
    "reference_target_raw",
    "reconstructed_target_raw",
    "known_mask",
    "tracker_pos_world",
    "configured",
    "measured_valid",
    "missing_age",
    "scenario",
    "eval_frame_mask",
    "fps",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估 140D 动态 Tracker 姿态补全结果。")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_json", default="")
    return parser


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(first, -1, -2) @ second
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _axis_angle_component_error_deg(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """返回 `[N,T,21,3]` 的轴角分量误差，口径与论文公开评估代码一致。"""

    leading = predicted.shape[:-3]
    predicted_axis_angle = Rotation.from_matrix(predicted.reshape(-1, 3, 3)).as_rotvec().reshape(
        *leading, predicted.shape[-3], 3
    )
    reference_axis_angle = Rotation.from_matrix(reference.reshape(-1, 3, 3)).as_rotvec().reshape(
        *leading, reference.shape[-3], 3
    )
    difference = predicted_axis_angle - reference_axis_angle
    difference = (difference + np.pi) % (2.0 * np.pi) - np.pi
    return np.degrees(np.abs(difference))


def _mean_or_none(values: np.ndarray, mask: np.ndarray, scale: float = 1.0) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    selected = array[np.broadcast_to(np.asarray(mask, dtype=bool), array.shape)]
    if selected.size == 0:
        return None
    return float(selected.mean() * scale)


def _mean_or_zero(values: np.ndarray, mask: np.ndarray, scale: float = 1.0) -> float:
    value = _mean_or_none(values, mask, scale=scale)
    return 0.0 if value is None else value


def _sum_count(values: np.ndarray, mask: np.ndarray, scale: float = 1.0) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    selected = array[np.broadcast_to(np.asarray(mask, dtype=bool), array.shape)]
    return {
        "sum": float(selected.sum() * scale),
        "count": int(selected.size),
    }


def _metric_from_stats(stats: dict[str, float | int], empty: float | None = 0.0) -> float | None:
    count = int(stats["count"])
    return float(stats["sum"]) / count if count else empty


def _load_result(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_RESULT_FIELDS.difference(data.files))
        if missing:
            raise KeyError(f"{path} 缺少 140D 评估字段：{missing}")
        values = {key: np.asarray(data[key]) for key in REQUIRED_RESULT_FIELDS}
        values["known_rotation_max_error"] = np.asarray(
            data["known_rotation_max_error"] if "known_rotation_max_error" in data.files else 0.0,
            dtype=np.float64,
        )
    return values


def _group_metrics(
    frame_select: np.ndarray,
    pair_select: np.ndarray,
    triple_select: np.ndarray,
    mpjre_per_frame: np.ndarray,
    mpjpe_per_frame_m: np.ndarray,
    mpjve_per_frame_m_s: np.ndarray,
    mpjae_per_frame_m_s2: np.ndarray,
    unknown_rotation_per_frame: np.ndarray,
    root_yaw_error: np.ndarray,
    root_unknown: np.ndarray,
    root_xz_error: np.ndarray,
    hip_height_error: np.ndarray,
    tracker_position_per_frame: np.ndarray,
) -> dict[str, float | int | None]:
    root_yaw_select = frame_select & root_unknown
    return {
        "samples": int(frame_select.sum()),
        "velocity_pairs": int(pair_select.sum()),
        "acceleration_triplets": int(triple_select.sum()),
        "root_yaw_samples": int(root_yaw_select.sum()),
        "mpjre_deg": _mean_or_zero(mpjre_per_frame, frame_select),
        "mpjpe_cm": _mean_or_zero(mpjpe_per_frame_m, frame_select, scale=100.0),
        "mpjve_cm_s": _mean_or_none(mpjve_per_frame_m_s, pair_select, scale=100.0),
        "mpjae_cm_s2": _mean_or_none(mpjae_per_frame_m_s2, triple_select, scale=100.0),
        "unknown_rotation_deg": _mean_or_zero(unknown_rotation_per_frame, frame_select, scale=180.0 / np.pi),
        "root_yaw_error_deg": _mean_or_zero(root_yaw_error, root_yaw_select, scale=180.0 / np.pi),
        "root_xz_error_m": _mean_or_zero(root_xz_error, frame_select),
        "hip_height_error_m": _mean_or_zero(hip_height_error, frame_select),
        "tracker_position_error_m": _mean_or_zero(tracker_position_per_frame, frame_select),
    }


def evaluate_file(path: Path) -> dict[str, object]:
    values = _load_result(path)
    reference = values["reference_target_raw"].astype(np.float64)
    predicted = values["reconstructed_target_raw"].astype(np.float64)
    if reference.ndim != 3 or reference.shape[-1] != 140 or predicted.shape != reference.shape:
        raise ValueError(f"{path} target 必须是匹配的 [N,T,140]，实际为 {reference.shape}/{predicted.shape}")
    sequence_count, steps = reference.shape[:2]
    frame_shape = (sequence_count, steps)
    eval_mask = values["eval_frame_mask"].reshape(frame_shape).astype(bool)
    fps = float(np.asarray(values["fps"]).reshape(()))

    reference_local = rotation_6d_to_matrix_np(
        values["reference_body_local_delta_6d"].reshape(sequence_count, steps, 24, 6)
    )
    predicted_local = rotation_6d_to_matrix_np(
        values["predicted_body_local_delta_6d"].reshape(sequence_count, steps, 24, 6)
    )
    mpjre_components = _axis_angle_component_error_deg(
        predicted_local[:, :, PAPER_BODY_ROTATION_SLICE],
        reference_local[:, :, PAPER_BODY_ROTATION_SLICE],
    )
    mpjre_per_frame = mpjre_components.mean(axis=(-1, -2))

    reference_joints = values["reference_joints_world"].reshape(sequence_count, steps, 24, 3).astype(np.float64)
    predicted_joints = values["predicted_joints_world"].reshape(sequence_count, steps, 24, 3).astype(np.float64)
    joint_distance = np.linalg.norm(
        predicted_joints[:, :, PAPER_JOINT_SLICE] - reference_joints[:, :, PAPER_JOINT_SLICE],
        axis=-1,
    )
    mpjpe_per_frame_m = joint_distance.mean(axis=-1)

    pair_mask = eval_mask[:, 1:] & eval_mask[:, :-1]
    predicted_velocity = np.diff(predicted_joints[:, :, PAPER_JOINT_SLICE], axis=1) * fps
    reference_velocity = np.diff(reference_joints[:, :, PAPER_JOINT_SLICE], axis=1) * fps
    velocity_error = np.linalg.norm(predicted_velocity - reference_velocity, axis=-1)
    mpjve_per_frame_m_s = np.zeros(frame_shape, dtype=np.float64)
    mpjve_per_frame_m_s[:, 1:] = velocity_error.mean(axis=-1)
    pair_frame_mask = np.zeros(frame_shape, dtype=bool)
    pair_frame_mask[:, 1:] = pair_mask

    # 加速度误差需要三个连续有效帧；结果对齐到三元组的最后一帧。
    triple_mask = eval_mask[:, 2:] & eval_mask[:, 1:-1] & eval_mask[:, :-2]
    predicted_acceleration = np.diff(predicted_velocity, axis=1) * fps
    reference_acceleration = np.diff(reference_velocity, axis=1) * fps
    acceleration_error = np.linalg.norm(predicted_acceleration - reference_acceleration, axis=-1)
    mpjae_per_frame_m_s2 = np.zeros(frame_shape, dtype=np.float64)
    mpjae_per_frame_m_s2[:, 2:] = acceleration_error.mean(axis=-1)
    triple_frame_mask = np.zeros(frame_shape, dtype=bool)
    triple_frame_mask[:, 2:] = triple_mask

    known = values["known_mask"].reshape(sequence_count, steps, 140).astype(bool)
    reference_global = rotation_6d_to_matrix_np(
        reference[..., :JOINT_GLOBAL_ROTATION_DIM].reshape(sequence_count, steps, 23, 6)
    )
    predicted_global = rotation_6d_to_matrix_np(
        predicted[..., :JOINT_GLOBAL_ROTATION_DIM].reshape(sequence_count, steps, 23, 6)
    )
    rotation_error = _rotation_angle(predicted_global, reference_global)
    unknown_joint = ~known[..., :JOINT_GLOBAL_ROTATION_DIM].reshape(sequence_count, steps, 23, 6).all(axis=-1)
    unknown_rotation_per_frame = (
        (rotation_error * unknown_joint).sum(axis=-1) / np.maximum(unknown_joint.sum(axis=-1), 1)
    )

    reference_root_yaw = values["reference_root_yaw_world"].reshape(frame_shape).astype(np.float64)
    predicted_root_yaw = values["predicted_root_yaw_world"].reshape(frame_shape).astype(np.float64)
    root_yaw_error = np.abs(
        np.arctan2(
            np.sin(predicted_root_yaw - reference_root_yaw),
            np.cos(predicted_root_yaw - reference_root_yaw),
        )
    )
    root_unknown = ~known[..., ROOT_YAW_RELATIVE_START:].all(axis=-1)

    reference_root = values["reference_root_position_world"].reshape(sequence_count, steps, 3).astype(np.float64)
    predicted_root = values["predicted_root_position_world"].reshape(sequence_count, steps, 3).astype(np.float64)
    root_xz_error = np.linalg.norm(predicted_root[..., [0, 2]] - reference_root[..., [0, 2]], axis=-1)
    hip_height_error = np.abs(
        values["predicted_hip_height"].reshape(frame_shape).astype(np.float64)
        - values["reference_hip_height"].reshape(frame_shape).astype(np.float64)
    )

    tracker_pos = values["tracker_pos_world"].reshape(sequence_count, steps, 6, 3).astype(np.float64)
    measured = values["measured_valid"].reshape(sequence_count, steps, 6).astype(bool)
    tracker_ids = np.asarray([*HAND_TRACKER_INDICES, *FOOT_TRACKER_INDICES], dtype=np.int64)
    joint_ids = np.asarray([TRACKER_TO_JOINT[index] for index in tracker_ids], dtype=np.int64)
    tracker_distance = np.linalg.norm(
        predicted_joints[:, :, joint_ids] - tracker_pos[:, :, tracker_ids],
        axis=-1,
    )
    tracker_valid = measured[:, :, tracker_ids]
    tracker_position_per_frame = (
        (tracker_distance * tracker_valid).sum(axis=-1) / np.maximum(tracker_valid.sum(axis=-1), 1)
    )

    scenarios = values["scenario"].reshape(frame_shape).astype(str)
    missing_age = values["missing_age"].reshape(sequence_count, steps, 6).astype(np.int64)
    max_missing_age = missing_age.max(axis=-1)

    per_scenario: dict[str, dict[str, float | int | None]] = {}
    for scenario in sorted(set(scenarios[eval_mask].tolist())):
        frame_select = eval_mask & (scenarios == scenario)
        pair_select = pair_frame_mask & (scenarios == scenario)
        triple_select = triple_frame_mask & (scenarios == scenario)
        per_scenario[scenario] = _group_metrics(
            frame_select,
            pair_select,
            triple_select,
            mpjre_per_frame,
            mpjpe_per_frame_m,
            mpjve_per_frame_m_s,
            mpjae_per_frame_m_s2,
            unknown_rotation_per_frame,
            root_yaw_error,
            root_unknown,
            root_xz_error,
            hip_height_error,
            tracker_position_per_frame,
        )

    by_missing_age: dict[str, dict[str, float | int | None]] = {}
    for name, lower, upper in MISSING_AGE_BUCKETS:
        frame_select = eval_mask & (max_missing_age >= lower) & (max_missing_age <= upper)
        pair_select = pair_frame_mask & (max_missing_age >= lower) & (max_missing_age <= upper)
        triple_select = triple_frame_mask & (max_missing_age >= lower) & (max_missing_age <= upper)
        by_missing_age[name] = _group_metrics(
            frame_select,
            pair_select,
            triple_select,
            mpjre_per_frame,
            mpjpe_per_frame_m,
            mpjve_per_frame_m_s,
            mpjae_per_frame_m_s2,
            unknown_rotation_per_frame,
            root_yaw_error,
            root_unknown,
            root_xz_error,
            hip_height_error,
            tracker_position_per_frame,
        )

    tracker_observation_mask = eval_mask[..., None] & tracker_valid
    metric_stats = {
        "mpjre_deg": _sum_count(mpjre_components, eval_mask[..., None, None]),
        "mpjpe_cm": _sum_count(joint_distance, eval_mask[..., None], scale=100.0),
        "mpjve_cm_s": _sum_count(velocity_error, pair_mask[..., None], scale=100.0),
        "mpjae_cm_s2": _sum_count(acceleration_error, triple_mask[..., None], scale=100.0),
        "unknown_rotation_deg": _sum_count(unknown_rotation_per_frame, eval_mask, scale=180.0 / np.pi),
        "root_yaw_error_deg": _sum_count(root_yaw_error, eval_mask & root_unknown, scale=180.0 / np.pi),
        "root_xz_error_m": _sum_count(root_xz_error, eval_mask),
        "hip_height_error_m": _sum_count(hip_height_error, eval_mask),
        "tracker_position_error_m": _sum_count(tracker_distance, tracker_observation_mask),
    }
    known_consistency = np.asarray(values["known_rotation_max_error"], dtype=np.float64)
    result: dict[str, object] = {
        "path": str(path),
        "sequences": sequence_count,
        "samples": int(eval_mask.sum()),
        "velocity_pairs": int(pair_mask.sum()),
        "acceleration_triplets": int(triple_mask.sum()),
        "mpjre_deg": _metric_from_stats(metric_stats["mpjre_deg"]),
        "mpjpe_cm": _metric_from_stats(metric_stats["mpjpe_cm"]),
        "mpjve_cm_s": _metric_from_stats(metric_stats["mpjve_cm_s"], empty=None),
        "mpjae_cm_s2": _metric_from_stats(metric_stats["mpjae_cm_s2"], empty=None),
        "unknown_rotation_deg": _metric_from_stats(metric_stats["unknown_rotation_deg"]),
        "root_yaw_error_deg": _metric_from_stats(metric_stats["root_yaw_error_deg"]),
        "root_xz_error_m": _metric_from_stats(metric_stats["root_xz_error_m"]),
        "hip_height_error_m": _metric_from_stats(metric_stats["hip_height_error_m"]),
        "tracker_position_error_m": _metric_from_stats(metric_stats["tracker_position_error_m"]),
        "known_tracker_rotation_max_error_deg": float(np.degrees(known_consistency.max(initial=0.0))),
        "by_scenario": per_scenario,
        "by_missing_age": by_missing_age,
        "_metric_stats": metric_stats,
    }
    return result


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise RuntimeError("没有可评估的 140D reconstruction 文件。")
    metric_keys = (
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
    summary: dict[str, object] = {
        "file_count": len(results),
        "sequences": int(sum(int(result["sequences"]) for result in results)),
        "samples": int(sum(int(result["samples"]) for result in results)),
        "velocity_pairs": int(sum(int(result["velocity_pairs"]) for result in results)),
        "acceleration_triplets": int(sum(int(result["acceleration_triplets"]) for result in results)),
    }
    for key in metric_keys:
        stats = {
            "sum": sum(float(result["_metric_stats"][key]["sum"]) for result in results),
            "count": sum(int(result["_metric_stats"][key]["count"]) for result in results),
        }
        summary[key] = _metric_from_stats(
            stats,
            empty=None if key in {"mpjve_cm_s", "mpjae_cm_s2"} else 0.0,
        )
    summary["known_tracker_rotation_max_error_deg"] = max(
        float(result["known_tracker_rotation_max_error_deg"]) for result in results
    )
    summary["by_scenario"] = _summarize_groups(results, "by_scenario")
    summary["by_missing_age"] = _summarize_groups(results, "by_missing_age")
    return summary


def _summarize_groups(results: list[dict[str, object]], group_key: str) -> dict[str, dict[str, object]]:
    metric_keys = (
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
    group_names = sorted(
        {
            name
            for result in results
            for name in result[group_key].keys()
        }
    )
    aggregated: dict[str, dict[str, object]] = {}
    for name in group_names:
        groups = [result[group_key][name] for result in results if name in result[group_key]]
        samples = sum(int(group["samples"]) for group in groups)
        velocity_pairs = sum(int(group["velocity_pairs"]) for group in groups)
        acceleration_triplets = sum(int(group["acceleration_triplets"]) for group in groups)
        root_yaw_samples = sum(int(group["root_yaw_samples"]) for group in groups)
        values: dict[str, object] = {
            "samples": samples,
            "velocity_pairs": velocity_pairs,
            "acceleration_triplets": acceleration_triplets,
            "root_yaw_samples": root_yaw_samples,
        }
        for key in metric_keys:
            if key == "mpjve_cm_s":
                weight_key = "velocity_pairs"
            elif key == "mpjae_cm_s2":
                weight_key = "acceleration_triplets"
            elif key == "root_yaw_error_deg":
                weight_key = "root_yaw_samples"
            else:
                weight_key = "samples"
            weighted = [
                (float(group[key]), int(group[weight_key]))
                for group in groups
                if group[key] is not None and int(group[weight_key]) > 0
            ]
            total_weight = sum(weight for _, weight in weighted)
            values[key] = (
                sum(value * weight for value, weight in weighted) / total_weight
                if total_weight
                else None if key in {"mpjve_cm_s", "mpjae_cm_s2"} else 0.0
            )
        aggregated[name] = values
    return aggregated


def public_result(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    results = [evaluate_file(path) for path in sorted(input_dir.rglob("*.npz"))]
    summary = summarize(results)
    output_json = Path(args.output_json).resolve() if args.output_json else input_dir / "realtime_pose_eval_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "files": [public_result(result) for result in results]},
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[evaluate_realtime_pose] wrote {output_json}")
    return summary


if __name__ == "__main__":
    main()
