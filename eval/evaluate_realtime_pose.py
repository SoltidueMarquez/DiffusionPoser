from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import (
    FOOT_TRACKER_INDICES,
    HEAD_TRACKER_INDEX,
    NON_HEAD_TRACKER_INDICES,
    REALTIME_POSE_TARGET_DIM,
    SMPL_JOINT_COUNT,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_TO_JOINT,
)


PAPER_JOINT_SLICE = slice(0, 22)
PAPER_BODY_ROTATION_SLICE = slice(1, 22)
DURATION_BUCKETS = (("1-5", 1, 5), ("6-15", 6, 15), ("16-30", 16, 30), ("31-60", 31, 60))
STARTUP_PHASE_BUCKETS = (
    ("frames_0_14", 0, 14),
    ("frames_15_29", 15, 29),
    ("frames_30_59", 30, 59),
    ("frames_60_119", 60, 119),
    ("frames_120_299", 120, 299),
    ("frames_300_plus", 300, None),
)
REQUIRED_RESULT_FIELDS = {
    "reference_target_raw",
    "raw_pred_target_raw",
    "deployed_pred_target_raw",
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
    "tracker_pos_world",
    "current_tracker_raw",
    "configured",
    "measured_valid",
    "d_off",
    "d_on",
    "hard_rotation_state",
    "current_trajectory",
    "contact_target",
    "contact_logits",
    "future_leg_target",
    "future_leg_prediction",
    "scenario",
    "eval_frame_mask",
    "fps",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估动态 Tracker 的 raw/deployed 姿态结果。")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_json", default="")
    return parser


def _rotation_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(first, -1, -2) @ second
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _load_result(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_RESULT_FIELDS.difference(data.files))
        if missing:
            raise KeyError(f"{path} 缺少当前评估字段：{missing}；旧结果不可复用。")
        values = {key: np.asarray(data[key]) for key in REQUIRED_RESULT_FIELDS}
        if "history_length" in data.files:
            values["history_length"] = np.asarray(data["history_length"], dtype=np.int64)
        if "absolute_frame_index" in data.files:
            values["absolute_frame_index"] = np.asarray(
                data["absolute_frame_index"], dtype=np.int64
            )
        for name in (
            "current_head_yaw_world",
            "taid_prior_root_head",
            "taid_prior_joints_head",
        ):
            if name in data.files:
                values[name] = np.asarray(data[name])
        values["hard_rotation_max_error"] = np.asarray(
            data["hard_rotation_max_error"] if "hard_rotation_max_error" in data.files else 0.0,
            dtype=np.float64,
        )
    return values


def _stats(values: np.ndarray, mask: np.ndarray, scale: float = 1.0) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    selected_mask = np.broadcast_to(np.asarray(mask, dtype=bool), array.shape) & np.isfinite(array)
    selected = array[selected_mask]
    return {"sum": float(selected.sum() * scale), "count": int(selected.size)}


def _metric(stats: dict[str, float | int]) -> float | None:
    return float(stats["sum"]) / int(stats["count"]) if int(stats["count"]) else None


def _build_startup_phase_masks(frame_indices: np.ndarray) -> dict[str, np.ndarray]:
    """按每条序列的绝对帧号拆分启动、锁模和稳定阶段。"""

    indices = np.asarray(frame_indices, dtype=np.int64)
    return {
        name: (
            indices >= lower
            if upper is None
            else (indices >= lower) & (indices <= upper)
        )
        for name, lower, upper in STARTUP_PHASE_BUCKETS
    }


def _summarize_root_yaw_diagnostics(
    samples: np.ndarray,
    sequence_stats: list[dict[str, int | None]],
) -> dict[str, float | int | None]:
    """汇总 Root yaw 的 π 模态分布与时序进入/退出次数。"""

    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    first_frames = [
        int(item["first_error_over_150_frame"])
        for item in sequence_stats
        if item.get("first_error_over_150_frame") is not None
    ]
    return {
        "samples": int(values.size),
        "median_deg": float(np.median(values)) if values.size else None,
        "p90_deg": float(np.percentile(values, 90)) if values.size else None,
        "p95_deg": float(np.percentile(values, 95)) if values.size else None,
        "error_over_90_ratio": float(np.mean(values > 90.0)) if values.size else None,
        "error_over_150_ratio": float(np.mean(values > 150.0)) if values.size else None,
        "earliest_error_over_150_frame": min(first_frames) if first_frames else None,
        "pi_mode_entry_count": sum(
            int(item["pi_mode_entry_count"]) for item in sequence_stats
        ),
        "pi_mode_exit_count": sum(
            int(item["pi_mode_exit_count"]) for item in sequence_stats
        ),
        "pi_mode_transition_count": sum(
            int(item["pi_mode_transition_count"]) for item in sequence_stats
        ),
        "pi_majority_sequence_count": sum(
            int(item["error_over_150_count"]) * 2 > int(item["sample_count"])
            for item in sequence_stats
            if int(item["sample_count"]) > 0
        ),
        "sequence_count": sum(int(item["sample_count"]) > 0 for item in sequence_stats),
    }


def _root_yaw_diagnostic_inputs(
    root_yaw_error_deg: np.ndarray,
    frame_mask: np.ndarray,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int | None]]]:
    """保留可精确合并的逐帧误差和逐序列 π 模态时序统计。"""

    values = np.asarray(root_yaw_error_deg, dtype=np.float64)
    mask = np.broadcast_to(np.asarray(frame_mask, dtype=bool), values.shape)
    indices = np.broadcast_to(np.asarray(frame_indices, dtype=np.int64), values.shape)
    finite_mask = mask & np.isfinite(values)
    samples = values[finite_mask]
    sequence_stats: list[dict[str, int | None]] = []
    for sequence_index in range(values.shape[0]):
        valid = finite_mask[sequence_index]
        bad = valid & (values[sequence_index] > 150.0)
        first_positions = np.flatnonzero(bad)
        adjacent = valid[1:] & valid[:-1]
        entries = int(np.count_nonzero(adjacent & bad[1:] & ~bad[:-1]))
        exits = int(np.count_nonzero(adjacent & ~bad[1:] & bad[:-1]))
        sequence_stats.append(
            {
                "sample_count": int(valid.sum()),
                "error_over_150_count": int(bad.sum()),
                "first_error_over_150_frame": (
                    int(indices[sequence_index, first_positions[0]])
                    if first_positions.size
                    else None
                ),
                "pi_mode_entry_count": entries,
                "pi_mode_exit_count": exits,
                "pi_mode_transition_count": entries + exits,
            }
        )
    return samples, sequence_stats


def _group_metrics(
    frame_mask: np.ndarray,
    metric_values: dict[str, np.ndarray],
    metric_scales: dict[str, float],
    frame_indices: np.ndarray,
) -> dict[str, object]:
    stats = {
        name: _stats(values, frame_mask, metric_scales.get(name, 1.0))
        for name, values in metric_values.items()
    }
    root_yaw_samples, root_yaw_sequence_stats = _root_yaw_diagnostic_inputs(
        metric_values["root_yaw_error_deg"],
        frame_mask,
        frame_indices,
    )
    return {
        "samples": int(np.asarray(frame_mask, dtype=bool).sum()),
        **{name: _metric(value) for name, value in stats.items()},
        "root_yaw_diagnostics": _summarize_root_yaw_diagnostics(
            root_yaw_samples,
            root_yaw_sequence_stats,
        ),
        "_metric_stats": stats,
        "_root_yaw_samples": root_yaw_samples,
        "_root_yaw_sequence_stats": root_yaw_sequence_stats,
    }


def evaluate_file(
    path: Path,
    eval_frame_mask_override: np.ndarray | None = None,
) -> dict[str, object]:
    values = _load_result(path)
    reference = values["reference_target_raw"].astype(np.float64)
    raw = values["raw_pred_target_raw"].astype(np.float64)
    deployed = values["deployed_pred_target_raw"].astype(np.float64)
    if reference.ndim != 3 or reference.shape[-1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError("reference_target_raw 必须为 [N,T,144]。")
    if raw.shape != reference.shape or deployed.shape != reference.shape:
        raise ValueError("raw/deployed target 必须与 reference 同形。")
    sequence_count, steps = reference.shape[:2]
    frame_shape = (sequence_count, steps)
    eval_mask = values["eval_frame_mask"].reshape(frame_shape).astype(bool)
    if eval_frame_mask_override is not None:
        override = np.asarray(eval_frame_mask_override, dtype=bool)
        if override.shape == (steps,):
            override = np.broadcast_to(override[None], frame_shape)
        elif override.shape != frame_shape:
            raise ValueError(
                "eval_frame_mask_override 必须为 [T] 或 [N,T]，"
                f"实际为 {override.shape}。"
            )
        # 诊断分组只能进一步收窄原评估范围，不能绕过 condition 自带的 eval mask。
        eval_mask &= override
    absolute_frame_index = values.get("absolute_frame_index")
    if absolute_frame_index is None:
        frame_indices = np.broadcast_to(np.arange(steps, dtype=np.int64), frame_shape)
    else:
        absolute_frame_index = np.asarray(absolute_frame_index, dtype=np.int64)
        if absolute_frame_index.shape == (steps,):
            frame_indices = np.broadcast_to(absolute_frame_index[None], frame_shape)
        elif absolute_frame_index.shape == frame_shape:
            frame_indices = absolute_frame_index
        else:
            raise ValueError(
                "absolute_frame_index 必须为 [T] 或 [N,T]，"
                f"实际为 {absolute_frame_index.shape}。"
            )
    fps = float(np.asarray(values["fps"]).reshape(()))

    reference_global = rotation_6d_to_matrix_np(reference.reshape(sequence_count, steps, 24, 6))
    raw_global = rotation_6d_to_matrix_np(raw.reshape(sequence_count, steps, 24, 6))
    deployed_global = rotation_6d_to_matrix_np(deployed.reshape(sequence_count, steps, 24, 6))
    raw_rotation = np.degrees(_rotation_angle(raw_global, reference_global)).mean(axis=-1)
    deployed_rotation = np.degrees(_rotation_angle(deployed_global, reference_global)).mean(axis=-1)

    reference_local = rotation_6d_to_matrix_np(
        values["reference_body_local_delta_6d"].reshape(sequence_count, steps, 24, 6)
    )
    predicted_local = rotation_6d_to_matrix_np(
        values["predicted_body_local_delta_6d"].reshape(sequence_count, steps, 24, 6)
    )
    mpjre = np.degrees(
        _rotation_angle(
            predicted_local[:, :, PAPER_BODY_ROTATION_SLICE],
            reference_local[:, :, PAPER_BODY_ROTATION_SLICE],
        )
    ).mean(axis=-1)

    reference_joints = values["reference_joints_world"].reshape(sequence_count, steps, 24, 3).astype(np.float64)
    predicted_joints = values["predicted_joints_world"].reshape(sequence_count, steps, 24, 3).astype(np.float64)
    mpjpe = np.linalg.norm(
        predicted_joints[:, :, PAPER_JOINT_SLICE] - reference_joints[:, :, PAPER_JOINT_SLICE], axis=-1
    ).mean(axis=-1)
    mpjve = np.full(frame_shape, np.nan, dtype=np.float64)
    mpjae = np.full(frame_shape, np.nan, dtype=np.float64)
    if steps >= 2:
        predicted_velocity = np.diff(predicted_joints[:, :, PAPER_JOINT_SLICE], axis=1) * fps
        reference_velocity = np.diff(reference_joints[:, :, PAPER_JOINT_SLICE], axis=1) * fps
        mpjve[:, 1:] = np.linalg.norm(predicted_velocity - reference_velocity, axis=-1).mean(axis=-1)
    if steps >= 3:
        predicted_acceleration = np.diff(predicted_velocity, axis=1) * fps
        reference_acceleration = np.diff(reference_velocity, axis=1) * fps
        mpjae[:, 2:] = np.linalg.norm(
            predicted_acceleration - reference_acceleration, axis=-1
        ).mean(axis=-1)

    tracker_raw = values["current_tracker_raw"].reshape(
        sequence_count,
        steps,
        TRACKER_COUNT,
        TRACKER_FEATURE_DIM,
    )
    tracker_rotation = rotation_6d_to_matrix_np(tracker_raw[..., 3:9])
    tracker_joint_indices = np.asarray(TRACKER_TO_JOINT, dtype=np.int64)
    raw_tracker_angle = np.degrees(
        _rotation_angle(raw_global[:, :, tracker_joint_indices], tracker_rotation)
    )
    deployed_tracker_angle = np.degrees(
        _rotation_angle(deployed_global[:, :, tracker_joint_indices], tracker_rotation)
    )
    measured = values["measured_valid"].reshape(sequence_count, steps, TRACKER_COUNT).astype(bool)
    hard = values["hard_rotation_state"].reshape(sequence_count, steps, TRACKER_COUNT).astype(bool)
    raw_hard_rotation = _masked_tracker_mean(raw_tracker_angle, hard)
    deployed_hard_rotation = _masked_tracker_mean(deployed_tracker_angle, hard)
    soft_rotation_mask = measured & ~hard
    soft_rotation = _masked_tracker_mean(deployed_tracker_angle, soft_rotation_mask)
    tracker_pos = values["tracker_pos_world"].reshape(sequence_count, steps, TRACKER_COUNT, 3)
    tracker_joint_position = predicted_joints[:, :, tracker_joint_indices]
    tracker_position_error = np.linalg.norm(tracker_joint_position - tracker_pos, axis=-1)
    # Position 从未执行 hard projection，因此稳定 Tracker 仍应参与误差统计。
    # Head 世界位置由 Resolver 解析式满足，若计入会用近似零误差稀释四肢和 Hip 的误差。
    measured_non_head = np.zeros_like(measured)
    measured_non_head[:, :, NON_HEAD_TRACKER_INDICES] = measured[:, :, NON_HEAD_TRACKER_INDICES]
    measured_non_head_position = _masked_tracker_mean(tracker_position_error, measured_non_head)

    reference_root = values["reference_root_position_world"].reshape(sequence_count, steps, 3)
    predicted_root = values["predicted_root_position_world"].reshape(sequence_count, steps, 3)
    root_xz = np.linalg.norm(predicted_root[..., [0, 2]] - reference_root[..., [0, 2]], axis=-1)
    root_yaw_ref = values["reference_root_yaw_world"].reshape(frame_shape)
    root_yaw_pred = values["predicted_root_yaw_world"].reshape(frame_shape)
    root_yaw = np.degrees(np.abs(np.arctan2(
        np.sin(root_yaw_pred - root_yaw_ref), np.cos(root_yaw_pred - root_yaw_ref)
    )))
    hip_height = np.abs(
        values["predicted_hip_height"].reshape(frame_shape)
        - values["reference_hip_height"].reshape(frame_shape)
    )
    root_step_delta = np.full(frame_shape, np.nan, dtype=np.float64)
    if steps >= 2:
        root_step_delta[:, 1:] = np.linalg.norm(
            np.diff(predicted_root[..., [0, 2]], axis=1)
            - np.diff(reference_root[..., [0, 2]], axis=1),
            axis=-1,
        )

    prior_root_head = np.asarray(
        values.get(
            "taid_prior_root_head",
            np.full((*frame_shape, 4), np.nan, dtype=np.float64),
        ),
        dtype=np.float64,
    ).reshape(sequence_count, steps, 4)
    prior_joints_head = np.asarray(
        values.get(
            "taid_prior_joints_head",
            np.full((*frame_shape, SMPL_JOINT_COUNT, 3), np.nan, dtype=np.float64),
        ),
        dtype=np.float64,
    ).reshape(sequence_count, steps, SMPL_JOINT_COUNT, 3)
    current_head_yaw = np.asarray(
        values.get(
            "current_head_yaw_world",
            np.full(frame_shape, np.nan, dtype=np.float64),
        ),
        dtype=np.float64,
    ).reshape(frame_shape)
    prior_available = (
        np.isfinite(prior_root_head).all(axis=-1)
        & np.isfinite(prior_joints_head).all(axis=(-1, -2))
        & np.isfinite(current_head_yaw)
    )
    prior_origin_world = np.stack(
        (
            tracker_pos[:, :, HEAD_TRACKER_INDEX, 0],
            reference_root[:, :, 1],
            tracker_pos[:, :, HEAD_TRACKER_INDEX, 2],
        ),
        axis=-1,
    )
    head_yaw_rotation = make_yaw_rotation_np(current_head_yaw.reshape(-1)).reshape(
        sequence_count, steps, 3, 3
    )
    prior_root_world = prior_origin_world + np.einsum(
        "...ij,...j->...i", head_yaw_rotation, prior_root_head[..., :3]
    )
    prior_joints_world = prior_origin_world[..., None, :] + np.einsum(
        "...ij,...aj->...ai", head_yaw_rotation, prior_joints_head
    )
    prior_root_yaw_world = current_head_yaw + prior_root_head[..., 3]

    def _mask_unavailable(array: np.ndarray) -> np.ndarray:
        return np.where(prior_available, array, np.nan)

    taid_prior_root_xz = _mask_unavailable(
        np.linalg.norm(
            prior_root_world[..., [0, 2]] - reference_root[..., [0, 2]], axis=-1
        )
    )
    taid_prior_root_y = _mask_unavailable(
        np.abs(prior_root_world[..., 1] - reference_root[..., 1])
    )
    taid_prior_vs_deployed_root = _mask_unavailable(
        np.linalg.norm(prior_root_world - predicted_root, axis=-1)
    )
    taid_prior_vs_deployed_yaw = _mask_unavailable(
        np.degrees(
            np.abs(
                np.arctan2(
                    np.sin(prior_root_yaw_world - root_yaw_pred),
                    np.cos(prior_root_yaw_world - root_yaw_pred),
                )
            )
        )
    )
    taid_prior_internal_mpjpe = _mask_unavailable(
        np.linalg.norm(
            prior_joints_world[:, :, PAPER_JOINT_SLICE]
            - reference_joints[:, :, PAPER_JOINT_SLICE],
            axis=-1,
        ).mean(axis=-1)
    )
    taid_prior_vs_deployed_joint = _mask_unavailable(
        np.linalg.norm(prior_joints_world - predicted_joints, axis=-1).mean(axis=-1)
    )

    contact_target = values["contact_target"].reshape(sequence_count, steps, 2) >= 0.5
    contact_logits = values["contact_logits"].reshape(sequence_count, steps, 2).astype(np.float64)
    contact_accuracy = np.nanmean(
        ((contact_logits >= 0.0) == contact_target).astype(np.float64), axis=-1
    )
    contact_accuracy[~np.isfinite(contact_logits).all(axis=-1)] = np.nan
    foot_slide = np.full(frame_shape, np.nan, dtype=np.float64)
    if steps >= 2:
        feet = np.asarray([TRACKER_TO_JOINT[index] for index in FOOT_TRACKER_INDICES], dtype=np.int64)
        foot_speed = np.linalg.norm(np.diff(predicted_joints[:, :, feet], axis=1), axis=-1) * fps
        # 只有相邻两帧都处于接触状态才属于脚底滑动；落地和离地边沿不计入。
        adjacent_contact = contact_target[:, 1:] & contact_target[:, :-1]
        numerator = (foot_speed * adjacent_contact).sum(axis=-1)
        denominator = adjacent_contact.sum(axis=-1)
        foot_slide[:, 1:] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )

    future_target = values["future_leg_target"].reshape(sequence_count, steps, 3, 8, 6)
    future_prediction = values["future_leg_prediction"].reshape(sequence_count, steps, 3, 8, 6)
    future_rotation = np.degrees(_rotation_angle(
        rotation_6d_to_matrix_np(future_prediction),
        rotation_6d_to_matrix_np(future_target),
    )).mean(axis=(-1, -2))

    metric_values = {
        "raw_rotation_deg": raw_rotation,
        "deployed_rotation_deg": deployed_rotation,
        "mpjre_deg": mpjre,
        "mpjpe_cm": mpjpe,
        "mpjve_cm_s": mpjve,
        "mpjae_cm_s2": mpjae,
        "raw_hard_tracker_rotation_deg": raw_hard_rotation,
        "deployed_hard_tracker_rotation_deg": deployed_hard_rotation,
        "soft_tracker_rotation_deg": soft_rotation,
        "measured_non_head_tracker_position_error_m": measured_non_head_position,
        "root_yaw_error_deg": root_yaw,
        "root_xz_error_m": root_xz,
        "hip_height_error_m": hip_height,
        "root_step_delta_error_m": root_step_delta,
        "future_leg_rotation_deg": future_rotation,
        "contact_accuracy": contact_accuracy,
        "foot_slide_m_s": foot_slide,
        "taid_prior_root_xz_error_m": taid_prior_root_xz,
        "taid_prior_root_y_error_m": taid_prior_root_y,
        "taid_prior_vs_deployed_root_gap_m": taid_prior_vs_deployed_root,
        "taid_prior_vs_deployed_yaw_deg": taid_prior_vs_deployed_yaw,
        "taid_prior_internal_mpjpe_cm": taid_prior_internal_mpjpe,
        "taid_prior_vs_deployed_joint_gap_cm": taid_prior_vs_deployed_joint,
        "taid_prior_available_ratio": prior_available.astype(np.float64),
    }
    metric_scales = {
        "mpjpe_cm": 100.0,
        "mpjve_cm_s": 100.0,
        "mpjae_cm_s2": 100.0,
        "taid_prior_internal_mpjpe_cm": 100.0,
        "taid_prior_vs_deployed_joint_gap_cm": 100.0,
    }
    result = _group_metrics(eval_mask, metric_values, metric_scales, frame_indices)
    result.update(
        path=str(path),
        sequences=sequence_count,
        hard_tracker_rotation_max_error_deg=float(
            np.degrees(np.nanmax(values["hard_rotation_max_error"]))
        ),
    )
    scenarios = values["scenario"].reshape(frame_shape).astype(str)
    result["by_scenario"] = {
        scenario: _group_metrics(
            eval_mask & (scenarios == scenario), metric_values, metric_scales, frame_indices
        )
        for scenario in sorted(set(scenarios[eval_mask].tolist()))
    }
    d_off = values["d_off"].reshape(sequence_count, steps, TRACKER_COUNT)
    max_d_off = d_off[:, :, NON_HEAD_TRACKER_INDICES].max(axis=-1)
    result["by_d_off"] = {
        name: _group_metrics(
            eval_mask & (max_d_off >= lower) & (max_d_off <= upper),
            metric_values,
            metric_scales,
            frame_indices,
        )
        for name, lower, upper in DURATION_BUCKETS
    }
    d_on = values["d_on"].reshape(sequence_count, steps, TRACKER_COUNT)
    reconnect_scenario = scenarios == "two_point_dropout_reconnect"
    result["by_reconnect_d_on"] = {
        str(duration): _group_metrics(
            eval_mask
            & reconnect_scenario
            & np.any(d_on[:, :, NON_HEAD_TRACKER_INDICES] == duration, axis=-1),
            metric_values,
            metric_scales,
            frame_indices,
        )
        for duration in range(1, 16)
    }
    hard_count = hard.sum(axis=-1)
    result["by_rotation_state"] = {
        "head_only_hard": _group_metrics(
            eval_mask & (hard_count == 1), metric_values, metric_scales, frame_indices
        ),
        "mixed_hard_soft": _group_metrics(
            eval_mask & (hard_count > 1) & soft_rotation_mask.any(axis=-1),
            metric_values,
            metric_scales,
            frame_indices,
        ),
        "all_configured_hard": _group_metrics(
            eval_mask & (hard_count == values["configured"].reshape(sequence_count, steps, 6).sum(axis=-1)),
            metric_values,
            metric_scales,
            frame_indices,
        ),
    }
    result["by_startup_phase"] = {
        name: _group_metrics(
            eval_mask & phase_mask,
            metric_values,
            metric_scales,
            frame_indices,
        )
        for name, phase_mask in _build_startup_phase_masks(frame_indices).items()
    }
    if "history_length" in values:
        history_length = values["history_length"].reshape(frame_shape)
        if np.any((history_length < 0) | (history_length > 60)):
            raise ValueError("history_length 必须位于 [0,60]。")
        result["by_history_phase"] = {
            "cold_start_0_59": _group_metrics(
                eval_mask & (history_length < 60),
                metric_values,
                metric_scales,
                frame_indices,
            ),
            "steady_state_60_plus": _group_metrics(
                eval_mask & (history_length >= 60),
                metric_values,
                metric_scales,
                frame_indices,
            ),
        }
    return result


def _masked_tracker_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    numerator = np.where(selected, values, 0.0).sum(axis=-1)
    denominator = selected.sum(axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def _merge_groups(results: list[dict[str, object]], group_name: str) -> dict[str, object]:
    names = sorted({name for result in results for name in result.get(group_name, {})})
    merged: dict[str, object] = {}
    for name in names:
        groups = [result[group_name][name] for result in results if name in result.get(group_name, {})]
        merged[name] = _summarize_metric_stats(groups)
    return merged


def _summarize_metric_stats(results: list[dict[str, object]]) -> dict[str, object]:
    metric_names = sorted({name for result in results for name in result.get("_metric_stats", {})})
    stats = {}
    for name in metric_names:
        values = [result.get("_metric_stats", {}).get(name, {"sum": 0.0, "count": 0}) for result in results]
        stats[name] = {
            "sum": sum(float(value["sum"]) for value in values),
            "count": sum(int(value["count"]) for value in values),
        }
    root_yaw_samples = np.concatenate(
        [
            np.asarray(result.get("_root_yaw_samples", []), dtype=np.float64).reshape(-1)
            for result in results
        ]
    )
    root_yaw_sequence_stats = [
        item
        for result in results
        for item in result.get("_root_yaw_sequence_stats", [])
    ]
    return {
        "samples": sum(int(result.get("samples", 0)) for result in results),
        **{name: _metric(value) for name, value in stats.items()},
        "root_yaw_diagnostics": _summarize_root_yaw_diagnostics(
            root_yaw_samples,
            root_yaw_sequence_stats,
        ),
        "_metric_stats": stats,
    }


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise ValueError("没有可汇总的评估结果。")
    summary = _summarize_metric_stats(results)
    summary["sequences"] = sum(int(result.get("sequences", 0)) for result in results)
    max_values = [
        float(result.get("hard_tracker_rotation_max_error_deg", result.get("known_tracker_rotation_max_error_deg", 0.0)))
        for result in results
    ]
    summary["hard_tracker_rotation_max_error_deg"] = max(max_values, default=0.0)
    # 保留汇总调用的旧键仅作为报告别名，不再参与任何数据/模型契约。
    summary["known_tracker_rotation_max_error_deg"] = summary["hard_tracker_rotation_max_error_deg"]
    for group_name in (
        "by_scenario",
        "by_d_off",
        "by_reconnect_d_on",
        "by_rotation_state",
        "by_history_phase",
        "by_startup_phase",
    ):
        summary[group_name] = _merge_groups(results, group_name)
    return summary


def public_result(result: dict[str, object]) -> dict[str, object]:
    return {
        key: (
            {name: public_result(value) for name, value in values.items()}
            if key.startswith("by_") and isinstance(values, dict)
            else values
        )
        for key, values in result.items()
        if not key.startswith("_")
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    paths = sorted(input_dir.rglob("*.npz"))
    results = [evaluate_file(path) for path in paths]
    summary = summarize(results)
    output_json = Path(args.output_json).resolve() if args.output_json else input_dir / "eval_summary.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {"summary": public_result(summary), "files": [public_result(result) for result in results]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[evaluate_realtime_pose] wrote {output_json}")
    return summary


if __name__ == "__main__":
    main()
