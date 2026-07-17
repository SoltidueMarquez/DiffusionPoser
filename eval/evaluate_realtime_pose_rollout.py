from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    TRACKER_JOINT_INDICES,
    make_yaw_rotation_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    HIP_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
    RIGHT_FOOT_TRACKER_INDEX,
    get_schema_spec,
)
from eval.stationary_signal_metrics import compute_stationary_signal_metrics


LOWER_BODY_JOINT_INDICES = np.asarray(
    [
        JOINT_INDEX["left_hip"],
        JOINT_INDEX["right_hip"],
        JOINT_INDEX["left_knee"],
        JOINT_INDEX["right_knee"],
        JOINT_INDEX["left_ankle"],
        JOINT_INDEX["right_ankle"],
        JOINT_INDEX["left_foot"],
        JOINT_INDEX["right_foot"],
    ],
    dtype=np.int64,
)
FOOT_JOINT_INDICES = np.asarray(
    [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]],
    dtype=np.int64,
)
NO_HIP_DURATION_BUCKETS = (
    ("1_2", 1, 2),
    ("3_10", 3, 10),
    ("11_30", 11, 30),
    ("gt30", 31, None),
)


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
        preliminary_joints = read_optional_array(data, "preliminary_joints_world")
        preliminary_root_yaw = read_optional_array(data, "preliminary_root_yaw")
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
        tracker_ref_root_yaw = read_optional_array(data, "tracker_ref_root_yaw")
        schema_name = (
            str(np.asarray(data["schema_name"]).item())
            if "schema_name" in data.files
            else DEFAULT_REALTIME_POSE_SCHEMA_NAME
        )
        timing_arrays = {
            name: read_optional_array(data, name)
            for name in ("codec_fk_elapsed_ms", "ddim_elapsed_ms", "resolver_elapsed_ms", "end_to_end_elapsed_ms")
        }

    if reference_features.shape != predicted_features.shape:
        raise ValueError(f"{path} feature shape 不匹配：{reference_features.shape} vs {predicted_features.shape}")
    if reference_joints.shape != predicted_joints.shape:
        raise ValueError(f"{path} joints shape 不匹配：{reference_joints.shape} vs {predicted_joints.shape}")
    if preliminary_joints is not None and np.asarray(preliminary_joints).shape != reference_joints.shape:
        raise ValueError(
            f"{path} preliminary joints shape 不匹配：{np.asarray(preliminary_joints).shape} vs {reference_joints.shape}"
        )
    if not eval_frame_mask.any():
        raise ValueError(f"{path} eval_frame_mask 没有可评估帧。")

    schema = get_schema_spec(schema_name)
    joint_error = np.linalg.norm(predicted_joints - reference_joints, axis=-1)
    reference_relative = reference_joints - reference_joints[..., 0:1, :]
    predicted_relative = predicted_joints - predicted_joints[..., 0:1, :]
    root_relative_error = np.linalg.norm(predicted_relative - reference_relative, axis=-1)
    mpjpe_by_time = masked_time_mean(joint_error, eval_frame_mask)
    yaw_error = np.abs(wrap_radians(root_yaw_predicted - root_yaw_reference))
    yaw_by_time = masked_time_mean(yaw_error[..., None], eval_frame_mask)
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
        "tracker_reprojection_pos_error": 0.0,
        "tracker_reprojection_pos_mean_cm": 0.0,
        "tracker_reprojection_pos_p95_cm": 0.0,
        "tracker_reprojection_rot_mean_deg": 0.0,
        "tracker_reprojection_rot_p95_deg": 0.0,
        "contact_foot_velocity_left_mps": 0.0,
        "contact_foot_velocity_right_mps": 0.0,
        "contact_foot_velocity_mean_mps": 0.0,
        "floating_foot_ratio": 0.0,
        "ground_penetration_ratio": 0.0,
        "no_hip_contact_foot_velocity_mean_mps": 0.0,
        "no_hip_floating_foot_ratio": 0.0,
        "lower_body_acceleration_mean_mps2": 0.0,
        "lower_body_jerk_mean_mps3": 0.0,
        "lower_body_acceleration_error_mps2": 0.0,
        "lower_body_jerk_error_mps3": 0.0,
    }
    motion_quality = compute_motion_quality_metrics(
        predicted_features=predicted_features,
        reference_features=reference_features,
        predicted_joints=predicted_joints,
        reference_joints=reference_joints,
        pose_slice=schema.body_pose_slice(),
        eval_frame_mask=eval_frame_mask,
        timestamps=timestamps,
    )
    result.update(motion_quality["summary"])
    root_error = None
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

    no_hip_mask = np.zeros_like(eval_frame_mask, dtype=bool)
    sensor_valid_array = None
    if sensor_valid is not None:
        sensor_valid_array = np.asarray(sensor_valid, dtype=bool)
        if sensor_valid_array.shape != (*eval_frame_mask.shape, 6):
            raise ValueError(
                f"{path} sensor_valid 应为 {(*eval_frame_mask.shape, 6)}，实际为 {sensor_valid_array.shape}"
            )
        no_hip_mask = (~sensor_valid_array[..., HIP_TRACKER_INDEX]) & eval_frame_mask
        state_masks = tracker_state_group_masks(
            sensor_valid=sensor_valid_array,
            eval_frame_mask=eval_frame_mask,
        )
        for group_name, group_mask in state_masks.items():
            add_motion_group_metrics(
                result=result,
                prefix=group_name,
                group_mask=group_mask,
                joint_error=joint_error,
                rotation_error_deg=motion_quality["rotation_error_deg"],
                velocity_error_cmps=motion_quality["velocity_error_cmps"],
                jitter_mps3=motion_quality["jitter_mps3_by_frame"],
            )
        result.update(
            compute_transition_pj_auj(
                predicted_joints=predicted_joints,
                reference_joints=reference_joints,
                sensor_valid=sensor_valid_array,
                eval_frame_mask=eval_frame_mask,
                timestamps=timestamps,
            )
        )
    result["no_hip_evaluated_frames"] = int(no_hip_mask.sum())

    preliminary_metrics = None
    if preliminary_joints is not None:
        preliminary_array = np.asarray(preliminary_joints, dtype=np.float32)
        preliminary_yaw_array = (
            root_yaw_predicted
            if preliminary_root_yaw is None
            else np.asarray(preliminary_root_yaw, dtype=np.float32)
        )
        if preliminary_yaw_array.shape != root_yaw_reference.shape:
            raise ValueError(
                f"{path} preliminary_root_yaw 应为 {root_yaw_reference.shape}，实际为 {preliminary_yaw_array.shape}"
            )
        common_ref_yaw = (
            root_yaw_reference
            if tracker_ref_root_yaw is None
            else np.asarray(tracker_ref_root_yaw, dtype=np.float32)
        )
        preliminary_metrics = compute_preliminary_geometry_metrics(
            preliminary_joints=preliminary_array,
            reference_joints=reference_joints,
            preliminary_root_yaw=preliminary_yaw_array,
            reference_root_yaw=root_yaw_reference,
            common_ref_yaw=common_ref_yaw,
        )
        preliminary_valid = np.isfinite(preliminary_array).all(axis=(-1, -2))
        no_hip_preliminary_mask = no_hip_mask & preliminary_valid
        result["no_hip_preliminary_frames"] = int(no_hip_preliminary_mask.sum())
        result["no_hip_preliminary_p2h_xz_error_cm"] = masked_scalar_mean(
            preliminary_metrics["p2h_xz_error"], no_hip_preliminary_mask, scale=100.0
        )
        result["no_hip_preliminary_p2h_height_error_cm"] = masked_scalar_mean(
            preliminary_metrics["p2h_height_error"], no_hip_preliminary_mask, scale=100.0
        )
        result["no_hip_preliminary_lower_body_aligned_mpjpe_mm"] = masked_scalar_mean(
            preliminary_metrics["lower_body_aligned_error"],
            no_hip_preliminary_mask[..., None],
            scale=1000.0,
        )
    else:
        result.update(
            {
                "no_hip_preliminary_frames": 0,
                "no_hip_preliminary_p2h_xz_error_cm": 0.0,
                "no_hip_preliminary_p2h_height_error_cm": 0.0,
                "no_hip_preliminary_lower_body_aligned_mpjpe_mm": 0.0,
            }
        )

    lower_rotation_error = compute_lower_body_local_rotation_error_deg(
        predicted_features=predicted_features,
        reference_features=reference_features,
        pose_slice=schema.body_pose_slice(),
    )
    result["no_hip_lower_body_local_rotation_error_deg"] = masked_scalar_mean(
        lower_rotation_error,
        no_hip_mask[..., None],
    )

    result.update(
        compute_lower_body_temporal_metrics(
            predicted_joints=predicted_joints,
            reference_joints=reference_joints,
            eval_frame_mask=eval_frame_mask,
            timestamps=timestamps,
        )
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

    if reference_features.shape[-1] == schema.feature_dim and schema.supports_stationary_prob:
        reference_stationary = reference_features[..., schema.stationary_prob_slice()]
        predicted_stationary = predicted_features[..., schema.stationary_prob_slice()]
        stationary = reference_stationary >= 0.7
        selected_reference = reference_stationary[eval_frame_mask]
        selected_prediction = predicted_stationary[eval_frame_mask]
        if selected_reference.size and selected_prediction.size:
            stationary_reports = compute_stationary_signal_metrics(
                selected_reference,
                selected_prediction,
                thresholds=(0.5, 0.7),
            )["thresholds"]
            legacy_metrics = stationary_reports["0.5"]["aggregate"]
            runtime_metrics = stationary_reports["0.7"]["aggregate"]
            result["stationary_f1"] = float(legacy_metrics["f1"])
            result["stationary_false_lock_rate"] = float(legacy_metrics["false_lock_rate"])
            result["stationary_missed_lock_rate"] = float(legacy_metrics["missed_lock_rate"])
            result["stationary_f1_at_0_7"] = float(runtime_metrics["f1"])
            result["stationary_false_lock_rate_at_0_7"] = float(runtime_metrics["false_lock_rate"])
            result["stationary_missed_lock_rate_at_0_7"] = float(runtime_metrics["missed_lock_rate"])
            result["stationary_jitter"] = float(runtime_metrics["prob_jitter_mean_abs"])
            result["stationary_clamp_pre_out_of_bounds_ratio"] = float(
                runtime_metrics["clamp_pre_out_of_bounds_ratio"]
            )
        floor = np.zeros(reference_features.shape[:2], dtype=np.float32) if floor_y is None else np.asarray(floor_y)
        if floor.shape != reference_features.shape[:2]:
            raise ValueError(f"{path} floor_y 应为 {reference_features.shape[:2]}，实际为 {floor.shape}")
        reference_feet = reference_joints[..., FOOT_JOINT_INDICES, :]
        predicted_feet = predicted_joints[..., FOOT_JOINT_INDICES, :]
        gt_foot_height = reference_feet[..., 1] - floor[..., None]
        predicted_foot_height = predicted_feet[..., 1] - floor[..., None]
        gt_contact = stationary[..., 1:3] & (gt_foot_height <= 0.05) & eval_frame_mask[..., None]

        left_velocity = foot_slide_m_per_second(
            predicted_feet[..., 0, :], gt_contact[..., 0], floor, timestamps, eval_frame_mask
        )
        right_velocity = foot_slide_m_per_second(
            predicted_feet[..., 1, :], gt_contact[..., 1], floor, timestamps, eval_frame_mask
        )
        result["foot_skating_left"] = left_velocity
        result["foot_skating_right"] = right_velocity
        result["contact_foot_velocity_left_mps"] = left_velocity
        result["contact_foot_velocity_right_mps"] = right_velocity
        active_velocity_values = [
            value
            for value, contact in zip((left_velocity, right_velocity), np.moveaxis(gt_contact, -1, 0))
            if contact.any()
        ]
        result["contact_foot_velocity_mean_mps"] = (
            float(np.mean(active_velocity_values)) if active_velocity_values else 0.0
        )
        contact_count = int(gt_contact.sum())
        result["floating_foot_ratio"] = (
            float(((predicted_foot_height > 0.05) & gt_contact).sum() / contact_count)
            if contact_count
            else 0.0
        )
        evaluated_feet = np.broadcast_to(eval_frame_mask[..., None], predicted_foot_height.shape)
        result["ground_penetration_ratio"] = (
            float(((predicted_foot_height < -0.01) & evaluated_feet).sum() / evaluated_feet.sum())
            if evaluated_feet.any()
            else 0.0
        )

        no_hip_contact = gt_contact & no_hip_mask[..., None]
        result["no_hip_floating_foot_ratio"] = (
            float(((predicted_foot_height > 0.05) & no_hip_contact).sum() / no_hip_contact.sum())
            if no_hip_contact.any()
            else 0.0
        )
        no_hip_foot_velocities = [
            foot_slide_m_per_second(
                predicted_feet[..., foot_index, :],
                no_hip_contact[..., foot_index],
                floor,
                timestamps,
                no_hip_mask,
            )
            for foot_index in range(2)
        ]
        active_no_hip_feet = [
            no_hip_foot_velocities[index]
            for index in range(2)
            if no_hip_contact[..., index].any()
        ]
        result["no_hip_contact_foot_velocity_mean_mps"] = (
            float(np.mean(active_no_hip_feet)) if active_no_hip_feet else 0.0
        )

    add_no_hip_duration_bucket_metrics(
        result=result,
        sensor_valid=sensor_valid,
        eval_frame_mask=eval_frame_mask,
        joint_error=joint_error,
        yaw_error=yaw_error,
        root_error=root_error,
        preliminary_metrics=preliminary_metrics,
        preliminary_joints=preliminary_joints,
        lower_rotation_error=lower_rotation_error,
    )
    if sensor_valid_array is not None:
        for label, duration_mask in no_hip_duration_bucket_masks(sensor_valid_array).items():
            add_motion_group_metrics(
                result=result,
                prefix=f"no_hip_duration_{label}",
                group_mask=duration_mask & eval_frame_mask,
                joint_error=joint_error,
                rotation_error_deg=motion_quality["rotation_error_deg"],
                velocity_error_cmps=motion_quality["velocity_error_cmps"],
                jitter_mps3=motion_quality["jitter_mps3_by_frame"],
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


def masked_scalar_mean(values: np.ndarray, mask: np.ndarray, scale: float = 1.0) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    mask_array = np.broadcast_to(np.asarray(mask, dtype=bool), values_array.shape)
    valid = mask_array & np.isfinite(values_array)
    if not valid.any():
        return 0.0
    return float(np.mean(values_array[valid]) * float(scale))


def compute_motion_quality_metrics(
    *,
    predicted_features: np.ndarray,
    reference_features: np.ndarray,
    predicted_joints: np.ndarray,
    reference_joints: np.ndarray,
    pose_slice: slice,
    eval_frame_mask: np.ndarray,
    timestamps: np.ndarray | None,
) -> dict[str, dict[str, float] | np.ndarray]:
    """计算全身 MPJRE、MPJVE 与 RPM 同量纲的关节 jerk。"""

    predicted_pose = np.asarray(predicted_features[..., pose_slice], dtype=np.float64).reshape(
        *predicted_features.shape[:2], 24, 6
    )
    reference_pose = np.asarray(reference_features[..., pose_slice], dtype=np.float64).reshape(
        *reference_features.shape[:2], 24, 6
    )
    predicted_rot = rotation_6d_to_matrix_np(predicted_pose)
    reference_rot = rotation_6d_to_matrix_np(reference_pose)
    relative = np.swapaxes(reference_rot, -1, -2) @ predicted_rot
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    rotation_error_deg = np.rad2deg(np.arccos(cosine))

    predicted = np.asarray(predicted_joints, dtype=np.float64)
    reference = np.asarray(reference_joints, dtype=np.float64)
    frame_mask = np.asarray(eval_frame_mask, dtype=bool)
    time = _broadcast_timestamps(timestamps, frame_mask.shape)
    velocity_error_cmps = np.full(predicted.shape[:-1], np.nan, dtype=np.float64)
    jitter_by_frame = np.full(frame_mask.shape, np.nan, dtype=np.float64)
    if predicted.shape[1] >= 2:
        dt = np.maximum(np.diff(time, axis=1), 1e-6)
        predicted_velocity = np.diff(predicted, axis=1) / dt[..., None, None]
        reference_velocity = np.diff(reference, axis=1) / dt[..., None, None]
        velocity_error_cmps[:, 1:] = np.linalg.norm(
            predicted_velocity - reference_velocity, axis=-1
        ) * 100.0
    if predicted.shape[1] >= 4:
        acceleration_dt = np.maximum((dt[:, 1:] + dt[:, :-1]) * 0.5, 1e-6)
        predicted_acceleration = np.diff(predicted_velocity, axis=1) / acceleration_dt[..., None, None]
        jerk_dt = np.maximum(
            (dt[:, 2:] + 2.0 * dt[:, 1:-1] + dt[:, :-2]) * 0.25,
            1e-6,
        )
        predicted_jerk = np.diff(predicted_acceleration, axis=1) / jerk_dt[..., None, None]
        jitter_by_frame[:, 3:] = np.linalg.norm(predicted_jerk, axis=-1).mean(axis=-1)

    return {
        "summary": {
            "mpjre_deg": masked_scalar_mean(rotation_error_deg, frame_mask[..., None]),
            "mpjve_cmps": masked_scalar_mean(velocity_error_cmps, frame_mask[..., None]),
            "jitter_mps3": masked_scalar_mean(jitter_by_frame, frame_mask),
        },
        "rotation_error_deg": rotation_error_deg,
        "velocity_error_cmps": velocity_error_cmps,
        "jitter_mps3_by_frame": jitter_by_frame,
    }


def _tracker_state_flags(sensor_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(sensor_valid, dtype=bool)
    full_six = valid.all(axis=-1)
    standard_pattern = np.asarray([True, True, True, False, False, False], dtype=bool)
    standard_three = (valid == standard_pattern).all(axis=-1)
    return full_six, standard_three


def _transition_window_mask(events: np.ndarray, frame_count: int, duration_frames: int = 30) -> np.ndarray:
    result = np.zeros((events.shape[0], frame_count), dtype=bool)
    for batch_index, frame_index in np.argwhere(events):
        result[batch_index, frame_index : min(frame_count, frame_index + duration_frames)] = True
    return result


def tracker_state_group_masks(
    *,
    sensor_valid: np.ndarray,
    eval_frame_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    valid = np.asarray(sensor_valid, dtype=bool)
    eval_mask = np.asarray(eval_frame_mask, dtype=bool)
    full_six, standard_three = _tracker_state_flags(valid)
    six_to_three_events = np.zeros_like(full_six)
    three_to_six_events = np.zeros_like(full_six)
    six_to_three_events[:, 1:] = full_six[:, :-1] & standard_three[:, 1:]
    three_to_six_events[:, 1:] = standard_three[:, :-1] & full_six[:, 1:]
    return {
        "full_six": full_six & eval_mask,
        "standard_three": standard_three & eval_mask,
        "left_foot_missing": (
            valid[..., HIP_TRACKER_INDEX]
            & (~valid[..., LEFT_FOOT_TRACKER_INDEX])
            & valid[..., RIGHT_FOOT_TRACKER_INDEX]
            & eval_mask
        ),
        "right_foot_missing": (
            valid[..., HIP_TRACKER_INDEX]
            & valid[..., LEFT_FOOT_TRACKER_INDEX]
            & (~valid[..., RIGHT_FOOT_TRACKER_INDEX])
            & eval_mask
        ),
        "hip_missing": (~valid[..., HIP_TRACKER_INDEX]) & eval_mask,
        "transition_6_to_3": _transition_window_mask(
            six_to_three_events, valid.shape[1]
        )
        & eval_mask,
        "transition_3_to_6_reconnect": _transition_window_mask(
            three_to_six_events, valid.shape[1]
        )
        & eval_mask,
    }


def add_motion_group_metrics(
    *,
    result: dict[str, float | int | str | list[float]],
    prefix: str,
    group_mask: np.ndarray,
    joint_error: np.ndarray,
    rotation_error_deg: np.ndarray,
    velocity_error_cmps: np.ndarray,
    jitter_mps3: np.ndarray,
) -> None:
    mask = np.asarray(group_mask, dtype=bool)
    result[f"{prefix}_frames"] = int(mask.sum())
    result[f"{prefix}_mpjpe_cm"] = masked_scalar_mean(joint_error, mask[..., None], scale=100.0)
    result[f"{prefix}_mpjre_deg"] = masked_scalar_mean(rotation_error_deg, mask[..., None])
    result[f"{prefix}_mpjve_cmps"] = masked_scalar_mean(velocity_error_cmps, mask[..., None])
    result[f"{prefix}_jitter_mps3"] = masked_scalar_mean(jitter_mps3, mask)


def _joint_jerk_scalar_by_frame(
    joints: np.ndarray,
    timestamps: np.ndarray | None,
) -> np.ndarray:
    motion = np.asarray(joints, dtype=np.float64)
    result = np.full(motion.shape[:2], np.nan, dtype=np.float64)
    if motion.shape[1] < 4:
        return result
    time = _broadcast_timestamps(timestamps, motion.shape[:2])
    dt = np.maximum(np.diff(time, axis=1), 1e-6)
    velocity = np.diff(motion, axis=1) / dt[..., None, None]
    acceleration_dt = np.maximum((dt[:, 1:] + dt[:, :-1]) * 0.5, 1e-6)
    acceleration = np.diff(velocity, axis=1) / acceleration_dt[..., None, None]
    jerk_dt = np.maximum(
        (dt[:, 2:] + 2.0 * dt[:, 1:-1] + dt[:, :-2]) * 0.25,
        1e-6,
    )
    jerk = np.diff(acceleration, axis=1) / jerk_dt[..., None, None]
    # RPM transition jerk 对关节取 max，再对相同 transition offset 求均值。
    result[:, 3:] = np.linalg.norm(jerk, axis=-1).max(axis=-1)
    return result


def _pj_auj_from_events(
    *,
    predicted_jerk: np.ndarray,
    reference_jerk: np.ndarray,
    events: np.ndarray,
    eval_frame_mask: np.ndarray,
    duration_frames: int = 30,
) -> tuple[float, float, int]:
    predicted_segments = []
    reference_segments = []
    for batch_index, frame_index in np.argwhere(events):
        stop = min(predicted_jerk.shape[1], frame_index + duration_frames)
        pred_segment = np.full(duration_frames, np.nan, dtype=np.float64)
        ref_segment = np.full(duration_frames, np.nan, dtype=np.float64)
        length = stop - frame_index
        valid = np.asarray(eval_frame_mask[batch_index, frame_index:stop], dtype=bool)
        pred_values = predicted_jerk[batch_index, frame_index:stop]
        ref_values = reference_jerk[batch_index, frame_index:stop]
        finite = valid & np.isfinite(pred_values) & np.isfinite(ref_values)
        pred_segment[:length][finite] = pred_values[finite]
        ref_segment[:length][finite] = ref_values[finite]
        if np.isfinite(pred_segment).any():
            predicted_segments.append(pred_segment)
            reference_segments.append(ref_segment)
    if not predicted_segments:
        return 0.0, 0.0, 0
    with np.errstate(invalid="ignore"):
        predicted_curve = np.nanmean(np.stack(predicted_segments), axis=0) * 0.01
        reference_curve = np.nanmean(np.stack(reference_segments), axis=0) * 0.01
    finite = np.isfinite(predicted_curve) & np.isfinite(reference_curve)
    if not finite.any():
        return 0.0, 0.0, len(predicted_segments)
    pj = float(np.max(predicted_curve[finite]))
    auj = float(np.sum(np.abs(predicted_curve[finite] - reference_curve[finite])))
    return pj, auj, len(predicted_segments)


def compute_transition_pj_auj(
    *,
    predicted_joints: np.ndarray,
    reference_joints: np.ndarray,
    sensor_valid: np.ndarray,
    eval_frame_mask: np.ndarray,
    timestamps: np.ndarray | None,
) -> dict[str, float | int]:
    valid = np.asarray(sensor_valid, dtype=bool)
    full_six, standard_three = _tracker_state_flags(valid)
    six_to_three = np.zeros_like(full_six)
    three_to_six = np.zeros_like(full_six)
    six_to_three[:, 1:] = full_six[:, :-1] & standard_three[:, 1:]
    three_to_six[:, 1:] = standard_three[:, :-1] & full_six[:, 1:]
    predicted_jerk = _joint_jerk_scalar_by_frame(predicted_joints, timestamps)
    reference_jerk = _joint_jerk_scalar_by_frame(reference_joints, timestamps)
    pj_63, auj_63, count_63 = _pj_auj_from_events(
        predicted_jerk=predicted_jerk,
        reference_jerk=reference_jerk,
        events=six_to_three,
        eval_frame_mask=eval_frame_mask,
    )
    pj_36, auj_36, count_36 = _pj_auj_from_events(
        predicted_jerk=predicted_jerk,
        reference_jerk=reference_jerk,
        events=three_to_six,
        eval_frame_mask=eval_frame_mask,
    )
    combined = six_to_three | three_to_six
    pj, auj, count = _pj_auj_from_events(
        predicted_jerk=predicted_jerk,
        reference_jerk=reference_jerk,
        events=combined,
        eval_frame_mask=eval_frame_mask,
    )
    return {
        "pj": pj,
        "auj": auj,
        "transition_event_count": count,
        "transition_6_to_3_pj": pj_63,
        "transition_6_to_3_auj": auj_63,
        "transition_6_to_3_event_count": count_63,
        "transition_3_to_6_reconnect_pj": pj_36,
        "transition_3_to_6_reconnect_auj": auj_36,
        "transition_3_to_6_reconnect_event_count": count_36,
    }


def rotate_vectors_to_heading_local(vectors: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    vectors_array = np.asarray(vectors, dtype=np.float64)
    yaw_array = np.asarray(yaw, dtype=np.float64)
    if vectors_array.shape[:2] != yaw_array.shape:
        raise ValueError(f"heading local 对齐 shape 不匹配：{vectors_array.shape[:2]} vs {yaw_array.shape}")
    rotations = make_yaw_rotation_np(yaw_array.reshape(-1)).reshape(*yaw_array.shape, 3, 3)
    return np.einsum("btij,btkj->btki", np.swapaxes(rotations, -1, -2), vectors_array)


def compute_preliminary_geometry_metrics(
    preliminary_joints: np.ndarray,
    reference_joints: np.ndarray,
    preliminary_root_yaw: np.ndarray,
    reference_root_yaw: np.ndarray,
    common_ref_yaw: np.ndarray,
) -> dict[str, np.ndarray]:
    predicted = np.asarray(preliminary_joints, dtype=np.float64)
    reference = np.asarray(reference_joints, dtype=np.float64)
    pelvis = JOINT_INDEX["pelvis"]
    head = JOINT_INDEX["head"]

    predicted_p2h = predicted[..., head, :] - predicted[..., pelvis, :]
    reference_p2h = reference[..., head, :] - reference[..., pelvis, :]
    predicted_p2h_local = rotate_vectors_to_heading_local(
        predicted_p2h[..., None, :], common_ref_yaw
    )[..., 0, :]
    reference_p2h_local = rotate_vectors_to_heading_local(
        reference_p2h[..., None, :], common_ref_yaw
    )[..., 0, :]
    p2h_delta = predicted_p2h_local - reference_p2h_local

    predicted_lower = predicted[..., LOWER_BODY_JOINT_INDICES, :] - predicted[..., pelvis : pelvis + 1, :]
    reference_lower = reference[..., LOWER_BODY_JOINT_INDICES, :] - reference[..., pelvis : pelvis + 1, :]
    predicted_lower_local = rotate_vectors_to_heading_local(predicted_lower, preliminary_root_yaw)
    reference_lower_local = rotate_vectors_to_heading_local(reference_lower, reference_root_yaw)
    return {
        "p2h_xz_error": np.linalg.norm(p2h_delta[..., [0, 2]], axis=-1),
        "p2h_height_error": np.abs(predicted_p2h[..., 1] - reference_p2h[..., 1]),
        "lower_body_aligned_error": np.linalg.norm(
            predicted_lower_local - reference_lower_local,
            axis=-1,
        ),
    }


def compute_lower_body_local_rotation_error_deg(
    predicted_features: np.ndarray,
    reference_features: np.ndarray,
    pose_slice: slice,
) -> np.ndarray:
    predicted_pose = np.asarray(predicted_features[..., pose_slice], dtype=np.float64)
    reference_pose = np.asarray(reference_features[..., pose_slice], dtype=np.float64)
    predicted_pose = predicted_pose.reshape(*predicted_pose.shape[:2], 24, 6)[..., LOWER_BODY_JOINT_INDICES, :]
    reference_pose = reference_pose.reshape(*reference_pose.shape[:2], 24, 6)[..., LOWER_BODY_JOINT_INDICES, :]
    predicted_rotations = rotation_6d_to_matrix_np(predicted_pose)
    reference_rotations = rotation_6d_to_matrix_np(reference_pose)
    relative = np.swapaxes(reference_rotations, -1, -2) @ predicted_rotations
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _broadcast_timestamps(
    timestamps: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray:
    if timestamps is None:
        values = np.arange(shape[1], dtype=np.float64)[None] / 60.0
        return np.broadcast_to(values, shape)
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim == 1:
        values = np.broadcast_to(values[None], shape)
    if values.shape != shape:
        raise ValueError(f"timestamp_seconds 应为 {shape} 或 [{shape[1]}]，实际为 {values.shape}")
    return values


def _masked_vector_norm_mean(values: np.ndarray, mask: np.ndarray) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    mask_array = np.asarray(mask, dtype=bool)
    if values_array.shape[:2] != mask_array.shape:
        raise ValueError(f"时间导数 mask shape 不匹配：{values_array.shape[:2]} vs {mask_array.shape}")
    finite = np.isfinite(values_array).all(axis=(-1, -2))
    valid = mask_array & finite
    if not valid.any():
        return 0.0
    return float(np.linalg.norm(values_array[valid], axis=-1).mean())


def compute_lower_body_temporal_metrics(
    predicted_joints: np.ndarray,
    reference_joints: np.ndarray,
    eval_frame_mask: np.ndarray,
    timestamps: np.ndarray | None,
) -> dict[str, float]:
    predicted = np.asarray(predicted_joints, dtype=np.float64)[..., LOWER_BODY_JOINT_INDICES, :]
    reference = np.asarray(reference_joints, dtype=np.float64)[..., LOWER_BODY_JOINT_INDICES, :]
    frame_mask = np.asarray(eval_frame_mask, dtype=bool)
    if predicted.shape[1] < 4:
        return {
            "lower_body_acceleration_mean_mps2": 0.0,
            "lower_body_jerk_mean_mps3": 0.0,
            "lower_body_acceleration_error_mps2": 0.0,
            "lower_body_jerk_error_mps3": 0.0,
        }
    time = _broadcast_timestamps(timestamps, frame_mask.shape)
    dt = np.maximum(np.diff(time, axis=1), 1e-6)
    predicted_velocity = np.diff(predicted, axis=1) / dt[..., None, None]
    reference_velocity = np.diff(reference, axis=1) / dt[..., None, None]
    acceleration_dt = np.maximum((dt[:, 1:] + dt[:, :-1]) * 0.5, 1e-6)
    predicted_acceleration = np.diff(predicted_velocity, axis=1) / acceleration_dt[..., None, None]
    reference_acceleration = np.diff(reference_velocity, axis=1) / acceleration_dt[..., None, None]
    jerk_dt = np.maximum(
        (dt[:, 2:] + 2.0 * dt[:, 1:-1] + dt[:, :-2]) * 0.25,
        1e-6,
    )
    predicted_jerk = np.diff(predicted_acceleration, axis=1) / jerk_dt[..., None, None]
    reference_jerk = np.diff(reference_acceleration, axis=1) / jerk_dt[..., None, None]
    acceleration_mask = frame_mask[:, 2:] & frame_mask[:, 1:-1] & frame_mask[:, :-2]
    jerk_mask = (
        frame_mask[:, 3:]
        & frame_mask[:, 2:-1]
        & frame_mask[:, 1:-2]
        & frame_mask[:, :-3]
    )
    return {
        "lower_body_acceleration_mean_mps2": _masked_vector_norm_mean(
            predicted_acceleration, acceleration_mask
        ),
        "lower_body_jerk_mean_mps3": _masked_vector_norm_mean(predicted_jerk, jerk_mask),
        "lower_body_acceleration_error_mps2": _masked_vector_norm_mean(
            predicted_acceleration - reference_acceleration,
            acceleration_mask,
        ),
        "lower_body_jerk_error_mps3": _masked_vector_norm_mean(
            predicted_jerk - reference_jerk,
            jerk_mask,
        ),
    }


def no_hip_duration_bucket_masks(sensor_valid: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.asarray(sensor_valid, dtype=bool)
    if valid.ndim != 3 or valid.shape[-1] != 6:
        raise ValueError(f"sensor_valid 应为 [B,T,6]，实际为 {valid.shape}")
    no_hip = ~valid[..., HIP_TRACKER_INDEX]
    bucket_masks = {
        label: np.zeros(no_hip.shape, dtype=bool)
        for label, _, _ in NO_HIP_DURATION_BUCKETS
    }
    for batch_index in range(no_hip.shape[0]):
        frame = 0
        while frame < no_hip.shape[1]:
            if not no_hip[batch_index, frame]:
                frame += 1
                continue
            end = frame + 1
            while end < no_hip.shape[1] and no_hip[batch_index, end]:
                end += 1
            duration = end - frame
            for label, minimum, maximum in NO_HIP_DURATION_BUCKETS:
                if duration >= minimum and (maximum is None or duration <= maximum):
                    bucket_masks[label][batch_index, frame:end] = True
                    break
            frame = end
    return bucket_masks


def add_no_hip_duration_bucket_metrics(
    result: dict[str, float | int | str | list[float]],
    sensor_valid: np.ndarray | None,
    eval_frame_mask: np.ndarray,
    joint_error: np.ndarray,
    yaw_error: np.ndarray,
    root_error: np.ndarray | None,
    preliminary_metrics: dict[str, np.ndarray] | None,
    preliminary_joints: np.ndarray | None,
    lower_rotation_error: np.ndarray,
) -> None:
    if sensor_valid is None:
        bucket_masks = {
            label: np.zeros_like(eval_frame_mask, dtype=bool)
            for label, _, _ in NO_HIP_DURATION_BUCKETS
        }
    else:
        bucket_masks = no_hip_duration_bucket_masks(np.asarray(sensor_valid))
    preliminary_valid = (
        np.zeros_like(eval_frame_mask, dtype=bool)
        if preliminary_joints is None
        else np.isfinite(np.asarray(preliminary_joints)).all(axis=(-1, -2))
    )
    for label, _, _ in NO_HIP_DURATION_BUCKETS:
        mask = bucket_masks[label] & eval_frame_mask
        prefix = f"no_hip_duration_{label}"
        result[f"{prefix}_frames"] = int(mask.sum())
        result[f"{prefix}_mpjpe_cm"] = masked_scalar_mean(
            joint_error,
            mask[..., None],
            scale=100.0,
        )
        result[f"{prefix}_yaw_error_deg"] = masked_scalar_mean(
            yaw_error,
            mask,
            scale=180.0 / np.pi,
        )
        result[f"{prefix}_root_xz_error_cm"] = (
            0.0 if root_error is None else masked_scalar_mean(root_error, mask, scale=100.0)
        )
        result[f"{prefix}_local_rotation_error_deg"] = masked_scalar_mean(
            lower_rotation_error,
            mask[..., None],
        )
        geometry_mask = mask & preliminary_valid
        result[f"{prefix}_p2h_xz_error_cm"] = (
            0.0
            if preliminary_metrics is None
            else masked_scalar_mean(
                preliminary_metrics["p2h_xz_error"], geometry_mask, scale=100.0
            )
        )
        result[f"{prefix}_p2h_height_error_cm"] = (
            0.0
            if preliminary_metrics is None
            else masked_scalar_mean(
                preliminary_metrics["p2h_height_error"], geometry_mask, scale=100.0
            )
        )
        result[f"{prefix}_lower_body_aligned_mpjpe_mm"] = (
            0.0
            if preliminary_metrics is None
            else masked_scalar_mean(
                preliminary_metrics["lower_body_aligned_error"],
                geometry_mask[..., None],
                scale=1000.0,
            )
        )


def read_optional_array(data: np.lib.npyio.NpzFile, name: str) -> np.ndarray | None:
    return np.asarray(data[name]) if name in data.files else None


def root_drift_cm_per_minute(
    root_error: np.ndarray,
    mask: np.ndarray,
    timestamps: np.ndarray | None,
    root_source: np.ndarray | None = None,
) -> float:
    errors = np.asarray(root_error, dtype=np.float64)
    # 后续还会复用同一个 eval_frame_mask 计算 no-Hip/contact 指标，不能原地改写调用方数组。
    valid = np.asarray(mask, dtype=bool).copy()
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
    eval_frame_mask: np.ndarray | None = None,
) -> float:
    foot = np.asarray(foot_positions, dtype=np.float64)
    del floor_y  # 接触由 GT stationary + GT 脚高定义，不能再用预测脚高筛选。
    active = np.asarray(contact, dtype=bool).copy()
    if eval_frame_mask is not None:
        active &= np.asarray(eval_frame_mask, dtype=bool)
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
        "mpjre_deg",
        "mpjve_cmps",
        "jitter_mps3",
        "pj",
        "auj",
        "transition_6_to_3_pj",
        "transition_6_to_3_auj",
        "transition_3_to_6_reconnect_pj",
        "transition_3_to_6_reconnect_auj",
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
        "contact_foot_velocity_left_mps",
        "contact_foot_velocity_right_mps",
        "contact_foot_velocity_mean_mps",
        "floating_foot_ratio",
        "no_hip_contact_foot_velocity_mean_mps",
        "no_hip_floating_foot_ratio",
        "no_hip_preliminary_p2h_xz_error_cm",
        "no_hip_preliminary_p2h_height_error_cm",
        "no_hip_preliminary_lower_body_aligned_mpjpe_mm",
        "no_hip_lower_body_local_rotation_error_deg",
        "lower_body_acceleration_mean_mps2",
        "lower_body_jerk_mean_mps3",
        "lower_body_acceleration_error_mps2",
        "lower_body_jerk_error_mps3",
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
        "stationary_f1_at_0_7",
        "stationary_false_lock_rate_at_0_7",
        "stationary_missed_lock_rate_at_0_7",
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
    bucket_metric_suffixes = (
        "mpjpe_cm",
        "mpjre_deg",
        "mpjve_cmps",
        "jitter_mps3",
        "yaw_error_deg",
        "root_xz_error_cm",
        "local_rotation_error_deg",
        "p2h_xz_error_cm",
        "p2h_height_error_cm",
        "lower_body_aligned_mpjpe_mm",
    )
    metric_names = metric_names + tuple(
        f"no_hip_duration_{label}_{suffix}"
        for label, _, _ in NO_HIP_DURATION_BUCKETS
        for suffix in bucket_metric_suffixes
    )
    state_group_names = (
        "full_six",
        "standard_three",
        "left_foot_missing",
        "right_foot_missing",
        "hip_missing",
        "transition_6_to_3",
        "transition_3_to_6_reconnect",
    )
    metric_names = metric_names + tuple(
        f"{group}_{suffix}"
        for group in state_group_names
        for suffix in ("mpjpe_cm", "mpjre_deg", "mpjve_cmps", "jitter_mps3")
    )
    summary: dict[str, float | int] = {
        "file_count": len(results),
        "frames": int(sum(int(item["frames"]) for item in results)),
        "evaluated_frames": int(sum(int(item.get("evaluated_frames", item["frames"])) for item in results)),
        "warmup_frames": int(sum(int(item.get("warmup_frames", 0)) for item in results)),
    }
    for name in metric_names:
        summary[name] = float(np.mean([float(item.get(name, 0.0)) for item in results]))
    for name in (
        "no_hip_evaluated_frames",
        "no_hip_preliminary_frames",
        "transition_event_count",
        "transition_6_to_3_event_count",
        "transition_3_to_6_reconnect_event_count",
        *(f"{group}_frames" for group in state_group_names),
        *(f"no_hip_duration_{label}_frames" for label, _, _ in NO_HIP_DURATION_BUCKETS),
    ):
        summary[name] = int(sum(int(item.get(name, 0)) for item in results))
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
