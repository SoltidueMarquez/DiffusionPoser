from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from data_loaders.realtime_pose_kinematics import (
    global_to_parent_local_rotations,
)


RPM_BODY_JOINT_COUNT = 22
RPM_P2_MC_METRIC_KEYS = (
    "mpjre_deg",
    "mpjpe_cm",
    "mpjve_cm_per_s",
    "pred_jitter_m_per_s3",
    "gt_jitter_m_per_s3",
)
RPM_TRANSITION_TYPES = ("tracking_to_synthesis", "synthesis_to_tracking")


def compute_rpm_p2_mc_metrics(
    *,
    predicted_global_rotations: np.ndarray,
    target_global_rotations: np.ndarray,
    predicted_joint_positions: np.ndarray,
    target_joint_positions: np.ndarray,
    fps: float,
) -> dict[str, float | None]:
    """计算一条序列的普通 RPM-P2/MC 指标。

    旋转输入为 `[T,24,3,3]`，位置输入为 `[T,J,3]`。按照当前实验假设，
    Body-FBX 与 SMPL24 共用关节顺序和局部坐标；RPM 指标只使用前 22 个关节。
    MPJRE 复现官方对 parent-local axis-angle 分量取周期化绝对误差的定义。
    """

    predicted_rotations = _validate_rotations(
        predicted_global_rotations, "predicted_global_rotations"
    )
    target_rotations = _validate_rotations(
        target_global_rotations, "target_global_rotations"
    )
    predicted_positions = _validate_positions(
        predicted_joint_positions, "predicted_joint_positions"
    )
    target_positions = _validate_positions(
        target_joint_positions, "target_joint_positions"
    )
    frame_count = predicted_rotations.shape[0]
    if target_rotations.shape[0] != frame_count:
        raise ValueError("预测与 GT rotation 帧数必须一致。")
    if (
        predicted_positions.shape[0] != frame_count
        or target_positions.shape[0] != frame_count
    ):
        raise ValueError("rotation 与 position 帧数必须一致。")
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps 必须是有限正数。")

    predicted_local = global_to_parent_local_rotations(predicted_rotations)[
        :, :RPM_BODY_JOINT_COUNT
    ]
    target_local = global_to_parent_local_rotations(target_rotations)[
        :, :RPM_BODY_JOINT_COUNT
    ]
    predicted_axis_angle = Rotation.from_matrix(
        predicted_local.reshape(-1, 3, 3)
    ).as_rotvec().reshape(frame_count, RPM_BODY_JOINT_COUNT, 3)
    target_axis_angle = Rotation.from_matrix(
        target_local.reshape(-1, 3, 3)
    ).as_rotvec().reshape(frame_count, RPM_BODY_JOINT_COUNT, 3)
    rotation_difference = (
        target_axis_angle - predicted_axis_angle + math.pi
    ) % (2.0 * math.pi) - math.pi

    predicted_positions = predicted_positions[:, :RPM_BODY_JOINT_COUNT]
    target_positions = target_positions[:, :RPM_BODY_JOINT_COUNT]
    position_error = np.linalg.norm(
        target_positions - predicted_positions, axis=-1
    )

    result: dict[str, float | None] = {
        "mpjre_deg": float(np.degrees(np.abs(rotation_difference)).mean()),
        "mpjpe_cm": float(position_error.mean() * 100.0),
        "mpjve_cm_per_s": None,
        "pred_jitter_m_per_s3": None,
        "gt_jitter_m_per_s3": None,
    }
    if frame_count >= 2:
        predicted_velocity = np.diff(predicted_positions, axis=0) * float(fps)
        target_velocity = np.diff(target_positions, axis=0) * float(fps)
        result["mpjve_cm_per_s"] = float(
            np.linalg.norm(target_velocity - predicted_velocity, axis=-1).mean()
            * 100.0
        )
    if frame_count >= 4:
        result["pred_jitter_m_per_s3"] = _mean_jitter(
            predicted_positions, float(fps)
        )
        result["gt_jitter_m_per_s3"] = _mean_jitter(
            target_positions, float(fps)
        )
    return result


def aggregate_rpm_p2_mc_metrics(
    sequence_metrics: Iterable[dict[str, float | None]],
) -> dict[str, object]:
    """按 RPM 官方口径先算每条序列，再对有效序列做等权平均。"""

    sums = {key: 0.0 for key in RPM_P2_MC_METRIC_KEYS}
    counts = {key: 0 for key in RPM_P2_MC_METRIC_KEYS}
    for metrics in sequence_metrics:
        for key in RPM_P2_MC_METRIC_KEYS:
            value = metrics.get(key)
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                continue
            sums[key] += number
            counts[key] += 1
    return {
        **{
            key: sums[key] / counts[key] if counts[key] > 0 else None
            for key in RPM_P2_MC_METRIC_KEYS
        },
        "sequence_counts": counts,
    }


def extract_rpm_transition_jerk_segments(
    *,
    joint_positions: np.ndarray,
    hand_gap_intervals: Iterable[Iterable[tuple[int, int]]],
    fps: float,
    transition_seconds: float = 0.5,
) -> dict[str, np.ndarray]:
    """按 RPM 官方代码提取掉线/重连后的逐帧最大关节 Jerk。

    `joint_positions` 为正式计分区间的 `[T,J,3]`；gap 使用同一区间内的
    左闭右开帧索引。官方发布代码对每个手部 gap 分别采样，并在每一帧对
    22 个关节取最大 Jerk，因此双手同时掉线会保留为两个 transition event。
    """

    positions = _validate_positions(joint_positions, "joint_positions")[
        :, :RPM_BODY_JOINT_COUNT
    ]
    if not math.isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps 必须是有限正数。")
    duration = float(transition_seconds)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("transition_seconds 必须是有限正数。")
    window_frames = int(duration * float(fps))
    if window_frames <= 0:
        raise ValueError("transition_seconds 对应的 transition window 为空。")

    jerk = (
        positions[3:]
        - 3.0 * positions[2:-1]
        + 3.0 * positions[1:-2]
        - positions[:-3]
    ) * (float(fps) ** 3)
    # [T-3,22] -> [T-3]；这里复现 RPM transition_jerk 的 max-joint 口径。
    jerk = np.linalg.norm(jerk, axis=-1).max(axis=-1)
    segments: dict[str, list[np.ndarray]] = {
        transition_type: [] for transition_type in RPM_TRANSITION_TYPES
    }
    for hand_gaps in hand_gap_intervals:
        for raw_start, raw_end in hand_gaps:
            start = int(raw_start) - 3
            end = int(raw_end) - 3
            gap_frames = end - start
            if (
                start > 0
                and gap_frames >= window_frames
                and start + window_frames <= jerk.shape[0]
            ):
                segments["tracking_to_synthesis"].append(
                    jerk[start : start + window_frames]
                )
            if end > 0 and end + window_frames < jerk.shape[0]:
                segments["synthesis_to_tracking"].append(
                    jerk[end : end + window_frames]
                )
    return {
        transition_type: (
            np.stack(values, axis=0).astype(np.float64)
            if values
            else np.empty((0, window_frames), dtype=np.float64)
        )
        for transition_type, values in segments.items()
    }


def aggregate_rpm_transition_jerk_segments(
    *,
    predicted_segments: Iterable[dict[str, np.ndarray]],
    target_segments: Iterable[dict[str, np.ndarray]],
) -> dict[str, dict[str, object]]:
    """复现 RPM 发布代码对 transition jerk 曲线计算 PJ 与 AUJ 的方式。"""

    predicted_values = list(predicted_segments)
    target_values = list(target_segments)
    if len(predicted_values) != len(target_values):
        raise ValueError("预测与 GT transition segment 组数必须一致。")
    result: dict[str, dict[str, object]] = {}
    for transition_type in RPM_TRANSITION_TYPES:
        pred_arrays = [
            np.asarray(values[transition_type], dtype=np.float64)
            for values in predicted_values
            if np.asarray(values[transition_type]).shape[0] > 0
        ]
        gt_arrays = [
            np.asarray(values[transition_type], dtype=np.float64)
            for values in target_values
            if np.asarray(values[transition_type]).shape[0] > 0
        ]
        if not pred_arrays or not gt_arrays:
            result[transition_type] = {
                "event_count": 0,
                "pj_1e2_m_per_s3": None,
                "auj_1e2_m_per_s2": None,
                "predicted_mean_curve_1e2_m_per_s3": [],
                "target_mean_curve_1e2_m_per_s3": [],
            }
            continue
        predicted = np.concatenate(pred_arrays, axis=0)
        target = np.concatenate(gt_arrays, axis=0)
        if predicted.shape != target.shape:
            raise ValueError(
                "预测与 GT transition segment 形状必须一致："
                f"{predicted.shape} != {target.shape}。"
            )
        # 官方实现先对所有 event 求逐帧均值，再乘 0.01，并直接对曲线差求和。
        predicted_curve = predicted.mean(axis=0) * 0.01
        target_curve = target.mean(axis=0) * 0.01
        result[transition_type] = {
            "event_count": int(predicted.shape[0]),
            "pj_1e2_m_per_s3": float(predicted_curve.max()),
            "auj_1e2_m_per_s2": float(
                np.abs(predicted_curve - target_curve).sum()
            ),
            "predicted_mean_curve_1e2_m_per_s3": predicted_curve.tolist(),
            "target_mean_curve_1e2_m_per_s3": target_curve.tolist(),
        }
    return result


def _mean_jitter(positions: np.ndarray, fps: float) -> float:
    jerk = (
        positions[3:]
        - 3.0 * positions[2:-1]
        + 3.0 * positions[1:-2]
        - positions[:-3]
    ) * (float(fps) ** 3)
    return float(np.linalg.norm(jerk, axis=-1).mean())


def _validate_rotations(value: np.ndarray, name: str) -> np.ndarray:
    rotations = np.asarray(value, dtype=np.float64)
    if rotations.ndim != 4 or rotations.shape[1:] != (24, 3, 3):
        raise ValueError(f"{name} 必须为 [T,24,3,3]，实际为 {rotations.shape}。")
    if rotations.shape[0] <= 0 or not np.isfinite(rotations).all():
        raise ValueError(f"{name} 必须包含有限的非空序列。")
    return rotations


def _validate_positions(value: np.ndarray, name: str) -> np.ndarray:
    positions = np.asarray(value, dtype=np.float64)
    if (
        positions.ndim != 3
        or positions.shape[1] < RPM_BODY_JOINT_COUNT
        or positions.shape[2] != 3
    ):
        raise ValueError(f"{name} 必须为 [T,J,3] 且 J>=22，实际为 {positions.shape}。")
    if positions.shape[0] <= 0 or not np.isfinite(positions).all():
        raise ValueError(f"{name} 必须包含有限的非空序列。")
    return positions
