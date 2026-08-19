from __future__ import annotations

import numpy as np
import pytest

from eval.realtime_pose_metrics import (
    aggregate_rpm_p2_mc_metrics,
    compute_rpm_p2_mc_metrics,
)


def test_rpm_p2_mc_metrics_use_local_smpl22_and_30hz_derivatives():
    fps = 30.0
    frame_count = 4
    predicted_rotations = np.repeat(
        np.eye(3, dtype=np.float64)[None, None], frame_count * 24, axis=0
    ).reshape(frame_count, 24, 3, 3)
    target_rotations = predicted_rotations.copy()
    angle = np.pi / 2.0
    rotation_z = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # 所有关节施加相同全局旋转后，parent-local 误差只剩 root 的 90 度。
    target_rotations[:] = rotation_z

    frame = np.arange(frame_count, dtype=np.float64)
    positions = np.zeros((frame_count, 24, 3), dtype=np.float64)
    positions[:, :, 0] = (frame**3 / fps**3)[:, None]
    metrics = compute_rpm_p2_mc_metrics(
        predicted_global_rotations=predicted_rotations,
        target_global_rotations=target_rotations,
        predicted_joint_positions=positions,
        target_joint_positions=positions,
        fps=fps,
    )

    assert metrics["mpjre_deg"] == pytest.approx(90.0 / (22 * 3))
    assert metrics["mpjpe_cm"] == pytest.approx(0.0)
    assert metrics["mpjve_cm_per_s"] == pytest.approx(0.0)
    assert metrics["pred_jitter_m_per_s3"] == pytest.approx(6.0)
    assert metrics["gt_jitter_m_per_s3"] == pytest.approx(6.0)


def test_rpm_p2_mc_mpjve_reports_centimeters_per_second():
    rotations = np.repeat(
        np.eye(3, dtype=np.float64)[None, None], 3 * 24, axis=0
    ).reshape(3, 24, 3, 3)
    predicted_positions = np.zeros((3, 24, 3), dtype=np.float64)
    target_positions = predicted_positions.copy()
    target_positions[:, :, 0] = np.arange(3, dtype=np.float64)[:, None] / 30.0
    metrics = compute_rpm_p2_mc_metrics(
        predicted_global_rotations=rotations,
        target_global_rotations=rotations,
        predicted_joint_positions=predicted_positions,
        target_joint_positions=target_positions,
        fps=30.0,
    )

    assert metrics["mpjve_cm_per_s"] == pytest.approx(100.0)
    assert metrics["pred_jitter_m_per_s3"] is None
    assert metrics["gt_jitter_m_per_s3"] is None


def test_rpm_p2_mc_aggregation_is_sequence_weighted_and_skips_missing_derivatives():
    summary = aggregate_rpm_p2_mc_metrics(
        [
            {
                "mpjre_deg": 1.0,
                "mpjpe_cm": 2.0,
                "mpjve_cm_per_s": None,
                "pred_jitter_m_per_s3": None,
                "gt_jitter_m_per_s3": None,
            },
            {
                "mpjre_deg": 3.0,
                "mpjpe_cm": 4.0,
                "mpjve_cm_per_s": 5.0,
                "pred_jitter_m_per_s3": 6.0,
                "gt_jitter_m_per_s3": 7.0,
            },
        ]
    )

    assert summary["mpjre_deg"] == pytest.approx(2.0)
    assert summary["mpjpe_cm"] == pytest.approx(3.0)
    assert summary["mpjve_cm_per_s"] == pytest.approx(5.0)
    assert summary["sequence_counts"]["mpjre_deg"] == 2
    assert summary["sequence_counts"]["mpjve_cm_per_s"] == 1
