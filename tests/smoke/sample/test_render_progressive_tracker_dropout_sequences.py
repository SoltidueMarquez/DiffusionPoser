from __future__ import annotations

import numpy as np
import pytest

from sample.evaluate_progressive_tracker_dropout import (
    OPTIONAL_TRACKER_NAME_TO_INDEX,
    build_equal_quarter_tracker_schedule,
)
from sample.evaluate_progressive_tracker_addition import (
    build_equal_quarter_tracker_schedule as build_addition_tracker_schedule,
)
from sample.render_progressive_tracker_dropout_sequences import (
    ProgressiveSequenceResult,
    build_transition_schedule,
    compute_continuity_diagnostics,
    validate_add_order,
    validate_drop_order,
    warmup_tracker_available,
)


def build_result_with_boundary_jump() -> ProgressiveSequenceResult:
    frame_count = 16
    drop_indices = validate_drop_order(("right_foot", "left_foot", "hip"))
    schedule = build_equal_quarter_tracker_schedule(
        scored_frame_count=frame_count,
        drop_order=drop_indices,
    )
    target_positions = np.zeros((frame_count, 24, 3), dtype=np.float32)
    deployed_positions = np.zeros_like(target_positions)
    # 每帧正常移动 1 cm；进入五点阶段时额外跳 4 cm，便于验证边界诊断。
    deployed_positions[:, :, 0] = np.arange(frame_count, dtype=np.float32)[:, None] * 0.01
    deployed_positions[4:, :, 0] += 0.04
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32), (frame_count, 24, 3, 3)
    ).copy()
    return ProgressiveSequenceResult(
        frame_start=30,
        frame_end_exclusive=46,
        tracker_available=schedule.tracker_available,
        stage_indices=schedule.stage_indices,
        target_rotations=rotations,
        target_positions=target_positions,
        deployed_rotations=rotations,
        deployed_positions=deployed_positions,
        deployed_root_yaw=np.zeros((frame_count,), dtype=np.float32),
        tracker_positions=np.zeros((frame_count, 6, 3), dtype=np.float32),
    )


def test_validate_drop_order_requires_each_optional_tracker_once() -> None:
    indices = validate_drop_order(("right_foot", "left_foot", "hip"))
    assert indices == tuple(
        OPTIONAL_TRACKER_NAME_TO_INDEX[name]
        for name in ("right_foot", "left_foot", "hip")
    )
    with pytest.raises(ValueError, match="恰好包含"):
        validate_drop_order(("right_foot", "right_foot", "hip"))


def test_continuity_diagnostics_reports_all_three_switch_boundaries() -> None:
    reports = compute_continuity_diagnostics(build_result_with_boundary_jump())

    assert [report["source_frame"] for report in reports] == [34, 38, 42]
    assert [report["tracker_count"] for report in reports] == [5, 4, 3]
    assert [report["dropped_tracker"] for report in reports] == [
        "right_foot",
        "left_foot",
        "hip",
    ]
    assert reports[0]["predicted_mean_joint_step_cm"] == pytest.approx(5.0)
    assert reports[0]["predicted_step_to_local_median_ratio"] == pytest.approx(5.0)


def test_addition_render_schedule_and_warmup_match_official_protocol() -> None:
    order = ("hip", "left_foot", "right_foot")
    indices = validate_add_order(order)
    render_schedule = build_transition_schedule(
        scored_frame_count=16,
        direction="addition",
        transition_indices=indices,
    )
    official_schedule = build_addition_tracker_schedule(
        scored_frame_count=16,
        add_order=indices,
    )

    np.testing.assert_array_equal(
        render_schedule.tracker_available,
        official_schedule.tracker_available,
    )
    assert render_schedule.tracker_available.sum(axis=1).tolist() == [
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
    ]
    assert warmup_tracker_available("addition").tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_addition_continuity_diagnostics_reports_added_trackers() -> None:
    frame_count = 16
    indices = validate_add_order(("hip", "left_foot", "right_foot"))
    schedule = build_addition_tracker_schedule(
        scored_frame_count=frame_count,
        add_order=indices,
    )
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32), (frame_count, 24, 3, 3)
    ).copy()
    positions = np.zeros((frame_count, 24, 3), dtype=np.float32)
    positions[:, :, 0] = np.arange(frame_count, dtype=np.float32)[:, None] * 0.01
    result = ProgressiveSequenceResult(
        frame_start=30,
        frame_end_exclusive=46,
        tracker_available=schedule.tracker_available,
        stage_indices=schedule.stage_indices,
        target_rotations=rotations,
        target_positions=positions,
        deployed_rotations=rotations,
        deployed_positions=positions,
        deployed_root_yaw=np.zeros((frame_count,), dtype=np.float32),
        tracker_positions=np.zeros((frame_count, 6, 3), dtype=np.float32),
    )

    reports = compute_continuity_diagnostics(result, direction="addition")

    assert [report["tracker_count"] for report in reports] == [4, 5, 6]
    assert [report["added_tracker"] for report in reports] == [
        "hip",
        "left_foot",
        "right_foot",
    ]
    assert all(report["transition"] == "addition" for report in reports)
