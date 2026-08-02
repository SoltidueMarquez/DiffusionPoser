from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_config import TrackerReliabilityConfig
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    REALTIME_POSE_TARGET_START,
    RIGHT_HAND_TRACKER_INDEX,
    SCENARIO_TWO_POINT_DROPOUT_RECONNECT,
    TRACKER_PATTERN_CATEGORIES,
)
from data_loaders.generate_realtime_pose_tasks import (
    select_window_starts,
    validate_two_point_phase_balance,
)
from data_loaders.tracker_reliability import (
    compute_hard_rotation_state_np,
    compute_region_coverage_np,
    compute_tracker_reliability_np,
)
from data_loaders.tracker_timeline import (
    build_task_config_plan,
    build_tracker_timeline,
    candidate_source_window_starts,
    materialize_task_configurations,
)


def test_five_scenarios_and_two_point_dropout_contract():
    phases: list[str] = []
    for index in range(200):
        plans = build_task_config_plan(f"task-{index}", global_seed=10, max_rollout_steps=4)
        assert tuple(plan["scenario"] for plan in plans) == TRACKER_PATTERN_CATEGORIES
        dropout = plans[-1]
        assert dropout["scenario"] == SCENARIO_TWO_POINT_DROPOUT_RECONNECT
        assert len(dropout["dropped_trackers"]) == 2
        assert len(set(dropout["dropped_trackers"])) == 2
        assert HEAD_TRACKER_INDEX not in dropout["dropped_trackers"]
        assert 5 <= int(dropout["dropout_duration"]) <= 30
        phases.append(str(dropout["target_phase"]))
    # 计划由稳定 hash 独立抽样；用宽松边界保护 1:1 目标而不制造脆弱统计测试。
    assert 70 <= phases.count("dropout") <= 130
    assert 70 <= phases.count("reconnect") <= 130


def test_materialized_dropout_is_synchronous_and_reconnects_soft_first():
    plans = build_task_config_plan("contract", global_seed=3, max_rollout_steps=4)
    states = materialize_task_configurations(plans, frame_count=64)
    plan = plans[-1]
    dropped = np.asarray(plan["dropped_trackers"], dtype=np.int64)
    valid = states.measured_valid[-1]
    missing_rows = np.flatnonzero(~valid[:, dropped[0]])
    np.testing.assert_array_equal(~valid[:, dropped[0]], ~valid[:, dropped[1]])
    assert missing_rows.size <= int(plan["dropout_duration"])
    assert states.hard_rotation_state[-1, :, HEAD_TRACKER_INDEX].all()
    if missing_rows.size and missing_rows[-1] + 1 < valid.shape[0]:
        reconnect = int(missing_rows[-1] + 1)
        assert states.d_on[-1, reconnect, dropped[0]] == 1
        assert not states.hard_rotation_state[-1, reconnect, dropped[0]]


def test_reliability_uses_validity_and_modality_specific_recovery_windows():
    configured = np.ones((1, 6), dtype=bool)
    measured = np.ones_like(configured)
    measured[0, 2] = False
    d_on = np.full((1, 6), 5, dtype=np.uint8)
    d_on[0, 2] = 0
    d_on[0, 3] = 20
    config = TrackerReliabilityConfig(d_warm_pos=10, d_warm_rot=20)
    kappa_pos, kappa_rot = compute_tracker_reliability_np(
        configured,
        measured,
        d_on,
        config=config,
    )
    assert np.isclose(kappa_pos[0, 1], 0.5)
    assert np.isclose(kappa_rot[0, 1], 0.25)
    assert kappa_pos[0, 2] == 0.0 and kappa_rot[0, 2] == 0.0
    assert kappa_pos[0, 3] == 1.0 and kappa_rot[0, 3] == 1.0
    rho_pos, rho_rot = compute_region_coverage_np(kappa_pos, kappa_rot)
    assert np.isclose(rho_pos[0, 1], 0.5)
    assert np.isclose(rho_rot[0, 1], 0.25)
    # Head position不覆盖 torso，但 Head rotation覆盖 torso。
    assert np.isclose(rho_pos[0, 0], 1.0)  # Hip position 仍完整覆盖 torso。
    assert np.isclose(rho_rot[0, 0], 1.0)


def test_hard_rotation_uses_validity_and_recovery_duration_only():
    configured = np.ones((4, 6), dtype=bool)
    measured = np.ones_like(configured)
    d_on = np.full((4, 6), 14, dtype=np.uint8)
    d_on[1] = 15
    measured[2, 3] = False
    d_on[2, 3] = 0
    d_on[3, 3] = 1

    hard = compute_hard_rotation_state_np(configured, measured, d_on)

    assert hard[:, HEAD_TRACKER_INDEX].all()
    assert not hard[0, 1:].any()
    assert hard[1].all()
    assert not hard[2, 3]
    assert not hard[3, 3]


def test_source_absolute_timeline_overlap_is_identical():
    timeline = build_tracker_timeline("source/A", frame_count=720, global_seed=9)
    first = timeline.window(100, 100)
    second = timeline.window(140, 100)
    np.testing.assert_array_equal(first.configured[40:], second.configured[:60])
    np.testing.assert_array_equal(first.measured_valid[40:], second.measured_valid[:60])
    np.testing.assert_array_equal(first.d_on[40:], second.d_on[:60])
    np.testing.assert_array_equal(first.hard_rotation_state[40:], second.hard_rotation_state[:60])


def test_task_generation_uses_source_absolute_events_for_overlapping_windows():
    source_id = "source/absolute"
    starts = candidate_source_window_starts(source_id, 720, 4, global_seed=12)
    adjacent = next((first for first in starts if first + 1 in starts), None)
    assert adjacent is not None
    first_plans = build_task_config_plan(
        "first",
        12,
        4,
        source_id=source_id,
        start_frame=adjacent,
        source_frame_count=720,
    )
    second_plans = build_task_config_plan(
        "second",
        12,
        4,
        source_id=source_id,
        start_frame=adjacent + 1,
        source_frame_count=720,
    )
    first = materialize_task_configurations(
        first_plans, frame_count=64, absolute_start_frame=adjacent
    )
    second = materialize_task_configurations(
        second_plans, frame_count=64, absolute_start_frame=adjacent + 1
    )
    np.testing.assert_array_equal(first.configured[:, 1:], second.configured[:, :-1])
    np.testing.assert_array_equal(first.measured_valid[:, 1:], second.measured_valid[:, :-1])
    np.testing.assert_array_equal(first.d_off[:, 1:], second.d_off[:, :-1])
    np.testing.assert_array_equal(first.d_on[:, 1:], second.d_on[:, :-1])
    selected_starts = select_window_starts(
        frame_count=720,
        count=20,
        max_rollout_steps=4,
        global_seed=12,
        split="train",
        source_id=source_id,
    )
    phases = [
        build_task_config_plan(
            f"phase-{start}",
            12,
            4,
            source_id=source_id,
            start_frame=start,
            source_frame_count=720,
        )[-1]["target_phase"]
        for start in selected_starts
    ]
    assert len(selected_starts) == 20
    assert abs(phases.count("dropout") - phases.count("reconnect")) <= 1


def test_reconnect_candidates_cover_full_warmup_and_hand_hard_reentry():
    source_id = "source/full-reconnect"
    starts = candidate_source_window_starts(source_id, 720, 4, global_seed=21)
    recovery_offsets: set[int] = set()
    hard_entry: tuple[int, list[dict]] | None = None

    for start in starts:
        plans = build_task_config_plan(
            f"reconnect-{start}",
            21,
            4,
            source_id=source_id,
            start_frame=start,
            source_frame_count=720,
        )
        dropout = plans[-1]
        if dropout["target_phase"] != "reconnect":
            continue
        reconnect_frame = int(dropout["dropout_start"]) + int(dropout["dropout_duration"])
        target_frame = int(start) + REALTIME_POSE_TARGET_START
        recovery_offset = target_frame - reconnect_frame
        recovery_offsets.add(recovery_offset)
        if recovery_offset == 14:
            hard_entry = (start, plans)

    assert recovery_offsets == set(range(15))
    assert hard_entry is not None

    start, plans = hard_entry
    # 显式改成双手掉线，保护“手部只能通过两点重连覆盖 Hard 重入”的契约。
    hand_plans = [dict(plan) for plan in plans]
    hand_plans[-1]["dropped_trackers"] = [LEFT_HAND_TRACKER_INDEX, RIGHT_HAND_TRACKER_INDEX]
    states = materialize_task_configurations(
        hand_plans,
        frame_count=64,
        absolute_start_frame=start,
    )
    hand_indices = np.asarray([LEFT_HAND_TRACKER_INDEX, RIGHT_HAND_TRACKER_INDEX], dtype=np.int64)
    previous = REALTIME_POSE_TARGET_START - 1
    current = REALTIME_POSE_TARGET_START
    np.testing.assert_array_equal(states.d_on[-1, previous, hand_indices], np.full(2, 14))
    assert not states.hard_rotation_state[-1, previous, hand_indices].any()
    np.testing.assert_array_equal(states.d_on[-1, current, hand_indices], np.full(2, 15))
    assert states.hard_rotation_state[-1, current, hand_indices].all()


def test_split_phase_balance_validation_rejects_large_bias():
    def task(task_id: str, phase: str) -> dict:
        return {
            "task_id": task_id,
            "configs": [
                {
                    "scenario": SCENARIO_TWO_POINT_DROPOUT_RECONNECT,
                    "target_phase": phase,
                }
            ],
        }

    balanced = [task(f"dropout-{index}", "dropout") for index in range(5)]
    balanced += [task(f"reconnect-{index}", "reconnect") for index in range(5)]
    assert validate_two_point_phase_balance(balanced, "train") == {
        "dropout": 5,
        "reconnect": 5,
    }

    biased = [task(f"dropout-{index}", "dropout") for index in range(9)]
    biased.append(task("reconnect-0", "reconnect"))
    with np.testing.assert_raises_regex(RuntimeError, "两点掉线阶段失衡"):
        validate_two_point_phase_balance(biased, "train")
