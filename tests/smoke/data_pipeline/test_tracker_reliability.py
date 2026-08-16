from __future__ import annotations

import numpy as np
import pytest
import torch

from data_loaders.realtime_pose_config import TrackerReliabilityConfig
from data_loaders.realtime_pose_ik import (
    DIRECTION_ONLY,
    INHERITED,
    build_ik_joint_source_reliability,
)
from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    HIP_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
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
    compute_tracker_online_confidence_np,
    compute_tracker_online_confidence_torch,
    compute_tracker_reliability_np,
    compute_ik_joint_confidence_torch,
)
from data_loaders.tracker_timeline import (
    build_isolated_condition_timeline,
    build_task_config_plan,
    build_tracker_timeline,
    candidate_source_window_starts,
    classify_tracker_frame,
    isolated_condition_eval_mask,
    materialize_task_configurations,
)
from eval.calibrate_realtime_pose_ik import fit_direction_confidence_parameters


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


def test_inpaint_tracker_confidence_uses_validity_and_warmup_only():
    valid = np.asarray([[True, True, True, False, True, True]], dtype=bool)
    d_on = np.asarray([[0, 10, 20, 20, 5, 30]], dtype=np.int64)
    expected = np.asarray([[0.0, 1.0, 1.0, 0.0, 0.5, 1.0]], dtype=np.float32)

    actual_np = compute_tracker_online_confidence_np(valid, d_on, warmup_frames=10)
    actual_torch = compute_tracker_online_confidence_torch(
        torch.from_numpy(valid), torch.from_numpy(d_on), warmup_frames=10
    )

    np.testing.assert_allclose(actual_np, expected)
    torch.testing.assert_close(actual_torch, torch.from_numpy(expected))


def test_ik_confidence_uses_constraint_quality_and_normalized_residual():
    source = torch.ones(1, 24) * 0.8
    constraint = torch.full((1, 24), 3, dtype=torch.long)
    constraint[:, :3] = torch.tensor([0, 2, 2])
    updated = torch.zeros(1, 24, dtype=torch.bool)
    updated[:, :3] = True
    residual = torch.zeros(1, 24)
    residual[:, :3] = torch.tensor([100.0, 0.01, 0.10])
    chain_length = torch.zeros(1, 24)
    chain_length[:, 1:3] = 1.0
    confidence = compute_ik_joint_confidence_torch(
        joint_source_reliability=source,
        constraint_type=constraint,
        updated_mask=updated,
        position_residual=residual,
        chain_length=chain_length,
        direction_only_quality=0.5,
        residual_scale=0.1,
    )

    # 直接 rotation 忽略位置残差，同来源下严格高于方向约束。
    assert confidence[0, 0] == 0.8
    assert confidence[0, 0] > confidence[0, 1] > confidence[0, 2]
    expected = 0.8 * 0.5 * torch.exp(torch.tensor(-0.1))
    torch.testing.assert_close(confidence[0, 1], expected)


def test_ik_confidence_forces_inherited_or_unupdated_joints_to_zero():
    constraint = torch.full((1, 24), 3, dtype=torch.long)
    constraint[:, :3] = torch.tensor([3, 0, 2])
    updated = torch.zeros(1, 24, dtype=torch.bool)
    updated[:, 2] = True
    chain_length = torch.zeros(1, 24)
    chain_length[:, 2] = 1.0
    confidence = compute_ik_joint_confidence_torch(
        joint_source_reliability=torch.ones(1, 24),
        constraint_type=constraint,
        updated_mask=updated,
        position_residual=torch.zeros(1, 24),
        chain_length=chain_length,
        direction_only_quality=0.4,
        residual_scale=0.1,
    )
    torch.testing.assert_close(confidence[0, :2], torch.zeros(2))


def test_multitracker_chain_source_reliability_uses_minimum():
    tracker_source = torch.tensor([[0.8, 0.6, 0.7, 0.3, 0.2, 0.9]])
    constraint = torch.full((1, 24), INHERITED, dtype=torch.long)
    for name in ("spine1", "left_shoulder", "left_hip"):
        constraint[:, JOINT_INDEX[name]] = DIRECTION_ONLY
    joint_source = build_ik_joint_source_reliability(tracker_source, constraint)

    assert joint_source[0, JOINT_INDEX["spine1"]] == torch.minimum(
        tracker_source[0, HEAD_TRACKER_INDEX], tracker_source[0, HIP_TRACKER_INDEX]
    )
    assert joint_source[0, JOINT_INDEX["left_shoulder"]] == tracker_source[
        0, LEFT_HAND_TRACKER_INDEX
    ]
    assert joint_source[0, JOINT_INDEX["left_hip"]] == torch.minimum(
        tracker_source[0, HIP_TRACKER_INDEX],
        tracker_source[0, LEFT_FOOT_TRACKER_INDEX],
    )


def test_offline_ik_calibration_recovers_synthetic_parameters():
    source = np.linspace(0.4, 1.0, 200)
    residual_ratio = np.linspace(0.001, 0.2, 200)
    expected_quality = 0.4
    expected_scale = 0.08
    target = source * expected_quality * np.exp(-residual_ratio / expected_scale)
    rotation_error = 2.0 * np.arccos(np.sqrt(target))
    fitted = fit_direction_confidence_parameters(
        source,
        residual_ratio,
        rotation_error,
    )
    assert fitted["direction_only_quality"] == pytest.approx(expected_quality, rel=0.03)
    assert fitted["residual_scale"] == pytest.approx(expected_scale, rel=0.03)


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


def test_isolated_condition_timelines_only_score_the_requested_condition():
    frame_count = 720
    for condition in TRACKER_PATTERN_CATEGORIES:
        timeline = build_isolated_condition_timeline(
            source_id="source/isolated",
            frame_count=frame_count,
            condition=condition,
            global_seed=10,
        )
        mask = isolated_condition_eval_mask(timeline, condition)
        labels = np.asarray(
            [classify_tracker_frame(timeline, index) for index in range(frame_count)]
        )

        assert mask.shape == (frame_count,)
        assert mask.any()
        assert np.all(labels[mask] == condition)
        if condition in {"fixed_six", "fixed_three"}:
            assert mask.all()
        if condition in {"three_to_six", "six_to_three"}:
            assert int(mask.sum()) == 15
        if condition == SCENARIO_TWO_POINT_DROPOUT_RECONNECT:
            missing = timeline.configured & ~timeline.measured_valid
            assert np.any(missing)
            assert np.all(missing.sum(axis=1) <= 2)
            assert int(mask.sum()) > int(np.any(missing, axis=1).sum())


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
