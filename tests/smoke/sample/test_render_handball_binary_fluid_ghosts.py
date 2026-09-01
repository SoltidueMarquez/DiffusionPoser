from types import SimpleNamespace

import numpy as np
import pytest

from sample import render_handball_binary_fluid_ghosts as ghosts


def test_handball_selected_indices_are_dense_around_switch() -> None:
    selected = ghosts.build_selected_indices(
        transition_index=20,
        frame_count=50,
    )

    np.testing.assert_array_equal(
        selected,
        20 + np.asarray(ghosts.DISPLAY_OFFSETS, dtype=np.int64),
    )
    assert selected[ghosts.TRANSITION_SLOT] == 20
    assert selected[ghosts.TRANSITION_SLOT - 1] == 19
    assert selected[ghosts.TRANSITION_SLOT + 1] == 21


def test_handball_selected_indices_reject_out_of_range_window() -> None:
    with pytest.raises(ValueError, match="越界"):
        ghosts.build_selected_indices(transition_index=4, frame_count=50)


def test_handball_alpha_schedule_emphasizes_transition() -> None:
    selected = np.arange(ghosts.DISPLAY_FRAME_COUNT, dtype=np.int64)
    alphas = ghosts.build_body_alphas(selected)

    assert alphas.shape == (ghosts.DISPLAY_FRAME_COUNT,)
    assert np.argmax(alphas) == ghosts.TRANSITION_SLOT
    assert alphas[ghosts.TRANSITION_SLOT - 1] >= 0.70
    assert alphas[ghosts.TRANSITION_SLOT + 1] >= 0.70


def test_boundary_tracker_step_uses_tracker_joint_mapping() -> None:
    joints = np.zeros((3, 24, 3), dtype=np.float32)
    right_foot_tracker = int(tuple(ghosts.TRACKER_NAMES).index("right_foot"))
    right_foot_joint = int(ghosts.TRACKER_TO_JOINT[right_foot_tracker])
    joints[1, right_foot_joint, 0] = 0.25
    arrays = SimpleNamespace(joints_world=joints)

    result = ghosts.compute_boundary_tracker_steps_cm(
        arrays,
        transition_index=1,
        tracker_indices=np.asarray([right_foot_tracker], dtype=np.int64),
    )

    assert result[right_foot_tracker] == pytest.approx(25.0)


def test_available_tracker_positions_follow_three_to_five_transition() -> None:
    tracker_count = len(ghosts.TRACKER_NAMES)
    positions = np.arange(4 * tracker_count * 3, dtype=np.float32).reshape(
        4, tracker_count, 3
    )
    available = np.zeros((4, tracker_count), dtype=bool)
    available[:, :3] = True
    left_foot = int(tuple(ghosts.TRACKER_NAMES).index("left_foot"))
    right_foot = int(tuple(ghosts.TRACKER_NAMES).index("right_foot"))
    available[2:, [left_foot, right_foot]] = True

    head_points = ghosts.select_available_tracker_positions(
        positions,
        available,
        tracker_index=0,
    )
    left_foot_points = ghosts.select_available_tracker_positions(
        positions,
        available,
        tracker_index=left_foot,
    )
    right_foot_points = ghosts.select_available_tracker_positions(
        positions,
        available,
        tracker_index=right_foot,
    )
    right_foot_without_transition = ghosts.select_available_tracker_positions(
        positions,
        available,
        tracker_index=right_foot,
        excluded_slots=(2,),
    )

    np.testing.assert_array_equal(head_points, positions[:, 0])
    np.testing.assert_array_equal(left_foot_points, positions[2:, left_foot])
    np.testing.assert_array_equal(right_foot_points, positions[2:, right_foot])
    np.testing.assert_array_equal(
        right_foot_without_transition,
        positions[3:, right_foot],
    )


def test_only_foot_trajectories_receive_distinct_colors() -> None:
    left_foot = int(tuple(ghosts.TRACKER_NAMES).index("left_foot"))
    right_foot = int(tuple(ghosts.TRACKER_NAMES).index("right_foot"))

    assert ghosts.tracker_trajectory_color(left_foot) == ghosts.LEFT_FOOT_COLOR
    assert ghosts.tracker_trajectory_color(right_foot) == ghosts.RIGHT_FOOT_COLOR
    with pytest.raises(ValueError, match="仅脚部"):
        ghosts.tracker_trajectory_color(0)
