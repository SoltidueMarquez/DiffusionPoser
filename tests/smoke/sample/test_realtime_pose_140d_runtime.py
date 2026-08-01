from __future__ import annotations

import numpy as np

from data_loaders.generate_realtime_pose_tasks import compute_source_joint_rotations_world
from data_loaders.realtime_pose_geometry import build_pose_target_np, extract_forward_yaw_np
from data_loaders.realtime_pose_kinematics import JOINT_INDEX, rotation_6d_to_matrix_np
from sample.realtime_pose_runtime import (
    WorldPoseState,
    advance_missing_age,
    build_online_conditioning,
    decode_and_resolve_pose,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_online_conditioning_and_head_first_root_resolver():
    source = build_toy_realtime_source(frame_count=80)
    rotations_world = compute_source_joint_rotations_world(source)
    pose_history = [
        WorldPoseState(
            joint_rotations_world=rotations_world[index],
            root_yaw_world=float(source["root_yaw"][index]),
            hip_height=float(source["pelvis_height"][index, 0]),
            root_position_world=source["root_pos_world"][index],
        )
        for index in range(60)
    ]
    configured = np.ones((61, 6), dtype=bool)
    measured = configured.copy()
    age = np.zeros((61, 6), dtype=np.int64)
    conditioning = build_online_conditioning(
        pose_history,
        source["tracker_pos_world"][:61],
        source["tracker_rot_world_6d"][:61],
        configured,
        measured,
        age,
        floor_y=0.0,
    )
    assert conditioning["pose_history"].shape == (60, 140)
    assert conditioning["tracker_window"].shape == (61, 6, 12)
    np.testing.assert_allclose(
        conditioning["tracker_window_raw"][-1, 0, [0, 2]],
        0.0,
        atol=1e-6,
    )

    head_rotations = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"][:61, 0])
    current_head_yaw = float(extract_forward_yaw_np(head_rotations)[-1])
    target = build_pose_target_np(
        rotations_world[60:61],
        source["root_yaw"][60:61],
        current_head_yaw,
    )[0]
    resolved = decode_and_resolve_pose(
        target,
        conditioning["tracker_window_raw"][-1],
        current_head_yaw,
        source["tracker_pos_world"][60, 0],
        0.0,
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
    np.testing.assert_allclose(
        resolved.joints_world[JOINT_INDEX["pelvis"]],
        source["tracker_pos_world"][60, 3],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        resolved.root_yaw_world,
        source["root_yaw"][60],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        resolved.hip_height,
        source["pelvis_height"][60, 0],
        atol=1e-5,
    )
    assert resolved.known_rotation_max_error < 1e-5

    tracker_without_hip = conditioning["tracker_window_raw"][-1].copy()
    tracker_without_hip[3, :9] = 0.0
    tracker_without_hip[3, 10] = 0.0
    tracker_without_hip[3, 11] = 1.0 / 60.0
    resolved_without_hip = decode_and_resolve_pose(
        target,
        tracker_without_hip,
        current_head_yaw,
        source["tracker_pos_world"][60, 0],
        0.0,
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
    )
    np.testing.assert_allclose(
        resolved_without_hip.joints_world[JOINT_INDEX["head"]],
        source["tracker_pos_world"][60, 0],
        atol=1e-5,
    )
    np.testing.assert_allclose(resolved_without_hip.root_position_world[1], 0.0, atol=1e-7)


def test_online_missing_age_never_uses_stale_measurement():
    previous = np.zeros(6, dtype=np.int64)
    configured = np.asarray([1, 1, 1, 0, 0, 0], dtype=bool)
    valid = configured.copy()
    valid[1:3] = False
    first = advance_missing_age(previous, configured, valid)
    second = advance_missing_age(first, configured, valid)
    assert first.tolist() == [0, 1, 1, 0, 0, 0]
    assert second.tolist() == [0, 2, 2, 0, 0, 0]
    reconnected = advance_missing_age(second, configured, configured)
    assert reconnected.tolist() == [0, 0, 0, 0, 0, 0]
