from __future__ import annotations

import numpy as np
import torch

from data_loaders.generate_realtime_pose_tasks import compute_source_joint_rotations_world
from data_loaders.realtime_pose_geometry import (
    assemble_tracker_features_np,
    build_pose_target_np,
    build_tracker_measurements_np,
    extract_forward_yaw_np,
    pelvis_relative_joint_positions_torch,
)
from data_loaders.realtime_pose_ik import (
    build_current_ik_pose,
    shortest_arc_rotation,
    solve_fabrik_chain,
)
from data_loaders.realtime_pose_kinematics import (
    rotation_6d_to_matrix_np,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import LEFT_HAND_TRACKER_INDEX
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_fixed_two_pass_fabrik_preserves_lengths_and_reaches_target():
    points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    target = torch.tensor([[1.0, 1.0, 0.0]])
    solved = solve_fabrik_chain(points, target, torch.tensor([True]), iterations=2)

    torch.testing.assert_close(
        torch.linalg.norm(solved[:, 1:] - solved[:, :-1], dim=-1),
        torch.ones(1, 2),
        atol=1e-5,
        rtol=0.0,
    )
    assert torch.linalg.norm(solved[:, -1] - target, dim=-1).item() < 1e-3


def test_fabrik_unreachable_and_missing_targets_are_finite_and_nonworsening():
    points = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ]
    )
    target = torch.tensor([[4.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    active = torch.tensor([True, False])
    solved = solve_fabrik_chain(points, target, active, iterations=2)
    before = torch.linalg.norm(points[:, -1] - target, dim=-1)
    after = torch.linalg.norm(solved[:, -1] - target, dim=-1)

    assert torch.isfinite(solved).all()
    assert after[0] <= before[0]
    torch.testing.assert_close(solved[1], points[1])


def test_shortest_arc_is_deterministic_and_keeps_nearby_bends_continuous():
    source = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 1e-4]])
    rotation = shortest_arc_rotation(source, target)
    mapped = torch.einsum("bij,bj->bi", rotation, source)

    torch.testing.assert_close(
        torch.nn.functional.normalize(mapped, dim=-1),
        torch.nn.functional.normalize(target, dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.det(rotation).min() > 0.9999
    assert torch.linalg.norm(rotation[0] - rotation[1]) < 1e-3

    # shortest-arc 通过左乘共同 swing 更新 global rotation，因此旧姿态之间的
    # 轴向 twist 相对量保持不变，不会在相邻帧随机翻转。
    angle = torch.tensor(0.4)
    twist = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(angle), -torch.sin(angle)],
            [0.0, torch.sin(angle), torch.cos(angle)],
        ]
    )
    updated_plain = rotation[0]
    updated_twist = rotation[0] @ twist
    relative = updated_plain.transpose(0, 1) @ updated_twist
    torch.testing.assert_close(relative, twist, atol=1e-6, rtol=1e-6)


def _toy_ik_inputs():
    source = build_toy_realtime_source(frame_count=2)
    rotations_world = compute_source_joint_rotations_world(source)
    tracker_rotations = source["tracker_rot_world_6d"][1:2]
    head_yaw = float(
        extract_forward_yaw_np(
            rotation_6d_to_matrix_np(tracker_rotations[:, 0]), initial_yaw=0.0
        )[0]
    )
    previous_pose = build_pose_target_np(rotations_world[0:1], head_yaw)
    tracker_measurements = build_tracker_measurements_np(
        source["tracker_pos_world"][1:2],
        tracker_rotations,
        source["tracker_pos_world"][1, 0],
        float(source["root_pos_world"][1, 1]),
        head_yaw,
    )
    configured = np.ones((1, 6), dtype=bool)
    measured = np.ones((1, 6), dtype=bool)
    tracker_raw = assemble_tracker_features_np(
        tracker_measurements,
        configured,
        measured,
        np.zeros((1, 6), dtype=np.int64),
        np.full((1, 6), 60, dtype=np.int64),
    )
    return source, previous_pose, tracker_raw


def test_current_ik_pose_is_finite_when_a_chain_tracker_is_missing():
    source, previous_pose, tracker_raw = _toy_ik_inputs()
    tracker_raw[0, LEFT_HAND_TRACKER_INDEX, 10] = 0.0
    tracker_raw[0, LEFT_HAND_TRACKER_INDEX, :9] = 0.0
    current_pose = build_current_ik_pose(
        previous_pose_raw=torch.from_numpy(previous_pose),
        previous_pose_valid=torch.tensor([True]),
        current_tracker_raw=torch.from_numpy(tracker_raw),
        joint_offsets_parent=torch.from_numpy(source["joint_offsets_parent"])[None],
        joint_rest_local_rotations_6d=torch.from_numpy(
            source["joint_rest_local_rotations_6d"]
        )[None],
    )

    assert current_pose.shape == (1, 144)
    assert torch.isfinite(current_pose).all()

    rotations = rotation_6d_to_matrix_torch(current_pose.reshape(1, 24, 6))
    joints = pelvis_relative_joint_positions_torch(
        rotations, torch.from_numpy(source["joint_offsets_parent"])[None]
    )
    lengths = torch.linalg.norm(
        joints[:, 1:] - joints[:, :-1], dim=-1
    )
    assert torch.isfinite(lengths).all()
