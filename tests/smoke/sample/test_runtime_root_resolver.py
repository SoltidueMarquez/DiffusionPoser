from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np, rotation_6d_forward_up_np
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    HIP_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    RIGHT_HAND_TRACKER_INDEX,
    TRACKER_COUNT,
)
from sample.runtime_root_resolver import RootSource, RuntimeRootResolver, RuntimeRootResolverConfig


PELVIS_OFFSET = np.asarray([0.1, 0.9, -0.05], dtype=np.float32)


def fake_fk(root: np.ndarray, yaw: float, pelvis_height: float) -> np.ndarray:
    joints = np.repeat(np.asarray(root, dtype=np.float32)[None], 24, axis=0)
    rotation = make_yaw_rotation_np(np.asarray([yaw], dtype=np.float64))[0]
    pelvis_offset = np.asarray([PELVIS_OFFSET[0], pelvis_height, PELVIS_OFFSET[2]], dtype=np.float64)
    joints[0] = root + (rotation @ pelvis_offset).astype(np.float32)
    head_offset = np.asarray([0.0, pelvis_height + 0.8, 0.15], dtype=np.float64)
    joints[15] = root + (rotation @ head_offset).astype(np.float32)
    return joints


def tracker_rotations(yaw: float = 0.0) -> np.ndarray:
    rotation = make_yaw_rotation_np(np.asarray([yaw], dtype=np.float64))[0]
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None], TRACKER_COUNT, axis=0)
    matrices[HIP_TRACKER_INDEX] = rotation
    return rotation_6d_forward_up_np(matrices).astype(np.float32)


def valid_standard_three() -> np.ndarray:
    valid = np.zeros(TRACKER_COUNT, dtype=bool)
    valid[[HEAD_TRACKER_INDEX, LEFT_HAND_TRACKER_INDEX, RIGHT_HAND_TRACKER_INDEX]] = True
    return valid


def test_missing_hip_uses_head_fk_and_previous_final_yaw_for_delta():
    resolver = RuntimeRootResolver(PELVIS_OFFSET, RuntimeRootResolverConfig(hip_filter_time_constant_seconds=0.0))
    positions = np.zeros((TRACKER_COUNT, 3), dtype=np.float32)
    positions[HEAD_TRACKER_INDEX] = np.asarray([2.0, 1.7, -1.0], dtype=np.float32)

    result = resolver.resolve(
        tracker_pos_world=positions,
        tracker_rot_world_6d=tracker_rotations(),
        sensor_valid=valid_standard_three(),
        timestamp=0.0,
        floor_y=0.0,
        tracking_origin_revision=0,
        model_root_delta_xz_ref=np.asarray([1.0, 0.0]),
        model_yaw_delta_sincos=np.asarray([1.0, 0.0]),
        model_pelvis_height=0.9,
        fk_callback=fake_fk,
    )

    assert result.root_source == RootSource.RESET
    predicted_offset = fake_fk(np.zeros(3, dtype=np.float32), np.pi / 2.0, 0.9)[15]
    np.testing.assert_allclose(
        result.final_root_pos_world[[0, 2]],
        positions[HEAD_TRACKER_INDEX, [0, 2]] - predicted_offset[[0, 2]],
        atol=1e-5,
    )
    np.testing.assert_allclose(result.final_root_delta_xz_ref, 0.0, atol=1e-6)


def test_hip_reconnect_uses_time_based_smoothstep_and_finishes_on_target():
    resolver = RuntimeRootResolver(
        PELVIS_OFFSET,
        RuntimeRootResolverConfig(reconnect_duration_seconds=0.1, hip_filter_time_constant_seconds=0.0),
    )
    positions = np.zeros((TRACKER_COUNT, 3), dtype=np.float32)
    positions[HEAD_TRACKER_INDEX] = np.asarray([0.0, 1.7, 0.0], dtype=np.float32)
    missing = valid_standard_three()
    resolver.resolve(
        tracker_pos_world=positions,
        tracker_rot_world_6d=tracker_rotations(),
        sensor_valid=missing,
        timestamp=0.0,
        floor_y=0.0,
        tracking_origin_revision=0,
        model_root_delta_xz_ref=np.zeros(2),
        model_yaw_delta_sincos=np.asarray([0.0, 1.0]),
        model_pelvis_height=0.9,
        fk_callback=fake_fk,
    )

    valid = np.ones(TRACKER_COUNT, dtype=bool)
    target_root = np.asarray([1.5, 0.0, -0.5], dtype=np.float32)
    yaw = 0.4
    rotation = make_yaw_rotation_np(np.asarray([yaw], dtype=np.float64))[0]
    positions[HIP_TRACKER_INDEX] = target_root + (rotation @ PELVIS_OFFSET.astype(np.float64)).astype(np.float32)
    midway = resolver.resolve(
        tracker_pos_world=positions,
        tracker_rot_world_6d=tracker_rotations(yaw),
        sensor_valid=valid,
        timestamp=0.05,
        floor_y=0.0,
        tracking_origin_revision=0,
        model_root_delta_xz_ref=np.zeros(2),
        model_yaw_delta_sincos=np.asarray([0.0, 1.0]),
        model_pelvis_height=0.9,
        fk_callback=fake_fk,
    )
    assert midway.root_source == RootSource.RECONNECT
    assert np.isclose(midway.reconnect_alpha, 0.5)

    final = resolver.resolve(
        tracker_pos_world=positions,
        tracker_rot_world_6d=tracker_rotations(yaw),
        sensor_valid=valid,
        timestamp=0.10,
        floor_y=0.0,
        tracking_origin_revision=0,
        model_root_delta_xz_ref=np.zeros(2),
        model_yaw_delta_sincos=np.asarray([0.0, 1.0]),
        model_pelvis_height=0.9,
        fk_callback=fake_fk,
    )
    assert final.root_source == RootSource.HIP
    assert np.isclose(final.reconnect_alpha, 1.0)
    np.testing.assert_allclose(final.final_root_pos_world, target_root, atol=1e-5)
    assert np.isclose(final.final_root_yaw, yaw, atol=1e-5)


def test_timestamp_gap_marks_reset_and_zeroes_final_deltas():
    resolver = RuntimeRootResolver(PELVIS_OFFSET)
    positions = np.zeros((TRACKER_COUNT, 3), dtype=np.float32)
    positions[HEAD_TRACKER_INDEX] = np.asarray([0.0, 1.7, 0.0], dtype=np.float32)
    kwargs = dict(
        tracker_pos_world=positions,
        tracker_rot_world_6d=tracker_rotations(),
        sensor_valid=valid_standard_three(),
        floor_y=0.0,
        tracking_origin_revision=0,
        model_root_delta_xz_ref=np.asarray([0.2, 0.0]),
        model_yaw_delta_sincos=np.asarray([0.0, 1.0]),
        model_pelvis_height=0.9,
        fk_callback=fake_fk,
    )
    resolver.resolve(timestamp=0.0, **kwargs)
    reset = resolver.resolve(timestamp=1.0, **kwargs)
    assert reset.root_source == RootSource.RESET
    np.testing.assert_allclose(reset.final_root_delta_xz_ref, 0.0)
    np.testing.assert_allclose(reset.final_yaw_delta_sincos, [0.0, 1.0])


def test_python_matches_unity_contract_v2_head_fk_and_reconnect_golden():
    def golden_fk(root: np.ndarray, yaw: float, pelvis_height: float) -> np.ndarray:
        joints = np.repeat(np.asarray(root, dtype=np.float32)[None], 24, axis=0)
        joints[:, 1] += float(pelvis_height)
        return joints

    resolver = RuntimeRootResolver(
        np.zeros(3, dtype=np.float32),
        RuntimeRootResolverConfig(reconnect_duration_seconds=0.1, hip_filter_time_constant_seconds=0.03),
    )
    positions = np.zeros((TRACKER_COUNT, 3), dtype=np.float32)
    positions[HEAD_TRACKER_INDEX] = [2.0, 1.0, 3.0]
    positions[LEFT_HAND_TRACKER_INDEX] = [1.0, 1.0, 3.0]
    positions[RIGHT_HAND_TRACKER_INDEX] = [3.0, 1.0, 3.0]
    first = resolver.resolve(
        tracker_pos_world=positions,
        tracker_rot_world_6d=tracker_rotations(),
        sensor_valid=valid_standard_three(),
        timestamp=0.0,
        floor_y=0.0,
        tracking_origin_revision=0,
        model_root_delta_xz_ref=np.zeros(2, dtype=np.float32),
        model_yaw_delta_sincos=np.asarray([0.0, 1.0], dtype=np.float32),
        model_pelvis_height=1.0,
        fk_callback=golden_fk,
    )
    assert first.root_source == RootSource.RESET
    np.testing.assert_allclose(first.final_root_pos_world, [2.0, 0.0, 3.0], atol=1e-4)

    positions[HIP_TRACKER_INDEX] = [4.0, 1.0, 5.0]
    valid = valid_standard_three()
    valid[HIP_TRACKER_INDEX] = True
    second = resolver.resolve(
        tracker_pos_world=positions,
        tracker_rot_world_6d=tracker_rotations(np.pi / 2.0),
        sensor_valid=valid,
        timestamp=1.0 / 60.0,
        floor_y=0.0,
        tracking_origin_revision=0,
        model_root_delta_xz_ref=np.zeros(2, dtype=np.float32),
        model_yaw_delta_sincos=np.asarray([0.0, 1.0], dtype=np.float32),
        model_pelvis_height=1.0,
        fk_callback=golden_fk,
    )
    alpha = 1.0 / 6.0
    blend = alpha * alpha * (3.0 - 2.0 * alpha)
    assert second.root_source == RootSource.RECONNECT
    assert np.isclose(second.reconnect_alpha, alpha, atol=1e-4)
    np.testing.assert_allclose(
        second.final_root_pos_world,
        (1.0 - blend) * np.asarray([2.0, 0.0, 3.0]) + blend * np.asarray([4.0, 0.0, 5.0]),
        atol=1e-4,
    )
    assert np.isclose(second.final_root_yaw, (np.pi / 2.0) * blend, atol=1e-4)
