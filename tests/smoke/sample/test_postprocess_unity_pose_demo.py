from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from sample.postprocess_unity_pose_demo import (
    FilterSegment,
    LandingBridge,
    SMPL_JOINT_COUNT,
    apply_landing_bridges,
    build_joint_filter_alpha,
    filter_local_rotations,
    parse_filter_segment,
    parse_landing_bridge,
)


def test_parse_filter_segment_validates_contract() -> None:
    assert parse_filter_segment("24.2:26.0:left") == FilterSegment(24.2, 26.0, "left")
    with pytest.raises(ValueError, match="start:end"):
        parse_filter_segment("24.2:26.0")
    with pytest.raises(ValueError, match="left、right"):
        parse_filter_segment("24.2:26.0:leg")


def test_parse_landing_bridge_uses_integer_frame_contract() -> None:
    assert parse_landing_bridge("770:779:left") == LandingBridge(770, 779, "left")
    with pytest.raises(ValueError, match="整数帧号"):
        parse_landing_bridge("25.5:779:left")
    with pytest.raises(ValueError, match="start_frame < end_frame"):
        parse_landing_bridge("779:770:left")


def test_joint_filter_alpha_only_activates_selected_leg_and_segment() -> None:
    times = np.arange(20, dtype=np.float64) / 10.0
    alpha = build_joint_filter_alpha(
        times=times,
        segments=(FilterSegment(0.5, 1.5, "left"),),
        blend_frames=2,
    )

    assert not alpha[:5].any()
    assert not alpha[16:].any()
    assert alpha[:, JOINT_INDEX["left_knee"]].max() == pytest.approx(1.0)
    assert not alpha[:, JOINT_INDEX["right_knee"]].any()
    assert not alpha[:, JOINT_INDEX["pelvis"]].any()


def test_local_filter_reduces_single_frame_leg_noise_without_touching_upper_body() -> None:
    frame_count = 15
    identity = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rotations = np.broadcast_to(
        identity,
        (frame_count, SMPL_JOINT_COUNT, 4),
    ).copy()
    knee_index = JOINT_INDEX["left_knee"]
    rotations[7, knee_index] = Rotation.from_rotvec([0.0, 0.35, 0.0]).as_quat()
    alpha = np.zeros((frame_count, SMPL_JOINT_COUNT), dtype=np.float64)
    alpha[3:12, knee_index] = 1.0

    filtered = filter_local_rotations(
        rotations,
        joint_alpha=alpha,
        window_frames=5,
        strength=0.30,
    )

    original_angle = np.linalg.norm(Rotation.from_quat(rotations[7, knee_index]).as_rotvec())
    filtered_angle = np.linalg.norm(Rotation.from_quat(filtered[7, knee_index]).as_rotvec())
    assert filtered_angle < original_angle
    np.testing.assert_array_equal(filtered[:, JOINT_INDEX["spine3"]], rotations[:, JOINT_INDEX["spine3"]])
    np.testing.assert_array_equal(filtered[:3, knee_index], rotations[:3, knee_index])
    np.testing.assert_allclose(np.linalg.norm(filtered, axis=-1), 1.0, atol=1e-6)


def test_landing_bridge_replaces_noise_and_only_partially_moves_knee() -> None:
    frame_count = 30
    identity = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rotations = np.broadcast_to(
        identity,
        (frame_count, SMPL_JOINT_COUNT, 4),
    ).copy()
    noisy = Rotation.from_rotvec([0.0, 1.0, 0.0]).as_quat()
    for joint_name in ("left_knee", "left_ankle", "left_foot"):
        rotations[10:14, JOINT_INDEX[joint_name]] = noisy

    bridged = apply_landing_bridges(
        rotations,
        bridges=(LandingBridge(10, 14, "left"),),
        pre_anchor_frames=4,
        post_anchor_frames=5,
        lock_frames=3,
        knee_strength=0.30,
        ankle_strength=1.0,
        foot_strength=1.0,
    )

    ankle_angle = np.linalg.norm(
        Rotation.from_quat(bridged[11, JOINT_INDEX["left_ankle"]]).as_rotvec()
    )
    knee_angle = np.linalg.norm(
        Rotation.from_quat(bridged[11, JOINT_INDEX["left_knee"]]).as_rotvec()
    )
    assert ankle_angle == pytest.approx(0.0, abs=1e-6)
    assert knee_angle == pytest.approx(0.7, abs=1e-6)
    np.testing.assert_array_equal(bridged[:10], rotations[:10])
    np.testing.assert_array_equal(bridged[18:], rotations[18:])
    np.testing.assert_array_equal(
        bridged[:, JOINT_INDEX["right_ankle"]],
        rotations[:, JOINT_INDEX["right_ankle"]],
    )
