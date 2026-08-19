from __future__ import annotations

import torch

from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_ik import build_current_ik
from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from data_loaders.sensor_masking import TRACKER_TO_JOINT
from diffusion.realtime_pose_projection import (
    project_realtime_pose_xstart,
    project_rotation_6d_to_so3,
)


IDENTITY_6D = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def _inputs(optional=(False, False, False)):
    initial = IDENTITY_6D.repeat(24)[None]
    tracker = torch.zeros(1, 6, 10)
    tracker[..., 3:9] = IDENTITY_6D
    tracker[..., 9] = torch.tensor([[True, True, True, *optional]])
    tracker[:, 0, :3] = torch.tensor([0.0, 1.6, 0.0])
    tracker[:, 1, :3] = torch.tensor([-0.5, 1.2, 0.0])
    tracker[:, 2, :3] = torch.tensor([0.5, 1.2, 0.0])
    tracker[:, 3, :3] = torch.tensor([0.0, 0.9, 0.0])
    tracker[:, 4, :3] = torch.tensor([-0.1, 0.05, 0.0])
    tracker[:, 5, :3] = torch.tensor([0.1, 0.05, 0.0])
    offsets = torch.zeros(1, 24, 3)
    offsets[:, 1:, 1] = 0.1
    return initial, tracker, offsets


def _config():
    return IKInpaintingConfig(
        fabrik_iterations=1,
        direction_only_quality=0.8,
        residual_scale=0.5,
        gap_low=0.1,
        gap_high=0.5,
    )


def test_ik_initializes_from_predictor_current_and_core_trackers_solve_arms():
    initial, tracker, offsets = _inputs()
    result = build_current_ik(initial, tracker, offsets, _config())
    assert result.pose.shape == (1, 24, 6)
    assert result.updated_mask[0, JOINT_INDEX["left_elbow"]]
    assert result.updated_mask[0, JOINT_INDEX["right_elbow"]]
    assert not result.updated_mask[0, JOINT_INDEX["left_knee"]]
    assert not result.updated_mask[0, JOINT_INDEX["left_collar"]]
    torch.testing.assert_close(
        result.pose[0, JOINT_INDEX["left_collar"]],
        initial.reshape(1, 24, 6)[0, JOINT_INDEX["left_collar"]],
    )


def test_foot_without_hip_is_direct_condition_but_does_not_run_leg_fabrik():
    initial, tracker, offsets = _inputs((False, True, False))
    result = build_current_ik(initial, tracker, offsets, _config())
    assert result.direct_rotation_mask[0, JOINT_INDEX["left_foot"]]
    assert not result.updated_mask[0, JOINT_INDEX["left_knee"]]
    assert tracker[0, 4, 9] == 1


def test_projection_overwrites_available_and_preserves_absent_tracker_joint():
    prediction = torch.randn(1, 144)
    _, tracker, _ = _inputs((False, True, False))
    tracker[:, 4, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    deployed = project_realtime_pose_xstart(prediction, tracker).reshape(1, 24, 6)
    projected_prediction = project_rotation_6d_to_so3(
        prediction.reshape(1, 24, 6)
    )
    torch.testing.assert_close(
        deployed[0, TRACKER_TO_JOINT[4]], tracker[0, 4, 3:9], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        deployed[0, TRACKER_TO_JOINT[5]],
        projected_prediction[0, TRACKER_TO_JOINT[5]],
        atol=1e-5,
        rtol=1e-5,
    )
