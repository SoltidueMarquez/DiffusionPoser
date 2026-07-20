from __future__ import annotations

import numpy as np
import pytest
import torch

from data_loaders.realtime_pose_kinematics import fk_body_fbx_local_torch
from data_loaders.sensor_masking import HEAD_TRACKER_INDEX, HIP_TRACKER_INDEX, TRACKER_COUNT
from diffusion.realtime_pose import resolve_realtime_pose_frame_torch
from sample.runtime_root_resolver import RuntimeRootResolver, RuntimeRootResolverState
from tests.smoke.realtime_pose_fixtures import IDENTITY_6D


def _build_case(*, hip_valid: bool, previous_hip_valid: bool, reconnect_elapsed: float):
    pose = torch.as_tensor(IDENTITY_6D, dtype=torch.float32).repeat(1, 24)
    root_delta = torch.tensor([[0.2, -0.1]], dtype=torch.float32)
    yaw_delta = torch.tensor([[np.sin(0.3), np.cos(0.3)]], dtype=torch.float32)
    pelvis_height = torch.tensor([0.9], dtype=torch.float32)
    offsets = torch.zeros(1, 24, 3)
    offsets[:, 0, 1] = 0.9
    tracker_pos = torch.zeros(1, TRACKER_COUNT, 3)
    tracker_pos[:, HEAD_TRACKER_INDEX] = torch.tensor([1.0, 1.7, 2.0])
    tracker_pos[:, HIP_TRACKER_INDEX] = torch.tensor([0.5, 0.9, -0.2])
    tracker_rot = torch.as_tensor(IDENTITY_6D, dtype=torch.float32).repeat(1, TRACKER_COUNT, 1)
    valid = torch.ones(1, TRACKER_COUNT, dtype=torch.bool)
    valid[:, HIP_TRACKER_INDEX] = hip_valid
    y = {
        "tracker_ref_root_pos_world": torch.zeros(1, 3),
        "tracker_ref_root_yaw": torch.zeros(1),
        "target_tracker_pos_ref": tracker_pos,
        "target_tracker_rot_ref_6d": tracker_rot,
        "target_sensor_valid": valid,
        "target_floor_y": torch.zeros(1),
        "joint_offsets_parent": offsets,
        "resolver_before_target_root_pos_world": torch.tensor([[0.1, 0.0, 0.2]]),
        "resolver_before_target_root_yaw": torch.tensor([0.1]),
        "resolver_before_target_pelvis_height": torch.tensor([0.85]),
        "resolver_before_target_hip_valid": torch.tensor([previous_hip_valid]),
        "resolver_before_target_reconnect_start_root_pos_world": torch.tensor([[0.0, 0.0, 0.1]]),
        "resolver_before_target_reconnect_start_root_yaw": torch.tensor([0.05]),
        "resolver_before_target_reconnect_start_pelvis_height": torch.tensor([0.82]),
        "resolver_before_target_reconnect_elapsed_seconds": torch.tensor([reconnect_elapsed]),
        "resolver_before_target_last_timestamp_seconds": torch.tensor([0.0]),
        "resolver_before_target_tracking_origin_revision": torch.tensor([0]),
        "target_timestamp_seconds": torch.tensor([1.0 / 60.0]),
        "target_tracking_origin_revision": torch.tensor([0]),
    }
    return pose, root_delta, yaw_delta, pelvis_height, y


@pytest.mark.parametrize(
    ("hip_valid", "previous_hip_valid", "reconnect_elapsed"),
    [
        (False, False, 0.0),
        (True, True, 0.0),
        (True, False, 0.0),
        (True, True, 0.04),
    ],
)
def test_differentiable_resolver_matches_numpy_runtime_paths(
    hip_valid: bool,
    previous_hip_valid: bool,
    reconnect_elapsed: float,
):
    pose, root_delta, yaw_delta, pelvis_height, y = _build_case(
        hip_valid=hip_valid,
        previous_hip_valid=previous_hip_valid,
        reconnect_elapsed=reconnect_elapsed,
    )
    actual = resolve_realtime_pose_frame_torch(
        pred_pose=pose,
        pred_root_delta_xz_ref=root_delta,
        pred_yaw_delta_sincos=yaw_delta,
        pred_pelvis_height=pelvis_height,
        y=y,
    )

    offsets = y["joint_offsets_parent"][0].numpy()
    state = RuntimeRootResolverState(
        initialized=True,
        final_root_pos_world=y["resolver_before_target_root_pos_world"][0].numpy(),
        final_root_yaw=float(y["resolver_before_target_root_yaw"][0]),
        final_pelvis_height=float(y["resolver_before_target_pelvis_height"][0]),
        final_joints_world=np.zeros((24, 3), dtype=np.float32),
        hip_was_valid=previous_hip_valid,
        reconnect_active=0.0 < reconnect_elapsed < 0.1,
        reconnect_elapsed_seconds=reconnect_elapsed,
        reconnect_start_root_pos_world=y["resolver_before_target_reconnect_start_root_pos_world"][0].numpy(),
        reconnect_start_root_yaw=float(y["resolver_before_target_reconnect_start_root_yaw"][0]),
        reconnect_start_pelvis_height=float(y["resolver_before_target_reconnect_start_pelvis_height"][0]),
        last_timestamp=0.0,
        floor_y=0.0,
        tracking_origin_revision=0,
    )

    def fk_callback(root: np.ndarray, yaw: float, height: float) -> np.ndarray:
        local_offsets = offsets.copy()
        local_offsets[0, 1] = height
        joints = fk_body_fbx_local_torch(
            body_pose_local_delta_6d=pose,
            actor_root_pos_world=torch.from_numpy(root[None]).float(),
            root_heading=torch.tensor([yaw], dtype=torch.float32),
            rest_local_positions=torch.from_numpy(local_offsets[None]).float(),
        )
        return joints[0].detach().numpy()

    expected = RuntimeRootResolver(pelvis_offset_parent=offsets[0], state=state).resolve(
        tracker_pos_world=y["target_tracker_pos_ref"][0].numpy(),
        tracker_rot_world_6d=y["target_tracker_rot_ref_6d"][0].numpy(),
        sensor_valid=y["target_sensor_valid"][0].numpy(),
        timestamp=1.0 / 60.0,
        floor_y=0.0,
        tracking_origin_revision=0,
        model_root_delta_xz_ref=root_delta[0].numpy(),
        model_yaw_delta_sincos=yaw_delta[0].numpy(),
        model_pelvis_height=float(pelvis_height[0]),
        fk_callback=fk_callback,
    )

    np.testing.assert_allclose(actual.final_root_pos_world[0].detach().numpy(), expected.final_root_pos_world, atol=1e-5)
    np.testing.assert_allclose(actual.final_root_yaw[0].detach().numpy(), expected.final_root_yaw, atol=1e-5)
    np.testing.assert_allclose(actual.final_pelvis_height[0].detach().numpy(), expected.final_pelvis_height, atol=1e-5)
    np.testing.assert_allclose(actual.final_joints_world[0].detach().numpy(), expected.final_joints_world, atol=1e-5)


def test_nohip_head_anchor_keeps_root_delta_and_pose_gradients():
    pose, root_delta, yaw_delta, pelvis_height, y = _build_case(
        hip_valid=False,
        previous_hip_valid=False,
        reconnect_elapsed=0.0,
    )
    pose = pose.clone().requires_grad_(True)
    root_delta = root_delta.clone().requires_grad_(True)
    result = resolve_realtime_pose_frame_torch(
        pred_pose=pose,
        pred_root_delta_xz_ref=root_delta,
        pred_yaw_delta_sincos=yaw_delta,
        pred_pelvis_height=pelvis_height,
        y=y,
    )
    result.final_joints_world.square().sum().backward()

    assert root_delta.grad is not None
    assert torch.isfinite(root_delta.grad).all()
    assert torch.any(root_delta.grad.abs() > 1e-6)
    assert pose.grad is not None
    assert torch.isfinite(pose.grad).all()


def test_differentiable_resolver_resets_filter_state_on_tracking_origin_change():
    pose, root_delta, yaw_delta, pelvis_height, y = _build_case(
        hip_valid=True,
        previous_hip_valid=True,
        reconnect_elapsed=0.0,
    )
    y["target_tracking_origin_revision"] = torch.tensor([1])

    result = resolve_realtime_pose_frame_torch(
        pred_pose=pose,
        pred_root_delta_xz_ref=root_delta,
        pred_yaw_delta_sincos=yaw_delta,
        pred_pelvis_height=pelvis_height,
        y=y,
    )

    # reset 后 Hip 路径不能再和旧 root/filter state 混合。
    assert torch.allclose(result.final_root_pos_world, torch.tensor([[0.5, 0.0, -0.2]]), atol=1e-6)
    assert torch.allclose(result.final_root_yaw, torch.zeros(1), atol=1e-6)
    assert torch.allclose(result.final_pelvis_height, torch.tensor([0.9]), atol=1e-6)
