from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_loaders.realtime_pose_geometry import (
    build_known_target_np,
    build_pose_target_np,
    build_tracker_window_np,
    decode_target_head_rotations_np,
    extract_forward_yaw_np,
    global_head_rotations_to_local_delta_6d_np,
    resolve_root_head_reference_np,
)
from data_loaders.realtime_pose_kinematics import (
    make_yaw_rotation_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.sensor_masking import (
    MISSING_AGE_CAP,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_TO_JOINT,
)


@dataclass(frozen=True)
class WorldPoseState:
    joint_rotations_world: np.ndarray
    root_yaw_world: float
    hip_height: float
    root_position_world: np.ndarray


@dataclass(frozen=True)
class ResolvedPose:
    target_raw: np.ndarray
    joint_rotations_world: np.ndarray
    body_local_delta_6d: np.ndarray
    root_yaw_world: float
    hip_height: float
    root_position_world: np.ndarray
    joints_world: np.ndarray
    known_rotation_max_error: float

    def as_world_state(self) -> WorldPoseState:
        return WorldPoseState(
            joint_rotations_world=self.joint_rotations_world.copy(),
            root_yaw_world=float(self.root_yaw_world),
            hip_height=float(self.hip_height),
            root_position_world=self.root_position_world.copy(),
        )


def advance_missing_age(
    previous_missing_age: np.ndarray,
    configured: np.ndarray,
    measured_valid: np.ndarray,
    cap: int = MISSING_AGE_CAP,
) -> np.ndarray:
    """在线端与离线 task 完全相同的单帧 missing-age 递推。"""

    previous = np.asarray(previous_missing_age, dtype=np.int64).reshape(TRACKER_COUNT)
    configured = np.asarray(configured, dtype=bool).reshape(TRACKER_COUNT)
    measured = np.asarray(measured_valid, dtype=bool).reshape(TRACKER_COUNT)
    if np.any(measured & ~configured):
        raise ValueError("measured_valid 必须是 configured 子集。")
    current = np.zeros(TRACKER_COUNT, dtype=np.int64)
    missing = configured & ~measured
    current[missing] = np.minimum(previous[missing] + 1, int(cap))
    current[0] = 0
    return current


def build_online_conditioning(
    pose_history_world: list[WorldPoseState],
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    configured: np.ndarray,
    measured_valid: np.ndarray,
    missing_age: np.ndarray,
    floor_y: float,
    normalizer=None,
    initial_head_yaw: float = 0.0,
) -> dict[str, np.ndarray]:
    """将 60 帧世界姿态和 61 帧 Tracker 统一重表达到当前 Head 参考系。"""

    if len(pose_history_world) != REALTIME_POSE_HISTORY_LENGTH:
        raise ValueError("在线采样需要恰好 60 帧 WorldPoseState 历史。")
    tracker_pos_world = np.asarray(tracker_pos_world, dtype=np.float32)
    tracker_rot_world_6d = np.asarray(tracker_rot_world_6d, dtype=np.float32)
    configured = np.asarray(configured, dtype=bool)
    measured_valid = np.asarray(measured_valid, dtype=bool)
    missing_age = np.asarray(missing_age, dtype=np.int64)
    if tracker_pos_world.shape != (61, TRACKER_COUNT, 3):
        raise ValueError("tracker_pos_world 必须为 [61,6,3]。")
    if tracker_rot_world_6d.shape != (61, TRACKER_COUNT, 6):
        raise ValueError("tracker_rot_world_6d 必须为 [61,6,6]。")
    if configured.shape != (61, TRACKER_COUNT) or measured_valid.shape != configured.shape:
        raise ValueError("configured/measured_valid 必须为 [61,6]。")
    if missing_age.shape != configured.shape:
        raise ValueError("missing_age 必须为 [61,6]。")
    if not configured[:, 0].all() or not measured_valid[:, 0].all():
        raise ValueError("Head 必须始终有效。")

    tracker_rot_world = rotation_6d_to_matrix_np(tracker_rot_world_6d)
    head_yaw_history = extract_forward_yaw_np(
        tracker_rot_world[:, 0],
        initial_yaw=float(initial_head_yaw),
    )
    current_head_yaw = float(head_yaw_history[-1])
    current_head_position = tracker_pos_world[-1, 0]
    rotations_world = np.stack([state.joint_rotations_world for state in pose_history_world], axis=0)
    pose_history_raw = build_pose_target_np(rotations_world, current_head_yaw)
    tracker_raw = build_tracker_window_np(
        tracker_pos_world,
        tracker_rot_world_6d,
        current_head_position,
        floor_y,
        current_head_yaw,
        configured,
        measured_valid,
        np.minimum(missing_age, MISSING_AGE_CAP).astype(np.float32) / float(MISSING_AGE_CAP),
    )
    known_target_raw, known_mask = build_known_target_np(tracker_raw[-1])
    if normalizer is None:
        pose_history = pose_history_raw
        tracker_window = tracker_raw
        known_target = known_target_raw
    else:
        pose_history = np.asarray(normalizer.normalize_pose(pose_history_raw), dtype=np.float32)
        tracker_window = np.asarray(normalizer.normalize_tracker(tracker_raw), dtype=np.float32)
        known_target = np.asarray(normalizer.normalize_pose(known_target_raw), dtype=np.float32)
    known_target = np.where(known_mask, known_target, 0.0).astype(np.float32)
    return {
        "pose_history": pose_history.astype(np.float32),
        "tracker_window": tracker_window.astype(np.float32),
        "tracker_window_raw": tracker_raw,
        "known_target": known_target,
        "known_mask": known_mask,
        "valid_frame_mask": np.ones(REALTIME_POSE_HISTORY_LENGTH, dtype=bool),
        "current_head_yaw_world": np.asarray(current_head_yaw, dtype=np.float32),
        "current_head_position_world": current_head_position.astype(np.float32),
        "floor_y": np.asarray(floor_y, dtype=np.float32),
    }


def sample_online_target(
    model,
    diffusion,
    conditioning: dict[str, np.ndarray | torch.Tensor],
    device: torch.device,
    normalizer=None,
) -> np.ndarray:
    """执行 DDIM hard inpainting，返回物理空间的 144 维目标。"""

    pose_history = torch.as_tensor(conditioning["pose_history"], device=device, dtype=torch.float32).unsqueeze(0)
    tracker_window = torch.as_tensor(conditioning["tracker_window"], device=device, dtype=torch.float32).unsqueeze(0)
    known_target = torch.as_tensor(conditioning["known_target"], device=device, dtype=torch.float32).unsqueeze(0)
    known_mask = torch.as_tensor(conditioning["known_mask"], device=device, dtype=torch.bool).unsqueeze(0)
    valid_frame_mask = torch.as_tensor(
        conditioning["valid_frame_mask"], device=device, dtype=torch.bool
    ).unsqueeze(0)
    model_kwargs = {
        "inpaint_cond": ~known_mask,
        "known_mask": known_mask,
        "pose_history": pose_history,
        "tracker_window": tracker_window,
        "valid_frame_mask": valid_frame_mask,
        "y": {"mask": ~known_mask, "inpainted_motion": known_target},
    }
    with torch.no_grad():
        sampled = diffusion.ddim_sample_loop(
            model,
            shape=(1, REALTIME_POSE_TARGET_DIM),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=device,
        )
    sampled = torch.where(known_mask, known_target, sampled)[0].cpu()
    if normalizer is not None:
        sampled = normalizer.inverse_pose(sampled)
    return np.asarray(sampled, dtype=np.float32)


def decode_and_resolve_pose(
    target_raw: np.ndarray,
    tracker_current_raw: np.ndarray,
    current_head_yaw_world: float,
    current_head_position_world: np.ndarray,
    floor_y: float,
    joint_offsets_parent: np.ndarray,
    joint_rest_local_rotations_6d: np.ndarray,
) -> ResolvedPose:
    """无积分、无滤波、无 IK 的 Head-first Root Resolver。"""

    target = np.asarray(target_raw, dtype=np.float32).reshape(REALTIME_POSE_TARGET_DIM)
    tracker = np.asarray(tracker_current_raw, dtype=np.float32).reshape(TRACKER_COUNT, TRACKER_FEATURE_DIM)
    rest_rot = rotation_6d_to_matrix_np(joint_rest_local_rotations_6d)
    rotations_head, root_yaw_head = decode_target_head_rotations_np(target)
    measured = tracker[:, 10] > 0.5
    root_head, hip_height, joints_head = resolve_root_head_reference_np(
        rotations_head,
        float(root_yaw_head),
        joint_offsets_parent,
        observed_head_height=float(tracker[0, 1]),
    )

    head_yaw_rotation = make_yaw_rotation_np(np.asarray([current_head_yaw_world], dtype=np.float64))[0]
    rotations_world = np.einsum("ij,ajk->aik", head_yaw_rotation, rotations_head)
    origin = np.asarray(
        [current_head_position_world[0], float(floor_y), current_head_position_world[2]],
        dtype=np.float64,
    )
    root_world = origin + head_yaw_rotation @ root_head.astype(np.float64)
    root_world[1] = float(floor_y)
    joints_world = origin[None] + np.einsum("ij,aj->ai", head_yaw_rotation, joints_head)
    local_delta = global_head_rotations_to_local_delta_6d_np(
        rotations_head,
        root_heading_head=root_yaw_head,
        rest_local_rotations=rest_rot,
    )
    root_yaw_world = float(current_head_yaw_world + float(root_yaw_head))

    known_error = 0.0
    tracker_rot_head = rotation_6d_to_matrix_np(tracker[:, 3:9])
    for tracker_index in range(TRACKER_COUNT):
        if not measured[tracker_index]:
            continue
        joint_index = TRACKER_TO_JOINT[tracker_index]
        difference = rotations_head[joint_index].T @ tracker_rot_head[tracker_index]
        cosine = np.clip((np.trace(difference) - 1.0) * 0.5, -1.0, 1.0)
        known_error = max(known_error, float(np.arccos(cosine)))
    if known_error > 1e-4:
        raise ValueError(f"hard inpainting 后已知 Tracker 旋转不一致：{known_error:.6g} rad")

    return ResolvedPose(
        target_raw=target,
        joint_rotations_world=rotations_world.astype(np.float32),
        body_local_delta_6d=local_delta.astype(np.float32),
        root_yaw_world=root_yaw_world,
        hip_height=float(hip_height),
        root_position_world=root_world.astype(np.float32),
        joints_world=joints_world.astype(np.float32),
        known_rotation_max_error=known_error,
    )
