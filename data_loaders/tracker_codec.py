from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_kinematics import (
    make_yaw_rotation_np,
    make_yaw_rotation_torch,
    rotation_6d_forward_up_np,
    rotation_6d_forward_up_torch,
    rotation_6d_to_matrix_np,
    rotation_6d_to_matrix_torch,
)
from data_loaders.sensor_masking import HIP_TRACKER_INDEX, TRACKER_COUNT


TRACKER_CODEC_VERSION = "tracker_codec_v2"
REFERENCE_POLICY_VERSION = "hip_current_else_previous_final_v1"
TRACKER_REF_SOURCE_PREVIOUS_FINAL = 0
TRACKER_REF_SOURCE_CURRENT_HIP = 1


def encode_tracker_positions_np(
    tracker_pos_world: np.ndarray,
    ref_root_pos_world: np.ndarray,
    ref_root_yaw: np.ndarray,
) -> np.ndarray:
    """按列向量契约编码世界系 Tracker 位置。"""

    tracker = np.asarray(tracker_pos_world, dtype=np.float64)
    root = np.asarray(ref_root_pos_world, dtype=np.float64)
    yaw = np.asarray(ref_root_yaw, dtype=np.float64)
    if tracker.shape[-2:] != (TRACKER_COUNT, 3):
        raise ValueError(f"tracker_pos_world 应以 [{TRACKER_COUNT},3] 结尾，实际为 {tracker.shape}")
    rotations = make_yaw_rotation_np(yaw.reshape(-1)).reshape((*yaw.shape, 3, 3))
    return np.einsum("...ij,...sj->...si", rotations.swapaxes(-1, -2), tracker - root[..., None, :]).astype(
        np.float32
    )


def decode_tracker_positions_np(
    tracker_pos_ref: np.ndarray,
    ref_root_pos_world: np.ndarray,
    ref_root_yaw: np.ndarray,
) -> np.ndarray:
    tracker = np.asarray(tracker_pos_ref, dtype=np.float64)
    root = np.asarray(ref_root_pos_world, dtype=np.float64)
    yaw = np.asarray(ref_root_yaw, dtype=np.float64)
    rotations = make_yaw_rotation_np(yaw.reshape(-1)).reshape((*yaw.shape, 3, 3))
    return (np.einsum("...ij,...sj->...si", rotations, tracker) + root[..., None, :]).astype(np.float32)


def encode_tracker_rotations_np(tracker_rot_world_6d: np.ndarray, ref_root_yaw: np.ndarray) -> np.ndarray:
    world = rotation_6d_to_matrix_np(tracker_rot_world_6d)
    yaw = np.asarray(ref_root_yaw, dtype=np.float64)
    rotations = make_yaw_rotation_np(yaw.reshape(-1)).reshape((*yaw.shape, 3, 3))
    local = np.einsum("...ij,...sjk->...sik", rotations.swapaxes(-1, -2), world)
    return rotation_6d_forward_up_np(local).astype(np.float32)


def decode_tracker_rotations_np(tracker_rot_ref_6d: np.ndarray, ref_root_yaw: np.ndarray) -> np.ndarray:
    local = rotation_6d_to_matrix_np(tracker_rot_ref_6d)
    yaw = np.asarray(ref_root_yaw, dtype=np.float64)
    rotations = make_yaw_rotation_np(yaw.reshape(-1)).reshape((*yaw.shape, 3, 3))
    world = np.einsum("...ij,...sjk->...sik", rotations, local)
    return rotation_6d_forward_up_np(world).astype(np.float32)


def encode_tracker_positions_torch(
    tracker_pos_world: torch.Tensor,
    ref_root_pos_world: torch.Tensor,
    ref_root_yaw: torch.Tensor,
) -> torch.Tensor:
    rotations = make_yaw_rotation_torch(ref_root_yaw)
    return torch.einsum(
        "...ij,...sj->...si",
        rotations.transpose(-1, -2),
        tracker_pos_world - ref_root_pos_world.unsqueeze(-2),
    )


def decode_tracker_positions_torch(
    tracker_pos_ref: torch.Tensor,
    ref_root_pos_world: torch.Tensor,
    ref_root_yaw: torch.Tensor,
) -> torch.Tensor:
    rotations = make_yaw_rotation_torch(ref_root_yaw)
    return torch.einsum("...ij,...sj->...si", rotations, tracker_pos_ref) + ref_root_pos_world.unsqueeze(-2)


def encode_tracker_rotations_torch(tracker_rot_world_6d: torch.Tensor, ref_root_yaw: torch.Tensor) -> torch.Tensor:
    world = rotation_6d_to_matrix_torch(tracker_rot_world_6d)
    rotations = make_yaw_rotation_torch(ref_root_yaw)
    local = torch.einsum("...ij,...sjk->...sik", rotations.transpose(-1, -2), world)
    return rotation_6d_forward_up_torch(local)


def decode_tracker_rotations_torch(tracker_rot_ref_6d: torch.Tensor, ref_root_yaw: torch.Tensor) -> torch.Tensor:
    local = rotation_6d_to_matrix_torch(tracker_rot_ref_6d)
    rotations = make_yaw_rotation_torch(ref_root_yaw)
    world = torch.einsum("...ij,...sjk->...sik", rotations, local)
    return rotation_6d_forward_up_torch(world)


def yaw_from_rotation_6d_np(rotation_6d: np.ndarray) -> np.ndarray:
    rotations = rotation_6d_to_matrix_np(rotation_6d)
    return np.arctan2(rotations[..., 0, 2], rotations[..., 2, 2]).astype(np.float32)


def build_tracker_reference_np(
    tracker_pos_world: np.ndarray,
    tracker_rot_world_6d: np.ndarray,
    sensor_valid: np.ndarray,
    previous_final_root_pos_world: np.ndarray,
    previous_final_root_yaw: np.ndarray,
    pelvis_offset_parent: np.ndarray,
    floor_y: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造推理前即可得到的 Hip/previous-final Tracker reference。"""

    tracker_pos = np.asarray(tracker_pos_world, dtype=np.float64)
    tracker_rot = np.asarray(tracker_rot_world_6d, dtype=np.float64)
    valid = np.asarray(sensor_valid, dtype=bool)
    prev_pos = np.asarray(previous_final_root_pos_world, dtype=np.float64)
    prev_yaw = np.asarray(previous_final_root_yaw, dtype=np.float64)
    floor = np.broadcast_to(np.asarray(floor_y, dtype=np.float64), prev_yaw.shape)
    if tracker_pos.shape[:-2] != prev_yaw.shape or tracker_pos.shape[-2:] != (TRACKER_COUNT, 3):
        raise ValueError("Tracker reference 输入前导维不一致。")
    ref_pos = prev_pos.copy()
    ref_yaw = prev_yaw.copy()
    source = np.full(prev_yaw.shape, TRACKER_REF_SOURCE_PREVIOUS_FINAL, dtype=np.int8)
    hip_valid = valid[..., HIP_TRACKER_INDEX]
    if np.any(hip_valid):
        hip_yaw = yaw_from_rotation_6d_np(tracker_rot[..., HIP_TRACKER_INDEX, :]).astype(np.float64)
        hip_rotation = make_yaw_rotation_np(hip_yaw.reshape(-1)).reshape((*hip_yaw.shape, 3, 3))
        offset = np.asarray(pelvis_offset_parent, dtype=np.float64).reshape(3)
        root_from_hip = tracker_pos[..., HIP_TRACKER_INDEX, :] - np.einsum("...ij,j->...i", hip_rotation, offset)
        root_from_hip[..., 1] = floor
        ref_pos[hip_valid] = root_from_hip[hip_valid]
        ref_yaw[hip_valid] = hip_yaw[hip_valid]
        source[hip_valid] = TRACKER_REF_SOURCE_CURRENT_HIP
    return ref_pos.astype(np.float32), ref_yaw.astype(np.float32), source
