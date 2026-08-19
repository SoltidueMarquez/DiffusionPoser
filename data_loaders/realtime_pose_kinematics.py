from __future__ import annotations

import math

import numpy as np
import torch

from data_loaders.sensor_masking import (
    REALTIME_POSE_FPS,
    STATIONARY_JOINT_INDICES,
    STATIONARY_PROB_DIM,
)


SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
JOINT_INDEX = {name: index for index, name in enumerate(SMPL_JOINT_NAMES)}
SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
FOOT_CONTACT_FULL_HEIGHT_M = 0.05
FOOT_CONTACT_NO_HEIGHT_M = 0.10
TRACKER_JOINT_INDICES = np.array(
    [
        JOINT_INDEX["head"],
        JOINT_INDEX["left_wrist"],
        JOINT_INDEX["right_wrist"],
        JOINT_INDEX["pelvis"],
        JOINT_INDEX["left_foot"],
        JOINT_INDEX["right_foot"],
    ],
    dtype=np.int64,
)


def make_yaw_rotation_np(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotations = np.zeros((yaw.shape[0], 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = cos_yaw
    rotations[:, 0, 2] = sin_yaw
    rotations[:, 1, 1] = 1.0
    rotations[:, 2, 0] = -sin_yaw
    rotations[:, 2, 2] = cos_yaw
    return rotations


def make_yaw_rotation_torch(yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotations = torch.zeros((*yaw.shape, 3, 3), dtype=yaw.dtype, device=yaw.device)
    rotations[..., 0, 0] = cos_yaw
    rotations[..., 0, 2] = sin_yaw
    rotations[..., 1, 1] = 1.0
    rotations[..., 2, 0] = -sin_yaw
    rotations[..., 2, 2] = cos_yaw
    return rotations


def encode_root_delta_xz_ref(root_pos_world: np.ndarray, root_yaw: np.ndarray) -> np.ndarray:
    """
    把相邻 root 位移编码到 previous-root-yaw 参考系。

    输入 `root_pos_world` 为 `[T,3]`，`root_yaw` 为 `[T]`。第 0 帧没有历史位移，
    固定写成 `[0,0]`；第 t 帧使用 `root_yaw[t-1]` 作为参考朝向，和 tracker ref
    的运行时约定保持一致。
    """

    roots = np.asarray(root_pos_world, dtype=np.float64)
    yaws = np.asarray(root_yaw, dtype=np.float64)
    if roots.ndim != 2 or roots.shape[1] != 3:
        raise ValueError(f"root_pos_world 应为 [T,3]，实际为 {roots.shape}")
    if yaws.shape != (roots.shape[0],):
        raise ValueError(f"root_yaw 应为 [T]，实际为 {yaws.shape}")

    delta_world = np.zeros_like(roots)
    if roots.shape[0] > 1:
        delta_world[1:] = roots[1:] - roots[:-1]
    ref_yaw = np.concatenate([yaws[:1], yaws[:-1]], axis=0)
    rotations = make_yaw_rotation_np(ref_yaw)
    delta_ref = np.einsum("ti,tij->tj", delta_world, rotations)
    return delta_ref[:, [0, 2]].astype(np.float32)


def integrate_root_delta_xz_ref(
    prev_root_pos_world: np.ndarray,
    prev_root_yaw: np.ndarray,
    root_delta_xz_ref: np.ndarray,
) -> np.ndarray:
    """
    将 previous-yaw 参考系下的 xz 位移积分回世界系 root 位置。

    `prev_root_pos_world`、`prev_root_yaw` 和 `root_delta_xz_ref` 的前导维一致；
    返回 `[N,3]`，y 分量保持 previous root 的 ground height。
    """

    prev_pos = np.asarray(prev_root_pos_world, dtype=np.float64)
    prev_yaw = np.asarray(prev_root_yaw, dtype=np.float64)
    delta_ref = np.asarray(root_delta_xz_ref, dtype=np.float64)
    if prev_pos.shape[-1] != 3 or delta_ref.shape[-1] != 2:
        raise ValueError(f"root integration shape 不匹配：prev={prev_pos.shape}, delta={delta_ref.shape}")
    flat_prev = prev_pos.reshape(-1, 3)
    flat_yaw = prev_yaw.reshape(-1)
    flat_delta = delta_ref.reshape(-1, 2)
    delta_3d = np.zeros((flat_delta.shape[0], 3), dtype=np.float64)
    delta_3d[:, 0] = flat_delta[:, 0]
    delta_3d[:, 2] = flat_delta[:, 1]
    rotations = make_yaw_rotation_np(flat_yaw)
    delta_world = np.einsum("ti,tji->tj", delta_3d, rotations)
    result = flat_prev + delta_world
    return result.reshape(prev_pos.shape).astype(np.float32)



def derive_stationary_prob_5(
    joints_world: np.ndarray,
    fps: float = REALTIME_POSE_FPS,
    speed_full_motion: float = 0.25,
    median_window: int = 5,
) -> np.ndarray:
    """从 5 个 GlobalPose 候选接触关节的世界速度派生静止概率 `[T,5]`。

    概率只表达“近似静止”，后续物理模块可以再结合接触可行性决定最终 contact set。
    """

    joints = np.asarray(joints_world, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"joints_world 应为 [T,24,3]，实际为 {joints.shape}")
    if float(fps) <= 0.0:
        raise ValueError(f"fps 必须为正数，实际为 {fps}")
    if float(speed_full_motion) <= 0.0:
        raise ValueError(f"speed_full_motion 必须为正数，实际为 {speed_full_motion}")

    joint_indices = np.asarray(STATIONARY_JOINT_INDICES, dtype=np.int64)
    if joint_indices.shape != (STATIONARY_PROB_DIM,):
        raise ValueError(f"stationary joint 数量必须为 {STATIONARY_PROB_DIM}，实际为 {joint_indices.shape}")
    joint_pos = joints[:, joint_indices]
    speed = np.zeros((joints.shape[0], STATIONARY_PROB_DIM), dtype=np.float64)
    if joints.shape[0] > 1:
        speed[1:] = np.linalg.norm(joint_pos[1:] - joint_pos[:-1], axis=-1) * float(fps)
        speed[0] = speed[1]

    smoothed_speed = median_filter_time(speed, window=int(median_window))
    stationary_prob = np.clip(1.0 - smoothed_speed / float(speed_full_motion), 0.0, 1.0)
    return stationary_prob.astype(np.float32)


def derive_foot_contact_prob_2(
    stationary_prob_5: np.ndarray,
    joints_world: np.ndarray,
    floor_y: np.ndarray | float,
    height_full_contact: float = FOOT_CONTACT_FULL_HEIGHT_M,
    height_no_contact: float = FOOT_CONTACT_NO_HEIGHT_M,
) -> np.ndarray:
    """用脚部静止概率与离地高度派生左右脚接触概率 ``[...,2]``。

    脚高不超过 ``height_full_contact`` 时保留原静止概率，达到
    ``height_no_contact`` 时将接触概率压到零，中间区间线性过渡。这样既排除
    “悬空但静止”的脚，也避免硬高度阈值在边界处产生标签跳变。
    """

    stationary = np.asarray(stationary_prob_5, dtype=np.float64)
    joints = np.asarray(joints_world, dtype=np.float64)
    if stationary.ndim < 1 or stationary.shape[-1] != STATIONARY_PROB_DIM:
        raise ValueError(
            f"stationary_prob_5 最后一维必须为 {STATIONARY_PROB_DIM}，"
            f"实际为 {stationary.shape}"
        )
    expected_joint_shape = (*stationary.shape[:-1], 24, 3)
    if joints.shape != expected_joint_shape:
        raise ValueError(
            f"joints_world 应为 {expected_joint_shape}，实际为 {joints.shape}"
        )

    full_height = float(height_full_contact)
    no_height = float(height_no_contact)
    if not math.isfinite(full_height) or not math.isfinite(no_height):
        raise ValueError("脚接触高度阈值必须为有限数。")
    if no_height <= full_height:
        raise ValueError("height_no_contact 必须大于 height_full_contact。")

    leading_shape = stationary.shape[:-1]
    try:
        floor = np.broadcast_to(np.asarray(floor_y, dtype=np.float64), leading_shape)
    except ValueError as exc:
        raise ValueError(
            f"floor_y 必须能广播到前导维 {leading_shape}，实际为 "
            f"{np.asarray(floor_y).shape}"
        ) from exc

    foot_indices = np.asarray(
        [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]],
        dtype=np.int64,
    )
    foot_height = joints[..., foot_indices, 1] - floor[..., None]
    height_probability = np.clip(
        (no_height - foot_height) / (no_height - full_height),
        0.0,
        1.0,
    )
    foot_stationary = np.clip(stationary[..., 1:3], 0.0, 1.0)
    return (foot_stationary * height_probability).astype(np.float32)


def median_filter_time(values: np.ndarray, window: int) -> np.ndarray:
    """沿时间维做 edge-padded median filter，保持 `[T,D]` 形状不变。"""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"median_filter_time 输入应为 [T,D]，实际为 {array.shape}")
    window = int(window)
    if window <= 1 or array.shape[0] <= 1:
        return array.copy()
    if window % 2 == 0:
        raise ValueError(f"median_window 必须为奇数，实际为 {window}")
    radius = window // 2
    padded = np.pad(array, ((radius, radius), (0, 0)), mode="edge")
    stacked = np.stack([padded[offset : offset + array.shape[0]] for offset in range(window)], axis=0)
    return np.median(stacked, axis=0)


def rotation_6d_forward_up_np(rotations: np.ndarray) -> np.ndarray:
    forward = rotations[..., :, 2]
    up = rotations[..., :, 1]
    return np.concatenate([forward, up], axis=-1)


def rotation_6d_forward_up_torch(rotations: torch.Tensor) -> torch.Tensor:
    forward = rotations[..., :, 2]
    up = rotations[..., :, 1]
    return torch.cat([forward, up], dim=-1)


def rotation_6d_to_matrix_np(rotation_6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rotation_6d, dtype=np.float64)
    forward = values[..., 0:3]
    up = values[..., 3:6]
    forward = forward / np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8)
    right = np.cross(up, forward)
    right = right / np.maximum(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8)
    up = np.cross(forward, right)
    return np.stack([right, up, forward], axis=-1)


def rotation_6d_to_matrix_torch(rotation_6d: torch.Tensor) -> torch.Tensor:
    forward = rotation_6d[..., 0:3]
    up = rotation_6d[..., 3:6]
    forward = torch.nn.functional.normalize(forward, dim=-1, eps=1e-8)
    right = torch.cross(up, forward, dim=-1)
    right = torch.nn.functional.normalize(right, dim=-1, eps=1e-8)
    up = torch.cross(forward, right, dim=-1)
    return torch.stack([right, up, forward], dim=-1)


def wrap_radians(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def extract_yaw_from_rotations(rotations: np.ndarray) -> np.ndarray:
    forward = np.asarray(rotations, dtype=np.float64)[..., 2]
    horizontal_norm = np.linalg.norm(forward[:, [0, 2]], axis=-1)
    yaws = np.arctan2(forward[:, 0], forward[:, 2])
    unstable = horizontal_norm < 1e-6
    for index in range(1, len(yaws)):
        if unstable[index]:
            yaws[index] = yaws[index - 1]
    if len(yaws) and unstable[0]:
        yaws[0] = 0.0
    return yaws


def global_to_parent_local_rotations(global_rotations: np.ndarray, parents: np.ndarray = SMPL_PARENTS) -> np.ndarray:
    global_rotations = np.asarray(global_rotations, dtype=np.float64)
    local = np.empty_like(global_rotations)
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            local[:, joint_index] = global_rotations[:, joint_index]
        else:
            local[:, joint_index] = np.swapaxes(global_rotations[:, parent_index], -1, -2) @ global_rotations[:, joint_index]
    return local


def build_body_pose_parent_6d(global_rotations: np.ndarray, root_yaws: np.ndarray) -> np.ndarray:
    local = global_to_parent_local_rotations(global_rotations)
    root_inv = np.swapaxes(make_yaw_rotation_np(root_yaws), -1, -2)
    local[:, 0] = root_inv @ global_rotations[:, 0]
    return rotation_6d_forward_up_np(local).reshape(global_rotations.shape[0], -1).astype(np.float32)


def build_body_pose_root_global_6d(global_rotations: np.ndarray, root_yaws: np.ndarray) -> np.ndarray:
    """Encode SMPL24 global rotations after removing only the root yaw."""

    global_rotations = np.asarray(global_rotations, dtype=np.float64)
    if global_rotations.ndim != 4 or global_rotations.shape[1:] != (24, 3, 3):
        raise ValueError(f"global_rotations should be [T,24,3,3], got {global_rotations.shape}")
    root_yaws = np.asarray(root_yaws, dtype=np.float64)
    if root_yaws.shape != (global_rotations.shape[0],):
        raise ValueError(f"root_yaws should be [T], got {root_yaws.shape}")
    root_inv = np.swapaxes(make_yaw_rotation_np(root_yaws), -1, -2)
    root_relative_global = root_inv[:, None] @ global_rotations
    return rotation_6d_forward_up_np(root_relative_global).reshape(global_rotations.shape[0], -1).astype(np.float32)


def root_global_6d_to_global_rot_torch(body_pose_root_global_6d: torch.Tensor, root_yaw: torch.Tensor) -> torch.Tensor:
    """Decode root-yaw-relative global 6D pose to world/global rotations, shape `[B,24,3,3]`."""

    batch_size = body_pose_root_global_6d.shape[0]
    root_relative_global = rotation_6d_to_matrix_torch(body_pose_root_global_6d.reshape(batch_size, 24, 6))
    yaw_rot = make_yaw_rotation_torch(root_yaw)
    return torch.matmul(yaw_rot[:, None], root_relative_global)


def root_global_6d_to_parent_local_torch(body_pose_root_global_6d: torch.Tensor) -> torch.Tensor:
    """Convert current root-global 6D pose to parent-local 6D for debug/retarget-only consumers."""

    batch_size = body_pose_root_global_6d.shape[0]
    root_relative_global = rotation_6d_to_matrix_torch(body_pose_root_global_6d.reshape(batch_size, 24, 6))
    local_rot: list[torch.Tensor] = []
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            joint_rot = root_relative_global[:, joint_index]
        else:
            joint_rot = root_relative_global[:, parent_index].transpose(-1, -2) @ root_relative_global[:, joint_index]
        local_rot.append(joint_rot)
    local = torch.stack(local_rot, dim=1)
    return rotation_6d_forward_up_torch(local).reshape(batch_size, -1)


def estimate_root_global_offsets(
    joints_world: np.ndarray,
    body_pose_root_global_6d: np.ndarray,
    root_yaws: np.ndarray,
    root_pos_world: np.ndarray | None = None,
    parents: np.ndarray = SMPL_PARENTS,
) -> np.ndarray:
    """Estimate rest offsets from the first frame under root-yaw-relative global pose semantics."""

    joints = np.asarray(joints_world, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"joints_world should be [T,24,3], got {joints.shape}")
    root_relative_global = rotation_6d_to_matrix_np(np.asarray(body_pose_root_global_6d[0], dtype=np.float64).reshape(24, 6))
    root_yaw_rotation = make_yaw_rotation_np(np.asarray([root_yaws[0]], dtype=np.float64))[0]
    global_rot = root_yaw_rotation[None] @ root_relative_global
    offsets = np.zeros((24, 3), dtype=np.float32)
    if root_pos_world is not None:
        root_pos = np.asarray(root_pos_world, dtype=np.float64)
        root_offset_world = joints[0, 0] - root_pos[0]
        offsets[0] = (root_yaw_rotation.T @ root_offset_world).astype(np.float32)
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            continue
        world_offset = joints[0, joint_index] - joints[0, parent_index]
        offsets[joint_index] = (global_rot[parent_index].T @ world_offset).astype(np.float32)
    return offsets


def estimate_parent_offsets(
    joints_world: np.ndarray,
    body_pose_parent_6d: np.ndarray,
    root_yaws: np.ndarray,
    root_pos_world: np.ndarray | None = None,
    parents: np.ndarray = SMPL_PARENTS,
) -> np.ndarray:
    """用第一帧估计 parent-local 骨骼 offset，供轻量 FK loss 和可视化使用。"""

    joints = np.asarray(joints_world, dtype=np.float64)
    rotations = rotation_6d_to_matrix_np(np.asarray(body_pose_parent_6d[0], dtype=np.float64).reshape(24, 6))
    root_rotation = make_yaw_rotation_np(np.asarray([root_yaws[0]], dtype=np.float64))[0] @ rotations[0]
    global_rot = np.empty((24, 3, 3), dtype=np.float64)
    global_rot[0] = root_rotation
    offsets = np.zeros((24, 3), dtype=np.float32)
    if root_pos_world is not None:
        root_pos = np.asarray(root_pos_world, dtype=np.float64)
        root_offset_world = joints[0, 0] - root_pos[0]
        root_yaw_rotation = make_yaw_rotation_np(np.asarray([root_yaws[0]], dtype=np.float64))[0]
        offsets[0] = (root_yaw_rotation.T @ root_offset_world).astype(np.float32)
    for joint_index, parent_index in enumerate(parents):
        if parent_index < 0:
            continue
        global_rot[joint_index] = global_rot[parent_index] @ rotations[joint_index]
        world_offset = joints[0, joint_index] - joints[0, parent_index]
        offsets[joint_index] = (global_rot[parent_index].T @ world_offset).astype(np.float32)
    return offsets


def fk_parent_local_torch(
    body_pose_parent_6d: torch.Tensor,
    root_pos_world: torch.Tensor,
    root_yaw: torch.Tensor,
    parent_offsets: torch.Tensor,
    return_global_rot: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """从 parent-local 6D pose 做可微 FK，输入形状 `[B,144]`，输出 `[B,24,3]`。"""

    batch_size = body_pose_parent_6d.shape[0]
    local_rot = rotation_6d_to_matrix_torch(body_pose_parent_6d.reshape(batch_size, 24, 6))
    yaw_rot = make_yaw_rotation_torch(root_yaw)
    global_rot: list[torch.Tensor] = []
    joints: list[torch.Tensor] = []
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            joint_rot = yaw_rot @ local_rot[:, joint_index]
            root_offset = parent_offsets[:, joint_index]
            joint_pos = root_pos_world + torch.einsum("bij,bj->bi", yaw_rot, root_offset)
        else:
            parent_rot = global_rot[parent_index]
            joint_rot = parent_rot @ local_rot[:, joint_index]
            offset = parent_offsets[:, joint_index]
            joint_pos = joints[parent_index] + torch.einsum("bij,bj->bi", parent_rot, offset)
        global_rot.append(joint_rot)
        joints.append(joint_pos)
    stacked_joints = torch.stack(joints, dim=1)
    if return_global_rot:
        return stacked_joints, torch.stack(global_rot, dim=1)
    return stacked_joints


def fk_root_global_torch(
    body_pose_root_global_6d: torch.Tensor,
    root_pos_world: torch.Tensor,
    root_yaw: torch.Tensor,
    parent_offsets: torch.Tensor,
    return_global_rot: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Differentiable FK for root-yaw-relative global 6D pose, input `[B,144]`, output `[B,24,3]`."""

    global_rot = root_global_6d_to_global_rot_torch(
        body_pose_root_global_6d=body_pose_root_global_6d,
        root_yaw=root_yaw,
    )
    yaw_rot = make_yaw_rotation_torch(root_yaw)
    joints: list[torch.Tensor] = []
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            root_offset = parent_offsets[:, joint_index]
            joint_pos = root_pos_world + torch.einsum("bij,bj->bi", yaw_rot, root_offset)
        else:
            parent_rot = global_rot[:, parent_index]
            offset = parent_offsets[:, joint_index]
            joint_pos = joints[parent_index] + torch.einsum("bij,bj->bi", parent_rot, offset)
        joints.append(joint_pos)
    stacked_joints = torch.stack(joints, dim=1)
    if return_global_rot:
        return stacked_joints, global_rot
    return stacked_joints


def fk_body_fbx_local_torch(
    body_pose_local_delta_6d: torch.Tensor,
    actor_root_pos_world: torch.Tensor,
    root_heading: torch.Tensor,
    rest_local_positions: torch.Tensor,
    rest_local_rotations_6d: torch.Tensor | None = None,
    return_global_rot: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """body.fbx local delta FK，输入 `[B,144]`，输出 `[B,24,3]`。"""

    batch_size = body_pose_local_delta_6d.shape[0]
    delta_rot = rotation_6d_to_matrix_torch(body_pose_local_delta_6d.reshape(batch_size, 24, 6))
    if rest_local_rotations_6d is None:
        rest_local_rot = torch.eye(3, dtype=delta_rot.dtype, device=delta_rot.device).expand(batch_size, 24, 3, 3)
    else:
        rest_6d = rest_local_rotations_6d.to(device=delta_rot.device, dtype=delta_rot.dtype)
        if rest_6d.ndim == 2:
            rest_6d = rest_6d.unsqueeze(0).expand(batch_size, -1, -1)
        rest_local_rot = rotation_6d_to_matrix_torch(rest_6d.reshape(batch_size, 24, 6))
    local_rot = rest_local_rot @ delta_rot
    heading_rot = make_yaw_rotation_torch(root_heading)

    offsets = rest_local_positions.to(device=delta_rot.device, dtype=delta_rot.dtype)
    if offsets.ndim == 2:
        offsets = offsets.unsqueeze(0).expand(batch_size, -1, -1)
    global_rot: list[torch.Tensor] = []
    joints: list[torch.Tensor] = []
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            joint_rot = heading_rot @ local_rot[:, joint_index]
            joint_pos = actor_root_pos_world + torch.einsum("bij,bj->bi", heading_rot, offsets[:, joint_index])
        else:
            parent_rot = global_rot[parent_index]
            joint_rot = parent_rot @ local_rot[:, joint_index]
            joint_pos = joints[parent_index] + torch.einsum("bij,bj->bi", parent_rot, offsets[:, joint_index])
        global_rot.append(joint_rot)
        joints.append(joint_pos)
    stacked_joints = torch.stack(joints, dim=1)
    stacked_rot = torch.stack(global_rot, dim=1)
    if return_global_rot:
        return stacked_joints, stacked_rot
    return stacked_joints
