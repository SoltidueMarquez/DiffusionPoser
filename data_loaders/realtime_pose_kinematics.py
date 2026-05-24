from __future__ import annotations

import math

import numpy as np
import torch


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


def derive_foot_contact(
    joints_world: np.ndarray,
    fps: float = 60.0,
    height_threshold: float = 0.05,
    speed_threshold: float = 0.05,
) -> np.ndarray:
    """
    从 GT joints 派生左右脚接触标签 `[T,2]`。

    `speed_threshold` 使用 m/frame 语义；保留 fps 参数是为了调用处能显式记录数据帧率。
    """

    del fps
    joints = np.asarray(joints_world, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1:] != (24, 3):
        raise ValueError(f"joints_world 应为 [T,24,3]，实际为 {joints.shape}")
    foot_indices = [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]]
    foot_pos = joints[:, foot_indices]
    velocity_xz = np.zeros((joints.shape[0], 2), dtype=np.float64)
    if joints.shape[0] > 1:
        velocity_xz[1:] = np.linalg.norm(foot_pos[1:, :, [0, 2]] - foot_pos[:-1, :, [0, 2]], axis=-1)
    low = foot_pos[:, :, 1] < float(height_threshold)
    still = velocity_xz < float(speed_threshold)
    return (low & still).astype(np.float32)


def rotation_6d_forward_up_np(rotations: np.ndarray) -> np.ndarray:
    forward = rotations[..., :, 2]
    up = rotations[..., :, 1]
    return np.concatenate([forward, up], axis=-1)


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
) -> torch.Tensor:
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
    return torch.stack(joints, dim=1)
