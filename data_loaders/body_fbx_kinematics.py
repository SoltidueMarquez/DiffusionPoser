from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.realtime_pose_kinematics import (
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    TRACKER_JOINT_INDICES,
    global_to_parent_local_rotations,
    make_yaw_rotation_np,
    rotation_6d_forward_up_np,
)


SMPL_JOINT_COUNT = len(SMPL_JOINT_NAMES)
IDENTITY_3X3 = np.eye(3, dtype=np.float64)
IDENTITY_6D = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

# Unity 旧 SmplWorldDeltaRetargeter 中的 SourceFkToBodyFbxBasis:
# Quaternion(0, 0.7071068, 0.7071068, 0)，等价于 [x,y,z] -> [-x,z,y]。
SOURCE_FK_TO_BODY_FBX_BASIS = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class BodyFbxRest:
    bone_names: tuple[str, ...]
    parents: np.ndarray
    rest_local_positions: np.ndarray
    rest_local_rotations: np.ndarray
    tracker_joint_indices: np.ndarray
    source_path: Path | None = None

    @property
    def pelvis_local_position(self) -> np.ndarray:
        return self.rest_local_positions[0]


def default_body_fbx_rest_json_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return (
        project_root.parent
        / "SIGGRAPH2024Unity"
        / "Assets"
        / "Projects"
        / "RealtimePose"
        / "Models"
        / "DiffusionPoser"
        / "body_fbx_rest.json"
    )


def load_body_fbx_rest(path: str | Path | None) -> BodyFbxRest:
    rest_path = Path(path).resolve() if path else default_body_fbx_rest_json_path().resolve()
    if not rest_path.exists():
        raise FileNotFoundError(
            f"body.fbx-local source 需要 Unity Editor 导出的 body_fbx_rest.json，当前找不到：{rest_path}"
        )
    with rest_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return body_fbx_rest_from_payload(payload, source_path=rest_path)


def body_fbx_rest_from_payload(payload: dict[str, Any], source_path: Path | None = None) -> BodyFbxRest:
    bone_names = tuple(str(value) for value in payload.get("boneNames", []))
    if len(bone_names) != SMPL_JOINT_COUNT:
        raise ValueError(f"body_fbx_rest.json boneNames 必须是 24 个，实际为 {len(bone_names)}")
    expected = tuple(SMPL_JOINT_NAMES)
    if tuple(_canonical_name(name) for name in bone_names) != tuple(_canonical_name(name) for name in expected):
        raise ValueError("body_fbx_rest.json boneNames 必须按 SMPL24/body.fbx runtime 顺序导出。")

    parents = np.asarray(payload.get("parents", []), dtype=np.int64)
    if parents.shape != (SMPL_JOINT_COUNT,):
        raise ValueError(f"body_fbx_rest.json parents 必须是 [24]，实际为 {parents.shape}")
    if not np.array_equal(parents, SMPL_PARENTS):
        raise ValueError("body_fbx_rest.json parents 与 runtime SMPL24 parent 顺序不一致。")

    rest_positions = _read_vec3_array(payload, "restLocalPositions", SMPL_JOINT_COUNT)
    rest_rotations = _read_quaternion_array(payload, "restLocalRotations", SMPL_JOINT_COUNT)
    tracker_indices = np.asarray(
        payload.get("trackerJointIndices", TRACKER_JOINT_INDICES.tolist()),
        dtype=np.int64,
    )
    if tracker_indices.shape != TRACKER_JOINT_INDICES.shape:
        raise ValueError(f"body_fbx_rest.json trackerJointIndices 必须是 [6]，实际为 {tracker_indices.shape}")
    return BodyFbxRest(
        bone_names=bone_names,
        parents=parents,
        rest_local_positions=rest_positions.astype(np.float32),
        rest_local_rotations=quaternion_xyzw_to_matrix_np(rest_rotations).astype(np.float32),
        tracker_joint_indices=tracker_indices,
        source_path=source_path,
    )


def build_synthetic_body_fbx_rest() -> BodyFbxRest:
    """Smoke test 用的最小 body.fbx rest；真实训练必须使用 Unity 导出的 JSON。"""

    rest_positions = np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float32)
    rest_positions[:, 1] = 0.05
    rest_positions[0, 1] = 0.9
    rest_rotations = np.repeat(IDENTITY_3X3[None], SMPL_JOINT_COUNT, axis=0).astype(np.float32)
    return BodyFbxRest(
        bone_names=tuple(SMPL_JOINT_NAMES),
        parents=SMPL_PARENTS.copy(),
        rest_local_positions=rest_positions,
        rest_local_rotations=rest_rotations,
        tracker_joint_indices=TRACKER_JOINT_INDICES.copy(),
        source_path=None,
    )


def source_global_rotations_to_body_fbx_local_delta_6d(
    global_rotations: np.ndarray,
    root_heading: np.ndarray,
) -> np.ndarray:
    """把 AMASS/SMPL global rotations 转成 body.fbx local delta 6D，形状 `[T,144]`。

    Pelvis 不能再被固定为单位 delta。这里先从源 pelvis 全局旋转中移除
    actor heading，再把剩余的 pitch/roll/yaw residual 转到 body.fbx 基底；
    后续 FK 会按 ``heading @ rest @ residual`` 恢复完整 pelvis 朝向。
    """

    global_rot = np.asarray(global_rotations, dtype=np.float64)
    if global_rot.ndim != 4 or global_rot.shape[1:] != (SMPL_JOINT_COUNT, 3, 3):
        raise ValueError(f"global_rotations 应为 [T,24,3,3]，实际为 {global_rot.shape}")
    headings = np.asarray(root_heading, dtype=np.float64).reshape(-1)
    if headings.shape != (global_rot.shape[0],):
        raise ValueError(f"root_heading 应为 [T]，实际为 {headings.shape}")
    source_local = global_to_parent_local_rotations(global_rot)
    heading_inv = np.swapaxes(make_yaw_rotation_np(headings), -1, -2)
    source_local[:, 0] = heading_inv @ global_rot[:, 0]
    basis = SOURCE_FK_TO_BODY_FBX_BASIS
    body_delta = basis[None, None] @ source_local @ basis.T[None, None]
    return rotation_6d_forward_up_np(body_delta).reshape(global_rot.shape[0], -1).astype(np.float32)


def extract_root_heading_from_source_pelvis_up(global_pelvis_rotations: np.ndarray) -> np.ndarray:
    """使用 Unity 显示规则：source pelvis world rotation 的 up 轴水平投影作为 heading。"""

    rotations = np.asarray(global_pelvis_rotations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"global_pelvis_rotations 应为 [T,3,3]，实际为 {rotations.shape}")
    up_axis = rotations[:, :, 1]
    horizontal_norm = np.linalg.norm(up_axis[:, [0, 2]], axis=-1)
    headings = np.arctan2(up_axis[:, 0], up_axis[:, 2])
    unstable = horizontal_norm < 1e-6
    for index in range(1, len(headings)):
        if unstable[index]:
            headings[index] = headings[index - 1]
    if len(headings) and unstable[0]:
        headings[0] = 0.0
    return headings.astype(np.float32)


def actor_root_positions_from_pelvis(
    pelvis_world: np.ndarray,
    root_heading: np.ndarray,
    pelvis_rest_local_position: np.ndarray,
) -> np.ndarray:
    pelvis = np.asarray(pelvis_world, dtype=np.float64)
    headings = np.asarray(root_heading, dtype=np.float64)
    rest_pelvis = np.asarray(pelvis_rest_local_position, dtype=np.float64)
    if pelvis.ndim != 2 or pelvis.shape[1] != 3:
        raise ValueError(f"pelvis_world 应为 [T,3]，实际为 {pelvis.shape}")
    if headings.shape != (pelvis.shape[0],):
        raise ValueError(f"root_heading 应为 [T]，实际为 {headings.shape}")
    root_heading_rot = make_yaw_rotation_np(headings)
    root_to_pelvis = np.einsum("tij,j->ti", root_heading_rot, rest_pelvis)
    return (pelvis - root_to_pelvis).astype(np.float32)


def fk_body_fbx_local_delta(
    body_pose_local_delta_6d: np.ndarray,
    actor_root_pos_world: np.ndarray,
    root_heading: np.ndarray,
    rest: BodyFbxRest,
    local_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """body.fbx local delta FK，输入 `[T,144]`，输出 world positions/rotations。"""

    from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np

    pose = np.asarray(body_pose_local_delta_6d, dtype=np.float64)
    root_pos = np.asarray(actor_root_pos_world, dtype=np.float64)
    headings = np.asarray(root_heading, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != SMPL_JOINT_COUNT * 6:
        raise ValueError(f"body_pose_local_delta_6d 应为 [T,144]，实际为 {pose.shape}")
    if root_pos.shape != (pose.shape[0], 3):
        raise ValueError(f"actor_root_pos_world 应为 [T,3]，实际为 {root_pos.shape}")
    if headings.shape != (pose.shape[0],):
        raise ValueError(f"root_heading 应为 [T]，实际为 {headings.shape}")
    if local_offsets is None:
        offsets = np.repeat(rest.rest_local_positions.astype(np.float64)[None], pose.shape[0], axis=0)
    else:
        offsets = np.asarray(local_offsets, dtype=np.float64)
        if offsets.shape != (pose.shape[0], SMPL_JOINT_COUNT, 3):
            raise ValueError(f"local_offsets 应为 [T,{SMPL_JOINT_COUNT},3]，实际为 {offsets.shape}")

    delta = rotation_6d_to_matrix_np(pose.reshape(pose.shape[0], SMPL_JOINT_COUNT, 6))
    local_rot = rest.rest_local_rotations.astype(np.float64)[None] @ delta
    heading_rot = make_yaw_rotation_np(headings)
    positions = np.zeros((pose.shape[0], SMPL_JOINT_COUNT, 3), dtype=np.float64)
    rotations = np.zeros((pose.shape[0], SMPL_JOINT_COUNT, 3, 3), dtype=np.float64)
    for bone_index, parent_index in enumerate(rest.parents.tolist()):
        local_offset = offsets[:, bone_index]
        if parent_index < 0:
            rotations[:, bone_index] = heading_rot @ local_rot[:, bone_index]
            positions[:, bone_index] = root_pos + np.einsum("tij,tj->ti", heading_rot, local_offset)
        else:
            parent_rot = rotations[:, parent_index]
            rotations[:, bone_index] = parent_rot @ local_rot[:, bone_index]
            positions[:, bone_index] = positions[:, parent_index] + np.einsum("tij,tj->ti", parent_rot, local_offset)
    return positions.astype(np.float32), rotations.astype(np.float32)


def fk_body_fbx_local_delta_root_y0(
    body_pose_local_delta_6d: np.ndarray,
    actor_root_pos_world: np.ndarray,
    root_heading: np.ndarray,
    pelvis_height: np.ndarray,
    rest: BodyFbxRest,
) -> tuple[np.ndarray, np.ndarray]:
    """
    root-y0 body.fbx FK。

    `actor_root_pos_world` 的 y 固定为 0；`pelvis_height` 表示 pelvis 世界高度，
    因此 FK 前把第 0 个 bone 的 local offset y 动态替换为逐帧 pelvis 高度。
    输入形状：pose `[T,144]`，root `[T,3]`，heading `[T]`，pelvis_height `[T]` 或 `[T,1]`。
    """

    pose = np.asarray(body_pose_local_delta_6d)
    root_pos = np.asarray(actor_root_pos_world, dtype=np.float32).copy()
    height = np.asarray(pelvis_height, dtype=np.float32).reshape(-1)
    if pose.ndim != 2:
        raise ValueError(f"body_pose_local_delta_6d 应为 [T,144]，实际为 {pose.shape}")
    if height.shape != (pose.shape[0],):
        raise ValueError(f"pelvis_height 应为 [T] 或 [T,1]，实际为 {np.asarray(pelvis_height).shape}")
    root_pos[:, 1] = 0.0
    offsets = np.repeat(rest.rest_local_positions.astype(np.float32)[None], pose.shape[0], axis=0)
    offsets[:, 0, 1] = height
    return fk_body_fbx_local_delta(
        body_pose_local_delta_6d=pose,
        actor_root_pos_world=root_pos,
        root_heading=root_heading,
        rest=rest,
        local_offsets=offsets,
    )


def _read_vec3_array(payload: dict[str, Any], key: str, count: int) -> np.ndarray:
    value = payload.get(key)
    if value is None:
        raise KeyError(f"body_fbx_rest.json 缺少 {key}")
    array = np.asarray(_objects_to_vectors(value, ("x", "y", "z")), dtype=np.float64)
    if array.shape != (count, 3):
        raise ValueError(f"body_fbx_rest.json {key} 必须是 [{count},3]，实际为 {array.shape}")
    return array


def _read_quaternion_array(payload: dict[str, Any], key: str, count: int) -> np.ndarray:
    value = payload.get(key)
    if value is None:
        raise KeyError(f"body_fbx_rest.json 缺少 {key}")
    array = np.asarray(_objects_to_vectors(value, ("x", "y", "z", "w")), dtype=np.float64)
    if array.shape != (count, 4):
        raise ValueError(f"body_fbx_rest.json {key} 必须是 [{count},4]，实际为 {array.shape}")
    return array


def _objects_to_vectors(values: Any, fields: tuple[str, ...]) -> list[list[float]]:
    result: list[list[float]] = []
    for item in values:
        if isinstance(item, dict):
            result.append([float(item[field]) for field in fields])
        else:
            result.append([float(value) for value in item])
    return result


def quaternion_xyzw_to_matrix_np(quaternions: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternions, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"quaternion 应以 xyzw 排列，最后一维为 4，实际为 {quat.shape}")
    norm = np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = np.moveaxis(quat / norm, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.empty((*quat.shape[:-1], 3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[..., 0, 1] = 2.0 * (xy - wz)
    matrix[..., 0, 2] = 2.0 * (xz + wy)
    matrix[..., 1, 0] = 2.0 * (xy + wz)
    matrix[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[..., 1, 2] = 2.0 * (yz - wx)
    matrix[..., 2, 0] = 2.0 * (xz - wy)
    matrix[..., 2, 1] = 2.0 * (yz + wx)
    matrix[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def _canonical_name(name: str) -> str:
    value = str(name).strip().lower()
    if value.startswith("m_avg_"):
        value = value[len("m_avg_") :]
    aliases = {
        "l_hip": "left_hip",
        "r_hip": "right_hip",
        "l_knee": "left_knee",
        "r_knee": "right_knee",
        "l_ankle": "left_ankle",
        "r_ankle": "right_ankle",
        "l_foot": "left_foot",
        "r_foot": "right_foot",
        "l_collar": "left_collar",
        "r_collar": "right_collar",
        "l_shoulder": "left_shoulder",
        "r_shoulder": "right_shoulder",
        "l_elbow": "left_elbow",
        "r_elbow": "right_elbow",
        "l_wrist": "left_wrist",
        "r_wrist": "right_wrist",
        "l_hand": "left_hand",
        "r_hand": "right_hand",
    }
    return aliases.get(value, value)
