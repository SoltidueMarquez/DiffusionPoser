from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from sample.visualization import make_yaw_rotation
from visual_editor.x277 import decode_root_trajectory


def normalize_vector(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """
    将 X277 的 forward/up 6D 旋转近似转成旋转矩阵。

    X277 元数据约定 6D 为 `forward_xyz_then_up_xyz`；这里用 Gram-Schmidt 正交化，
    让可选 SMPL 预览即使遇到轻微数值噪声也能保持有效旋转。
    """

    values = np.asarray(rotation_6d, dtype=np.float64).reshape(-1, 6)
    forward = normalize_vector(values[:, 0:3])
    up_hint = values[:, 3:6]
    right = normalize_vector(np.cross(up_hint, forward))
    up = normalize_vector(np.cross(forward, right))
    matrices = np.stack([right, up, forward], axis=-1)
    return matrices.astype(np.float32)


def matrix_to_axis_angle(rotations: np.ndarray) -> np.ndarray:
    matrices = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    axis_angles = np.zeros((matrices.shape[0], 3), dtype=np.float32)
    for index, matrix in enumerate(matrices):
        cos_theta = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
        theta = math.acos(cos_theta)
        if theta < 1e-6:
            continue
        axis = np.asarray(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ],
            dtype=np.float64,
        )
        axis = axis / max(2.0 * math.sin(theta), 1e-8)
        axis_angles[index] = (axis * theta).astype(np.float32)
    return axis_angles


def build_smpl_mesh_payload(*, smpl_model_dir: Path, x277: np.ndarray, frame: int) -> dict[str, Any]:
    """
    基于当前帧 X277 body rotation 生成可选 SMPL mesh 预览。

    该能力依赖用户本地授权的 SMPL 模型和可选 Python 包；默认环境不自动下载这些资产。
    """

    try:
        import smplx
        import torch
    except ImportError as exc:
        raise RuntimeError("SMPL mesh 预览需要在 visual_editor/.venv 中安装 torch 和 smplx。") from exc

    if not smpl_model_dir.exists():
        raise FileNotFoundError(f"smpl_model_dir 不存在: {smpl_model_dir}")
    frame_index = int(frame)
    if frame_index < 0 or frame_index >= x277.shape[0]:
        raise ValueError(f"frame 越界: {frame_index}, T={x277.shape[0]}")

    rotation_6d = x277[frame_index, 0:144].reshape(24, 6)
    rotmats = rotation_6d_to_matrix(rotation_6d)
    axis_angles = matrix_to_axis_angle(rotmats)
    root_positions, root_yaws = decode_root_trajectory(x277)
    yaw_rotation = make_yaw_rotation(np.asarray([float(root_yaws[frame_index])], dtype=np.float64))[0]
    global_orient_matrix = yaw_rotation @ rotmats[0]
    global_orient = matrix_to_axis_angle(global_orient_matrix[None])[0]

    device = torch.device("cpu")
    model = smplx.SMPL(model_path=str(smpl_model_dir), gender="neutral", batch_size=1).to(device)
    model.eval()
    with torch.no_grad():
        output = model(
            global_orient=torch.as_tensor(global_orient[None], dtype=torch.float32, device=device),
            body_pose=torch.as_tensor(axis_angles[1:24].reshape(1, -1), dtype=torch.float32, device=device),
            transl=torch.as_tensor(root_positions[frame_index][None], dtype=torch.float32, device=device),
        )
    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    return {
        "available": True,
        "frame": frame_index,
        "vertices": vertices.tolist(),
        "faces": np.asarray(model.faces, dtype=np.int32).tolist(),
        "model_type": "smpl",
    }
