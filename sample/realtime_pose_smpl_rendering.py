from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFont
from scipy.spatial.transform import Rotation

from data_converter.amass_smpl_utils import (
    AMASS_TO_UNITY,
    SOURCE_BODY_JOINT_COUNT,
    load_motion_source,
    normalize_gender,
)
from data_loaders.body_fbx_kinematics import (
    SOURCE_FK_TO_BODY_FBX_BASIS,
    BodyFbxRest,
    load_body_fbx_rest,
)
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    SMPL_PARENTS,
    global_to_parent_local_rotations,
    make_yaw_rotation_np,
    rotation_6d_to_matrix_np,
)


SCENE_AMBIENT_LIGHT = 0.45
METHOD_ORDER = ("GT", "Predictor-only", "+ Diffusion")


@dataclass(frozen=True)
class ComparisonClip:
    """SMPL 渲染所需的同步片段；时间轴均为 30 Hz source frame。"""

    fps: int
    source_frame_start: int
    source_frame_end_exclusive: int
    diffusion_variant: str
    gt_pose_axis_angle: np.ndarray
    gt_translation_amass: np.ndarray
    betas: np.ndarray
    gender: str
    body_fbx_rest: BodyFbxRest
    reference_rotations_world: np.ndarray
    reference_joints_world: np.ndarray
    predictor_rotations_world: np.ndarray
    predictor_joints_world: np.ndarray
    predictor_root_yaw: np.ndarray
    diffusion_rotations_world: np.ndarray
    diffusion_joints_world: np.ndarray
    diffusion_root_yaw: np.ndarray
    tracker_pos_world: np.ndarray
    tracker_rotations_world: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.gt_pose_axis_angle.shape[0])


@dataclass(frozen=True)
class SmplMeshSequence:
    """逐帧 SMPL-H 网格与关节，均已转换到项目的 Unity/y-up 坐标。"""

    vertices_world: np.ndarray
    joints_world: np.ndarray


@dataclass(frozen=True)
class CameraSpec:
    pose: np.ndarray
    target: np.ndarray
    yfov: float
    aspect_ratio: float


# region 输入校验


def require_file(path: Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} 不存在：{resolved}")
    return resolved


def require_directory(path: Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} 不存在：{resolved}")
    return resolved


def read_scalar_string(payload: Any, key: str) -> str:
    if key not in payload.files:
        raise KeyError(f"source npz 缺少 {key}，无法恢复 body.fbx rest pose。")
    value = np.asarray(payload[key])
    if value.ndim != 0:
        raise ValueError(f"{key} 应为标量字符串，实际为 {value.shape}")
    return str(value.item())


def validate_time_array(array: np.ndarray, name: str, frame_count: int, tail_shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array)
    expected_shape = (int(frame_count),) + tuple(tail_shape)
    if value.shape != expected_shape:
        raise ValueError(f"{name} 应为 {expected_shape}，实际为 {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} 含 NaN/Inf。")
    return value


def load_comparison_clip(
    *,
    comparison_npz: Path,
    report_json: Path,
    amass_npz: Path,
    source_frame_start: int,
    source_frame_end_exclusive: int,
    diffusion_variant: str = "core_only",
) -> ComparisonClip:
    """加载并对齐原始 AMASS、30 Hz source 和模型输出。

    four-way NPZ 的第 0 帧不是 source 第 0 帧，而是报告中的 ``frame_start``。
    这里统一使用 source frame 编号，避免把 60–119 错切成预测数组的 60–119。
    """

    comparison_path = require_file(comparison_npz, "comparison_npz")
    report_path = require_file(report_json, "report_json")
    amass_path = require_file(amass_npz, "amass_npz")
    if int(source_frame_start) < 0 or int(source_frame_end_exclusive) <= int(source_frame_start):
        raise ValueError(
            "source frame 范围必须满足 0 <= start < end，"
            f"实际为 [{source_frame_start},{source_frame_end_exclusive})"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    prediction_frame_offset = int(report["frame_start"])
    fps_value = float(report["fps"])
    fps = int(round(fps_value))
    if not math.isclose(fps_value, float(fps), rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"渲染要求整数 FPS，报告实际为 {fps_value}")
    prediction_start = int(source_frame_start) - prediction_frame_offset
    prediction_end = int(source_frame_end_exclusive) - prediction_frame_offset
    if prediction_start < 0:
        raise ValueError(
            f"source_frame_start={source_frame_start} 早于预测起始帧 {prediction_frame_offset}。"
        )
    frame_count = int(source_frame_end_exclusive) - int(source_frame_start)
    if diffusion_variant not in {"core_only", "all_six"}:
        raise ValueError(
            f"diffusion_variant 必须是 core_only/all_six，实际为 {diffusion_variant}"
        )

    source_path = require_file(Path(report["source_path"]), "报告中的 source_path")
    with np.load(source_path, allow_pickle=False) as source_payload:
        body_fbx_rest_path = require_file(
            Path(read_scalar_string(source_payload, "body_fbx_rest_json")),
            "body_fbx_rest_json",
        )
        tracker_rotations_world = rotation_6d_to_matrix_np(
            np.asarray(
                source_payload["tracker_rot_world_6d"][
                    int(source_frame_start) : int(source_frame_end_exclusive)
                ],
                dtype=np.float32,
            )
        )
    body_fbx_rest = load_body_fbx_rest(body_fbx_rest_path)

    # load_motion_source 复用 converter 的 Slerp 与 translation 插值规则，保证 GT
    # 的第 N 帧与 converted source 的第 N 帧处于同一个 30 Hz 时间点。
    amass_source = load_motion_source(
        path=amass_path,
        amass_dir=amass_path.parent,
        target_fps=float(fps),
    )
    if int(source_frame_end_exclusive) > amass_source.poses.shape[0]:
        raise ValueError(
            f"请求结束帧 {source_frame_end_exclusive} 超过重采样 AMASS 长度 "
            f"{amass_source.poses.shape[0]}。"
        )
    gt_pose_axis_angle = amass_source.poses[
        int(source_frame_start) : int(source_frame_end_exclusive),
        : SOURCE_BODY_JOINT_COUNT * 3,
    ].reshape(frame_count, SOURCE_BODY_JOINT_COUNT, 3)
    gt_translation_amass = amass_source.trans[
        int(source_frame_start) : int(source_frame_end_exclusive)
    ]

    with np.load(comparison_path, allow_pickle=False) as comparison:
        available_frames = int(comparison["reference_joints_world"].shape[0])
        if prediction_end > available_frames:
            raise ValueError(
                f"预测切片 [{prediction_start},{prediction_end}) 超过结果长度 {available_frames}。"
            )
        selected = slice(prediction_start, prediction_end)
        arrays = {
            key: np.asarray(comparison[key][selected], dtype=np.float32)
            for key in (
                "reference_rotations_world",
                "reference_joints_world",
                "predictor_only_rotations_world",
                "predictor_only_joints_world",
                "predictor_only_root_yaw",
                f"{diffusion_variant}_rotations_world",
                f"{diffusion_variant}_joints_world",
                f"{diffusion_variant}_root_yaw",
                "tracker_pos_world",
            )
        }

    validate_time_array(arrays["reference_rotations_world"], "reference_rotations_world", frame_count, (24, 3, 3))
    validate_time_array(arrays["reference_joints_world"], "reference_joints_world", frame_count, (24, 3))
    validate_time_array(arrays["predictor_only_rotations_world"], "predictor_only_rotations_world", frame_count, (24, 3, 3))
    validate_time_array(arrays["predictor_only_joints_world"], "predictor_only_joints_world", frame_count, (24, 3))
    validate_time_array(arrays["predictor_only_root_yaw"], "predictor_only_root_yaw", frame_count, ())
    diffusion_rotations_key = f"{diffusion_variant}_rotations_world"
    diffusion_joints_key = f"{diffusion_variant}_joints_world"
    diffusion_yaw_key = f"{diffusion_variant}_root_yaw"
    validate_time_array(arrays[diffusion_rotations_key], diffusion_rotations_key, frame_count, (24, 3, 3))
    validate_time_array(arrays[diffusion_joints_key], diffusion_joints_key, frame_count, (24, 3))
    validate_time_array(arrays[diffusion_yaw_key], diffusion_yaw_key, frame_count, ())
    validate_time_array(arrays["tracker_pos_world"], "tracker_pos_world", frame_count, (6, 3))
    validate_time_array(
        tracker_rotations_world,
        "tracker_rotations_world",
        frame_count,
        (6, 3, 3),
    )

    return ComparisonClip(
        fps=fps,
        source_frame_start=int(source_frame_start),
        source_frame_end_exclusive=int(source_frame_end_exclusive),
        diffusion_variant=str(diffusion_variant),
        gt_pose_axis_angle=np.asarray(gt_pose_axis_angle, dtype=np.float32),
        gt_translation_amass=np.asarray(gt_translation_amass, dtype=np.float32),
        betas=np.asarray(amass_source.betas, dtype=np.float32).reshape(-1),
        gender=normalize_gender(amass_source.gender),
        body_fbx_rest=body_fbx_rest,
        reference_rotations_world=arrays["reference_rotations_world"],
        reference_joints_world=arrays["reference_joints_world"],
        predictor_rotations_world=arrays["predictor_only_rotations_world"],
        predictor_joints_world=arrays["predictor_only_joints_world"],
        predictor_root_yaw=arrays["predictor_only_root_yaw"],
        diffusion_rotations_world=arrays[diffusion_rotations_key],
        diffusion_joints_world=arrays[diffusion_joints_key],
        diffusion_root_yaw=arrays[diffusion_yaw_key],
        tracker_pos_world=arrays["tracker_pos_world"],
        tracker_rotations_world=np.asarray(tracker_rotations_world, dtype=np.float32),
    )


# endregion


# region body.fbx 到 SMPL 姿态


def body_fbx_world_to_smpl_local_rotations(
    global_rotations_world: np.ndarray,
    root_yaw: np.ndarray,
    rest_local_rotations: np.ndarray,
    parents: np.ndarray = SMPL_PARENTS,
) -> np.ndarray:
    """把 body.fbx 世界旋转逆变换为 AMASS/SMPL 父局部旋转。

    输入分别为 `[T,24,3,3]`、`[T]`、`[24,3,3]`，返回
    `[T,24,3,3]`。该函数严格逆转 source converter 中的 rest rotation、
    root heading 与 ``SOURCE_FK_TO_BODY_FBX_BASIS``，不能用一次普通的
    global-to-local 替代，否则 SMPL 网格会出现约 90 度的基底偏转。
    """

    global_rotations = np.asarray(global_rotations_world, dtype=np.float64)
    headings = np.asarray(root_yaw, dtype=np.float64).reshape(-1)
    rest_rotations = np.asarray(rest_local_rotations, dtype=np.float64)
    parent_indices = np.asarray(parents, dtype=np.int64)
    if global_rotations.ndim != 4 or global_rotations.shape[1:] != (24, 3, 3):
        raise ValueError(
            "global_rotations_world 应为 [T,24,3,3]，"
            f"实际为 {global_rotations.shape}"
        )
    if headings.shape != (global_rotations.shape[0],):
        raise ValueError(f"root_yaw 应为 [T]，实际为 {headings.shape}")
    if rest_rotations.shape != (24, 3, 3):
        raise ValueError(
            f"rest_local_rotations 应为 [24,3,3]，实际为 {rest_rotations.shape}"
        )
    if parent_indices.shape != (24,):
        raise ValueError(f"parents 应为 [24]，实际为 {parent_indices.shape}")

    body_local = global_to_parent_local_rotations(
        global_rotations,
        parents=parent_indices,
    )
    heading_rotations = make_yaw_rotation_np(headings)
    body_local[:, 0] = np.swapaxes(heading_rotations, -1, -2) @ global_rotations[:, 0]
    body_delta = np.swapaxes(rest_rotations, -1, -2)[None] @ body_local

    basis = SOURCE_FK_TO_BODY_FBX_BASIS
    source_local_unity = basis.T[None, None] @ body_delta @ basis[None, None]
    # converter 对 pelvis 只在右侧换基；逆变换后还需把 actor heading 乘回去，
    # 才得到 SMPL 所需的完整 global_orient。
    root_residual_unity = body_delta[:, 0] @ basis
    source_local_unity[:, 0] = heading_rotations @ root_residual_unity

    source_local_amass = (
        AMASS_TO_UNITY.T[None, None]
        @ source_local_unity
        @ AMASS_TO_UNITY[None, None]
    )
    if not np.isfinite(source_local_amass).all():
        raise ValueError("逆变换后的 SMPL 局部旋转含 NaN/Inf。")
    return source_local_amass.astype(np.float32)


def rotation_matrices_to_axis_angle(rotation_matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(rotation_matrices, dtype=np.float64)
    if matrices.ndim != 4 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"rotation_matrices 应为 [T,J,3,3]，实际为 {matrices.shape}")
    axis_angle = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec()
    return axis_angle.reshape(matrices.shape[:-2] + (3,)).astype(np.float32)


def create_smplh_model(model_dir: Path, gender: str, batch_size: int):
    """加载当前 AMASS 配套的 10-beta SMPL-H，规避模型文件与 16-beta 参数不兼容。"""

    try:
        import smplx
    except ImportError as exc:
        raise ImportError("缺少 smplx，无法生成 SMPL-H 网格。") from exc

    model = smplx.SMPLH(
        model_path=str(model_dir),
        gender=gender,
        batch_size=int(batch_size),
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
        ext="npz",
    )
    model.eval()
    return model


def run_smplh_forward(
    *,
    model,
    pose_axis_angle: np.ndarray,
    betas: np.ndarray,
    translation_amass: np.ndarray,
) -> SmplMeshSequence:
    """用共享身形和根平移生成 SMPL-H 网格，输入 pose 为 `[T,22,3]`。"""

    import torch

    poses = np.asarray(pose_axis_angle, dtype=np.float32)
    translations = np.asarray(translation_amass, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (SOURCE_BODY_JOINT_COUNT, 3):
        raise ValueError(f"pose_axis_angle 应为 [T,22,3]，实际为 {poses.shape}")
    if translations.shape != (poses.shape[0], 3):
        raise ValueError(
            f"translation_amass 应为 [T,3]，实际为 {translations.shape}"
        )
    fitted_betas = np.zeros((10,), dtype=np.float32)
    source_betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    fitted_betas[: min(10, source_betas.shape[0])] = source_betas[:10]
    frame_count = poses.shape[0]
    device = next(model.parameters()).device
    parameters = {
        "global_orient": torch.as_tensor(poses[:, 0], device=device),
        "body_pose": torch.as_tensor(poses[:, 1:].reshape(frame_count, -1), device=device),
        "left_hand_pose": torch.zeros((frame_count, 45), dtype=torch.float32, device=device),
        "right_hand_pose": torch.zeros((frame_count, 45), dtype=torch.float32, device=device),
        "betas": torch.as_tensor(
            np.repeat(fitted_betas[None], frame_count, axis=0), device=device
        ),
        "transl": torch.as_tensor(translations, device=device),
        "return_verts": True,
    }
    with torch.no_grad():
        output = model(**parameters)
    vertices_amass = output.vertices.detach().cpu().numpy()
    joints_amass = output.joints[:, :SOURCE_BODY_JOINT_COUNT].detach().cpu().numpy()
    vertices_world = vertices_amass @ AMASS_TO_UNITY.T
    joints_world = joints_amass @ AMASS_TO_UNITY.T
    if not np.isfinite(vertices_world).all() or not np.isfinite(joints_world).all():
        raise ValueError("SMPL-H forward 输出含 NaN/Inf。")
    return SmplMeshSequence(
        vertices_world=vertices_world.astype(np.float32),
        joints_world=joints_world.astype(np.float32),
    )


def transform_faces_to_unity_winding(faces: np.ndarray) -> np.ndarray:
    """匹配 AMASS→Unity 基变换后的三角面绕序，不修改调用方数组。

    ``AMASS_TO_UNITY`` 交换 Y/Z，行列式为负，因此它不仅改变坐标轴名称，
    还反转了空间手性。顶点完成该反射后必须交换每个三角形的后两个顶点，
    否则 PyRender 会把人体外表面当作背面剔除，并从近侧“看穿”到远侧表面。
    """

    source_faces = np.asarray(faces, dtype=np.int64)
    if source_faces.ndim != 2 or source_faces.shape[1] != 3:
        raise ValueError(f"faces 应为 [F,3]，实际为 {source_faces.shape}")
    determinant = float(np.linalg.det(AMASS_TO_UNITY))
    if abs(determinant) < 1e-8:
        raise ValueError("AMASS_TO_UNITY 必须是可逆坐标基变换。")
    if determinant < 0.0:
        return source_faces[:, [0, 2, 1]].copy()
    return source_faces.copy()


def build_mesh_sequences(
    clip: ComparisonClip,
    smpl_model_dir: Path,
) -> tuple[dict[str, SmplMeshSequence], np.ndarray]:
    model_dir = require_directory(smpl_model_dir, "smpl_model_dir")
    predictor_local = body_fbx_world_to_smpl_local_rotations(
        clip.predictor_rotations_world,
        clip.predictor_root_yaw,
        clip.body_fbx_rest.rest_local_rotations,
        clip.body_fbx_rest.parents,
    )
    diffusion_local = body_fbx_world_to_smpl_local_rotations(
        clip.diffusion_rotations_world,
        clip.diffusion_root_yaw,
        clip.body_fbx_rest.rest_local_rotations,
        clip.body_fbx_rest.parents,
    )
    predictor_pose = rotation_matrices_to_axis_angle(
        predictor_local[:, :SOURCE_BODY_JOINT_COUNT]
    )
    diffusion_pose = rotation_matrices_to_axis_angle(
        diffusion_local[:, :SOURCE_BODY_JOINT_COUNT]
    )

    model = create_smplh_model(
        model_dir=model_dir,
        gender=clip.gender,
        batch_size=clip.frame_count,
    )
    sequences = {
        "GT": run_smplh_forward(
            model=model,
            pose_axis_angle=clip.gt_pose_axis_angle,
            betas=clip.betas,
            translation_amass=clip.gt_translation_amass,
        ),
        "Predictor-only": run_smplh_forward(
            model=model,
            pose_axis_angle=predictor_pose,
            betas=clip.betas,
            translation_amass=clip.gt_translation_amass,
        ),
        "+ Diffusion": run_smplh_forward(
            model=model,
            pose_axis_angle=diffusion_pose,
            betas=clip.betas,
            translation_amass=clip.gt_translation_amass,
        ),
    }
    return sequences, transform_faces_to_unity_winding(model.faces)


# endregion


# region 共享渲染基础


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError("无法归一化零向量。")
    return value / norm


def camera_pose_look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """构建 pyrender 相机位姿；相机本地 -Z 指向目标，+Y 尽量保持世界向上。"""

    eye_value = np.asarray(eye, dtype=np.float64)
    target_value = np.asarray(target, dtype=np.float64)
    camera_z = normalize_vector(eye_value - target_value)
    camera_x = normalize_vector(np.cross(np.asarray([0.0, 1.0, 0.0]), camera_z))
    camera_y = normalize_vector(np.cross(camera_z, camera_x))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack([camera_x, camera_y, camera_z], axis=1)
    pose[:3, 3] = eye_value
    return pose


def create_grid_mesh(center: np.ndarray, floor_y: float, size: float):
    import trimesh

    floor = trimesh.creation.box(extents=(size, 0.018, size))
    floor.apply_translation([float(center[0]), float(floor_y) - 0.012, float(center[2])])
    line_parts = []
    half = float(size) * 0.5
    spacing = 0.25
    line_positions = np.arange(-half, half + spacing * 0.5, spacing)
    for offset in line_positions:
        line_x = trimesh.creation.box(extents=(size, 0.008, 0.008))
        line_x.apply_translation(
            [float(center[0]), float(floor_y) + 0.002, float(center[2] + offset)]
        )
        line_parts.append(line_x)
        line_z = trimesh.creation.box(extents=(0.008, 0.008, size))
        line_z.apply_translation(
            [float(center[0] + offset), float(floor_y) + 0.002, float(center[2])]
        )
        line_parts.append(line_z)
    return floor, trimesh.util.concatenate(line_parts)


def create_static_scene(
    camera_spec: CameraSpec,
    floor_y: float,
    grid_size: float,
    grid_center: np.ndarray | None = None,
):
    """创建共享相机、柔和双灯布光与棋盘地面，并返回可移动的 camera node。"""

    import pyrender

    scene = pyrender.Scene(
        bg_color=np.asarray([0.96, 0.97, 0.98, 1.0]),
        # 纯色 SMPL 没有纹理帮助解释表面，高环境光可避免局部出现类似脏污的
        # 大块暗面；仍保留主光和补光，让胸背、四肢维持必要的体积感。
        ambient_light=np.asarray(
            [SCENE_AMBIENT_LIGHT] * 3,
            dtype=np.float64,
        ),
    )
    camera_node = scene.add(
        pyrender.PerspectiveCamera(
            yfov=camera_spec.yfov,
            aspectRatio=camera_spec.aspect_ratio,
        ),
        pose=camera_spec.pose,
    )

    camera_x = camera_spec.pose[:3, 0]
    camera_z = camera_spec.pose[:3, 2]
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    # 主光从相机左前上方斜射，但强度刻意低于旧版高反差布光，避免三路略微
    # 不同的表面法线被渲染成夸张的明暗差异。
    key_pose = camera_pose_look_at(
        camera_spec.target
        + camera_z * 2.2
        - camera_x * 2.0
        + world_up * 2.6,
        camera_spec.target,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.7),
        pose=key_pose,
    )
    # 冷色补光负责抬起背光面，使手臂与双腿的前后关系清楚但不过度变暗。
    fill_pose = camera_pose_look_at(
        camera_spec.target
        + camera_z * 1.5
        + camera_x * 2.4
        + world_up * 1.4,
        camera_spec.target,
    )
    scene.add(
        pyrender.DirectionalLight(
            color=np.asarray([0.76, 0.84, 1.0]),
            intensity=0.9,
        ),
        pose=fill_pose,
    )

    floor_trimesh, grid_trimesh = create_grid_mesh(
        center=(
            camera_spec.target
            if grid_center is None
            else np.asarray(grid_center, dtype=np.float64)
        ),
        floor_y=floor_y,
        size=grid_size,
    )
    floor_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.90, 0.91, 0.92, 1.0),
        metallicFactor=0.0,
        roughnessFactor=0.95,
    )
    grid_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.56, 0.59, 0.63, 1.0),
        metallicFactor=0.0,
        roughnessFactor=0.95,
    )
    scene.add(pyrender.Mesh.from_trimesh(floor_trimesh, material=floor_material, smooth=False))
    scene.add(pyrender.Mesh.from_trimesh(grid_trimesh, material=grid_material, smooth=False))
    return scene, camera_node


def create_sphere_cloud(points: np.ndarray, radius: float):
    import trimesh

    parts = []
    for point in np.asarray(points, dtype=np.float64).reshape(-1, 3):
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=float(radius))
        sphere.apply_translation(point)
        parts.append(sphere)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def build_surface_aligned_glyph_points(
    anchor_points: np.ndarray,
    camera_position: np.ndarray,
    body_vertices: np.ndarray,
    body_faces: np.ndarray,
    *,
    glyph_radius: float,
    outward_offset_ratio: float = 0.18,
) -> np.ndarray:
    """把关节图标沿观察射线贴到人体可见表面，返回 ``[K,3]``。

    ``anchor_points`` 表示图标应对应的渲染关节，而不是新的 tracker 数据。
    每条射线从相机穿过关节投影位置，并取首次命中的人体三角面，因此图标
    的屏幕中心不变，但不会再依赖固定的 12 cm 深度前移。图标中心只向相机
    外移少量半径，使球体与皮肤相交，看起来是贴附而不是悬浮。
    """

    anchors = np.asarray(anchor_points, dtype=np.float64)
    camera = np.asarray(camera_position, dtype=np.float64)
    vertices = np.asarray(body_vertices, dtype=np.float64)
    faces = np.asarray(body_faces, dtype=np.int64)
    radius = float(glyph_radius)
    offset_ratio = float(outward_offset_ratio)
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError(f"anchor_points 应为 [K,3]，实际为 {anchors.shape}")
    if camera.shape != (3,):
        raise ValueError(f"camera_position 应为 [3]，实际为 {camera.shape}")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"body_vertices 应为 [V,3]，实际为 {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"body_faces 应为 [F,3]，实际为 {faces.shape}")
    if faces.size and (int(np.min(faces)) < 0 or int(np.max(faces)) >= vertices.shape[0]):
        raise ValueError("body_faces 含越界顶点索引。")
    if not (
        np.isfinite(anchors).all()
        and np.isfinite(camera).all()
        and np.isfinite(vertices).all()
    ):
        raise ValueError("anchor/camera/body mesh 含 NaN/Inf。")
    if radius <= 0.0:
        raise ValueError("glyph_radius 必须为正数。")
    if not 0.0 <= offset_ratio <= 1.0:
        raise ValueError("outward_offset_ratio 必须位于 [0,1]。")
    if anchors.shape[0] == 0:
        return anchors.astype(np.float32)

    triangles = vertices[faces]
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    ray_directions = anchors - camera[None]
    ray_lengths = np.linalg.norm(ray_directions, axis=1, keepdims=True)
    if np.any(ray_lengths <= 1e-8):
        raise ValueError("anchor point 不能与相机位置重合。")
    ray_directions /= ray_lengths

    result = anchors.copy()
    epsilon = 1e-9
    for point_index, direction in enumerate(ray_directions):
        # Moller-Trumbore 求交只在当前 tracker 的一条射线上展开；每帧最多
        # 六条射线，避免引入 rtree 等额外运行时依赖。
        cross_direction_edge_2 = np.cross(
            np.broadcast_to(direction, edge_2.shape),
            edge_2,
        )
        determinant = np.einsum("ij,ij->i", edge_1, cross_direction_edge_2)
        valid = np.abs(determinant) > epsilon
        inverse_determinant = np.zeros_like(determinant)
        inverse_determinant[valid] = 1.0 / determinant[valid]
        camera_from_triangle = camera[None] - triangles[:, 0]
        barycentric_u = (
            np.einsum("ij,ij->i", camera_from_triangle, cross_direction_edge_2)
            * inverse_determinant
        )
        valid &= (barycentric_u >= -epsilon) & (barycentric_u <= 1.0 + epsilon)
        cross_origin_edge_1 = np.cross(camera_from_triangle, edge_1)
        barycentric_v = (
            np.einsum(
                "ij,ij->i",
                np.broadcast_to(direction, cross_origin_edge_1.shape),
                cross_origin_edge_1,
            )
            * inverse_determinant
        )
        valid &= (barycentric_v >= -epsilon) & (
            barycentric_u + barycentric_v <= 1.0 + epsilon
        )
        distances = (
            np.einsum("ij,ij->i", edge_2, cross_origin_edge_1)
            * inverse_determinant
        )
        valid &= distances > epsilon
        if not np.any(valid):
            # 极端姿态下关节投影可能落在人体轮廓外；保留关节锚点比随意
            # 吸附到另一块肢体更安全，也便于调用方发现真实的轮廓偏差。
            continue
        surface_distance = float(np.min(distances[valid]))
        surface_point = camera + direction * surface_distance
        result[point_index] = (
            surface_point - direction * radius * offset_ratio
        )
    return result.astype(np.float32)


def create_front_marker_mesh(center: np.ndarray, outward_normal: np.ndarray):
    """创建贴近胸口的小圆形徽标；它只标明身体正面，不编码额外姿态。"""

    import trimesh

    marker_center = np.asarray(center, dtype=np.float64).reshape(3)
    normal = normalize_vector(np.asarray(outward_normal, dtype=np.float64).reshape(3))
    marker = trimesh.creation.cylinder(radius=0.042, height=0.012, sections=24)
    alignment = trimesh.geometry.align_vectors(
        np.asarray([0.0, 0.0, 1.0]),
        normal,
    )
    marker.apply_transform(alignment)
    marker.apply_translation(marker_center)
    return marker


def build_horizontal_pelvis_follow_offsets(joints_world: np.ndarray) -> np.ndarray:
    """提取 `[T,3]` pelvis 水平位移，供腿部近景逐帧共享跟随。"""

    joints = np.asarray(joints_world, dtype=np.float64)
    if joints.ndim != 3 or joints.shape[1] <= JOINT_INDEX["pelvis"] or joints.shape[2] != 3:
        raise ValueError(f"joints_world 应为 [T,J,3]，实际为 {joints.shape}")
    if joints.shape[0] == 0 or not np.isfinite(joints).all():
        raise ValueError("joints_world 必须包含有限的逐帧关节。")
    offsets = joints[:, JOINT_INDEX["pelvis"]].copy()
    # 只跟随水平移动，保留统一地面高度；否则起跳时镜头会跟着升降，掩盖离地量。
    offsets[:, 1] = 0.0
    return offsets.astype(np.float32)


def load_font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=int(size))
    return ImageFont.load_default()


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def decode_png(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


# endregion
