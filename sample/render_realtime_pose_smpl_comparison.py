from __future__ import annotations

import argparse
from contextlib import suppress
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
from eval.realtime_pose_metrics import compute_rpm_p2_mc_metrics
from sample.render_realtime_pose_comparison import Mp4FrameWriter


OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
PANEL_WIDTH = OUTPUT_WIDTH // 3
HEADER_HEIGHT = 96
FOOTER_HEIGHT = 104
VIEWPORT_HEIGHT = OUTPUT_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT
INSET_WIDTH = 270
INSET_HEIGHT = 330
PERSPECTIVE_CAMERA_YFOV = math.radians(34.0)
PERSPECTIVE_CAMERA_SIDE_YAW = math.radians(25.0)
PERSPECTIVE_CAMERA_ELEVATION = math.radians(8.0)
SCENE_AMBIENT_LIGHT = 0.45
CORE_TRACKER_COUNT = 3
FOOT_TRAIL_FRAMES = 8
HEAD_FORWARD_COLOR = (1.0, 0.45, 0.02, 1.0)
TRAVEL_DIRECTION_COLOR = (0.10, 0.75, 0.32, 1.0)
FRONT_MARKER_COLOR = (1.0, 0.88, 0.18, 1.0)
LEFT_FOOT_TRAIL_COLOR = (0.05, 0.82, 0.92, 1.0)
RIGHT_FOOT_TRAIL_COLOR = (0.86, 0.24, 0.78, 1.0)
DIRECTION_ARROW_LENGTH = 0.38
DIRECTION_SMOOTHING_HALF_WINDOW = 5

METHOD_ORDER = ("GT", "Predictor-only", "+ Diffusion")
METHOD_COLORS = {
    "GT": (0.16, 0.43, 0.84, 1.0),
    "Predictor-only": (0.43, 0.46, 0.50, 1.0),
    "+ Diffusion": (0.86, 0.18, 0.20, 1.0),
}


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

    @property
    def diffusion_tracker_count(self) -> int:
        return 6 if self.diffusion_variant == "all_six" else CORE_TRACKER_COUNT

    @property
    def diffusion_header(self) -> str:
        return (
            "+ Diffusion (all 6 trackers)"
            if self.diffusion_variant == "all_six"
            else "+ Diffusion (same 3 trackers)"
        )


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


# region CLI 与输入校验


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render GT, RPM-backbone and three-tracker Diffusion with one SMPL-H mesh."
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--comparison_npz", required=True, type=Path)
    paths.add_argument("--report_json", required=True, type=Path)
    paths.add_argument("--amass_npz", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_mp4", required=True, type=Path)

    clip = parser.add_argument_group("clip")
    clip.add_argument(
        "--diffusion_variant",
        default="core_only",
        choices=("core_only", "all_six"),
        help="选择 three-tracker core-only 或 six-tracker Diffusion 结果。",
    )
    clip.add_argument("--source_frame_start", default=60, type=int)
    clip.add_argument("--source_frame_end_exclusive", default=120, type=int)
    clip.add_argument(
        "--peak_source_frame_start",
        default=-1,
        type=int,
        help="高亮时间优势最明显的 source 起始帧；-1 表示不高亮。",
    )
    clip.add_argument(
        "--peak_source_frame_end_exclusive",
        default=-1,
        type=int,
        help="高亮 source 结束帧（右开区间）；-1 表示不高亮。",
    )
    return parser


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


# region EGL mesh renderer


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


def fit_travel_oblique_perspective_camera(
    points: np.ndarray,
    travel_direction: np.ndarray,
    viewport_width: int,
    viewport_height: int,
    padding: float = 1.2,
) -> CameraSpec:
    """沿整段移动方向设置固定的轻微斜侧透视相机。

    ``travel_direction`` 落在 Unity XZ 平面。相机以侧视为主，并向人物前方偏转
    少量角度：运动方向仍接近画面横轴，但左右脚不再被纯侧视压到同一条线上。
    使用较窄视场角保留真实深度关系，同时避免旧三分之四广角相机夸大头部投影。
    """

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("points 必须包含有限的三维点。")
    if int(viewport_width) <= 0 or int(viewport_height) <= 0:
        raise ValueError("viewport_width/viewport_height 必须为正数。")
    if float(padding) <= 0.0:
        raise ValueError("padding 必须为正数。")

    travel = np.asarray(travel_direction, dtype=np.float64).reshape(3).copy()
    travel[1] = 0.0
    travel = normalize_vector(travel)
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    side_view = normalize_vector(np.cross(travel, world_up))
    # 从纯侧面向人物前方偏转 25 度，既显出胸背和两脚间距，也让动作继续从左
    # 向右展开。这里只改变镜头，不触碰三路姿态或共享 GT 根平移。
    horizontal_view = normalize_vector(
        side_view
        + travel * math.tan(PERSPECTIVE_CAMERA_SIDE_YAW)
    )
    view_direction = normalize_vector(
        horizontal_view
        + world_up * math.tan(PERSPECTIVE_CAMERA_ELEVATION)
    )

    mins = np.min(values, axis=0)
    maxs = np.max(values, axis=0)
    target = (mins + maxs) * 0.5
    # 先用单位距离获得相机基向量；下面的投影和距离拟合只依赖旋转。
    eye = target + view_direction
    pose = camera_pose_look_at(eye, target)

    # 世界轴包围盒中心在斜侧俯视下不一定是屏幕包围盒中心，先在相机平面内校正。
    projected = (values - target) @ pose[:3, :3]
    projected_center = (
        np.min(projected[:, :2], axis=0) + np.max(projected[:, :2], axis=0)
    ) * 0.5
    target = (
        target
        + pose[:3, 0] * projected_center[0]
        + pose[:3, 1] * projected_center[1]
    )
    projected = (values - target) @ pose[:3, :3]

    aspect = float(viewport_width) / float(viewport_height)
    tan_half_y = math.tan(PERSPECTIVE_CAMERA_YFOV * 0.5)
    tan_half_x = tan_half_y * aspect
    # 对每个三维点直接求满足水平、垂直视锥的最小相机距离。projected[:,2]
    # 是点沿 eye-target 方向的偏移，因此也计入近远深度，而不是只拟合二维包围盒。
    required_horizontal = (
        projected[:, 2]
        + np.abs(projected[:, 0]) * float(padding) / tan_half_x
    )
    required_vertical = (
        projected[:, 2]
        + np.abs(projected[:, 1]) * float(padding) / tan_half_y
    )
    distance = max(
        float(np.max(required_horizontal)),
        float(np.max(required_vertical)),
        float(np.max(projected[:, 2])) + 0.25,
        0.75,
    )
    eye = target + view_direction * distance
    pose = camera_pose_look_at(eye, target)
    return CameraSpec(
        pose=pose,
        target=target,
        yfov=float(PERSPECTIVE_CAMERA_YFOV),
        aspect_ratio=float(aspect),
    )


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


def build_horizontal_travel_directions(
    translation_amass: np.ndarray,
    half_window: int = DIRECTION_SMOOTHING_HALF_WINDOW,
) -> np.ndarray:
    """从共享 GT 根平移计算逐帧水平移动方向，返回 ``[T,3]``。

    单帧位移容易被起跳前的身体摆动放大，因此用对称时间窗计算方向。
    近乎静止时回落到整段位移，避免箭头因数值噪声随机翻转。
    """

    translations = np.asarray(translation_amass, dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError(
            f"translation_amass 应为 [T,3]，实际为 {translations.shape}"
        )
    if translations.shape[0] == 0:
        raise ValueError("translation_amass 至少需要 1 帧。")
    if int(half_window) < 1:
        raise ValueError("half_window 必须至少为 1。")

    positions_world = translations @ AMASS_TO_UNITY.T
    fallback = build_horizontal_clip_direction(translations).astype(np.float64)

    directions = np.empty_like(positions_world)
    last_index = translations.shape[0] - 1
    for frame_index in range(translations.shape[0]):
        start = max(0, frame_index - int(half_window))
        end = min(last_index, frame_index + int(half_window))
        delta = positions_world[end] - positions_world[start]
        delta[1] = 0.0
        norm = float(np.linalg.norm(delta))
        directions[frame_index] = fallback if norm < 1e-8 else delta / norm
    return directions.astype(np.float32)


def build_horizontal_clip_direction(translation_amass: np.ndarray) -> np.ndarray:
    """返回整段共享 GT 根平移的水平运动方向。"""

    translations = np.asarray(translation_amass, dtype=np.float64)
    if translations.ndim != 2 or translations.shape[1] != 3:
        raise ValueError(
            f"translation_amass 应为 [T,3]，实际为 {translations.shape}"
        )
    if translations.shape[0] == 0:
        raise ValueError("translation_amass 至少需要 1 帧。")
    positions_world = translations @ AMASS_TO_UNITY.T
    direction = positions_world[-1] - positions_world[0]
    direction[1] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return (direction / norm).astype(np.float32)


def create_arrow_mesh(
    origin: np.ndarray,
    direction: np.ndarray,
    length: float = DIRECTION_ARROW_LENGTH,
):
    """构造沿 ``direction`` 的箭头网格，用于直接对比 Head 朝向与位移方向。"""

    import trimesh

    start = np.asarray(origin, dtype=np.float64).reshape(3)
    vector = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("direction 不能为零向量。")
    if float(length) <= 0.0:
        raise ValueError("length 必须为正数。")

    unit = vector / norm
    head_length = min(float(length) * 0.30, 0.12)
    shaft_length = float(length) - head_length
    shaft = trimesh.creation.cylinder(
        radius=0.012,
        height=shaft_length,
        sections=12,
    )
    # cylinder 默认以原点为中心；先在局部 +Z 上放到箭杆中点。
    shaft.apply_translation([0.0, 0.0, shaft_length * 0.5])
    arrow_head = trimesh.creation.cone(
        radius=0.036,
        height=head_length,
        sections=16,
    )
    # cone 默认从 z=0 延伸到 z=height，把底面接到箭杆末端。
    arrow_head.apply_translation([0.0, 0.0, shaft_length])
    arrow = trimesh.util.concatenate((shaft, arrow_head))
    alignment = trimesh.geometry.align_vectors(
        np.asarray([0.0, 0.0, 1.0]),
        unit,
    )
    arrow.apply_transform(alignment)
    arrow.apply_translation(start)
    return arrow


def render_mesh_view(
    *,
    renderer,
    scene,
    vertices: np.ndarray,
    faces: np.ndarray,
    body_color: tuple[float, float, float, float],
    tracker_points: np.ndarray | None = None,
    trail_points: np.ndarray | None = None,
    front_marker: tuple[np.ndarray, np.ndarray] | None = None,
    direction_arrows: tuple[
        tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]], ...
    ] = (),
    enable_directional_shadows: bool = True,
) -> np.ndarray:
    import pyrender
    import trimesh

    dynamic_nodes = []
    body = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=body_color,
        metallicFactor=0.0,
        roughnessFactor=0.90,
    )
    dynamic_nodes.append(
        scene.add(pyrender.Mesh.from_trimesh(body, material=body_material, smooth=True))
    )
    if tracker_points is not None:
        tracker_cloud = create_sphere_cloud(tracker_points, radius=0.035)
        tracker_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(1.0, 0.55, 0.05, 1.0),
            metallicFactor=0.0,
            roughnessFactor=0.45,
        )
        dynamic_nodes.append(
            scene.add(
                pyrender.Mesh.from_trimesh(
                    tracker_cloud,
                    material=tracker_material,
                    smooth=True,
                )
            )
        )
    if trail_points is not None and len(trail_points):
        trails = np.asarray(trail_points, dtype=np.float64)
        if trails.ndim != 3 or trails.shape[1:] != (2, 3):
            raise ValueError(
                f"trail_points 应为 [N,2,3]（左脚、右脚），实际为 {trails.shape}"
            )
        for foot_index, trail_color in enumerate(
            (LEFT_FOOT_TRAIL_COLOR, RIGHT_FOOT_TRAIL_COLOR)
        ):
            trail_cloud = create_sphere_cloud(
                trails[:, foot_index],
                radius=0.018,
            )
            trail_material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=trail_color,
                metallicFactor=0.0,
                roughnessFactor=0.50,
            )
            dynamic_nodes.append(
                scene.add(
                    pyrender.Mesh.from_trimesh(
                        trail_cloud,
                        material=trail_material,
                        smooth=True,
                    )
                )
            )
    if front_marker is not None:
        marker_center, marker_normal = front_marker
        marker_mesh = create_front_marker_mesh(marker_center, marker_normal)
        marker_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=FRONT_MARKER_COLOR,
            metallicFactor=0.0,
            roughnessFactor=0.38,
        )
        dynamic_nodes.append(
            scene.add(
                pyrender.Mesh.from_trimesh(
                    marker_mesh,
                    material=marker_material,
                    smooth=True,
                )
            )
        )
    for origin, direction, color in direction_arrows:
        arrow = create_arrow_mesh(origin, direction)
        arrow_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.42,
        )
        dynamic_nodes.append(
            scene.add(
                pyrender.Mesh.from_trimesh(
                    arrow,
                    material=arrow_material,
                    smooth=True,
                )
            )
        )
    try:
        # 正式筛选视频会关闭投射阴影，避免阴影轮廓与脚部姿态互相干扰；
        # 原讲解型视频仍可显式保留阴影来提供落地关系。
        render_flags = (
            pyrender.RenderFlags.SHADOWS_DIRECTIONAL
            if enable_directional_shadows
            else pyrender.RenderFlags.NONE
        )
        color, _ = renderer.render(
            scene,
            flags=render_flags,
        )
        return np.asarray(color[..., :3], dtype=np.uint8)
    finally:
        for node in dynamic_nodes:
            scene.remove_node(node)


def load_font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=int(size))
    return ImageFont.load_default()


def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill) -> None:
    bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=3, align="center")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) * 0.5
    y = box[1] + (box[3] - box[1] - height) * 0.5
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=3, align="center")


def draw_direction_legend(draw: ImageDraw.ImageDraw, top: int) -> None:
    """标注两种朝向与胸口正面徽标，避免和 Tracker 圆点混淆。"""

    left = 18
    right = 330
    bottom = int(top) + 92
    draw.rounded_rectangle(
        (left, int(top), right, bottom),
        radius=8,
        fill=(31, 41, 55),
    )
    entries = (
        ("Head forward (3D)", HEAD_FORWARD_COLOR, "arrow"),
        ("Travel direction (+/-5F)", TRAVEL_DIRECTION_COLOR, "arrow"),
        ("Chest badge = body front", FRONT_MARKER_COLOR, "badge"),
    )
    for row, (label, color, symbol) in enumerate(entries):
        center_y = int(top) + 17 + row * 27
        rgb = tuple(int(round(channel * 255.0)) for channel in color[:3])
        if symbol == "arrow":
            draw.line((30, center_y, 58, center_y), fill=rgb, width=5)
            draw.polygon(
                ((58, center_y - 6), (69, center_y), (58, center_y + 6)),
                fill=rgb,
            )
        else:
            draw.ellipse(
                (43, center_y - 7, 57, center_y + 7),
                fill=rgb,
                outline=(255, 255, 255),
                width=1,
            )
        draw.text(
            (80, center_y - 9),
            label,
            font=load_font(13),
            fill=(255, 255, 255),
        )


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def decode_png(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def compose_panel(
    *,
    method_name: str,
    full_rgb: np.ndarray,
    inset_rgb: np.ndarray | None,
    source_frame: int,
    peak_source_frame_start: int,
    peak_source_frame_end_exclusive: int,
    header_text: str,
    tracker_label: str | None,
    playback_label: str,
    footer_text: str,
) -> Image.Image:
    canvas = Image.new("RGB", (PANEL_WIDTH, OUTPUT_HEIGHT), color=(247, 248, 250))
    draw = ImageDraw.Draw(canvas)
    header_color = tuple(int(channel * 255) for channel in METHOD_COLORS[method_name][:3])
    draw.rectangle((0, 0, PANEL_WIDTH, HEADER_HEIGHT), fill=header_color)
    draw_centered_text(
        draw,
        (0, 4, PANEL_WIDTH, HEADER_HEIGHT - 22),
        header_text,
        load_font(24),
        (255, 255, 255),
    )
    draw_centered_text(
        draw,
        (0, HEADER_HEIGHT - 29, PANEL_WIDTH, HEADER_HEIGHT - 2),
        playback_label,
        load_font(15),
        (250, 250, 250),
    )
    canvas.paste(Image.fromarray(full_rgb), (0, HEADER_HEIGHT))
    draw_direction_legend(draw, HEADER_HEIGHT + 18)

    if tracker_label is not None:
        draw.rounded_rectangle(
            (18, HEADER_HEIGHT + 120, 303, HEADER_HEIGHT + 153),
            radius=8,
            fill=(31, 41, 55),
        )
        draw.text(
            (29, HEADER_HEIGHT + 127),
            tracker_label,
            font=load_font(13),
            fill=(255, 255, 255),
        )

    if inset_rgb is not None:
        inset_x = PANEL_WIDTH - INSET_WIDTH - 18
        inset_y = HEADER_HEIGHT + VIEWPORT_HEIGHT - INSET_HEIGHT - 18
        peak_frame = (
            int(peak_source_frame_start) >= 0
            and int(peak_source_frame_start)
            <= int(source_frame)
            < int(peak_source_frame_end_exclusive)
        )
        border_color = (245, 158, 11) if peak_frame else (255, 255, 255)
        draw.rectangle(
            (
                inset_x - 5,
                inset_y - 34,
                inset_x + INSET_WIDTH + 5,
                inset_y + INSET_HEIGHT + 5,
            ),
            fill=(255, 255, 255),
            outline=border_color,
            width=5,
        )
        draw.text(
            (inset_x + 7, inset_y - 28),
            "LOWER BODY | L/R TRAILS (8F)",
            font=load_font(12),
            fill=(31, 41, 55),
        )
        canvas.paste(Image.fromarray(inset_rgb), (inset_x, inset_y))
        if peak_frame:
            draw.rounded_rectangle(
                (inset_x + 8, inset_y + 8, inset_x + 181, inset_y + 39),
                radius=7,
                fill=(245, 158, 11),
            )
            draw.text(
                (inset_x + 17, inset_y + 15),
                "PEAK TEMPORAL GAP",
                font=load_font(13),
                fill=(17, 24, 39),
            )

    footer_top = HEADER_HEIGHT + VIEWPORT_HEIGHT
    draw.rectangle((0, footer_top, PANEL_WIDTH, OUTPUT_HEIGHT), fill=(249, 250, 251))
    draw.line((0, footer_top, PANEL_WIDTH, footer_top), fill=(209, 213, 219), width=2)
    draw_centered_text(
        draw,
        (14, footer_top + 3, PANEL_WIDTH - 14, OUTPUT_HEIGHT - 3),
        footer_text,
        load_font(17),
        (31, 41, 55),
    )
    return canvas


def build_footer_texts(clip: ComparisonClip) -> dict[str, str]:
    predictor_metrics = compute_rpm_p2_mc_metrics(
        predicted_global_rotations=clip.predictor_rotations_world,
        target_global_rotations=clip.reference_rotations_world,
        predicted_joint_positions=clip.predictor_joints_world,
        target_joint_positions=clip.reference_joints_world,
        fps=float(clip.fps),
    )
    diffusion_metrics = compute_rpm_p2_mc_metrics(
        predicted_global_rotations=clip.diffusion_rotations_world,
        target_global_rotations=clip.reference_rotations_world,
        predicted_joint_positions=clip.diffusion_joints_world,
        target_joint_positions=clip.reference_joints_world,
        fps=float(clip.fps),
    )
    velocity_gain = (
        1.0
        - float(diffusion_metrics["mpjve_cm_per_s"])
        / float(predictor_metrics["mpjve_cm_per_s"])
    ) * 100.0
    jitter_gain = (
        1.0
        - float(diffusion_metrics["pred_jitter_m_per_s3"])
        / float(predictor_metrics["pred_jitter_m_per_s3"])
    ) * 100.0
    return {
        "GT": "Reference motion\nShared GT root translation",
        "Predictor-only": (
            f"MPJVE {predictor_metrics['mpjve_cm_per_s']:.2f} cm/s  |  "
            f"Jitter {predictor_metrics['pred_jitter_m_per_s3']:.1f} m/s^3\n"
            "Metrics from original 3-tracker output"
        ),
        "+ Diffusion": (
            f"MPJVE {diffusion_metrics['mpjve_cm_per_s']:.2f} cm/s  "
            f"(-{velocity_gain:.1f}%)\n"
            f"Jitter {diffusion_metrics['pred_jitter_m_per_s3']:.1f} m/s^3  "
            f"(-{jitter_gain:.1f}%)"
        ),
    }


def render_smpl_comparison_video(
    *,
    output_path: Path,
    clip: ComparisonClip,
    sequences: dict[str, SmplMeshSequence],
    faces: np.ndarray,
    peak_source_frame_start: int = -1,
    peak_source_frame_end_exclusive: int = -1,
) -> Path:
    """渲染 2 秒原速 + 4 秒半速 replay，总计 ``3*T`` 帧。"""

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError("缺少 pyrender，无法执行 SMPL-H 离屏渲染。") from exc

    reference = sequences["GT"]
    clip_travel_direction = build_horizontal_clip_direction(
        clip.gt_translation_amass
    )
    full_camera = fit_travel_oblique_perspective_camera(
        reference.vertices_world,
        travel_direction=clip_travel_direction,
        viewport_width=PANEL_WIDTH,
        viewport_height=VIEWPORT_HEIGHT,
        padding=1.15,
    )
    lower_joint_indices = np.asarray(
        [
            JOINT_INDEX["left_hip"],
            JOINT_INDEX["right_hip"],
            JOINT_INDEX["left_knee"],
            JOINT_INDEX["right_knee"],
            JOINT_INDEX["left_ankle"],
            JOINT_INDEX["right_ankle"],
            JOINT_INDEX["left_foot"],
            JOINT_INDEX["right_foot"],
        ],
        dtype=np.int64,
    )
    inset_follow_offsets = build_horizontal_pelvis_follow_offsets(
        reference.joints_world
    )
    # 在拟合腿部相机前去掉逐帧 pelvis 水平位移，使相机尺度只由人体下半身决定，
    # 而不是被整段轨迹长度拉远。渲染时再把同一偏移加回相机位姿。
    lower_reference_points = (
        reference.joints_world[:, lower_joint_indices]
        - inset_follow_offsets[:, None]
    )
    lower_camera = fit_travel_oblique_perspective_camera(
        lower_reference_points,
        travel_direction=clip_travel_direction,
        viewport_width=INSET_WIDTH,
        viewport_height=INSET_HEIGHT,
        padding=1.24,
    )
    floor_y = float(np.min(reference.vertices_world[..., 1]))
    horizontal_extent = np.ptp(reference.vertices_world[..., [0, 2]].reshape(-1, 2), axis=0)
    grid_size = max(4.0, float(np.max(horizontal_extent)) + 2.0)
    full_scene, _ = create_static_scene(
        full_camera,
        floor_y=floor_y,
        grid_size=grid_size,
    )
    inset_scene, inset_camera_node = create_static_scene(
        lower_camera,
        floor_y=floor_y,
        grid_size=grid_size,
        # 相机逐帧跟随 pelvis，但棋盘地面保持在完整世界轨迹中心。
        grid_center=full_camera.target,
    )
    full_renderer = pyrender.OffscreenRenderer(PANEL_WIDTH, VIEWPORT_HEIGHT)
    inset_renderer = pyrender.OffscreenRenderer(INSET_WIDTH, INSET_HEIGHT)
    foot_indices = np.asarray(
        [JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"]], dtype=np.int64
    )
    footer_texts = build_footer_texts(clip)
    header_texts = {
        "GT": "GT",
        "Predictor-only": "Predictor-only (RPM backbone)",
        "+ Diffusion": clip.diffusion_header,
    }
    tracker_labels = {
        "GT": None,
        "Predictor-only": "dots = same Head + wrists",
        "+ Diffusion": (
            "dots = all 6 trackers"
            if clip.diffusion_tracker_count == 6
            else "dots = same Head + wrists"
        ),
    }
    method_rotations_world = {
        "GT": clip.reference_rotations_world,
        "Predictor-only": clip.predictor_rotations_world,
        "+ Diffusion": clip.diffusion_rotations_world,
    }
    travel_directions = build_horizontal_travel_directions(
        clip.gt_translation_amass,
        half_window=DIRECTION_SMOOTHING_HALF_WINDOW,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: Mp4FrameWriter | None = None
    slow_frames: list[bytes] = []

    try:
        for frame_index in range(clip.frame_count):
            source_frame = clip.source_frame_start + frame_index
            travel_direction = travel_directions[frame_index]
            # Head tracker 的局部 +Z 是完整三维 forward；保留 pitch，才能区分
            # 落地时向前低头与真正的侧向歪头。
            head_forward = normalize_vector(
                clip.tracker_rotations_world[frame_index, 0, :, 2]
            ).astype(np.float32)
            head_position = clip.tracker_pos_world[frame_index, 0]
            direction_arrows = (
                (
                    head_position + np.asarray([0.0, 0.10, 0.0]),
                    head_forward,
                    HEAD_FORWARD_COLOR,
                ),
                (
                    head_position + np.asarray([0.0, -0.08, 0.0]),
                    travel_direction,
                    TRAVEL_DIRECTION_COLOR,
                ),
            )
            inset_camera_pose = lower_camera.pose.copy()
            inset_camera_pose[:3, 3] += inset_follow_offsets[frame_index]
            # 同一 source frame 的 GT/Predictor/Diffusion 共用这一近景相机。
            inset_scene.set_pose(inset_camera_node, pose=inset_camera_pose)
            normal_panels: list[Image.Image] = []
            slow_panels: list[Image.Image] = []
            for method_name in METHOD_ORDER:
                sequence = sequences[method_name]
                tracker_count = (
                    clip.diffusion_tracker_count
                    if method_name == "+ Diffusion"
                    else CORE_TRACKER_COUNT
                )
                trackers = None if method_name == "GT" else clip.tracker_pos_world[
                    frame_index, :tracker_count
                ]
                torso_front = -method_rotations_world[method_name][
                    frame_index,
                    JOINT_INDEX["spine3"],
                    :,
                    2,
                ]
                torso_front = normalize_vector(torso_front).astype(np.float32)
                front_marker = (
                    sequence.joints_world[
                        frame_index,
                        JOINT_INDEX["spine3"],
                    ]
                    + torso_front * 0.12,
                    torso_front,
                )
                full_rgb = render_mesh_view(
                    renderer=full_renderer,
                    scene=full_scene,
                    vertices=sequence.vertices_world[frame_index],
                    faces=faces,
                    body_color=METHOD_COLORS[method_name],
                    tracker_points=trackers,
                    front_marker=front_marker,
                    direction_arrows=direction_arrows,
                )
                trail_start = max(0, frame_index - FOOT_TRAIL_FRAMES + 1)
                trail_points = sequence.joints_world[
                    trail_start : frame_index + 1,
                    foot_indices,
                ]
                inset_rgb = render_mesh_view(
                    renderer=inset_renderer,
                    scene=inset_scene,
                    vertices=sequence.vertices_world[frame_index],
                    faces=faces,
                    body_color=METHOD_COLORS[method_name],
                    trail_points=trail_points,
                )
                normal_panels.append(
                    compose_panel(
                        method_name=method_name,
                        full_rgb=full_rgb,
                        inset_rgb=None,
                        source_frame=source_frame,
                        peak_source_frame_start=peak_source_frame_start,
                        peak_source_frame_end_exclusive=peak_source_frame_end_exclusive,
                        header_text=header_texts[method_name],
                        tracker_label=tracker_labels[method_name],
                        playback_label=f"1.0x | source frame {source_frame}",
                        footer_text=footer_texts[method_name],
                    )
                )
                slow_panels.append(
                    compose_panel(
                        method_name=method_name,
                        full_rgb=full_rgb,
                        inset_rgb=inset_rgb,
                        source_frame=source_frame,
                        peak_source_frame_start=peak_source_frame_start,
                        peak_source_frame_end_exclusive=peak_source_frame_end_exclusive,
                        header_text=header_texts[method_name],
                        tracker_label=tracker_labels[method_name],
                        playback_label=f"0.5x replay | source frame {source_frame}",
                        footer_text=footer_texts[method_name],
                    )
                )

            normal_frame = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            slow_frame = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            for column, panel in enumerate(normal_panels):
                normal_frame.paste(panel, (column * PANEL_WIDTH, 0))
            for column, panel in enumerate(slow_panels):
                slow_frame.paste(panel, (column * PANEL_WIDTH, 0))
            normal_rgb = np.asarray(normal_frame, dtype=np.uint8)
            if writer is None:
                writer = Mp4FrameWriter(
                    output_path=output,
                    frame_rgb=normal_rgb,
                    fps=clip.fps,
                )
            writer.append(normal_rgb)
            slow_frames.append(encode_png(slow_frame))
            print(
                f"[smpl-render] prepared {frame_index + 1}/{clip.frame_count} "
                f"(source frame {source_frame})",
                flush=True,
            )

        # 半速段只重复帧，不修改 pose，也不做可能人为降低 Jitter 的插值或平滑。
        for slow_frame in slow_frames:
            slow_rgb = decode_png(slow_frame)
            writer.append(slow_rgb)
            writer.append(slow_rgb)
    finally:
        # 两个 EGL OffscreenRenderer 共享同一 display 时，先销毁其中一个可能让
        # 另一个的 make_current 失败。视频必须先完整落盘，OpenGL 清理则采用
        # best-effort，并按最后使用的 inset context 逆序释放。
        if writer is not None:
            writer.close()
        with suppress(Exception):
            inset_renderer.delete()
        with suppress(Exception):
            full_renderer.delete()
    return output


# endregion


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    clip = load_comparison_clip(
        comparison_npz=args.comparison_npz,
        report_json=args.report_json,
        amass_npz=args.amass_npz,
        source_frame_start=args.source_frame_start,
        source_frame_end_exclusive=args.source_frame_end_exclusive,
        diffusion_variant=args.diffusion_variant,
    )
    print(
        f"[smpl-render] clip source frames "
        f"[{clip.source_frame_start},{clip.source_frame_end_exclusive}), "
        f"frames={clip.frame_count}, fps={clip.fps}, gender={clip.gender}",
        flush=True,
    )
    sequences, faces = build_mesh_sequences(
        clip=clip,
        smpl_model_dir=args.smpl_model_dir,
    )
    output = render_smpl_comparison_video(
        output_path=args.output_mp4,
        clip=clip,
        sequences=sequences,
        faces=faces,
        peak_source_frame_start=args.peak_source_frame_start,
        peak_source_frame_end_exclusive=args.peak_source_frame_end_exclusive,
    )
    print(f"[smpl-render] wrote: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
