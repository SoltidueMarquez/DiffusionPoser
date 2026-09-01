from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_geometry import (
    assemble_current_tracker_features_np,
    build_tracker_measurements_np,
)
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    SMPL_PARENTS,
    rotation_6d_to_matrix_np,
)
from data_loaders.realtime_pose_predictor_features import (
    build_predictor_step_features_np,
)
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    RIGHT_FOOT_TRACKER_INDEX,
    RIGHT_HAND_TRACKER_INDEX,
)
from diffusion.realtime_pose_inpainting import (
    build_current_realtime_pose_conditions,
)
from sample.realtime_pose_runtime import decode_and_resolve_pose
from sample.realtime_pose_smpl_rendering import (
    body_fbx_world_to_smpl_local_rotations,
    camera_pose_look_at,
    create_smplh_model,
    load_font,
    require_directory,
    require_file,
    rotation_matrices_to_axis_angle,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)
from utils.model_util import load_realtime_pose_predictor
from utils.normalizer import RealtimePoseNormalizer


DEFAULT_CURRENT_FRAME = 180
DEFAULT_OUTPUT = Path(
    "output/主方法图所需材料与参考/"
    "Predictor_IK_三链人体模块_frame180_绿色全骨架.png"
)
OUTPUT_WIDTH = 420
OUTPUT_HEIGHT = 900
BODY_PANEL_WIDTH = OUTPUT_WIDTH
BODY_PANEL_HEIGHT = OUTPUT_HEIGHT
BODY_COLOR = (0.61, 0.62, 0.63, 1.0)
PREDICTOR_COLOR = (214, 79, 90, 255)
IK_COLOR = (20, 155, 139, 255)
TRACKER_COLOR = (246, 169, 27, 255)
SKELETON_COLOR = (35, 163, 101, 255)
TEXT_COLOR = (35, 43, 55, 255)
MUTED_TEXT_COLOR = (91, 103, 120, 255)
CARD_FILL = (250, 251, 253, 255)
CARD_BORDER = (205, 212, 222, 255)
WHITE = (255, 255, 255, 255)

CHAIN_SPECS = (
    (
        "Left arm",
        (
            JOINT_INDEX["left_shoulder"],
            JOINT_INDEX["left_elbow"],
            JOINT_INDEX["left_wrist"],
        ),
        LEFT_HAND_TRACKER_INDEX,
    ),
    (
        "Right arm",
        (
            JOINT_INDEX["right_shoulder"],
            JOINT_INDEX["right_elbow"],
            JOINT_INDEX["right_wrist"],
        ),
        RIGHT_HAND_TRACKER_INDEX,
    ),
    (
        "Right leg",
        (
            JOINT_INDEX["right_hip"],
            JOINT_INDEX["right_knee"],
            JOINT_INDEX["right_ankle"],
            JOINT_INDEX["right_foot"],
        ),
        RIGHT_FOOT_TRACKER_INDEX,
    ),
)


@dataclass(frozen=True)
class PredictorIKSnapshot:
    """指定帧的 Predictor prior、IK proposal 与对应 Tracker。"""

    current_frame: int
    history_source_frames: np.ndarray  # [10]
    predictor_pose_head: np.ndarray  # [144]
    ik_pose_head: np.ndarray  # [144]
    predictor_rotations_world: np.ndarray  # [24,3,3]
    ik_rotations_world: np.ndarray  # [24,3,3]
    predictor_joints_world: np.ndarray  # [24,3]
    ik_joints_world: np.ndarray  # [24,3]
    predictor_root_yaw: float
    ik_root_yaw: float
    tracker_positions_world: np.ndarray  # [6,3]
    tracker_available: np.ndarray  # [6]
    ik_gap_deg: np.ndarray  # [24]
    ik_confidence: np.ndarray  # [24]
    denoise_strength: np.ndarray  # [24]


@dataclass(frozen=True)
class DisplayGeometry:
    """正面展示坐标中的男性 SMPL 网格、两路骨链和 Tracker。"""

    predictor_vertices: np.ndarray  # [V,3]
    faces: np.ndarray  # [F,3]
    predictor_joints: np.ndarray  # [24,3]
    ik_joints: np.ndarray  # [24,3]
    tracker_positions: np.ndarray  # [6,3]
    presentation_yaw_deg: float


@dataclass(frozen=True)
class PanelProjection:
    """中央人体正交相机的 2D 投影参数。"""

    target: np.ndarray
    xmag: float
    ymag: float
    width: int
    height: int


# region CLI 与输入


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用已有 deployed history，在一个指定帧只运行 Predictor 与 IK，"
            "生成带全身骨架的紧凑三链人体模块。"
        )
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--history_npz", required=True, type=Path)
    paths.add_argument("--history_json", required=True, type=Path)
    paths.add_argument("--source_npz", default=None, type=Path)
    paths.add_argument("--predictor_model_path", default=None, type=Path)
    paths.add_argument("--normalizer_dir", default=None, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_png", default=DEFAULT_OUTPUT, type=Path)
    frame = parser.add_argument_group("frame")
    frame.add_argument("--current_frame", default=DEFAULT_CURRENT_FRAME, type=int)
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="auto 在 CUDA 可用时使用 CUDA，否则使用 CPU。",
    )
    return parser


def load_json_object(path: Path, label: str) -> tuple[Path, dict]:
    resolved = require_file(path, label)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 顶层必须为 object：{resolved}")
    return resolved, value


def select_device(value: str) -> torch.device:
    name = str(value)
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前环境中 CUDA 不可用。")
    return torch.device(name)


def resolve_optional_path(
    explicit: Path | None,
    fallback: object,
    *,
    label: str,
    directory: bool = False,
) -> Path:
    value = explicit if explicit is not None else Path(str(fallback))
    return (
        require_directory(value, label)
        if directory
        else require_file(value, label)
    )


def build_ik_config(values: dict) -> IKInpaintingConfig:
    return IKInpaintingConfig(
        fabrik_iterations=int(values["fabrik_iterations"]),
        direction_only_quality=float(values["ik_direction_only_quality"]),
        residual_scale=float(values["ik_residual_scale"]),
        position_solved_quality=(
            None
            if values.get("ik_position_solved_quality") is None
            else float(values["ik_position_solved_quality"])
        ),
        gap_low=float(values["ik_gap_low"]),
        gap_high=float(values["ik_gap_high"]),
        direction_support=float(values.get("ik_direction_support", 0.35)),
        untracked_strength=float(values.get("ik_untracked_strength", 0.05)),
    ).validate()


# endregion


# region Predictor 与 IK 中间状态


def build_current_tracker_raw(
    *,
    source: dict[str, np.ndarray],
    current_frame: int,
    head_yaw_world: float,
    tracker_available: np.ndarray,
) -> np.ndarray:
    measurements = build_tracker_measurements_np(
        source["tracker_pos_world"][current_frame : current_frame + 1],
        source["tracker_rot_world_6d"][current_frame : current_frame + 1],
        source["tracker_pos_world"][current_frame, HEAD_TRACKER_INDEX],
        float(source["root_pos_world"][current_frame, 1]),
        float(head_yaw_world),
    )[0]
    return assemble_current_tracker_features_np(
        measurements,
        np.asarray(tracker_available, dtype=bool),
    ).astype(np.float32)


def extract_predictor_ik_snapshot(
    *,
    history_npz: Path,
    history_report: dict,
    source_npz: Path,
    predictor_model_path: Path,
    normalizer_dir: Path,
    ik_config: IKInpaintingConfig,
    current_frame: int,
    device: torch.device,
) -> tuple[PredictorIKSnapshot, dict[str, np.ndarray]]:
    """从保存的闭环历史恢复 Predictor 输入，并只执行一次 Predictor+IK。"""

    source = load_realtime_source(require_file(source_npz, "source_npz"))
    frame = int(current_frame)
    source_frame_count = int(source["tracker_pos_world"].shape[0])
    if not 12 <= frame < source_frame_count:
        raise ValueError(
            f"current_frame 必须位于 [12,{source_frame_count})，实际为 {frame}。"
        )
    report_start = int(history_report["frame_start"])
    report_end = int(history_report["frame_end_exclusive"])
    if not report_start + 10 <= frame < report_end:
        raise ValueError(
            f"history 只覆盖 [{report_start},{report_end})，无法恢复 frame {frame} 的 10 帧历史。"
        )
    history_start = frame - 10 - report_start
    history_end = frame - report_start
    current_index = frame - report_start
    history_path = require_file(history_npz, "history_npz")
    with np.load(history_path, allow_pickle=False) as payload:
        required = (
            "deployed_rotations_world",
            "deployed_root_yaw",
            "tracker_available",
        )
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"history_npz 缺少字段：{missing}")
        motion_rotations_world = np.asarray(
            payload["deployed_rotations_world"][history_start:history_end],
            dtype=np.float32,
        )
        previous_root_yaw = float(payload["deployed_root_yaw"][history_end - 1])
        recorded_tracker_available = np.asarray(
            payload["tracker_available"][current_index], dtype=bool
        )
    if motion_rotations_world.shape != (10, 24, 3, 3):
        raise ValueError(
            "恢复出的 deployed history 应为 [10,24,3,3]，"
            f"实际为 {motion_rotations_world.shape}。"
        )
    figure_tracker_available = np.asarray(
        [True, True, True, False, False, True], dtype=bool
    )
    expected_base_mask = np.asarray(
        [True, True, True, False, False], dtype=bool
    )
    if not np.array_equal(
        recorded_tracker_available[:5], expected_base_mask
    ):
        raise ValueError(
            "三链主图要求历史实验固定提供 Head + 双手且不提供双腿，"
            "实际 availability="
            f"{recorded_tracker_available.astype(int).tolist()}。"
        )
    # 模块图表达的是“右脚在当前帧重新可用”的状态。所选展开姿态位于原实验
    # 的断连区间，因此只覆盖当前帧的右脚 availability；前十帧 Predictor
    # history 完全沿用保存的 deployed 结果，不修改 Predictor 输入姿态。
    tracker_available = figure_tracker_available

    features = build_predictor_step_features_np(
        motion_rotations_world=motion_rotations_world,
        tracker_positions_world_with_previous=source["tracker_pos_world"][
            frame - 11 : frame + 1
        ],
        tracker_rotations_world_6d_with_previous=source[
            "tracker_rot_world_6d"
        ][frame - 11 : frame + 1],
        floor_y=float(source["root_pos_world"][frame, 1]),
    )
    normalizer = RealtimePoseNormalizer(normalizer_dir, disable=False)
    predictor = load_realtime_pose_predictor(
        predictor_model_path,
        device=device,
    )
    motion_normalized = np.asarray(
        normalizer.normalize_pose(features.motion_context), dtype=np.float32
    )
    sparse_normalized = np.asarray(
        normalizer.normalize_predictor_sparse(features.core_tracker_context),
        dtype=np.float32,
    )
    with torch.no_grad():
        predictor_normalized = predictor(
            torch.as_tensor(
                motion_normalized[None], device=device, dtype=torch.float32
            ),
            torch.as_tensor(
                sparse_normalized[None], device=device, dtype=torch.float32
            ),
        )[0]
        predictor_pose_horizon = np.asarray(
            normalizer.inverse_pose(predictor_normalized).detach().cpu(),
            dtype=np.float32,
        )
    if predictor_pose_horizon.shape != (11, 144):
        raise RuntimeError(
            "Predictor horizon 应为 [11,144]，"
            f"实际为 {predictor_pose_horizon.shape}。"
        )

    current_tracker_raw = build_current_tracker_raw(
        source=source,
        current_frame=frame,
        head_yaw_world=float(features.current_head_yaw_world),
        tracker_available=tracker_available,
    )
    pose_mean = normalizer.pose_mean.to(device)
    pose_scale = normalizer.pose_scale.to(device)
    tracker_mean = normalizer.tracker_mean.to(device)
    tracker_scale = (normalizer.tracker_std + normalizer.eps).to(device)
    with torch.no_grad():
        ik_result, ik_condition, _ = build_current_realtime_pose_conditions(
            initial_pose_raw=torch.as_tensor(
                predictor_pose_horizon[0:1], device=device, dtype=torch.float32
            ),
            current_tracker_raw=torch.as_tensor(
                current_tracker_raw[None], device=device, dtype=torch.float32
            ),
            joint_offsets_parent=torch.as_tensor(
                source["joint_offsets_parent"][None],
                device=device,
                dtype=torch.float32,
            ),
            pose_mean=pose_mean,
            pose_scale=pose_scale,
            tracker_mean=tracker_mean,
            tracker_scale=tracker_scale,
            config=ik_config,
        )
    predictor_pose = predictor_pose_horizon[0].copy()
    ik_pose = np.asarray(
        ik_result.pose[0].reshape(144).detach().cpu(), dtype=np.float32
    )
    predictor_resolved = decode_and_resolve_pose(
        predictor_pose,
        current_tracker_raw,
        float(features.current_head_yaw_world),
        source["tracker_pos_world"][frame, HEAD_TRACKER_INDEX],
        float(source["root_pos_world"][frame, 1]),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
        previous_root_yaw_world=previous_root_yaw,
    )
    ik_resolved = decode_and_resolve_pose(
        ik_pose,
        current_tracker_raw,
        float(features.current_head_yaw_world),
        source["tracker_pos_world"][frame, HEAD_TRACKER_INDEX],
        float(source["root_pos_world"][frame, 1]),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
        previous_root_yaw_world=previous_root_yaw,
    )
    snapshot = PredictorIKSnapshot(
        current_frame=frame,
        history_source_frames=np.arange(frame - 10, frame, dtype=np.int64),
        predictor_pose_head=predictor_pose,
        ik_pose_head=ik_pose,
        predictor_rotations_world=predictor_resolved.joint_rotations_world,
        ik_rotations_world=ik_resolved.joint_rotations_world,
        predictor_joints_world=predictor_resolved.joints_world,
        ik_joints_world=ik_resolved.joints_world,
        predictor_root_yaw=float(predictor_resolved.root_yaw_world),
        ik_root_yaw=float(ik_resolved.root_yaw_world),
        tracker_positions_world=np.asarray(
            source["tracker_pos_world"][frame], dtype=np.float32
        ),
        tracker_available=tracker_available,
        ik_gap_deg=np.degrees(
            np.asarray(ik_condition.ik_gap[0].detach().cpu(), dtype=np.float32)
        ),
        ik_confidence=np.asarray(
            ik_condition.ik_confidence[0].detach().cpu(), dtype=np.float32
        ),
        denoise_strength=np.asarray(
            ik_condition.denoise_strength[0].detach().cpu(), dtype=np.float32
        ),
    )
    return snapshot, source


# endregion


# region 男性 SMPL 与正面展示坐标


def build_front_rotation(
    predictor_rotations_world: np.ndarray,
) -> tuple[np.ndarray, float]:
    """把 Predictor 的胸腹面旋转到 +Z，相机随后从 +Z 正面观察。"""

    rotations = np.asarray(predictor_rotations_world, dtype=np.float64)
    if rotations.shape != (24, 3, 3):
        raise ValueError(
            f"predictor_rotations_world 应为 [24,3,3]，实际为 {rotations.shape}。"
        )
    torso_front = rotations[JOINT_INDEX["spine3"], :, 2]
    horizontal = torso_front[[0, 2]]
    if float(np.linalg.norm(horizontal)) <= 1e-8:
        raise ValueError("Spine3 正面方向水平分量为零，无法确定正面视角。")
    current_angle = math.atan2(float(horizontal[1]), float(horizontal[0]))
    presentation_yaw = math.pi * 0.5 - current_angle
    cos_yaw = math.cos(presentation_yaw)
    sin_yaw = math.sin(presentation_yaw)
    rotation = np.asarray(
        [
            [cos_yaw, 0.0, -sin_yaw],
            [0.0, 1.0, 0.0],
            [sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float64,
    )
    return rotation, math.degrees(presentation_yaw)


def build_display_geometry(
    *,
    snapshot: PredictorIKSnapshot,
    source: dict[str, np.ndarray],
    smpl_model_dir: Path,
) -> DisplayGeometry:
    """将两路旋转转移到标准男性 SMPL-H，并保留真实骨链/Tracker 差值。"""

    rest_rotations = rotation_6d_to_matrix_np(
        source["joint_rest_local_rotations_6d"]
    )
    rotations_world = np.stack(
        [
            snapshot.predictor_rotations_world,
            snapshot.ik_rotations_world,
        ],
        axis=0,
    )
    root_yaw = np.asarray(
        [snapshot.predictor_root_yaw, snapshot.ik_root_yaw], dtype=np.float32
    )
    local_rotations = body_fbx_world_to_smpl_local_rotations(
        rotations_world,
        root_yaw,
        rest_rotations,
        SMPL_PARENTS,
    )
    pose_axis_angle = rotation_matrices_to_axis_angle(
        local_rotations[:, :22]
    )
    model = create_smplh_model(
        model_dir=require_directory(smpl_model_dir, "smpl_model_dir"),
        gender="male",
        batch_size=2,
    )
    male_sequence = run_smplh_forward(
        model=model,
        pose_axis_angle=pose_axis_angle,
        betas=np.zeros((10,), dtype=np.float32),
        translation_amass=np.zeros((2, 3), dtype=np.float32),
    )

    # 模块图要把完整骨架准确叠到男性网格上，因此可视化坐标直接使用同一次
    # SMPL-H forward 得到的前 24 个 body joints。链级端点误差仍在 sidecar
    # 中使用 runtime 原始关节计算，避免改变定量结果的定义。
    predictor_body_joints = np.asarray(
        male_sequence.joints_world[0], dtype=np.float64
    )
    ik_body_joints = np.asarray(
        male_sequence.joints_world[1], dtype=np.float64
    )
    if predictor_body_joints.shape != (22, 3):
        raise RuntimeError(
            "SMPL-H body joints 应为 [22,3]，"
            f"实际为 {predictor_body_joints.shape}。"
        )

    # SMPL-H 渲染辅助只导出到左右 wrist 的 22 个 body joints，而 runtime
    # 骨架还包含两个 hand 末端。沿用 runtime 的 wrist→hand 世界向量补齐它们，
    # 可以完整展示 24 关节，同时不会让骨架主体偏离 SMPL 网格。
    hand_pairs = (
        (JOINT_INDEX["left_wrist"], JOINT_INDEX["left_hand"]),
        (JOINT_INDEX["right_wrist"], JOINT_INDEX["right_hand"]),
    )

    def append_hand_joints(
        body_joints: np.ndarray,
        runtime_joints: np.ndarray,
    ) -> np.ndarray:
        hand_joints = []
        for wrist_index, hand_index in hand_pairs:
            wrist_to_hand = (
                runtime_joints[hand_index] - runtime_joints[wrist_index]
            )
            hand_joints.append(body_joints[wrist_index] + wrist_to_hand)
        return np.concatenate(
            [body_joints, np.stack(hand_joints, axis=0)], axis=0
        )

    predictor_joints = append_hand_joints(
        predictor_body_joints,
        np.asarray(snapshot.predictor_joints_world, dtype=np.float64),
    )
    ik_joints = append_hand_joints(
        ik_body_joints,
        np.asarray(snapshot.ik_joints_world, dtype=np.float64),
    )
    pelvis_index = JOINT_INDEX["pelvis"]
    runtime_to_male = (
        predictor_joints[pelvis_index]
        - np.asarray(
            snapshot.predictor_joints_world[pelvis_index], dtype=np.float64
        )
    )
    tracker_positions = (
        np.asarray(snapshot.tracker_positions_world, dtype=np.float64)
        + runtime_to_male
    )
    predictor_vertices = np.asarray(
        male_sequence.vertices_world[0], dtype=np.float64
    )
    front_rotation, presentation_yaw_deg = build_front_rotation(
        snapshot.predictor_rotations_world
    )
    predictor_vertices = predictor_vertices @ front_rotation.T
    predictor_joints = predictor_joints @ front_rotation.T
    ik_joints = ik_joints @ front_rotation.T
    tracker_positions = tracker_positions @ front_rotation.T

    pelvis = predictor_joints[pelvis_index].copy()
    floor_y = float(np.min(predictor_vertices[:, 1]))
    display_offset = np.asarray(
        [-pelvis[0], -floor_y, -pelvis[2]], dtype=np.float64
    )
    return DisplayGeometry(
        predictor_vertices=(predictor_vertices + display_offset).astype(
            np.float32
        ),
        faces=transform_faces_to_unity_winding(model.faces),
        predictor_joints=(predictor_joints + display_offset).astype(np.float32),
        ik_joints=(ik_joints + display_offset).astype(np.float32),
        tracker_positions=(tracker_positions + display_offset).astype(np.float32),
        presentation_yaw_deg=float(presentation_yaw_deg),
    )


# endregion


# region 中央人体渲染


def render_predictor_body_panel(
    geometry: DisplayGeometry,
) -> tuple[Image.Image, PanelProjection]:
    try:
        import pyrender
        import trimesh
    except ImportError as exc:
        raise ImportError("缺少 pyrender/trimesh，无法渲染 Predictor 人体。") from exc

    vertices = np.asarray(geometry.predictor_vertices, dtype=np.float64)
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    target = 0.5 * (mins + maxs)
    content_width = float(maxs[0] - mins[0])
    content_height = float(maxs[1] - mins[1])
    aspect = float(BODY_PANEL_WIDTH) / float(BODY_PANEL_HEIGHT)
    ymag = max(content_height * 0.5, content_width * 0.5 / aspect) * 1.08
    xmag = ymag * aspect
    camera_pose = camera_pose_look_at(
        target + np.asarray([0.0, 0.0, 6.0]), target
    )
    scene = pyrender.Scene(
        bg_color=np.asarray([1.0, 1.0, 1.0, 1.0]),
        ambient_light=np.asarray([0.48, 0.48, 0.48]),
    )
    scene.add(
        pyrender.OrthographicCamera(
            xmag=float(xmag),
            ymag=float(ymag),
            znear=0.05,
            zfar=20.0,
        ),
        pose=camera_pose,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.9),
        pose=camera_pose_look_at(
            target + np.asarray([-2.4, 3.2, 4.4]), target
        ),
    )
    scene.add(
        pyrender.DirectionalLight(
            color=np.asarray([0.88, 0.92, 1.0]), intensity=0.7
        ),
        pose=camera_pose_look_at(
            target + np.asarray([2.8, 1.7, 3.8]), target
        ),
    )
    body = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(geometry.faces, dtype=np.int64),
        process=False,
    )
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=BODY_COLOR,
        metallicFactor=0.0,
        roughnessFactor=0.88,
    )
    scene.add(
        pyrender.Mesh.from_trimesh(body, material=material, smooth=True)
    )
    renderer = pyrender.OffscreenRenderer(BODY_PANEL_WIDTH, BODY_PANEL_HEIGHT)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
    finally:
        with suppress(Exception):
            renderer.delete()
    projection = PanelProjection(
        target=np.asarray(target, dtype=np.float64),
        xmag=float(xmag),
        ymag=float(ymag),
        width=BODY_PANEL_WIDTH,
        height=BODY_PANEL_HEIGHT,
    )
    return (
        Image.fromarray(np.asarray(color[..., :3], dtype=np.uint8)).convert(
            "RGBA"
        ),
        projection,
    )


def project_front_points(
    points: np.ndarray,
    projection: PanelProjection,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x = (
        (values[:, 0] - projection.target[0]) / projection.xmag
        * projection.width
        * 0.5
        + projection.width * 0.5
    )
    y = (
        projection.height * 0.5
        - (values[:, 1] - projection.target[1])
        / projection.ymag
        * projection.height
        * 0.5
    )
    return np.stack([x, y], axis=-1).astype(np.float32)


# endregion


# region 2D 链条绘制


def draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: float = 18.0,
    gap: float = 10.0,
) -> None:
    values = np.asarray(points, dtype=np.float64)
    for start, end in zip(values[:-1], values[1:]):
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length <= 1e-8:
            continue
        direction = vector / length
        cursor = 0.0
        while cursor < length:
            segment_end = min(cursor + float(dash), length)
            point_start = start + direction * cursor
            point_end = start + direction * segment_end
            draw.line(
                (*point_start.tolist(), *point_end.tolist()),
                fill=fill,
                width=int(width),
            )
            cursor += float(dash) + float(gap)


def draw_joint_nodes(
    draw: ImageDraw.ImageDraw,
    points: np.ndarray,
    *,
    fill: tuple[int, int, int, int],
    hollow: bool,
    radius: int,
) -> None:
    for point in np.asarray(points, dtype=np.float64):
        box = (
            int(round(point[0] - radius)),
            int(round(point[1] - radius)),
            int(round(point[0] + radius)),
            int(round(point[1] + radius)),
        )
        draw.ellipse(box, fill=WHITE, outline=fill, width=3)
        if not hollow:
            inner = max(2, radius - 4)
            draw.ellipse(
                (
                    int(round(point[0] - inner)),
                    int(round(point[1] - inner)),
                    int(round(point[0] + inner)),
                    int(round(point[1] + inner)),
                ),
                fill=fill,
            )


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: np.ndarray,
    end: np.ndarray,
    *,
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    point_start = np.asarray(start, dtype=np.float64)
    point_end = np.asarray(end, dtype=np.float64)
    vector = point_end - point_start
    length = float(np.linalg.norm(vector))
    if length < 8.0:
        return
    direction = vector / length
    normal = np.asarray([-direction[1], direction[0]])
    draw.line(
        (*point_start.tolist(), *point_end.tolist()),
        fill=fill,
        width=int(width),
    )
    head_base = point_end - direction * 13.0
    polygon = [
        tuple(point_end.tolist()),
        tuple((head_base + normal * 6.0).tolist()),
        tuple((head_base - normal * 6.0).tolist()),
    ]
    draw.polygon(polygon, fill=fill)


def draw_tracker_point(
    draw: ImageDraw.ImageDraw,
    point: np.ndarray,
    *,
    radius: int,
) -> None:
    value = np.asarray(point, dtype=np.float64)
    draw.ellipse(
        (
            int(round(value[0] - radius - 3)),
            int(round(value[1] - radius - 3)),
            int(round(value[0] + radius + 3)),
            int(round(value[1] + radius + 3)),
        ),
        fill=WHITE,
    )
    draw.ellipse(
        (
            int(round(value[0] - radius)),
            int(round(value[1] - radius)),
            int(round(value[0] + radius)),
            int(round(value[1] + radius)),
        ),
        fill=TRACKER_COLOR,
        outline=(206, 132, 10, 255),
        width=2,
    )


def draw_body_chain_overlay(
    panel: Image.Image,
    *,
    geometry: DisplayGeometry,
    projection: PanelProjection,
) -> Image.Image:
    """在人体网格上绘制完整 Predictor 骨架和三条 IK 对比链。"""

    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 绿色完整骨架表示 IK proposal。它补全躯干、头、左腿及手部末端，
    # 并与人体网格所表示的 Predictor output 形成直接空间对照。
    ik_all = project_front_points(geometry.ik_joints, projection)
    body_edges = [
        (int(parent), child)
        for child, parent in enumerate(SMPL_PARENTS[:24])
        if int(parent) >= 0
    ]
    for parent, child in body_edges:
        segment = [
            tuple(ik_all[parent]),
            tuple(ik_all[child]),
        ]
        draw.line(segment, fill=WHITE, width=11)
        draw.line(segment, fill=SKELETON_COLOR, width=6)
    draw_joint_nodes(
        draw,
        ik_all,
        fill=SKELETON_COLOR,
        hollow=False,
        radius=6,
    )

    # 三条关键链额外叠加红色 Predictor 虚线。绿色底线已经是完整 IK
    # 骨架，因此不再重复画第二层 IK 线，模块在缩小后仍保持简洁。
    for _, chain, _tracker_index in CHAIN_SPECS:
        indices = np.asarray(chain, dtype=np.int64)
        predictor = project_front_points(
            geometry.predictor_joints[indices], projection
        )
        draw_dashed_polyline(
            draw,
            predictor,
            fill=WHITE,
            width=10,
            dash=17.0,
            gap=10.0,
        )
        draw_dashed_polyline(
            draw,
            predictor,
            fill=PREDICTOR_COLOR,
            width=6,
            dash=17.0,
            gap=10.0,
        )
        draw_joint_nodes(
            draw,
            predictor,
            fill=PREDICTOR_COLOR,
            hollow=True,
            radius=9,
        )
    return Image.alpha_composite(panel, overlay)


def fit_chain_to_card(
    *,
    predictor_points: np.ndarray,
    ik_points: np.ndarray,
    tracker_point: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictor = np.asarray(predictor_points, dtype=np.float64)
    ik = np.asarray(ik_points, dtype=np.float64)
    tracker = np.asarray(tracker_point, dtype=np.float64)
    root = predictor[0].copy()
    predictor_2d = np.stack(
        [predictor[:, 0] - root[0], -(predictor[:, 1] - root[1])], axis=-1
    )
    ik_2d = np.stack(
        [ik[:, 0] - root[0], -(ik[:, 1] - root[1])], axis=-1
    )
    tracker_2d = np.asarray(
        [tracker[0] - root[0], -(tracker[1] - root[1])], dtype=np.float64
    )
    combined = np.concatenate(
        [predictor_2d, ik_2d, tracker_2d[None]], axis=0
    )
    mins = np.min(combined, axis=0)
    maxs = np.max(combined, axis=0)
    span = np.maximum(maxs - mins, 1e-4)
    inner_left = box[0] + 52
    inner_top = box[1] + 86
    inner_right = box[2] - 52
    inner_bottom = box[3] - 94
    scale = min(
        float(inner_right - inner_left) / float(span[0]),
        float(inner_bottom - inner_top) / float(span[1]),
    )
    content_center = 0.5 * (mins + maxs)
    card_center = np.asarray(
        [
            0.5 * (inner_left + inner_right),
            0.5 * (inner_top + inner_bottom),
        ],
        dtype=np.float64,
    )

    def transform(points: np.ndarray) -> np.ndarray:
        return (points - content_center[None]) * scale + card_center[None]

    return (
        transform(predictor_2d),
        transform(ik_2d),
        transform(tracker_2d[None])[0],
    )


def endpoint_error_cm(
    joints_world: np.ndarray,
    tracker_positions_world: np.ndarray,
    *,
    joint_index: int,
    tracker_index: int,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(joints_world[joint_index], dtype=np.float64)
            - np.asarray(
                tracker_positions_world[tracker_index], dtype=np.float64
            )
        )
        * 100.0
    )


def summarize_chain_metrics(
    snapshot: PredictorIKSnapshot,
) -> dict[str, dict[str, float]]:
    """计算三条链的定量结果，成图本身保持无文字、无图例。"""

    metrics: dict[str, dict[str, float]] = {}
    for name, chain, tracker_index in CHAIN_SPECS:
        indices = np.asarray(chain, dtype=np.int64)
        gaps = np.asarray(snapshot.ik_gap_deg[indices], dtype=np.float64)
        metrics[name] = {
            "max_ik_gap_deg": float(np.max(gaps)),
            "mean_ik_gap_deg": float(np.mean(gaps)),
            "predictor_endpoint_error_cm": endpoint_error_cm(
                snapshot.predictor_joints_world,
                snapshot.tracker_positions_world,
                joint_index=int(chain[-1]),
                tracker_index=tracker_index,
            ),
            "ik_endpoint_error_cm": endpoint_error_cm(
                snapshot.ik_joints_world,
                snapshot.tracker_positions_world,
                joint_index=int(chain[-1]),
                tracker_index=tracker_index,
            ),
            "mean_ik_confidence": float(
                np.mean(snapshot.ik_confidence[indices])
            ),
        }
    return metrics


def draw_chain_card(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    name: str,
    chain: tuple[int, ...],
    tracker_index: int,
    snapshot: PredictorIKSnapshot,
    geometry: DisplayGeometry,
) -> dict[str, float]:
    draw.rounded_rectangle(
        box,
        radius=28,
        fill=CARD_FILL,
        outline=CARD_BORDER,
        width=3,
    )
    indices = np.asarray(chain, dtype=np.int64)
    predictor, ik, tracker = fit_chain_to_card(
        predictor_points=geometry.predictor_joints[indices],
        ik_points=geometry.ik_joints[indices],
        tracker_point=geometry.tracker_positions[tracker_index],
        box=box,
    )
    draw.text(
        (box[0] + 28, box[1] + 18),
        name,
        font=load_font(31),
        fill=TEXT_COLOR,
    )
    draw_dashed_polyline(
        draw,
        predictor,
        fill=PREDICTOR_COLOR,
        width=7,
        dash=18.0,
        gap=10.0,
    )
    draw.line(
        [tuple(point) for point in ik],
        fill=IK_COLOR,
        width=9,
        joint="curve",
    )
    draw_joint_nodes(
        draw,
        predictor,
        fill=PREDICTOR_COLOR,
        hollow=True,
        radius=10,
    )
    draw_joint_nodes(
        draw,
        ik,
        fill=IK_COLOR,
        hollow=False,
        radius=10,
    )
    draw_tracker_point(draw, tracker, radius=12)
    draw_arrow(
        draw,
        predictor[-1],
        tracker,
        fill=(133, 144, 160, 255),
        width=3,
    )

    gaps = np.asarray(snapshot.ik_gap_deg[indices], dtype=np.float64)
    for joint_slot, (point, gap) in enumerate(zip(ik, gaps)):
        label_y_offset = -24 if joint_slot % 2 == 0 else 5
        draw.text(
            (int(point[0] + 12), int(point[1] + label_y_offset)),
            f"{gap:.1f}°",
            font=load_font(16),
            fill=MUTED_TEXT_COLOR,
        )
    endpoint_joint = int(chain[-1])
    predictor_error = endpoint_error_cm(
        snapshot.predictor_joints_world,
        snapshot.tracker_positions_world,
        joint_index=endpoint_joint,
        tracker_index=tracker_index,
    )
    ik_error = endpoint_error_cm(
        snapshot.ik_joints_world,
        snapshot.tracker_positions_world,
        joint_index=endpoint_joint,
        tracker_index=tracker_index,
    )
    footer = (
        f"max ΔR {float(np.max(gaps)):.1f}°   "
        f"endpoint {predictor_error:.1f} → {ik_error:.1f} cm"
    )
    draw.text(
        (box[0] + 28, box[3] - 56),
        footer,
        font=load_font(20),
        fill=TEXT_COLOR,
    )
    return {
        "max_ik_gap_deg": float(np.max(gaps)),
        "mean_ik_gap_deg": float(np.mean(gaps)),
        "predictor_endpoint_error_cm": predictor_error,
        "ik_endpoint_error_cm": ik_error,
        "mean_ik_confidence": float(
            np.mean(snapshot.ik_confidence[indices])
        ),
    }


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    x = 1710
    y = 1265
    draw.rounded_rectangle(
        (x, y, 2380, y + 150),
        radius=24,
        fill=(255, 255, 255, 245),
        outline=CARD_BORDER,
        width=2,
    )
    entries = (
        (PREDICTOR_COLOR, "Predictor chain", "dashed"),
        (IK_COLOR, "IK proposal", "solid"),
        (TRACKER_COLOR, "Tracker target", "point"),
    )
    for index, (color, label, style) in enumerate(entries):
        entry_x = x + 28 + index * 210
        center_y = y + 54
        if style == "dashed":
            draw_dashed_polyline(
                draw,
                np.asarray(
                    [[entry_x, center_y], [entry_x + 54, center_y]],
                    dtype=np.float32,
                ),
                fill=color,
                width=6,
                dash=13.0,
                gap=7.0,
            )
        elif style == "solid":
            draw.line(
                (entry_x, center_y, entry_x + 54, center_y),
                fill=color,
                width=7,
            )
        else:
            draw.ellipse(
                (entry_x + 17, center_y - 10, entry_x + 37, center_y + 10),
                fill=color,
            )
        draw.text(
            (entry_x, y + 84),
            label,
            font=load_font(17),
            fill=TEXT_COLOR,
        )


# endregion


# region 成图与 sidecar


def compose_comparison_image(
    *,
    snapshot: PredictorIKSnapshot,
    geometry: DisplayGeometry,
) -> tuple[Image.Image, dict[str, dict[str, float]]]:
    body_panel, projection = render_predictor_body_panel(geometry)
    body_panel = draw_body_chain_overlay(
        body_panel,
        geometry=geometry,
        projection=projection,
    )
    return body_panel.convert("RGB"), summarize_chain_metrics(snapshot)


def validate_output_image(image: Image.Image) -> None:
    if image.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        raise RuntimeError(
            f"输出尺寸错误：{image.size}，期望 {(OUTPUT_WIDTH, OUTPUT_HEIGHT)}。"
        )
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    corners = rgb[[0, 0, -1, -1], [0, -1, 0, -1]]
    if not np.all(corners == 255):
        raise RuntimeError("成图四角不是纯白背景。")
    if int(np.any(rgb < 248, axis=-1).sum()) < 10000:
        raise RuntimeError("成图前景像素过少，可能渲染失败。")


def write_sidecars(
    *,
    output_png: Path,
    snapshot: PredictorIKSnapshot,
    geometry: DisplayGeometry,
    chain_metrics: dict[str, dict[str, float]],
    history_npz: Path,
    source_npz: Path,
    predictor_model_path: Path,
    normalizer_dir: Path,
) -> tuple[Path, Path]:
    output = Path(output_png).expanduser().resolve()
    output_npz = output.with_suffix(".npz")
    output_json = output.with_suffix(".json")
    np.savez_compressed(
        output_npz,
        current_frame=np.asarray(snapshot.current_frame, dtype=np.int32),
        history_source_frames=snapshot.history_source_frames,
        predictor_pose_head=snapshot.predictor_pose_head,
        ik_pose_head=snapshot.ik_pose_head,
        predictor_rotations_world=snapshot.predictor_rotations_world,
        ik_rotations_world=snapshot.ik_rotations_world,
        predictor_joints_world=snapshot.predictor_joints_world,
        ik_joints_world=snapshot.ik_joints_world,
        tracker_positions_world=snapshot.tracker_positions_world,
        tracker_available=snapshot.tracker_available,
        ik_gap_deg=snapshot.ik_gap_deg,
        ik_confidence=snapshot.ik_confidence,
        denoise_strength=snapshot.denoise_strength,
    )
    report = {
        "experiment": "predictor_ik_three_chain_human_module",
        "current_frame": int(snapshot.current_frame),
        "history_source_frames": snapshot.history_source_frames.astype(
            int
        ).tolist(),
        "tracker_configuration": [
            "head",
            "left_wrist",
            "right_wrist",
            "right_foot",
        ],
        "tracker_configuration_note": (
            "右脚仅在 current_frame 设为可用；十帧 deployed history 沿用"
            "原实验的 Head + 双手配置。"
        ),
        "chains": chain_metrics,
        "render": {
            "body_model": "SMPL-H male",
            "betas": "zeros(10)",
            "projection": "orthographic",
            "view": "front",
            "presentation_yaw_deg": float(geometry.presentation_yaw_deg),
            "resolution": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
            "background_rgb": [255, 255, 255],
            "layout": "human_only_no_legend",
            "skeleton_overlay": (
                "full green IK skeleton + three red Predictor chains"
            ),
        },
        "inputs": {
            "history_npz": str(Path(history_npz).resolve()),
            "source_npz": str(Path(source_npz).resolve()),
            "predictor_model_path": str(Path(predictor_model_path).resolve()),
            "normalizer_dir": str(Path(normalizer_dir).resolve()),
        },
        "outputs": {
            "png": str(output),
            "npz": str(output_npz),
        },
    }
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_npz, output_json


def main(argv: list[str] | None = None) -> tuple[Path, Path, Path]:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    history_json_path, history_report = load_json_object(
        args.history_json, "history_json"
    )
    source_path = resolve_optional_path(
        args.source_npz,
        history_report.get("source_path", ""),
        label="source_npz",
    )
    predictor_path = resolve_optional_path(
        args.predictor_model_path,
        history_report.get("predictor_model_path", ""),
        label="predictor_model_path",
    )
    dit_model_path = require_file(
        Path(str(history_report.get("dit_model_path", ""))),
        "dit_model_path",
    )
    dit_args_path, dit_args = load_json_object(
        dit_model_path.with_name("args.json"), "DiT args.json"
    )
    normalizer_dir = resolve_optional_path(
        args.normalizer_dir,
        dit_args.get("normalizer_dir", ""),
        label="normalizer_dir",
        directory=True,
    )
    device = select_device(args.device)
    snapshot, source = extract_predictor_ik_snapshot(
        history_npz=args.history_npz,
        history_report=history_report,
        source_npz=source_path,
        predictor_model_path=predictor_path,
        normalizer_dir=normalizer_dir,
        ik_config=build_ik_config(dit_args),
        current_frame=int(args.current_frame),
        device=device,
    )
    geometry = build_display_geometry(
        snapshot=snapshot,
        source=source,
        smpl_model_dir=args.smpl_model_dir,
    )
    image, chain_metrics = compose_comparison_image(
        snapshot=snapshot,
        geometry=geometry,
    )
    validate_output_image(image)
    output_png = Path(args.output_png).expanduser().resolve()
    if output_png.suffix.lower() != ".png":
        raise ValueError(f"output_png 必须使用 .png 后缀：{output_png}")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    output_npz, output_json = write_sidecars(
        output_png=output_png,
        snapshot=snapshot,
        geometry=geometry,
        chain_metrics=chain_metrics,
        history_npz=args.history_npz,
        source_npz=source_path,
        predictor_model_path=predictor_path,
        normalizer_dir=normalizer_dir,
    )
    print(f"[predictor-ik-chains] device: {device}", flush=True)
    print(
        f"[predictor-ik-chains] history: "
        f"{snapshot.history_source_frames.tolist()} -> {snapshot.current_frame}",
        flush=True,
    )
    for name, metrics in chain_metrics.items():
        print(f"[predictor-ik-chains] {name}: {metrics}", flush=True)
    print(f"[predictor-ik-chains] wrote: {output_png}", flush=True)
    print(f"[predictor-ik-chains] wrote: {output_npz}", flush=True)
    print(f"[predictor-ik-chains] wrote: {output_json}", flush=True)
    del history_json_path, dit_args_path
    return output_png, output_npz, output_json


# endregion


if __name__ == "__main__":
    main()
