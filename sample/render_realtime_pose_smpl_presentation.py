from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from eval.realtime_pose_metrics import compute_rpm_p2_mc_metrics
from sample.render_realtime_pose_comparison import Mp4FrameWriter
from sample.render_realtime_pose_smpl_comparison import (
    CameraSpec,
    METHOD_ORDER,
    SmplMeshSequence,
    build_horizontal_pelvis_follow_offsets,
    build_mesh_sequences,
    camera_pose_look_at,
    create_front_marker_mesh,
    create_sphere_cloud,
    create_static_scene,
    decode_png,
    encode_png,
    load_comparison_clip,
    load_font,
    normalize_vector,
)


OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
INTRO_FRAME_COUNT = 21
CORE_TRACKER_COUNT = 3
METHOD_GAP_METERS = 0.25
CAMERA_FIT_PADDING = 1.16
CAMERA_ELEVATION_DEG = 14.0
CAMERA_AZIMUTH_DEG = -72.0
CAMERA_YFOV = math.radians(34.0)

METHOD_COLORS = {
    "GT": (0x90 / 255.0, 0xA9 / 255.0, 0xC2 / 255.0, 1.0),
    "Predictor-only": (0xC8 / 255.0, 0x92 / 255.0, 0x92 / 255.0, 1.0),
    "+ Diffusion": (0x59 / 255.0, 0xB9 / 255.0, 0xB7 / 255.0, 1.0),
}
METHOD_LABELS = {
    "GT": "GT",
    "Predictor-only": "Predictor-only (RPM backbone)",
    "+ Diffusion": "+ Diffusion (same 3 trackers)",
}
TRACKER_COLOR = (1.0, 0.55, 0.05, 1.0)
CHEST_MARKER_COLOR = (1.0, 0.88, 0.18, 1.0)
LEFT_FOOT_COLOR = (0.05, 0.82, 0.92, 1.0)
RIGHT_FOOT_COLOR = (0.86, 0.24, 0.78, 1.0)


@dataclass(frozen=True)
class PresentationLayout:
    """共享舞台的固定几何关系；所有数组均使用项目 Unity/y-up 坐标。"""

    method_offsets: np.ndarray  # [3,3]，仅用于展示，不参与指标
    follow_offsets: np.ndarray  # [T,3]，只包含 GT pelvis 的 XZ 平移
    base_camera: CameraSpec
    camera_poses: np.ndarray  # [T,4,4]，旋转、距离与高度恒定
    grid_center: np.ndarray
    grid_size: float
    stage_spacing: float


@dataclass(frozen=True)
class PresentationFrame:
    """最终 MP4 中一帧对应的 source frame 和展示状态。"""

    source_frame_index: int
    playback_label: str
    show_trackers: bool


@dataclass(frozen=True)
class PresentationMetrics:
    predictor_mpjve: float
    diffusion_mpjve: float
    predictor_jitter: float
    diffusion_jitter: float
    mpjve_gain_percent: float
    jitter_gain_percent: float


# region CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a StableMotion-style shared-stage SMPL-H comparison for "
            "GT, Predictor-only and same-three-tracker Diffusion."
        )
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--comparison_npz", required=True, type=Path)
    paths.add_argument("--report_json", required=True, type=Path)
    paths.add_argument("--amass_npz", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_mp4", required=True, type=Path)
    clip = parser.add_argument_group("clip")
    clip.add_argument("--source_frame_start", required=True, type=int)
    clip.add_argument("--source_frame_end_exclusive", required=True, type=int)
    return parser


# endregion


# region 共享舞台几何与相机


def presentation_view_direction_unity() -> np.ndarray:
    """把已验收的 Matplotlib 观察角转换为项目 Unity/y-up 眼睛方向。"""

    elevation = math.radians(CAMERA_ELEVATION_DEG)
    azimuth = math.radians(CAMERA_AZIMUTH_DEG)
    direction = np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.sin(azimuth),
        ],
        dtype=np.float64,
    )
    return normalize_vector(direction)


def validate_mesh_sequences(
    sequences: dict[str, SmplMeshSequence],
) -> tuple[int, int]:
    """校验三路网格契约，并返回 ``(frame_count, vertex_count)``。"""

    if tuple(sequences.keys()) != METHOD_ORDER:
        raise ValueError(
            f"sequences 必须按 {METHOD_ORDER} 排列，实际为 {tuple(sequences.keys())}"
        )
    reference = sequences["GT"]
    vertices = np.asarray(reference.vertices_world)
    joints = np.asarray(reference.joints_world)
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError(f"GT vertices_world 应为 [T,V,3]，实际为 {vertices.shape}")
    if joints.ndim != 3 or joints.shape[0] != vertices.shape[0] or joints.shape[-1] != 3:
        raise ValueError(f"GT joints_world 应为 [T,J,3]，实际为 {joints.shape}")
    expected_vertices = vertices.shape
    expected_joints = joints.shape
    for method_name in METHOD_ORDER:
        sequence = sequences[method_name]
        method_vertices = np.asarray(sequence.vertices_world)
        method_joints = np.asarray(sequence.joints_world)
        if method_vertices.shape != expected_vertices:
            raise ValueError(
                f"{method_name} vertices_world 应为 {expected_vertices}，"
                f"实际为 {method_vertices.shape}"
            )
        if method_joints.shape != expected_joints:
            raise ValueError(
                f"{method_name} joints_world 应为 {expected_joints}，"
                f"实际为 {method_joints.shape}"
            )
        if not (np.isfinite(method_vertices).all() and np.isfinite(method_joints).all()):
            raise ValueError(f"{method_name} 网格或关节含 NaN/Inf。")
    return int(vertices.shape[0]), int(vertices.shape[1])


def build_presentation_frame_schedule(
    frame_count: int,
    intro_frame_count: int = INTRO_FRAME_COUNT,
) -> tuple[PresentationFrame, ...]:
    """生成冻结开场、1× 正播和逐帧重复的 0.5× 重放时间表。"""

    count = int(frame_count)
    intro_count = int(intro_frame_count)
    if count <= 0:
        raise ValueError("frame_count 必须为正数。")
    if intro_count < 0:
        raise ValueError("intro_frame_count 不能为负数。")
    frames: list[PresentationFrame] = []
    frames.extend(
        PresentationFrame(0, "Input setup", True) for _ in range(intro_count)
    )
    frames.extend(
        PresentationFrame(frame_index, "1.0×", False)
        for frame_index in range(count)
    )
    for frame_index in range(count):
        frames.extend(
            (
                PresentationFrame(frame_index, "0.5× replay", False),
                PresentationFrame(frame_index, "0.5× replay", False),
            )
        )
    return tuple(frames)


def build_intro_tracker_points(
    core_tracker_frame: np.ndarray,
    method_offsets: np.ndarray,
) -> np.ndarray:
    """复制同一组三点到 Predictor 与 Diffusion 的展示位置，返回 ``[2,3,3]``。"""

    core_trackers = np.asarray(core_tracker_frame, dtype=np.float64)
    offsets = np.asarray(method_offsets, dtype=np.float64)
    if core_trackers.shape != (CORE_TRACKER_COUNT, 3):
        raise ValueError(
            f"core_tracker_frame 应为 {(CORE_TRACKER_COUNT, 3)}，"
            f"实际为 {core_trackers.shape}"
        )
    if offsets.shape != (len(METHOD_ORDER), 3):
        raise ValueError(f"method_offsets 应为 [3,3]，实际为 {offsets.shape}")
    return np.stack(
        [
            core_trackers + offsets[1],
            core_trackers + offsets[2],
        ],
        axis=0,
    ).astype(np.float32)


def build_stage_method_offsets(
    *,
    sequences: dict[str, SmplMeshSequence],
    follow_offsets: np.ndarray,
    camera_right: np.ndarray,
    method_gap: float = METHOD_GAP_METERS,
) -> tuple[np.ndarray, float]:
    """沿相机右轴横向排列三路，并保证整段投影包围盒至少留出固定间隔。"""

    frame_count, _ = validate_mesh_sequences(sequences)
    follow = np.asarray(follow_offsets, dtype=np.float64)
    if follow.shape != (frame_count, 3):
        raise ValueError(
            f"follow_offsets 应为 {(frame_count, 3)}，实际为 {follow.shape}"
        )
    right = normalize_vector(np.asarray(camera_right, dtype=np.float64).reshape(3))
    gap = float(method_gap)
    if gap < 0.0:
        raise ValueError("method_gap 不能为负数。")

    scalar_min = math.inf
    scalar_max = -math.inf
    for method_name in METHOD_ORDER:
        # [T,V,3] 减去只含 XZ 的 pelvis 跟随量；因此起跳高度和方法间差异都保留。
        centered = (
            np.asarray(sequences[method_name].vertices_world, dtype=np.float64)
            - follow[:, None, :]
        )
        scalar = centered @ right
        scalar_min = min(scalar_min, float(np.min(scalar)))
        scalar_max = max(scalar_max, float(np.max(scalar)))
    stage_spacing = max(float(scalar_max - scalar_min) + gap, gap + 0.25)
    method_offsets = (
        np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)[:, None]
        * right[None, :]
        * stage_spacing
    )
    return method_offsets.astype(np.float32), float(stage_spacing)


def fit_fixed_presentation_camera(
    *,
    sequences: dict[str, SmplMeshSequence],
    tracker_pos_world: np.ndarray,
    follow_offsets: np.ndarray,
    method_offsets: np.ndarray,
    viewport_width: int = OUTPUT_WIDTH,
    viewport_height: int = OUTPUT_HEIGHT,
    padding: float = CAMERA_FIT_PADDING,
) -> CameraSpec:
    """在 pelvis 局部移动坐标中一次性拟合相机，后续禁止逐帧缩放。"""

    frame_count, _ = validate_mesh_sequences(sequences)
    follow = np.asarray(follow_offsets, dtype=np.float64)
    offsets = np.asarray(method_offsets, dtype=np.float64)
    trackers = np.asarray(tracker_pos_world, dtype=np.float64)
    if follow.shape != (frame_count, 3):
        raise ValueError(f"follow_offsets 应为 {(frame_count, 3)}，实际为 {follow.shape}")
    if offsets.shape != (len(METHOD_ORDER), 3):
        raise ValueError(f"method_offsets 应为 [3,3]，实际为 {offsets.shape}")
    if trackers.shape != (frame_count, 6, 3):
        raise ValueError(f"tracker_pos_world 应为 {(frame_count, 6, 3)}，实际为 {trackers.shape}")
    if int(viewport_width) <= 0 or int(viewport_height) <= 0:
        raise ValueError("viewport_width/viewport_height 必须为正数。")

    stage_parts = []
    for method_index, method_name in enumerate(METHOD_ORDER):
        centered_vertices = (
            np.asarray(sequences[method_name].vertices_world, dtype=np.float64)
            - follow[:, None, :]
            + offsets[method_index]
        )
        stage_parts.append(centered_vertices.reshape(-1, 3))
    # 开场会在 Predictor 和 Diffusion 两侧各显示同一份三点，因此也必须纳入视锥。
    centered_core_trackers = trackers[:, :CORE_TRACKER_COUNT] - follow[:, None, :]
    stage_parts.extend(
        [
            (centered_core_trackers + offsets[1]).reshape(-1, 3),
            (centered_core_trackers + offsets[2]).reshape(-1, 3),
        ]
    )
    points = np.concatenate(stage_parts, axis=0)
    if not np.isfinite(points).all():
        raise ValueError("共享舞台点集含 NaN/Inf。")

    view_direction = presentation_view_direction_unity()
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    target = (mins + maxs) * 0.5
    rotation_pose = camera_pose_look_at(target + view_direction, target)

    # 世界包围盒中心在斜视相机中并不等于屏幕中心，先在相机平面校正一次。
    projected = (points - target) @ rotation_pose[:3, :3]
    projected_center = (
        np.min(projected[:, :2], axis=0) + np.max(projected[:, :2], axis=0)
    ) * 0.5
    target = (
        target
        + rotation_pose[:3, 0] * projected_center[0]
        + rotation_pose[:3, 1] * projected_center[1]
    )
    projected = (points - target) @ rotation_pose[:3, :3]

    aspect = float(viewport_width) / float(viewport_height)
    tan_half_y = math.tan(float(CAMERA_YFOV) * 0.5)
    tan_half_x = tan_half_y * aspect
    fit_padding = float(padding)
    distance = max(
        float(
            np.max(
                projected[:, 2]
                + np.abs(projected[:, 0]) * fit_padding / tan_half_x
            )
        ),
        float(
            np.max(
                projected[:, 2]
                + np.abs(projected[:, 1]) * fit_padding / tan_half_y
            )
        ),
        float(np.max(projected[:, 2])) + 0.25,
        0.75,
    )
    eye = target + view_direction * distance
    return CameraSpec(
        pose=camera_pose_look_at(eye, target),
        target=target.astype(np.float64),
        yfov=float(CAMERA_YFOV),
        aspect_ratio=float(aspect),
    )


def build_follow_camera_poses(
    base_camera_pose: np.ndarray,
    follow_offsets: np.ndarray,
) -> np.ndarray:
    """只把相机沿 GT pelvis 的 XZ 平移；高度、旋转和观察距离均保持不变。"""

    base = np.asarray(base_camera_pose, dtype=np.float64)
    follow = np.asarray(follow_offsets, dtype=np.float64)
    if base.shape != (4, 4):
        raise ValueError(f"base_camera_pose 应为 [4,4]，实际为 {base.shape}")
    if follow.ndim != 2 or follow.shape[1] != 3:
        raise ValueError(f"follow_offsets 应为 [T,3]，实际为 {follow.shape}")
    if not np.allclose(follow[:, 1], 0.0, atol=1e-7):
        raise ValueError("follow_offsets 的 Y 必须为 0，不能让相机跟随起跳高度。")
    poses = np.repeat(base[None], follow.shape[0], axis=0)
    poses[:, :3, 3] += follow
    return poses


def build_presentation_layout(
    *,
    sequences: dict[str, SmplMeshSequence],
    tracker_pos_world: np.ndarray,
) -> PresentationLayout:
    """构造三路共享舞台、固定尺度透视相机以及整段地面范围。"""

    frame_count, _ = validate_mesh_sequences(sequences)
    trackers = np.asarray(tracker_pos_world, dtype=np.float64)
    if trackers.shape != (frame_count, 6, 3):
        raise ValueError(f"tracker_pos_world 应为 {(frame_count, 6, 3)}，实际为 {trackers.shape}")
    if not np.isfinite(trackers).all():
        raise ValueError("tracker_pos_world 含 NaN/Inf。")

    follow_offsets = build_horizontal_pelvis_follow_offsets(
        sequences["GT"].joints_world
    ).astype(np.float64)
    provisional_pose = camera_pose_look_at(
        presentation_view_direction_unity(),
        np.zeros((3,), dtype=np.float64),
    )
    method_offsets, stage_spacing = build_stage_method_offsets(
        sequences=sequences,
        follow_offsets=follow_offsets,
        camera_right=provisional_pose[:3, 0],
    )
    base_camera = fit_fixed_presentation_camera(
        sequences=sequences,
        tracker_pos_world=trackers,
        follow_offsets=follow_offsets,
        method_offsets=method_offsets,
    )
    camera_poses = build_follow_camera_poses(base_camera.pose, follow_offsets)

    horizontal_mins = np.full((2,), np.inf, dtype=np.float64)
    horizontal_maxs = np.full((2,), -np.inf, dtype=np.float64)
    for method_index, method_name in enumerate(METHOD_ORDER):
        vertices = (
            np.asarray(sequences[method_name].vertices_world, dtype=np.float64)
            + method_offsets[method_index]
        )
        horizontal = vertices[..., [0, 2]].reshape(-1, 2)
        horizontal_mins = np.minimum(horizontal_mins, np.min(horizontal, axis=0))
        horizontal_maxs = np.maximum(horizontal_maxs, np.max(horizontal, axis=0))
    horizontal_center = (horizontal_mins + horizontal_maxs) * 0.5
    floor_y = float(np.min(sequences["GT"].vertices_world[..., 1]))
    grid_center = np.asarray(
        [horizontal_center[0], floor_y, horizontal_center[1]],
        dtype=np.float64,
    )
    grid_size = max(6.0, float(np.max(horizontal_maxs - horizontal_mins)) + 3.0)
    return PresentationLayout(
        method_offsets=np.asarray(method_offsets, dtype=np.float32),
        follow_offsets=np.asarray(follow_offsets, dtype=np.float32),
        base_camera=base_camera,
        camera_poses=np.asarray(camera_poses, dtype=np.float64),
        grid_center=grid_center,
        grid_size=float(grid_size),
        stage_spacing=float(stage_spacing),
    )


# endregion


# region 指标与画面排版


def safe_reduction_percent(before: float, after: float) -> float:
    if abs(float(before)) < 1e-12:
        return 0.0
    return (1.0 - float(after) / float(before)) * 100.0


def build_presentation_metrics(clip) -> PresentationMetrics:
    predictor = compute_rpm_p2_mc_metrics(
        predicted_global_rotations=clip.predictor_rotations_world,
        target_global_rotations=clip.reference_rotations_world,
        predicted_joint_positions=clip.predictor_joints_world,
        target_joint_positions=clip.reference_joints_world,
        fps=float(clip.fps),
    )
    diffusion = compute_rpm_p2_mc_metrics(
        predicted_global_rotations=clip.diffusion_rotations_world,
        target_global_rotations=clip.reference_rotations_world,
        predicted_joint_positions=clip.diffusion_joints_world,
        target_joint_positions=clip.reference_joints_world,
        fps=float(clip.fps),
    )
    predictor_mpjve = float(predictor["mpjve_cm_per_s"])
    diffusion_mpjve = float(diffusion["mpjve_cm_per_s"])
    predictor_jitter = float(predictor["pred_jitter_m_per_s3"])
    diffusion_jitter = float(diffusion["pred_jitter_m_per_s3"])
    return PresentationMetrics(
        predictor_mpjve=predictor_mpjve,
        diffusion_mpjve=diffusion_mpjve,
        predictor_jitter=predictor_jitter,
        diffusion_jitter=diffusion_jitter,
        mpjve_gain_percent=safe_reduction_percent(
            predictor_mpjve, diffusion_mpjve
        ),
        jitter_gain_percent=safe_reduction_percent(
            predictor_jitter, diffusion_jitter
        ),
    )


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    fill: tuple[int, int, int, int],
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - width) * 0.5
    y = box[1] + (box[3] - box[1] - height) * 0.5 - bounds[1]
    draw.text((x, y), text, font=font, fill=fill)


def rgba_color(color: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(int(round(channel * 255.0)) for channel in color)


def draw_method_labels(draw: ImageDraw.ImageDraw) -> None:
    centers = (OUTPUT_WIDTH // 6, OUTPUT_WIDTH // 2, OUTPUT_WIDTH * 5 // 6)
    widths = (190, 360, 390)
    for method_name, center_x, width in zip(METHOD_ORDER, centers, widths):
        left = int(center_x - width * 0.5)
        right = int(center_x + width * 0.5)
        draw.rounded_rectangle(
            (left, 22, right, 72),
            radius=18,
            fill=(255, 255, 255, 226),
            outline=(210, 214, 220, 230),
            width=2,
        )
        color = rgba_color(METHOD_COLORS[method_name])
        draw.rounded_rectangle(
            (left + 16, 36, left + 38, 58),
            radius=6,
            fill=color,
        )
        draw.text(
            (left + 50, 33),
            METHOD_LABELS[method_name],
            font=load_font(18),
            fill=(31, 41, 55, 255),
        )


def draw_depth_cues(draw: ImageDraw.ImageDraw) -> None:
    draw.rounded_rectangle(
        (24, 88, 337, 126),
        radius=13,
        fill=(31, 41, 55, 218),
    )
    entries = (
        (CHEST_MARKER_COLOR, "chest = front"),
        (LEFT_FOOT_COLOR, "L foot"),
        (RIGHT_FOOT_COLOR, "R foot"),
    )
    x = 39
    for color, label in entries:
        rgb = rgba_color(color)
        draw.ellipse((x, 100, x + 14, 114), fill=rgb)
        draw.text(
            (x + 20, 96),
            label,
            font=load_font(13),
            fill=(255, 255, 255, 255),
        )
        x += 121 if label == "chest = front" else 78


def compose_presentation_frame(
    *,
    viewport_rgb: np.ndarray,
    playback_label: str,
    metrics: PresentationMetrics,
    show_tracker_message: bool,
) -> Image.Image:
    image = Image.fromarray(np.asarray(viewport_rgb, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_method_labels(draw)
    draw_depth_cues(draw)

    playback_width = 178 if playback_label == "0.5× replay" else 132
    draw.rounded_rectangle(
        (OUTPUT_WIDTH - playback_width - 28, 89, OUTPUT_WIDTH - 28, 129),
        radius=13,
        fill=(31, 41, 55, 218),
    )
    draw_centered_text(
        draw,
        (OUTPUT_WIDTH - playback_width - 28, 89, OUTPUT_WIDTH - 28, 129),
        playback_label,
        load_font(15),
        (255, 255, 255, 255),
    )

    if show_tracker_message:
        draw.rounded_rectangle(
            (OUTPUT_WIDTH // 2 - 320, 88, OUTPUT_WIDTH // 2 + 320, 134),
            radius=15,
            fill=(255, 255, 255, 236),
            outline=(235, 142, 24, 245),
            width=2,
        )
        draw.ellipse(
            (OUTPUT_WIDTH // 2 - 286, 104, OUTPUT_WIDTH // 2 - 270, 120),
            fill=rgba_color(TRACKER_COLOR),
        )
        draw.text(
            (OUTPUT_WIDTH // 2 - 256, 99),
            "Same three inputs on both methods: Head + left/right wrist",
            font=load_font(16),
            fill=(31, 41, 55, 255),
        )

    footer_top = OUTPUT_HEIGHT - 82
    draw.rectangle(
        (0, footer_top, OUTPUT_WIDTH, OUTPUT_HEIGHT),
        fill=(255, 255, 255, 232),
    )
    draw.line(
        (0, footer_top, OUTPUT_WIDTH, footer_top),
        fill=(202, 207, 214, 240),
        width=2,
    )
    footer = (
        "Same Head + wrists  |  Shared GT root translation     "
        f"MPJVE {metrics.predictor_mpjve:.2f} → {metrics.diffusion_mpjve:.2f} cm/s "
        f"(−{metrics.mpjve_gain_percent:.1f}%)     "
        f"Jitter {metrics.predictor_jitter:.1f} → {metrics.diffusion_jitter:.1f} m/s³ "
        f"(−{metrics.jitter_gain_percent:.1f}%)"
    )
    draw_centered_text(
        draw,
        (22, footer_top + 4, OUTPUT_WIDTH - 22, OUTPUT_HEIGHT - 4),
        footer,
        load_font(19),
        (31, 41, 55, 255),
    )
    return Image.alpha_composite(image, overlay).convert("RGB")


# endregion


# region PyRender 共享场景


def create_material(pyrender, color, roughness: float):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color,
        metallicFactor=0.0,
        roughnessFactor=float(roughness),
    )


def render_presentation_view(
    *,
    renderer,
    scene,
    frame_index: int,
    clip,
    sequences: dict[str, SmplMeshSequence],
    faces: np.ndarray,
    method_offsets: np.ndarray,
    show_trackers: bool,
) -> np.ndarray:
    """在同一 scene 一次渲染三个人体；展示偏移不回写任何模型结果。"""

    import pyrender
    import trimesh

    dynamic_nodes = []
    rotations = {
        "GT": clip.reference_rotations_world,
        "Predictor-only": clip.predictor_rotations_world,
        "+ Diffusion": clip.diffusion_rotations_world,
    }
    foot_indices = (JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"])

    def add_trimesh(mesh, material, *, smooth: bool = True) -> None:
        node = scene.add(
            pyrender.Mesh.from_trimesh(
                mesh,
                material=material,
                smooth=bool(smooth),
            )
        )
        dynamic_nodes.append(node)

    try:
        for method_index, method_name in enumerate(METHOD_ORDER):
            offset = np.asarray(method_offsets[method_index], dtype=np.float64)
            sequence = sequences[method_name]
            vertices = (
                np.asarray(sequence.vertices_world[frame_index], dtype=np.float64)
                + offset
            )
            joints = (
                np.asarray(sequence.joints_world[frame_index], dtype=np.float64)
                + offset
            )
            if not (np.isfinite(vertices).all() and np.isfinite(joints).all()):
                raise ValueError(
                    f"{method_name} source frame {frame_index} 含 NaN/Inf。"
                )
            body = trimesh.Trimesh(
                vertices=vertices,
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            add_trimesh(
                body,
                create_material(pyrender, METHOD_COLORS[method_name], 0.92),
            )

            torso_front = -rotations[method_name][
                frame_index,
                JOINT_INDEX["spine3"],
                :,
                2,
            ]
            torso_front = normalize_vector(torso_front)
            chest_marker = create_front_marker_mesh(
                joints[JOINT_INDEX["spine3"]] + torso_front * 0.12,
                torso_front,
            )
            add_trimesh(
                chest_marker,
                create_material(pyrender, CHEST_MARKER_COLOR, 0.45),
            )

            for foot_index, foot_color in zip(
                foot_indices,
                (LEFT_FOOT_COLOR, RIGHT_FOOT_COLOR),
            ):
                foot_point = joints[foot_index] + np.asarray([0.0, 0.018, 0.0])
                foot_marker = create_sphere_cloud(foot_point[None], radius=0.027)
                add_trimesh(
                    foot_marker,
                    create_material(pyrender, foot_color, 0.52),
                )

        if show_trackers:
            core_trackers = np.asarray(
                clip.tracker_pos_world[frame_index, :CORE_TRACKER_COUNT],
                dtype=np.float64,
            )
            tracker_points = build_intro_tracker_points(
                core_trackers,
                method_offsets,
            )
            tracker_cloud = create_sphere_cloud(
                tracker_points.reshape(-1, 3),
                radius=0.036,
            )
            add_trimesh(
                tracker_cloud,
                create_material(pyrender, TRACKER_COLOR, 0.48),
            )

        # StableMotion 风格只保留柔和布光与地面透视，不使用会干扰脚姿的投射阴影。
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
        return np.asarray(color[..., :3], dtype=np.uint8)
    finally:
        for node in dynamic_nodes:
            scene.remove_node(node)


def render_presentation_video(
    *,
    output_path: Path,
    clip,
    sequences: dict[str, SmplMeshSequence],
    faces: np.ndarray,
) -> Path:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError("缺少 pyrender，无法执行 SMPL-H 离屏渲染。") from exc

    frame_count, _ = validate_mesh_sequences(sequences)
    if frame_count != int(clip.frame_count):
        raise ValueError(
            f"clip.frame_count={clip.frame_count} 与网格帧数 {frame_count} 不一致。"
        )
    layout = build_presentation_layout(
        sequences=sequences,
        tracker_pos_world=clip.tracker_pos_world,
    )
    metrics = build_presentation_metrics(clip)
    schedule = build_presentation_frame_schedule(frame_count)
    floor_y = float(np.min(sequences["GT"].vertices_world[..., 1]))
    scene, camera_node = create_static_scene(
        layout.base_camera,
        floor_y=floor_y,
        grid_size=layout.grid_size,
        grid_center=layout.grid_center,
    )
    renderer = pyrender.OffscreenRenderer(OUTPUT_WIDTH, OUTPUT_HEIGHT)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: Mp4FrameWriter | None = None
    slow_frames: list[bytes] = []
    written_frame_count = 0

    try:
        scene.set_pose(camera_node, pose=layout.camera_poses[0])
        intro_view = render_presentation_view(
            renderer=renderer,
            scene=scene,
            frame_index=0,
            clip=clip,
            sequences=sequences,
            faces=faces,
            method_offsets=layout.method_offsets,
            show_trackers=True,
        )
        intro_frame = compose_presentation_frame(
            viewport_rgb=intro_view,
            playback_label="Input setup",
            metrics=metrics,
            show_tracker_message=True,
        )
        intro_rgb = np.asarray(intro_frame, dtype=np.uint8)
        writer = Mp4FrameWriter(
            output_path=output,
            frame_rgb=intro_rgb,
            fps=int(clip.fps),
        )
        for _ in range(INTRO_FRAME_COUNT):
            writer.append(intro_rgb)
            written_frame_count += 1

        for frame_index in range(frame_count):
            scene.set_pose(camera_node, pose=layout.camera_poses[frame_index])
            viewport = render_presentation_view(
                renderer=renderer,
                scene=scene,
                frame_index=frame_index,
                clip=clip,
                sequences=sequences,
                faces=faces,
                method_offsets=layout.method_offsets,
                show_trackers=False,
            )
            normal_frame = compose_presentation_frame(
                viewport_rgb=viewport,
                playback_label="1.0×",
                metrics=metrics,
                show_tracker_message=False,
            )
            slow_frame = compose_presentation_frame(
                viewport_rgb=viewport,
                playback_label="0.5× replay",
                metrics=metrics,
                show_tracker_message=False,
            )
            writer.append(np.asarray(normal_frame, dtype=np.uint8))
            written_frame_count += 1
            slow_frames.append(encode_png(slow_frame))
            print(
                f"[smpl-presentation] rendered {frame_index + 1}/{frame_count} "
                f"(source frame {clip.source_frame_start + frame_index})",
                flush=True,
            )

        for slow_frame in slow_frames:
            slow_rgb = decode_png(slow_frame)
            writer.append(slow_rgb)
            writer.append(slow_rgb)
            written_frame_count += 2
        if written_frame_count != len(schedule):
            raise RuntimeError(
                f"输出帧数应为 {len(schedule)}，实际写入 {written_frame_count}。"
            )
    finally:
        if writer is not None:
            writer.close()
        with suppress(Exception):
            renderer.delete()

    print(
        f"[smpl-presentation] stage spacing={layout.stage_spacing:.3f} m, "
        f"frames={written_frame_count}, fps={clip.fps}",
        flush=True,
    )
    return output


# endregion


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    clip = load_comparison_clip(
        comparison_npz=args.comparison_npz,
        report_json=args.report_json,
        amass_npz=args.amass_npz,
        source_frame_start=int(args.source_frame_start),
        source_frame_end_exclusive=int(args.source_frame_end_exclusive),
        diffusion_variant="core_only",
    )
    sequences, faces = build_mesh_sequences(
        clip=clip,
        smpl_model_dir=args.smpl_model_dir,
    )
    output = render_presentation_video(
        output_path=args.output_mp4,
        clip=clip,
        sequences=sequences,
        faces=faces,
    )
    print(f"[smpl-presentation] wrote: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
