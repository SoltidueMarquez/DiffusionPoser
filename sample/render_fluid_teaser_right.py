from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_converter.amass_smpl_utils import AMASS_TO_UNITY, SOURCE_BODY_JOINT_COUNT
from data_loaders.sensor_masking import TRACKER_NAMES
from sample.realtime_pose_smpl_rendering import (
    CameraSpec,
    SmplMeshSequence,
    body_fbx_world_to_smpl_local_rotations,
    camera_pose_look_at,
    create_smplh_model,
    create_sphere_cloud,
    create_static_scene,
    load_font,
    normalize_vector,
    require_directory,
    require_file,
    rotation_matrices_to_axis_angle,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)
from sample.render_progressive_tracker_dropout_sequences import (
    load_progressive_mesh_inputs,
)
from sample.render_realtime_pose_smpl_presentation import (
    build_visible_tracker_glyph_points,
    create_material,
    draw_centered_text,
)


OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
SIDE_LABEL_WIDTH = 250
RENDER_WIDTH = OUTPUT_WIDTH - SIDE_LABEL_WIDTH
HEADER_HEIGHT = 70
ROW_HEIGHT = 455
ROW_GAP = 30

EXPECTED_FRAME_START = 127
EXPECTED_FRAME_END_EXCLUSIVE = 199
EXPECTED_FRAME_COUNT = EXPECTED_FRAME_END_EXCLUSIVE - EXPECTED_FRAME_START
EXPECTED_TRACKER_COUNTS = (3, 4, 5, 6)
EXPECTED_ADD_ORDER = ("hip", "left_foot", "right_foot")
EXPECTED_STAGE_LENGTH = 18
BOUNDARY_SOURCE_FRAMES = (145, 163, 181)
STAGE_LABELS = ("3 trackers", "4 trackers", "5 trackers", "6 trackers")
WINDOW_RADIUS = 3

# 12 个低透明度帧负责交代整段行走；三个边界各自补充前 3 / 后 3 帧。
# 两类帧取并集后约 27 帧，既形成完整 motion trail，也避免把 72 个网格
# 全部塞进不足两米的真实轨迹中。
CONTEXT_SOURCE_FRAMES = tuple(
    int(value)
    for value in np.rint(
        np.linspace(
            EXPECTED_FRAME_START,
            EXPECTED_FRAME_END_EXCLUSIVE - 1,
            num=12,
        )
    ).astype(np.int64)
)
BOUNDARY_WINDOW_SOURCE_FRAMES = tuple(
    source_frame
    for boundary in BOUNDARY_SOURCE_FRAMES
    for source_frame in range(boundary - WINDOW_RADIUS, boundary + WINDOW_RADIUS)
)
SELECTED_SOURCE_FRAMES = tuple(
    sorted(set(CONTEXT_SOURCE_FRAMES) | set(BOUNDARY_WINDOW_SOURCE_FRAMES))
)

BODY_COLOR = (0.04, 0.52, 0.57)
CONTEXT_ALPHA_START = 0.13
CONTEXT_ALPHA_END = 0.27
BOUNDARY_ALPHAS = (0.17, 0.24, 0.34, 0.48, 0.66, 0.86)
TRACKER_COLOR = (1.0, 0.0, 0.72, 1.0)
TRACKER_RADIUS = 0.024
TRAJECTORY_RADIUS = 0.006
CAMERA_YFOV = math.radians(31.0)
CAMERA_FIT_PADDING_X = 1.10
CAMERA_FIT_PADDING_Y = 1.12
PATH_LONGITUDINAL_STRETCH = 3.2


# region 数据契约


@dataclass(frozen=True)
class ProgressiveTeaserArrays:
    """一路连续 3→6 产物中右图需要的数组。"""

    rotations_world: np.ndarray  # [72,24,3,3]
    root_yaw: np.ndarray  # [72]
    tracker_pos_world: np.ndarray  # [72,6,3]
    tracker_available: np.ndarray  # [72,6]
    stage_indices: np.ndarray  # [72]


@dataclass(frozen=True)
class FluidTeaserRightInputs:
    """经严格对齐的 Binary/FLUID 输入与共享元数据。"""

    hard: ProgressiveTeaserArrays
    fluid: ProgressiveTeaserArrays
    source_path: Path
    amass_path: Path
    frame_start: int
    frame_end_exclusive: int
    selected_source_frames: np.ndarray  # [K]
    selected_indices: np.ndarray  # [K]
    context_indices: np.ndarray  # [12]
    boundary_indices: np.ndarray  # [3]
    changed_tracker_indices: np.ndarray  # [3]
    hard_boundary_steps_cm: np.ndarray  # [3]
    fluid_boundary_steps_cm: np.ndarray  # [3]


@dataclass(frozen=True)
class DisplayGeometry:
    """两行共享的真实场景几何和投影视图。"""

    sequences: dict[str, SmplMeshSequence]
    tracker_positions: np.ndarray  # [72,6,3]
    tracker_available: np.ndarray  # [72,6]
    body_alphas: np.ndarray  # [K]
    camera: CameraSpec
    floor_y: float
    grid_center: np.ndarray
    grid_size: float
    stage_label_centers_px: np.ndarray  # [4]
    boundary_centers_px: np.ndarray  # [3]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the FLUID teaser right panel as one continuous HumanEva "
            "walking motion trail with shared ground-truth root translation."
        )
    )
    paths = parser.add_argument_group("FLUID teaser right paths")
    paths.add_argument("--hard_npz", required=True, type=Path)
    paths.add_argument("--fluid_npz", required=True, type=Path)
    paths.add_argument("--amass_dir", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_png", required=True, type=Path)
    return parser


def load_json_sidecar(npz_path: Path) -> tuple[Path, dict]:
    report_path = require_file(Path(npz_path).with_suffix(".json"), "JSON sidecar")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"JSON sidecar 顶层必须是 object：{report_path}")
    return report_path, report


def load_progressive_arrays(npz_path: Path) -> ProgressiveTeaserArrays:
    path = require_file(npz_path, "progressive NPZ")
    required = {
        "deployed_rotations_world": (EXPECTED_FRAME_COUNT, 24, 3, 3),
        "deployed_root_yaw": (EXPECTED_FRAME_COUNT,),
        "tracker_pos_world": (EXPECTED_FRAME_COUNT, 6, 3),
        "tracker_available": (EXPECTED_FRAME_COUNT, 6),
        "stage_indices": (EXPECTED_FRAME_COUNT,),
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise KeyError(f"{path} 缺少字段：{missing}")
        values = {key: np.asarray(payload[key]) for key in required}
    for key, expected_shape in required.items():
        if values[key].shape != expected_shape:
            raise ValueError(
                f"{path.name}.{key} 应为 {expected_shape}，实际为 {values[key].shape}"
            )
    for key in (
        "deployed_rotations_world",
        "deployed_root_yaw",
        "tracker_pos_world",
    ):
        if not np.isfinite(values[key]).all():
            raise ValueError(f"{path.name}.{key} 含 NaN/Inf。")
    if values["tracker_available"].dtype != np.bool_:
        raise ValueError(f"{path.name}.tracker_available 必须为 bool。")
    return ProgressiveTeaserArrays(
        rotations_world=np.asarray(values["deployed_rotations_world"], dtype=np.float32),
        root_yaw=np.asarray(values["deployed_root_yaw"], dtype=np.float32),
        tracker_pos_world=np.asarray(values["tracker_pos_world"], dtype=np.float32),
        tracker_available=np.asarray(values["tracker_available"], dtype=bool),
        stage_indices=np.asarray(values["stage_indices"], dtype=np.int64),
    )


def _load_boundary_steps(report: dict, label: str) -> np.ndarray:
    steps = np.asarray(
        [
            float(item["predicted_mean_joint_step_cm"])
            for item in report.get("switch_boundary_diagnostics", ())
        ],
        dtype=np.float32,
    )
    if steps.shape != (3,) or not np.isfinite(steps).all():
        raise ValueError(f"{label} JSON 必须包含 3 个有限的边界诊断。")
    return steps


def load_teaser_inputs(
    *,
    hard_npz: Path,
    fluid_npz: Path,
) -> FluidTeaserRightInputs:
    hard_path = require_file(hard_npz, "hard_npz")
    fluid_path = require_file(fluid_npz, "fluid_npz")
    _, hard_report = load_json_sidecar(hard_path)
    _, fluid_report = load_json_sidecar(fluid_path)

    for label, report in (("hard", hard_report), ("fluid", fluid_report)):
        if report.get("experiment") != "progressive_tracker_addition_3_to_6_showcase":
            raise ValueError(f"{label} 不是 progressive 3→6 showcase 产物。")
        if int(report.get("frame_start", -1)) != EXPECTED_FRAME_START:
            raise ValueError(f"{label}.frame_start 必须为 {EXPECTED_FRAME_START}。")
        if int(report.get("frame_end_exclusive", -1)) != EXPECTED_FRAME_END_EXCLUSIVE:
            raise ValueError(
                f"{label}.frame_end_exclusive 必须为 "
                f"{EXPECTED_FRAME_END_EXCLUSIVE}。"
            )
        if tuple(report.get("tracker_counts", ())) != EXPECTED_TRACKER_COUNTS:
            raise ValueError(
                f"{label}.tracker_counts 必须为 {EXPECTED_TRACKER_COUNTS}。"
            )
        if tuple(report.get("add_order", ())) != EXPECTED_ADD_ORDER:
            raise ValueError(f"{label}.add_order 必须为 {EXPECTED_ADD_ORDER}。")

    hard_blend_frames = int(hard_report.get("activation_blend", {}).get("frames", 0))
    fluid_blend_frames = int(
        fluid_report.get("activation_blend", {}).get("frames", 0)
    )
    if hard_blend_frames != 0:
        raise ValueError("hard_npz 必须使用 activation_blend_frames=0。")
    if fluid_blend_frames != 10:
        raise ValueError("fluid_npz 必须使用 activation_blend_frames=10。")
    if int(hard_report.get("sampling_noise_seed", -1)) != int(
        fluid_report.get("sampling_noise_seed", -2)
    ):
        raise ValueError("hard/fluid 必须使用相同 diffusion noise seed。")

    source_paths = {
        Path(str(report.get("source_path", ""))).expanduser().resolve()
        for report in (hard_report, fluid_report)
    }
    amass_paths = {
        Path(str(report.get("amass_path", ""))).expanduser().resolve()
        for report in (hard_report, fluid_report)
    }
    if len(source_paths) != 1 or len(amass_paths) != 1:
        raise ValueError("hard/fluid JSON 必须引用同一条 source 和 AMASS。")
    source_path = require_file(next(iter(source_paths)), "JSON source_path")
    amass_path = require_file(next(iter(amass_paths)), "JSON amass_path")

    hard = load_progressive_arrays(hard_path)
    fluid = load_progressive_arrays(fluid_path)
    if not np.array_equal(hard.stage_indices, fluid.stage_indices):
        raise ValueError("hard/fluid stage_indices 不一致。")
    if not np.array_equal(hard.tracker_available, fluid.tracker_available):
        raise ValueError("hard/fluid tracker_available 不一致。")
    if not np.allclose(
        hard.tracker_pos_world,
        fluid.tracker_pos_world,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("hard/fluid 原始 tracker 位置不一致。")

    stage_lengths = tuple(
        int(np.count_nonzero(hard.stage_indices == stage_index))
        for stage_index in range(4)
    )
    if stage_lengths != (EXPECTED_STAGE_LENGTH,) * 4:
        raise ValueError(
            f"四个阶段必须各为 {EXPECTED_STAGE_LENGTH} 帧，实际为 {stage_lengths}。"
        )
    boundaries = np.flatnonzero(np.diff(hard.stage_indices) > 0) + 1
    expected_boundaries = np.asarray(BOUNDARY_SOURCE_FRAMES) - EXPECTED_FRAME_START
    if not np.array_equal(boundaries, expected_boundaries):
        raise ValueError(
            f"阶段边界应为 {expected_boundaries.tolist()}，"
            f"实际为 {boundaries.tolist()}。"
        )

    changed_tracker_indices = []
    for stage_index, boundary in enumerate(boundaries.tolist(), start=1):
        before = hard.tracker_available[boundary - 1]
        after = hard.tracker_available[boundary]
        added = np.flatnonzero(after & ~before)
        removed = np.flatnonzero(before & ~after)
        if added.shape != (1,) or removed.size:
            raise ValueError(f"边界 {boundary} 不是合法的单 Tracker 增加。")
        tracker_index = int(added[0])
        expected_name = EXPECTED_ADD_ORDER[stage_index - 1]
        if str(TRACKER_NAMES[tracker_index]) != expected_name:
            raise ValueError(
                f"边界 {boundary} 应新增 {expected_name}，"
                f"实际为 {TRACKER_NAMES[tracker_index]}。"
            )
        changed_tracker_indices.append(tracker_index)

    selected_source_frames = np.asarray(SELECTED_SOURCE_FRAMES, dtype=np.int64)
    selected_indices = selected_source_frames - EXPECTED_FRAME_START
    context_indices = (
        np.asarray(CONTEXT_SOURCE_FRAMES, dtype=np.int64) - EXPECTED_FRAME_START
    )
    if np.any(selected_indices < 0) or np.any(selected_indices >= EXPECTED_FRAME_COUNT):
        raise ValueError("右图选帧越出连续 progressive 产物范围。")
    return FluidTeaserRightInputs(
        hard=hard,
        fluid=fluid,
        source_path=source_path,
        amass_path=amass_path,
        frame_start=EXPECTED_FRAME_START,
        frame_end_exclusive=EXPECTED_FRAME_END_EXCLUSIVE,
        selected_source_frames=selected_source_frames,
        selected_indices=selected_indices,
        context_indices=context_indices,
        boundary_indices=boundaries.astype(np.int64),
        changed_tracker_indices=np.asarray(changed_tracker_indices, dtype=np.int64),
        hard_boundary_steps_cm=_load_boundary_steps(hard_report, "hard"),
        fluid_boundary_steps_cm=_load_boundary_steps(fluid_report, "fluid"),
    )


# endregion


# region SMPL-H 与连续场景


def build_body_alphas(selected_source_frames: np.ndarray) -> np.ndarray:
    """全局帧保持轻量，边界窗口按时间逐渐加深。"""

    frames = np.asarray(selected_source_frames, dtype=np.int64)
    progress = (frames - EXPECTED_FRAME_START) / float(EXPECTED_FRAME_COUNT - 1)
    alphas = CONTEXT_ALPHA_START + progress * (
        CONTEXT_ALPHA_END - CONTEXT_ALPHA_START
    )
    for boundary in BOUNDARY_SOURCE_FRAMES:
        for slot, source_frame in enumerate(
            range(boundary - WINDOW_RADIUS, boundary + WINDOW_RADIUS)
        ):
            selected = frames == source_frame
            alphas[selected] = np.maximum(alphas[selected], BOUNDARY_ALPHAS[slot])
    return np.asarray(alphas, dtype=np.float32)


def build_selected_mesh_sequences(
    *,
    inputs: FluidTeaserRightInputs,
    amass_dir: Path,
    smpl_model_dir: Path,
) -> tuple[dict[str, SmplMeshSequence], np.ndarray, np.ndarray]:
    """重建选中帧，同时保留整段 GT 根轨迹用于相机和标签。"""

    mesh_inputs = load_progressive_mesh_inputs(
        source_path=inputs.source_path,
        amass_path=inputs.amass_path,
        amass_dir=require_directory(amass_dir, "amass_dir"),
        frame_start=inputs.frame_start,
        frame_end_exclusive=inputs.frame_end_exclusive,
    )
    selected = inputs.selected_indices
    translations_amass = np.asarray(
        mesh_inputs.gt_translation_amass[selected], dtype=np.float32
    )
    model = create_smplh_model(
        model_dir=require_directory(smpl_model_dir, "smpl_model_dir"),
        gender=mesh_inputs.gender,
        batch_size=int(selected.shape[0]),
    )
    sequences: dict[str, SmplMeshSequence] = {}
    for row_name, arrays in (("hard", inputs.hard), ("fluid", inputs.fluid)):
        local_rotations = body_fbx_world_to_smpl_local_rotations(
            arrays.rotations_world[selected],
            arrays.root_yaw[selected],
            mesh_inputs.rest_local_rotations,
            mesh_inputs.parents,
        )
        pose_axis_angle = rotation_matrices_to_axis_angle(
            local_rotations[:, :SOURCE_BODY_JOINT_COUNT]
        )
        sequences[row_name] = run_smplh_forward(
            model=model,
            pose_axis_angle=pose_axis_angle,
            betas=mesh_inputs.betas,
            # 两行共享未经水平归一化的 GT 根平移，真实展示人在场景中的移动。
            translation_amass=translations_amass,
        )
    full_root_unity = (
        np.asarray(mesh_inputs.gt_translation_amass, dtype=np.float32)
        @ AMASS_TO_UNITY.T
    )
    return (
        sequences,
        transform_faces_to_unity_winding(model.faces),
        np.asarray(full_root_unity, dtype=np.float32),
    )


def fit_path_aligned_camera(
    points: np.ndarray,
    root_positions: np.ndarray,
) -> CameraSpec:
    """令行走首尾方向接近画面水平轴，并在共享点集上拟合相机。"""

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    roots = np.asarray(root_positions, dtype=np.float64)
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("相机拟合需要非空有限点集。")
    if roots.shape != (EXPECTED_FRAME_COUNT, 3) or not np.isfinite(roots).all():
        raise ValueError(
            f"root_positions 应为 {(EXPECTED_FRAME_COUNT, 3)}，实际为 {roots.shape}"
        )
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    path_direction = np.asarray(roots[-1] - roots[0], dtype=np.float64)
    path_direction[1] = 0.0
    path_direction = normalize_vector(path_direction)
    horizontal_view = normalize_vector(np.cross(path_direction, world_up))
    view_direction = normalize_vector(horizontal_view + world_up * 0.34)

    mins = np.min(values, axis=0)
    maxs = np.max(values, axis=0)
    target = (mins + maxs) * 0.5
    rotation_pose = camera_pose_look_at(target + view_direction, target)
    projected = (values - target) @ rotation_pose[:3, :3]
    projected_center = (
        np.min(projected[:, :2], axis=0) + np.max(projected[:, :2], axis=0)
    ) * 0.5
    target = (
        target
        + rotation_pose[:3, 0] * projected_center[0]
        + rotation_pose[:3, 1] * projected_center[1]
    )
    projected = (values - target) @ rotation_pose[:3, :3]
    aspect = float(RENDER_WIDTH) / float(ROW_HEIGHT)
    tan_half_y = math.tan(CAMERA_YFOV * 0.5)
    tan_half_x = tan_half_y * aspect
    distance = max(
        float(
            np.max(
                projected[:, 2]
                + np.abs(projected[:, 0]) * CAMERA_FIT_PADDING_X / tan_half_x
            )
        ),
        float(
            np.max(
                projected[:, 2]
                + np.abs(projected[:, 1]) * CAMERA_FIT_PADDING_Y / tan_half_y
            )
        ),
        float(np.max(projected[:, 2])) + 0.25,
        0.75,
    )
    eye = target + view_direction * distance
    return CameraSpec(
        pose=camera_pose_look_at(eye, target),
        target=np.asarray(target, dtype=np.float64),
        yfov=float(CAMERA_YFOV),
        aspect_ratio=float(aspect),
    )


def project_points_x_to_pixels(points: np.ndarray, camera: CameraSpec) -> np.ndarray:
    """把世界点的透视投影 X 坐标转为 row viewport 像素。"""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"projection points 应为 [N,3]，实际为 {values.shape}")
    homogeneous = np.concatenate(
        [values, np.ones((values.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera_points = (
        np.linalg.inv(np.asarray(camera.pose, dtype=np.float64)) @ homogeneous.T
    ).T[:, :3]
    forward_depth = -camera_points[:, 2]
    if np.any(forward_depth <= 1e-8):
        raise ValueError("标签锚点必须位于相机前方。")
    tan_half_x = math.tan(float(camera.yfov) * 0.5) * float(camera.aspect_ratio)
    normalized_x = camera_points[:, 0] / (forward_depth * tan_half_x)
    return ((normalized_x + 1.0) * 0.5 * float(RENDER_WIDTH)).astype(np.float32)


def build_display_geometry(
    *,
    inputs: FluidTeaserRightInputs,
    sequences: dict[str, SmplMeshSequence],
    full_root_unity: np.ndarray,
) -> DisplayGeometry:
    """保留根轨迹形状，并沿前进方向展开以适应论文横向版式。"""

    roots = np.asarray(full_root_unity, dtype=np.float64)
    path_direction = np.asarray(roots[-1] - roots[0], dtype=np.float64)
    path_direction[1] = 0.0
    path_direction = normalize_vector(path_direction)
    longitudinal_distance = (roots - roots[0]) @ path_direction
    display_offsets = (
        longitudinal_distance[:, None]
        * path_direction[None, :]
        * (PATH_LONGITUDINAL_STRETCH - 1.0)
    )
    display_roots = roots + display_offsets

    # 这里仅展开展示位置：同一帧的人体、关节和 Tracker 使用完全相同的
    # 平移量，因此 Tracker 与人体的相对关系以及 hard/FLUID 差异均不变。
    selected_offsets = display_offsets[inputs.selected_indices]
    display_sequences = {
        row_name: SmplMeshSequence(
            vertices_world=(
                np.asarray(sequence.vertices_world, dtype=np.float64)
                + selected_offsets[:, None, :]
            ).astype(np.float32),
            joints_world=(
                np.asarray(sequence.joints_world, dtype=np.float64)
                + selected_offsets[:, None, :]
            ).astype(np.float32),
        )
        for row_name, sequence in sequences.items()
    }
    display_trackers = (
        np.asarray(inputs.hard.tracker_pos_world, dtype=np.float64)
        + display_offsets[:, None, :]
    ).astype(np.float32)

    fit_parts = [
        np.asarray(sequence.vertices_world, dtype=np.float64).reshape(-1, 3)
        for sequence in display_sequences.values()
    ]
    for tracker_index in range(6):
        active = inputs.hard.tracker_available[:, tracker_index]
        fit_parts.append(
            np.asarray(display_trackers[active, tracker_index], dtype=np.float64)
        )
    fit_points = np.concatenate(fit_parts, axis=0)
    camera = fit_path_aligned_camera(fit_points, display_roots)

    stage_anchors = []
    for stage_index in range(4):
        stage_mask = inputs.hard.stage_indices == stage_index
        anchor = np.mean(display_roots[stage_mask], axis=0)
        anchor[1] = float(np.max(fit_points[:, 1]))
        stage_anchors.append(anchor)
    boundary_anchors = np.asarray(
        display_roots[inputs.boundary_indices], dtype=np.float64
    ).copy()
    boundary_anchors[:, 1] = float(np.max(fit_points[:, 1]))
    stage_centers_px = project_points_x_to_pixels(
        np.stack(stage_anchors, axis=0), camera
    )
    boundary_centers_px = project_points_x_to_pixels(boundary_anchors, camera)
    if np.any(np.diff(stage_centers_px) <= 0.0):
        raise RuntimeError(
            f"连续阶段投影未保持从左到右：{stage_centers_px.tolist()}"
        )

    floor_y = min(
        float(np.min(sequence.vertices_world[..., 1]))
        for sequence in display_sequences.values()
    )
    horizontal = fit_points[:, [0, 2]]
    horizontal_center = (
        np.min(horizontal, axis=0) + np.max(horizontal, axis=0)
    ) * 0.5
    grid_center = np.asarray(
        [horizontal_center[0], floor_y, horizontal_center[1]], dtype=np.float64
    )
    grid_size = max(4.0, float(np.max(np.ptp(horizontal, axis=0))) + 1.5)
    return DisplayGeometry(
        sequences=display_sequences,
        tracker_positions=display_trackers,
        tracker_available=np.asarray(inputs.hard.tracker_available, dtype=bool),
        body_alphas=build_body_alphas(inputs.selected_source_frames),
        camera=camera,
        floor_y=float(floor_y),
        grid_center=grid_center,
        grid_size=float(grid_size),
        stage_label_centers_px=stage_centers_px,
        boundary_centers_px=boundary_centers_px,
    )


# endregion


# region 离屏渲染


def create_ghost_material(pyrender, alpha: float):
    opacity = float(alpha)
    if not 0.0 < opacity <= 1.0:
        raise ValueError("ghost alpha 必须位于 (0,1]。")
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*BODY_COLOR, opacity),
        metallicFactor=0.0,
        roughnessFactor=0.86,
        alphaMode="BLEND",
        doubleSided=False,
    )


def create_polyline_tube(points: np.ndarray, radius: float):
    """将连续 tracker 轨迹转成可稳定离屏渲染的细圆柱管。"""

    import trimesh

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 2:
        raise ValueError(f"trajectory points 应为 [N>=2,3]，实际为 {values.shape}")
    parts = []
    for start, end in zip(values[:-1], values[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-8:
            continue
        segment = trimesh.creation.cylinder(
            radius=float(radius),
            height=length,
            sections=10,
        )
        alignment = trimesh.geometry.align_vectors(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            delta / length,
        )
        segment.apply_transform(alignment)
        segment.apply_translation((start + end) * 0.5)
        parts.append(segment)
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def render_row(
    *,
    row_name: str,
    inputs: FluidTeaserRightInputs,
    geometry: DisplayGeometry,
    faces: np.ndarray,
) -> np.ndarray:
    import pyrender
    import trimesh

    if row_name not in geometry.sequences:
        raise KeyError(f"未知右图行：{row_name}")
    scene, _ = create_static_scene(
        geometry.camera,
        floor_y=geometry.floor_y,
        grid_size=geometry.grid_size,
        grid_center=geometry.grid_center,
    )
    sequence = geometry.sequences[row_name]
    for frame_index, alpha in enumerate(geometry.body_alphas.tolist()):
        body = trimesh.Trimesh(
            vertices=np.asarray(sequence.vertices_world[frame_index], dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        scene.add(
            pyrender.Mesh.from_trimesh(
                body,
                material=create_ghost_material(pyrender, alpha),
                smooth=True,
            )
        )

    camera_position = np.asarray(geometry.camera.pose[:3, 3], dtype=np.float64)
    tracker_material = create_material(pyrender, TRACKER_COLOR, 0.50)
    # 每个 Tracker 只从真正可用的第一帧开始绘制；所以 Hip、左右脚的
    # 洋红轨迹会在三个切换点依次出现，而核心三点贯穿整段序列。
    for tracker_index in range(6):
        active = geometry.tracker_available[:, tracker_index]
        active_indices = np.flatnonzero(active)
        if active_indices.size < 2:
            continue
        trajectory_points = build_visible_tracker_glyph_points(
            geometry.tracker_positions[active_indices, tracker_index],
            camera_position,
            camera_offset=0.065,
        )
        trajectory_mesh = create_polyline_tube(
            trajectory_points,
            radius=TRAJECTORY_RADIUS,
        )
        if trajectory_mesh is not None:
            scene.add(
                pyrender.Mesh.from_trimesh(
                    trajectory_mesh,
                    material=tracker_material,
                    smooth=True,
                )
            )

    # Tracker 球只放在 12 个全局上下文帧，避免边界的 6 帧密集窗口把人体遮住。
    marker_points = []
    for frame_index in inputs.context_indices.tolist():
        available = geometry.tracker_available[frame_index]
        marker_points.append(geometry.tracker_positions[frame_index, available])
    visible_markers = build_visible_tracker_glyph_points(
        np.concatenate(marker_points, axis=0),
        camera_position,
        camera_offset=0.075,
    )
    tracker_cloud = create_sphere_cloud(visible_markers, radius=TRACKER_RADIUS)
    if tracker_cloud is not None:
        scene.add(
            pyrender.Mesh.from_trimesh(
                tracker_cloud,
                material=tracker_material,
                smooth=True,
            )
        )

    renderer = pyrender.OffscreenRenderer(
        viewport_width=RENDER_WIDTH,
        viewport_height=ROW_HEIGHT,
    )
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
        return np.asarray(color[..., :3], dtype=np.uint8)
    finally:
        renderer.delete()


def draw_dashed_vertical_line(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y_start: int,
    y_end: int,
) -> None:
    for y in range(y_start, y_end, 15):
        draw.line((x, y, x, min(y + 7, y_end)), fill=(128, 137, 151, 105), width=2)


def compose_teaser_png(
    *,
    hard_row: np.ndarray,
    fluid_row: np.ndarray,
    stage_label_centers_px: np.ndarray,
    boundary_centers_px: np.ndarray,
    output_png: Path,
) -> Path:
    expected_row_shape = (ROW_HEIGHT, RENDER_WIDTH, 3)
    for label, row in (("hard", hard_row), ("fluid", fluid_row)):
        if np.asarray(row).shape != expected_row_shape:
            raise ValueError(
                f"{label} row 应为 {expected_row_shape}，实际为 {np.asarray(row).shape}"
            )
    canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (246, 248, 251))
    top_y = HEADER_HEIGHT
    bottom_y = HEADER_HEIGHT + ROW_HEIGHT + ROW_GAP
    canvas.paste(Image.fromarray(hard_row), (SIDE_LABEL_WIDTH, top_y))
    canvas.paste(Image.fromarray(fluid_row), (SIDE_LABEL_WIDTH, bottom_y))
    draw = ImageDraw.Draw(canvas, "RGBA")

    stage_centers = np.asarray(stage_label_centers_px, dtype=np.float64)
    boundary_centers = np.asarray(boundary_centers_px, dtype=np.float64)
    if stage_centers.shape != (4,) or boundary_centers.shape != (3,):
        raise ValueError("阶段/边界标签投影形状错误。")
    for label, center_x in zip(STAGE_LABELS, stage_centers):
        center_x = SIDE_LABEL_WIDTH + int(round(float(center_x)))
        box = (center_x - 88, 13, center_x + 88, 57)
        draw.rounded_rectangle(
            box,
            radius=15,
            fill=(255, 255, 255, 235),
            outline=(205, 211, 220, 245),
            width=2,
        )
        draw_centered_text(
            draw,
            box,
            label,
            load_font(19),
            (31, 41, 55, 255),
        )

    for center_x in boundary_centers:
        x = SIDE_LABEL_WIDTH + int(round(float(center_x)))
        draw_dashed_vertical_line(draw, x=x, y_start=top_y + 8, y_end=top_y + ROW_HEIGHT)
        draw_dashed_vertical_line(
            draw,
            x=x,
            y_start=bottom_y + 8,
            y_end=bottom_y + ROW_HEIGHT,
        )

    row_labels = (
        ("Binary availability", top_y),
        ("FLUID", bottom_y),
    )
    for label, row_y in row_labels:
        box = (
            18,
            row_y + ROW_HEIGHT // 2 - 29,
            SIDE_LABEL_WIDTH - 18,
            row_y + ROW_HEIGHT // 2 + 29,
        )
        draw.rounded_rectangle(
            box,
            radius=16,
            fill=(255, 255, 255, 235),
            outline=(205, 211, 220, 245),
            width=2,
        )
        draw_centered_text(
            draw,
            box,
            label,
            load_font(19),
            (31, 41, 55, 255),
        )
    divider_y = HEADER_HEIGHT + ROW_HEIGHT + ROW_GAP // 2
    draw.line(
        (18, divider_y, OUTPUT_WIDTH - 18, divider_y),
        fill=(205, 211, 220, 220),
        width=2,
    )

    output = Path(output_png).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("output_png 必须使用 .png 后缀。")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    with Image.open(output) as written:
        if written.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
            raise RuntimeError(f"右图尺寸错误：{written.size}")
    print(f"[fluid-teaser-right] wrote: {output}", flush=True)
    return output


# endregion


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    inputs = load_teaser_inputs(
        hard_npz=args.hard_npz,
        fluid_npz=args.fluid_npz,
    )
    sequences, faces, full_root_unity = build_selected_mesh_sequences(
        inputs=inputs,
        amass_dir=args.amass_dir,
        smpl_model_dir=args.smpl_model_dir,
    )
    geometry = build_display_geometry(
        inputs=inputs,
        sequences=sequences,
        full_root_unity=full_root_unity,
    )
    hard_row = render_row(
        row_name="hard",
        inputs=inputs,
        geometry=geometry,
        faces=faces,
    )
    fluid_row = render_row(
        row_name="fluid",
        inputs=inputs,
        geometry=geometry,
        faces=faces,
    )
    print(
        "[fluid-teaser-right] boundary mean joint steps cm: "
        f"hard={np.round(inputs.hard_boundary_steps_cm, 3).tolist()}, "
        f"FLUID={np.round(inputs.fluid_boundary_steps_cm, 3).tolist()}",
        flush=True,
    )
    return compose_teaser_png(
        hard_row=hard_row,
        fluid_row=fluid_row,
        stage_label_centers_px=geometry.stage_label_centers_px,
        boundary_centers_px=geometry.boundary_centers_px,
        output_png=args.output_png,
    )


if __name__ == "__main__":
    main()
