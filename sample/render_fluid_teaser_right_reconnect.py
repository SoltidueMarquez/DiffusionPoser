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
from data_loaders.sensor_masking import TRACKER_NAMES, TRACKER_TO_JOINT
from sample.realtime_pose_smpl_rendering import (
    CameraSpec,
    SmplMeshSequence,
    body_fbx_world_to_smpl_local_rotations,
    camera_pose_look_at,
    create_smplh_model,
    create_sphere_cloud,
    create_static_scene,
    load_font,
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
    presentation_view_direction_unity,
)


OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
SIDE_LABEL_WIDTH = 250
RENDER_WIDTH = OUTPUT_WIDTH - SIDE_LABEL_WIDTH
HEADER_HEIGHT = 70
ROW_HEIGHT = 455
ROW_GAP = 30

EXPECTED_RECONNECT_TRACKER = "right_foot"
EXPECTED_SOFT_BLEND_FRAMES = 10
STAGE_LABELS = ("3 trackers", "4 trackers")
PRE_SWITCH_FRAME_COUNT = 5
SOFT_SEQUENCE_FRAME_COUNT = EXPECTED_SOFT_BLEND_FRAMES
DISPLAY_FRAME_COUNT = PRE_SWITCH_FRAME_COUNT + SOFT_SEQUENCE_FRAME_COUNT
TIME_PITCH_METERS = 0.24
BOUNDARY_TIME_PITCH_METERS = 0.07
BODY_RENDER_ALPHAS = (
    0.14,
    0.18,
    0.24,
    0.34,
    0.82,
    0.94,
    0.72,
    0.58,
    0.48,
    0.42,
    0.36,
    0.32,
    0.30,
    0.40,
    0.80,
)
TRACKER_MARKER_LOCAL_INDICES = (0, 3, 4, 5, 9, 14)

BODY_COLOR = (0.04, 0.52, 0.57)
TRACKER_COLOR = (1.0, 0.0, 0.72, 1.0)
TRACKER_RADIUS = 0.024
TRAJECTORY_RADIUS = 0.006
CAMERA_YFOV = math.radians(31.0)
CAMERA_FIT_PADDING_X = 1.08
CAMERA_FIT_PADDING_Y = 1.12


# region 数据契约


@dataclass(frozen=True)
class ReconnectionArrays:
    """一路 3→4 重连产物中试版右图需要的逐帧数组。"""

    rotations_world: np.ndarray  # [T,24,3,3]
    root_yaw: np.ndarray  # [T]
    tracker_pos_world: np.ndarray  # [T,6,3]
    tracker_available: np.ndarray  # [T,6]
    stage_indices: np.ndarray  # [T]


@dataclass(frozen=True)
class ReconnectionTeaserInputs:
    """从 reconnect-final JSON 恢复出的严格配对 hard/FLUID 输入。"""

    hard: ReconnectionArrays
    fluid: ReconnectionArrays
    source_path: Path
    amass_path: Path
    frame_start: int
    frame_end_exclusive: int
    transition_index: int
    transition_source_frame: int
    reconnect_tracker_index: int
    window_start_index: int
    window_end_index: int
    selected_indices: np.ndarray  # [K]，相对于完整 NPZ
    context_indices: np.ndarray  # [15]，相对于完整 NPZ
    hard_boundary_step_cm: float
    fluid_boundary_step_cm: float


@dataclass(frozen=True)
class DisplayGeometry:
    """两行共享的时间展开舞台与相机。"""

    sequences: dict[str, SmplMeshSequence]
    tracker_positions: dict[str, np.ndarray]  # 每行 [15,6,3]
    tracker_available: np.ndarray  # [15,6]
    body_alphas: np.ndarray  # [15]
    context_local_indices: np.ndarray  # [6]
    reconnect_tracker_index: int
    camera: CameraSpec
    floor_y: float
    grid_center: np.ndarray
    grid_size: float
    stage_label_centers_px: np.ndarray  # [2]
    boundary_center_px: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render the FLUID teaser right panel from a tracker_reconnection_final "
            "manifest as a continuous 3-to-4 motion trail."
        )
    )
    paths = parser.add_argument_group("FLUID teaser right reconnect paths")
    paths.add_argument("--final_json", required=True, type=Path)
    paths.add_argument("--amass_dir", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_png", required=True, type=Path)
    return parser


def load_json_object(path: Path, label: str) -> tuple[Path, dict]:
    resolved = require_file(path, label)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 顶层必须为 object：{resolved}")
    return resolved, value


def load_reconnection_arrays(npz_path: Path) -> ReconnectionArrays:
    path = require_file(npz_path, "reconnection NPZ")
    keys = (
        "deployed_rotations_world",
        "deployed_root_yaw",
        "tracker_pos_world",
        "tracker_available",
        "stage_indices",
    )
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload.files]
        if missing:
            raise KeyError(f"{path} 缺少字段：{missing}")
        values = {key: np.asarray(payload[key]) for key in keys}
    frame_count = int(values["stage_indices"].shape[0])
    expected_shapes = {
        "deployed_rotations_world": (frame_count, 24, 3, 3),
        "deployed_root_yaw": (frame_count,),
        "tracker_pos_world": (frame_count, 6, 3),
        "tracker_available": (frame_count, 6),
        "stage_indices": (frame_count,),
    }
    for key, expected_shape in expected_shapes.items():
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
    return ReconnectionArrays(
        rotations_world=np.asarray(values["deployed_rotations_world"], dtype=np.float32),
        root_yaw=np.asarray(values["deployed_root_yaw"], dtype=np.float32),
        tracker_pos_world=np.asarray(values["tracker_pos_world"], dtype=np.float32),
        tracker_available=np.asarray(values["tracker_available"], dtype=bool),
        stage_indices=np.asarray(values["stage_indices"], dtype=np.int64),
    )


def _load_boundary_step(report: dict, label: str) -> float:
    diagnostics = report.get("switch_boundary_diagnostics", ())
    if not isinstance(diagnostics, list) or len(diagnostics) != 1:
        raise ValueError(f"{label} JSON 必须恰好包含一个切换诊断。")
    value = float(diagnostics[0]["predicted_mean_joint_step_cm"])
    if not np.isfinite(value):
        raise ValueError(f"{label} 边界位移不是有限数。")
    return value


def load_teaser_inputs(final_json: Path) -> ReconnectionTeaserInputs:
    _, final_report = load_json_object(final_json, "tracker_reconnection_final JSON")
    if final_report.get("experiment") != (
        "tracker_reconnection_shared_gt_grid_with_inline_transition_detail"
    ):
        raise ValueError("final_json 不是 reconnect-final 成片清单。")
    source_relative_path = final_report.get("source_relative_path")
    if not isinstance(source_relative_path, str) or not source_relative_path:
        raise ValueError("final_json 缺少有效的 source_relative_path。")
    input_paths = final_report.get("inputs", {})
    hard_path = require_file(Path(str(input_paths.get("hard_npz", ""))), "hard_npz")
    fluid_path = require_file(Path(str(input_paths.get("soft_npz", ""))), "soft_npz")
    _, hard_report = load_json_object(hard_path.with_suffix(".json"), "hard JSON")
    _, fluid_report = load_json_object(fluid_path.with_suffix(".json"), "FLUID JSON")

    transition = final_report.get("transition", {})
    transition_source_frame = int(transition.get("source_frame", -1))
    if transition.get("reconnect_tracker") != EXPECTED_RECONNECT_TRACKER:
        raise ValueError("右侧 teaser 要求恢复 right_foot tracker。")
    if transition_source_frame < 0:
        raise ValueError("final_json 缺少有效的 reconnect source frame。")
    if int(transition.get("soft_blend_frames", -1)) != EXPECTED_SOFT_BLEND_FRAMES:
        raise ValueError("右侧 teaser 要求 FLUID 使用 soft10f。")

    for label, report, blend_frames in (
        ("hard", hard_report, 0),
        ("fluid", fluid_report, EXPECTED_SOFT_BLEND_FRAMES),
    ):
        if report.get("experiment") != "tracker_reconnection_3_to_4_showcase":
            raise ValueError(f"{label} 不是 3→4 reconnect 产物。")
        if report.get("tracker_counts") != [3, 4]:
            raise ValueError(f"{label}.tracker_counts 必须为 [3,4]。")
        if report.get("reconnect_tracker") != EXPECTED_RECONNECT_TRACKER:
            raise ValueError(f"{label} 恢复的不是 right_foot。")
        actual_blend = int(report.get("activation_blend", {}).get("frames", -1))
        if actual_blend != blend_frames:
            raise ValueError(
                f"{label} activation blend 应为 {blend_frames}，实际为 {actual_blend}。"
            )
    if int(hard_report.get("sampling_noise_seed", -1)) != int(
        fluid_report.get("sampling_noise_seed", -2)
    ):
        raise ValueError("hard/FLUID 必须使用相同 diffusion noise seed。")

    source_paths = {
        Path(str(report.get("source_path", ""))).expanduser().resolve()
        for report in (hard_report, fluid_report)
    }
    amass_paths = {
        Path(str(report.get("amass_path", ""))).expanduser().resolve()
        for report in (hard_report, fluid_report)
    }
    if len(source_paths) != 1 or len(amass_paths) != 1:
        raise ValueError("hard/FLUID 必须引用同一条 source 和 AMASS。")
    source_path = require_file(next(iter(source_paths)), "source_path")
    amass_path = require_file(next(iter(amass_paths)), "amass_path")

    frame_start = int(hard_report["frame_start"])
    frame_end_exclusive = int(hard_report["frame_end_exclusive"])
    if frame_start != int(fluid_report["frame_start"]) or frame_end_exclusive != int(
        fluid_report["frame_end_exclusive"]
    ):
        raise ValueError("hard/FLUID frame 范围不一致。")
    hard = load_reconnection_arrays(hard_path)
    fluid = load_reconnection_arrays(fluid_path)
    if hard.stage_indices.shape[0] != frame_end_exclusive - frame_start:
        raise ValueError("NPZ 帧数与 JSON frame 范围不一致。")
    if not np.array_equal(hard.stage_indices, fluid.stage_indices):
        raise ValueError("hard/FLUID stage_indices 不一致。")
    if not np.array_equal(hard.tracker_available, fluid.tracker_available):
        raise ValueError("hard/FLUID tracker_available 不一致。")
    if not np.allclose(
        hard.tracker_pos_world,
        fluid.tracker_pos_world,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("hard/FLUID 原始 tracker 位置不一致。")
    changes = np.flatnonzero(np.diff(hard.stage_indices) > 0) + 1
    if changes.shape != (1,):
        raise ValueError("3→4 结果必须恰好包含一个阶段边界。")
    transition_index = int(changes[0])
    if frame_start + transition_index != transition_source_frame:
        raise ValueError("final JSON 与 NPZ 的 reconnect source frame 不一致。")
    before = hard.tracker_available[transition_index - 1]
    after = hard.tracker_available[transition_index]
    added = np.flatnonzero(after & ~before)
    if int(before.sum()) != 3 or int(after.sum()) != 4 or added.shape != (1,):
        raise ValueError("重连边界必须只从 3 点增加到 4 点。")
    reconnect_tracker_index = int(added[0])
    if str(TRACKER_NAMES[reconnect_tracker_index]) != EXPECTED_RECONNECT_TRACKER:
        raise ValueError("NPZ 重连 tracker 与 final JSON 不一致。")

    inline = final_report.get("inline_slow_motion", {})
    available_pre_frames = int(inline.get("pre_frames", -1))
    available_post_frames = int(inline.get("post_frames", -1))
    if available_pre_frames < PRE_SWITCH_FRAME_COUNT or available_post_frames < (
        SOFT_SEQUENCE_FRAME_COUNT
    ):
        raise ValueError(
            "reconnect-final 过渡窗口不足以提供切换前 5 帧和完整 soft10f。"
        )
    window_start_index = transition_index - PRE_SWITCH_FRAME_COUNT
    window_end_index = transition_index + SOFT_SEQUENCE_FRAME_COUNT
    if window_start_index < 0 or window_end_index > hard.stage_indices.shape[0]:
        raise ValueError("reconnect-final 过渡窗口越出 NPZ 范围。")

    context_indices = np.arange(
        window_start_index,
        window_end_index,
        dtype=np.int64,
    )
    if context_indices.shape != (DISPLAY_FRAME_COUNT,):
        raise RuntimeError(
            f"右侧 teaser 应恰好选择 {DISPLAY_FRAME_COUNT} 帧，"
            f"实际为 {context_indices.shape[0]}。"
        )
    # 这版不再额外插入或抽样帧：左侧 5 帧是切换前状态，右侧 10 帧
    # 与 soft10f 的 alpha=1..10 完整一一对应。
    selected_indices = context_indices.copy()
    return ReconnectionTeaserInputs(
        hard=hard,
        fluid=fluid,
        source_path=source_path,
        amass_path=amass_path,
        frame_start=frame_start,
        frame_end_exclusive=frame_end_exclusive,
        transition_index=transition_index,
        transition_source_frame=transition_source_frame,
        reconnect_tracker_index=reconnect_tracker_index,
        window_start_index=window_start_index,
        window_end_index=window_end_index,
        selected_indices=selected_indices,
        context_indices=context_indices,
        hard_boundary_step_cm=_load_boundary_step(hard_report, "hard"),
        fluid_boundary_step_cm=_load_boundary_step(fluid_report, "FLUID"),
    )


# endregion


# region SMPL-H 与时间展开舞台


def build_body_alphas(inputs: ReconnectionTeaserInputs) -> np.ndarray:
    if inputs.selected_indices.shape != (DISPLAY_FRAME_COUNT,):
        raise ValueError("透明度映射要求严格的 15 帧时间序列。")
    # 恢复完整的切换前 5 帧和 soft10f 10 帧人体虚影；边界前后两帧保持
    # 最高不透明度，其他帧作为 motion trail 交代连续时间顺序。
    alphas = np.asarray(BODY_RENDER_ALPHAS, dtype=np.float32)
    if alphas.shape != (DISPLAY_FRAME_COUNT,):
        raise RuntimeError("人体虚影透明度必须与 15 帧展示窗口一一对应。")
    return alphas


def teaser_view_direction_unity() -> np.ndarray:
    """从左腿所在一侧观察；时间方向由 camera_right 独立保证。"""

    # 不再为了人物朝向翻转相机。这个 reconnect 片段需要比较左腿附近的
    # body-wide 恢复，使用原始已验收侧面才能避免左腿被另一条腿遮挡。
    return np.asarray(presentation_view_direction_unity(), dtype=np.float64)


def build_selected_mesh_sequences(
    *,
    inputs: ReconnectionTeaserInputs,
    amass_dir: Path,
    smpl_model_dir: Path,
) -> tuple[dict[str, SmplMeshSequence], np.ndarray, np.ndarray]:
    mesh_inputs = load_progressive_mesh_inputs(
        source_path=inputs.source_path,
        amass_path=inputs.amass_path,
        amass_dir=require_directory(amass_dir, "amass_dir"),
        frame_start=inputs.frame_start,
        frame_end_exclusive=inputs.frame_end_exclusive,
    )
    selected = inputs.selected_indices
    model = create_smplh_model(
        model_dir=require_directory(smpl_model_dir, "smpl_model_dir"),
        gender=mesh_inputs.gender,
        batch_size=int(selected.shape[0]),
    )
    translations_amass = np.asarray(
        mesh_inputs.gt_translation_amass[selected], dtype=np.float32
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
            # 两行共享 GT 根平移；时间展开只发生在随后的展示层。
            translation_amass=translations_amass,
        )
    window_root_unity = (
        np.asarray(
            mesh_inputs.gt_translation_amass[
                inputs.window_start_index : inputs.window_end_index
            ],
            dtype=np.float32,
        )
        @ AMASS_TO_UNITY.T
    )
    return (
        sequences,
        transform_faces_to_unity_winding(model.faces),
        np.asarray(window_root_unity, dtype=np.float32),
    )


def fit_fixed_camera(points: np.ndarray) -> CameraSpec:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("相机拟合需要非空有限点集。")
    view_direction = teaser_view_direction_unity()
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
    values = np.asarray(points, dtype=np.float64)
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
    inputs: ReconnectionTeaserInputs,
    sequences: dict[str, SmplMeshSequence],
    window_root_unity: np.ndarray,
) -> DisplayGeometry:
    window_frame_count = inputs.window_end_index - inputs.window_start_index
    view_direction = teaser_view_direction_unity()
    provisional_pose = camera_pose_look_at(
        view_direction,
        np.zeros((3,), dtype=np.float64),
    )
    camera_right = np.asarray(provisional_pose[:3, 0], dtype=np.float64)
    local_frames = np.arange(window_frame_count, dtype=np.float64)
    transition_local = inputs.transition_index - inputs.window_start_index
    time_coordinates = local_frames * TIME_PITCH_METERS
    # 仅压缩 218→219 的展示间距，使切换前后的两个高不透明度人体近乎
    # 重叠：hard 的脚部跳变会形成清晰双轮廓，FLUID 则保持连续轮廓。
    time_coordinates[transition_local:] -= (
        TIME_PITCH_METERS - BOUNDARY_TIME_PITCH_METERS
    )
    time_coordinates -= 0.5 * (
        float(time_coordinates[0]) + float(time_coordinates[-1])
    )
    # 当前观察侧能清楚看到恢复的脚，但人物朝原始画面的左侧。先让原始渲染
    # 沿该朝向从右向左推进，最后整行水平镜像；成片便同时满足人物朝右、
    # 时间从左向右以及 3→4 标签从左向右，避免产生“倒退”的视觉歧义。
    time_offsets = -time_coordinates[:, None] * camera_right[None, :]

    selected_local = inputs.selected_indices - inputs.window_start_index
    selected_offsets = time_offsets[selected_local]
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
    window_slice = slice(inputs.window_start_index, inputs.window_end_index)
    tracker_joint_indices = np.asarray(TRACKER_TO_JOINT, dtype=np.int64)
    display_trackers = {
        row_name: np.asarray(
            sequence.joints_world[:, tracker_joint_indices], dtype=np.float32
        )
        for row_name, sequence in display_sequences.items()
    }
    # 洋红球是 tracker 在当前展示骨架上的 attachment glyph。这里仅把 glyph
    # 吸附到每行 SMPL-H 关节，availability、恢复时刻和模型输入均仍取原始 NPZ；
    # 这样既消除 FBX/SMPL-H 骨架定义带来的悬空，也能直接读出 hard 的突变和
    # FLUID 的连续恢复。
    display_roots = np.asarray(window_root_unity, dtype=np.float64) + time_offsets

    fit_parts = [
        np.asarray(sequence.vertices_world, dtype=np.float64).reshape(-1, 3)
        for sequence in display_sequences.values()
    ]
    window_available = np.asarray(
        inputs.hard.tracker_available[window_slice], dtype=bool
    )
    for tracker_index in range(6):
        active = window_available[:, tracker_index]
        if np.any(active):
            for row_trackers in display_trackers.values():
                fit_parts.append(row_trackers[active, tracker_index])
    fit_points = np.concatenate(fit_parts, axis=0)
    camera = fit_fixed_camera(fit_points)

    stage_anchors = np.stack(
        [
            np.mean(display_roots[:transition_local], axis=0),
            np.mean(display_roots[transition_local:], axis=0),
        ],
        axis=0,
    )
    stage_anchors[:, 1] = float(np.max(fit_points[:, 1]))
    boundary_anchor = np.asarray(
        display_roots[transition_local : transition_local + 1], dtype=np.float64
    ).copy()
    boundary_anchor[:, 1] = float(np.max(fit_points[:, 1]))
    # render_row 会水平镜像离屏结果，所有叠加标签必须使用相同像素变换。
    stage_centers_px = RENDER_WIDTH - project_points_x_to_pixels(
        stage_anchors, camera
    )
    boundary_center_px = float(
        RENDER_WIDTH - project_points_x_to_pixels(boundary_anchor, camera)[0]
    )
    if np.any(np.diff(stage_centers_px) <= 0.0):
        raise RuntimeError("3/4 tracker 阶段投影未保持从左到右。")

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
    grid_size = max(6.0, float(np.max(np.ptp(horizontal, axis=0))) + 1.5)
    return DisplayGeometry(
        sequences=display_sequences,
        tracker_positions=display_trackers,
        tracker_available=window_available,
        body_alphas=build_body_alphas(inputs),
        context_local_indices=np.asarray(
            TRACKER_MARKER_LOCAL_INDICES, dtype=np.int64
        ),
        reconnect_tracker_index=int(inputs.reconnect_tracker_index),
        camera=camera,
        floor_y=float(floor_y),
        grid_center=grid_center,
        grid_size=float(grid_size),
        stage_label_centers_px=stage_centers_px,
        boundary_center_px=boundary_center_px,
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
    geometry: DisplayGeometry,
    faces: np.ndarray,
) -> np.ndarray:
    import pyrender
    import trimesh

    scene, _ = create_static_scene(
        geometry.camera,
        floor_y=geometry.floor_y,
        grid_size=geometry.grid_size,
        grid_center=geometry.grid_center,
    )
    sequence = geometry.sequences[row_name]
    for frame_index, alpha in enumerate(geometry.body_alphas.tolist()):
        if alpha <= 0.0:
            continue
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
    row_tracker_positions = geometry.tracker_positions[row_name]
    # 核心 3 点只保留关键帧上的 attachment glyph；连续轨迹仅画恢复的脚，
    # 让读者的注意力集中在 tracker 回来后的 hard snap / FLUID continuity。
    reconnect_active = np.flatnonzero(
        geometry.tracker_available[:, geometry.reconnect_tracker_index]
    )
    if reconnect_active.size >= 2:
        trajectory_points = build_visible_tracker_glyph_points(
            row_tracker_positions[reconnect_active, geometry.reconnect_tracker_index],
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

    marker_points = []
    for local_index in geometry.context_local_indices.tolist():
        available = geometry.tracker_available[local_index]
        marker_points.append(row_tracker_positions[local_index, available])
    # 恢复脚的 10 个 tracker 点全部保留，明确表示 soft10f 的完整时间范围；
    # 其他核心 tracker 只随 6 个关键人体显示，减少洋红元素遮挡。
    marker_points.append(
        row_tracker_positions[reconnect_active, geometry.reconnect_tracker_index]
    )
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

    renderer = pyrender.OffscreenRenderer(RENDER_WIDTH, ROW_HEIGHT)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
        # 保留能看清目标脚的 3D 观察侧，同时把最终阅读方向统一为左→右。
        return np.ascontiguousarray(
            np.asarray(color[..., :3], dtype=np.uint8)[:, ::-1]
        )
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
    geometry: DisplayGeometry,
    output_png: Path,
) -> Path:
    expected_shape = (ROW_HEIGHT, RENDER_WIDTH, 3)
    for label, row in (("hard", hard_row), ("fluid", fluid_row)):
        if np.asarray(row).shape != expected_shape:
            raise ValueError(
                f"{label} row 应为 {expected_shape}，实际为 {np.asarray(row).shape}"
            )
    canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (246, 248, 251))
    top_y = HEADER_HEIGHT
    bottom_y = HEADER_HEIGHT + ROW_HEIGHT + ROW_GAP
    canvas.paste(Image.fromarray(hard_row), (SIDE_LABEL_WIDTH, top_y))
    canvas.paste(Image.fromarray(fluid_row), (SIDE_LABEL_WIDTH, bottom_y))
    draw = ImageDraw.Draw(canvas, "RGBA")

    for label, center_x in zip(STAGE_LABELS, geometry.stage_label_centers_px):
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

    stage_centers = SIDE_LABEL_WIDTH + np.rint(
        geometry.stage_label_centers_px
    ).astype(np.int64)
    arrow_start = int(stage_centers[0] + 102)
    arrow_end = int(stage_centers[1] - 102)
    if arrow_end > arrow_start + 12:
        arrow_y = 35
        draw.line(
            (arrow_start, arrow_y, arrow_end, arrow_y),
            fill=(128, 137, 151, 220),
            width=3,
        )
        draw.polygon(
            (
                (arrow_end, arrow_y),
                (arrow_end - 9, arrow_y - 6),
                (arrow_end - 9, arrow_y + 6),
            ),
            fill=(128, 137, 151, 220),
        )

    boundary_x = SIDE_LABEL_WIDTH + int(round(float(geometry.boundary_center_px)))
    draw_dashed_vertical_line(
        draw,
        x=boundary_x,
        y_start=top_y + 8,
        y_end=top_y + ROW_HEIGHT,
    )
    draw_dashed_vertical_line(
        draw,
        x=boundary_x,
        y_start=bottom_y + 8,
        y_end=bottom_y + ROW_HEIGHT,
    )

    for label, row_y in (("Binary availability", top_y), ("FLUID", bottom_y)):
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
    print(f"[fluid-teaser-right-reconnect] wrote: {output}", flush=True)
    return output


# endregion


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    inputs = load_teaser_inputs(args.final_json)
    sequences, faces, window_root_unity = build_selected_mesh_sequences(
        inputs=inputs,
        amass_dir=args.amass_dir,
        smpl_model_dir=args.smpl_model_dir,
    )
    geometry = build_display_geometry(
        inputs=inputs,
        sequences=sequences,
        window_root_unity=window_root_unity,
    )
    hard_row = render_row(row_name="hard", geometry=geometry, faces=faces)
    fluid_row = render_row(row_name="fluid", geometry=geometry, faces=faces)
    print(
        "[fluid-teaser-right-reconnect] boundary mean joint step cm: "
        f"hard={inputs.hard_boundary_step_cm:.3f}, "
        f"FLUID={inputs.fluid_boundary_step_cm:.3f}",
        flush=True,
    )
    return compose_teaser_png(
        hard_row=hard_row,
        fluid_row=fluid_row,
        geometry=geometry,
        output_png=args.output_png,
    )


if __name__ == "__main__":
    main()
