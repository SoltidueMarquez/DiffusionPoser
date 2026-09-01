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
from sample.render_fluid_teaser_right_reconnect import create_polyline_tube
from sample.render_progressive_tracker_dropout_sequences import (
    load_progressive_mesh_inputs,
)
from sample.render_realtime_pose_smpl_presentation import (
    build_visible_tracker_glyph_points,
    create_material,
    draw_centered_text,
    presentation_view_direction_unity,
)


OUTPUT_WIDTH = 3840
OUTPUT_HEIGHT = 2160
SIDE_LABEL_WIDTH = 430
HEADER_HEIGHT = 150
FOOTER_HEIGHT = 100
ROW_GAP = 60
ROW_HEIGHT = (OUTPUT_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT - ROW_GAP) // 2
RENDER_WIDTH = OUTPUT_WIDTH - SIDE_LABEL_WIDTH

DISPLAY_OFFSETS = (-8, -5, -3, -1, 0, 1, 3, 6, 9)
DISPLAY_FRAME_COUNT = len(DISPLAY_OFFSETS)
TRANSITION_SLOT = DISPLAY_OFFSETS.index(0)
BODY_RENDER_ALPHAS = (0.12, 0.17, 0.26, 0.72, 0.98, 0.72, 0.30, 0.20, 0.14)
TIME_PITCH_METERS = 0.30
BOUNDARY_TIME_PITCH_METERS = 0.08

BODY_COLOR = (0.04, 0.52, 0.57)
AVAILABLE_TRACKER_COLOR = (1.0, 0.68, 0.02, 1.0)
RIGHT_FOOT_COLOR = (1.0, 0.0, 0.72, 1.0)
LEFT_FOOT_COLOR = (0.0, 0.78, 0.90, 1.0)
HARD_ACCENT = (0.96, 0.20, 0.06, 1.0)
FLUID_ACCENT = (0.04, 0.66, 0.42, 1.0)

TRACKER_RADIUS = 0.026
TRAJECTORY_RADIUS = 0.006
BOUNDARY_TRAJECTORY_RADIUS = 0.014
CAMERA_YFOV = math.radians(31.0)
CAMERA_FIT_PADDING_X = 1.08
CAMERA_FIT_PADDING_Y = 1.12


# region 数据契约


@dataclass(frozen=True)
class MotionArrays:
    """一路 Binary/FLUID 产物中绘图所需的逐帧数据。"""

    rotations_world: np.ndarray  # [T,24,3,3]
    root_yaw: np.ndarray  # [T]
    joints_world: np.ndarray  # [T,24,3]
    tracker_pos_world: np.ndarray  # [T,6,3]
    tracker_available: np.ndarray  # [T,6]
    stage_indices: np.ndarray  # [T]


@dataclass(frozen=True)
class HandballGhostInputs:
    """一张 Handball 残影图对应的严格配对输入。"""

    case_name: str
    sequence_label: str
    stage_labels: tuple[str, str]
    hard: MotionArrays
    fluid: MotionArrays
    source_path: Path
    amass_path: Path
    frame_start: int
    frame_end_exclusive: int
    transition_index: int
    transition_source_frame: int
    selected_indices: np.ndarray  # [9]
    changed_tracker_indices: np.ndarray
    focus_tracker_index: int
    hard_tracker_steps_cm: dict[int, float]
    fluid_tracker_steps_cm: dict[int, float]
    prereconnect_max_abs: float
    camera_yaw_offset_deg: float


@dataclass(frozen=True)
class DisplayGeometry:
    """Binary/FLUID 两行共享的舞台、相机和标注锚点。"""

    sequences: dict[str, SmplMeshSequence]
    tracker_positions: dict[str, np.ndarray]  # 每行 [9,6,3]
    tracker_available: np.ndarray  # [9,6]
    body_alphas: np.ndarray  # [9]
    changed_tracker_indices: np.ndarray
    focus_tracker_index: int
    camera: CameraSpec
    floor_y: float
    grid_center: np.ndarray
    grid_size: float
    stage_label_centers_px: np.ndarray  # [2]
    boundary_center_px: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "为 Handball 3→4 与 3→5 Tracker 重连输出两张独立 4K Binary/FLUID "
            "残影对比图。"
        )
    )
    paths = parser.add_argument_group("Handball ghost comparison paths")
    paths.add_argument("--three_to_four_json", required=True, type=Path)
    paths.add_argument("--three_to_five_json", required=True, type=Path)
    paths.add_argument("--amass_dir", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_dir", required=True, type=Path)
    return parser


def load_json_object(path: Path, label: str) -> tuple[Path, dict]:
    resolved = require_file(path, label)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 顶层必须为 object：{resolved}")
    return resolved, value


def load_motion_arrays(npz_path: Path) -> MotionArrays:
    path = require_file(npz_path, "Binary/FLUID NPZ")
    keys = (
        "deployed_rotations_world",
        "deployed_root_yaw",
        "deployed_joints_world",
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
        "deployed_joints_world": (frame_count, 24, 3),
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
        "deployed_joints_world",
        "tracker_pos_world",
    ):
        if not np.isfinite(values[key]).all():
            raise ValueError(f"{path.name}.{key} 含 NaN/Inf。")
    if values["tracker_available"].dtype != np.bool_:
        raise ValueError(f"{path.name}.tracker_available 必须为 bool。")
    return MotionArrays(
        rotations_world=np.asarray(values["deployed_rotations_world"], dtype=np.float32),
        root_yaw=np.asarray(values["deployed_root_yaw"], dtype=np.float32),
        joints_world=np.asarray(values["deployed_joints_world"], dtype=np.float32),
        tracker_pos_world=np.asarray(values["tracker_pos_world"], dtype=np.float32),
        tracker_available=np.asarray(values["tracker_available"], dtype=bool),
        stage_indices=np.asarray(values["stage_indices"], dtype=np.int64),
    )


def build_selected_indices(
    *,
    transition_index: int,
    frame_count: int,
) -> np.ndarray:
    """围绕切换点非均匀取 9 帧，并在 t=-1/0/1 处提高时间分辨率。"""

    selected = int(transition_index) + np.asarray(DISPLAY_OFFSETS, dtype=np.int64)
    if selected.shape != (DISPLAY_FRAME_COUNT,):
        raise RuntimeError("Handball 残影取帧数量与透明度契约不一致。")
    if np.any(selected < 0) or np.any(selected >= int(frame_count)):
        raise ValueError(
            f"切换索引 {transition_index} 的残影窗口越界：{selected.tolist()}"
        )
    if not np.all(np.diff(selected) > 0):
        raise RuntimeError("残影取帧必须严格递增。")
    return selected


def compute_boundary_tracker_steps_cm(
    arrays: MotionArrays,
    *,
    transition_index: int,
    tracker_indices: np.ndarray,
) -> dict[int, float]:
    """计算切换前一帧到切换帧的真实 Tracker 对应关节位移。"""

    index = int(transition_index)
    if index <= 0 or index >= arrays.joints_world.shape[0]:
        raise ValueError(f"transition_index 越界：{index}")
    result: dict[int, float] = {}
    for tracker_index in np.asarray(tracker_indices, dtype=np.int64).tolist():
        joint_index = int(TRACKER_TO_JOINT[tracker_index])
        delta = (
            arrays.joints_world[index, joint_index]
            - arrays.joints_world[index - 1, joint_index]
        )
        result[int(tracker_index)] = float(np.linalg.norm(delta) * 100.0)
    return result


def resolve_amass_path_from_source(source_path: Path, amass_dir: Path) -> Path:
    source = require_file(source_path, "source_path")
    parts = source.parts
    if "source_30hz" not in parts:
        raise ValueError(f"无法从 source_path 推导 AMASS 相对路径：{source}")
    marker = parts.index("source_30hz")
    relative = Path(*parts[marker + 1 :])
    return require_file(Path(amass_dir) / relative, "AMASS source")


def _validate_paired_arrays(
    *,
    hard: MotionArrays,
    fluid: MotionArrays,
    frame_count: int,
    transition_index: int,
    expected_after_count: int,
) -> np.ndarray:
    if hard.stage_indices.shape[0] != int(frame_count):
        raise ValueError("NPZ 帧数与 JSON frame 范围不一致。")
    if not np.array_equal(hard.stage_indices, fluid.stage_indices):
        raise ValueError("Binary/FLUID stage_indices 不一致。")
    if not np.array_equal(hard.tracker_available, fluid.tracker_available):
        raise ValueError("Binary/FLUID tracker_available 不一致。")
    if not np.allclose(
        hard.tracker_pos_world,
        fluid.tracker_pos_world,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("Binary/FLUID 原始 Tracker 位置不一致。")
    changes = np.flatnonzero(np.diff(hard.stage_indices) > 0) + 1
    if changes.shape != (1,) or int(changes[0]) != int(transition_index):
        raise ValueError("NPZ 必须在指定时刻恰好发生一次 Tracker 增加。")
    before = hard.tracker_available[transition_index - 1]
    after = hard.tracker_available[transition_index]
    changed = np.flatnonzero(after & ~before)
    if int(before.sum()) != 3 or int(after.sum()) != int(expected_after_count):
        raise ValueError(
            f"Tracker 数量应从 3 增加到 {expected_after_count}，"
            f"实际为 {int(before.sum())}→{int(after.sum())}。"
        )
    if changed.shape != (expected_after_count - 3,):
        raise ValueError("新增 Tracker 数量与目标点数不一致。")
    return np.asarray(changed, dtype=np.int64)


def _build_inputs(
    *,
    case_name: str,
    sequence_label: str,
    stage_labels: tuple[str, str],
    hard: MotionArrays,
    fluid: MotionArrays,
    source_path: Path,
    amass_path: Path,
    frame_start: int,
    frame_end_exclusive: int,
    transition_source_frame: int,
    expected_after_count: int,
    camera_yaw_offset_deg: float,
) -> HandballGhostInputs:
    frame_count = int(frame_end_exclusive) - int(frame_start)
    transition_index = int(transition_source_frame) - int(frame_start)
    changed = _validate_paired_arrays(
        hard=hard,
        fluid=fluid,
        frame_count=frame_count,
        transition_index=transition_index,
        expected_after_count=expected_after_count,
    )
    right_foot_index = int(tuple(TRACKER_NAMES).index("right_foot"))
    if right_foot_index not in changed.tolist():
        raise ValueError("Handball 主图要求新增 Tracker 中包含 right_foot。")
    selected = build_selected_indices(
        transition_index=transition_index,
        frame_count=frame_count,
    )
    prereconnect_max_abs = float(
        np.max(
            np.abs(
                hard.joints_world[:transition_index]
                - fluid.joints_world[:transition_index]
            )
        )
    )
    if prereconnect_max_abs > 1e-6:
        raise ValueError(
            "Binary/FLUID 在切换前不一致，不能作为严格配对对比："
            f"max_abs={prereconnect_max_abs}"
        )
    return HandballGhostInputs(
        case_name=case_name,
        sequence_label=sequence_label,
        stage_labels=stage_labels,
        hard=hard,
        fluid=fluid,
        source_path=require_file(source_path, "source_path"),
        amass_path=require_file(amass_path, "amass_path"),
        frame_start=int(frame_start),
        frame_end_exclusive=int(frame_end_exclusive),
        transition_index=int(transition_index),
        transition_source_frame=int(transition_source_frame),
        selected_indices=selected,
        changed_tracker_indices=changed,
        focus_tracker_index=right_foot_index,
        hard_tracker_steps_cm=compute_boundary_tracker_steps_cm(
            hard,
            transition_index=transition_index,
            tracker_indices=changed,
        ),
        fluid_tracker_steps_cm=compute_boundary_tracker_steps_cm(
            fluid,
            transition_index=transition_index,
            tracker_indices=changed,
        ),
        prereconnect_max_abs=prereconnect_max_abs,
        camera_yaw_offset_deg=float(camera_yaw_offset_deg),
    )


def load_three_to_four_inputs(
    final_json: Path,
    *,
    amass_dir: Path,
) -> HandballGhostInputs:
    _, final_report = load_json_object(final_json, "3→4 final JSON")
    inputs = final_report.get("inputs", {})
    hard_path = require_file(Path(str(inputs.get("hard_npz", ""))), "3→4 hard NPZ")
    fluid_path = require_file(Path(str(inputs.get("soft_npz", ""))), "3→4 FLUID NPZ")
    _, hard_report = load_json_object(hard_path.with_suffix(".json"), "3→4 hard JSON")
    _, fluid_report = load_json_object(
        fluid_path.with_suffix(".json"), "3→4 FLUID JSON"
    )
    if hard_report.get("tracker_counts") != [3, 4] or fluid_report.get(
        "tracker_counts"
    ) != [3, 4]:
        raise ValueError("3→4 输入的 tracker_counts 必须为 [3,4]。")
    transition = final_report.get("transition", {})
    transition_source_frame = int(transition.get("source_frame", -1))
    source_path = require_file(Path(str(hard_report["source_path"])), "3→4 source")
    amass_value = hard_report.get("amass_path")
    if isinstance(amass_value, str) and amass_value:
        amass_path = require_file(Path(amass_value), "3→4 AMASS")
    else:
        amass_path = resolve_amass_path_from_source(source_path, amass_dir)
    frame_start = int(hard_report["frame_start"])
    frame_end_exclusive = int(hard_report["frame_end_exclusive"])
    return _build_inputs(
        case_name="three_to_four",
        sequence_label="BMLhandball S08 · Trial upper-left 103",
        stage_labels=("3 trackers", "4 trackers"),
        hard=load_motion_arrays(hard_path),
        fluid=load_motion_arrays(fluid_path),
        source_path=source_path,
        amass_path=amass_path,
        frame_start=frame_start,
        frame_end_exclusive=frame_end_exclusive,
        transition_source_frame=transition_source_frame,
        expected_after_count=4,
        camera_yaw_offset_deg=0.0,
    )


def load_three_to_five_inputs(
    summary_json: Path,
    *,
    amass_dir: Path,
) -> HandballGhostInputs:
    _, report = load_json_object(summary_json, "3→5 summary JSON")
    hard_block = report.get("hard", {})
    fluid_block = report.get("soft10f", {})
    hard_path = require_file(Path(str(hard_block.get("npz", ""))), "3→5 hard NPZ")
    fluid_path = require_file(
        Path(str(fluid_block.get("npz", ""))), "3→5 FLUID NPZ"
    )
    source_path = require_file(Path(str(report["source_path"])), "3→5 source")
    amass_path = resolve_amass_path_from_source(source_path, amass_dir)
    return _build_inputs(
        case_name="three_to_five",
        sequence_label="BMLhandball S08 · Trial upper-right 197",
        stage_labels=("3 trackers", "5 trackers"),
        hard=load_motion_arrays(hard_path),
        fluid=load_motion_arrays(fluid_path),
        source_path=source_path,
        amass_path=amass_path,
        frame_start=int(report["frame_start"]),
        frame_end_exclusive=int(report["frame_end_exclusive"]),
        transition_source_frame=int(report["boundary_source_frame"]),
        expected_after_count=5,
        # 双脚恢复时从略偏侧面的方向观察，避免前后两条腿在投影中重合。
        camera_yaw_offset_deg=12.0,
    )


# endregion


# region SMPL-H 与展示舞台


def build_body_alphas(selected_indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.shape != (DISPLAY_FRAME_COUNT,):
        raise ValueError(f"透明度映射要求严格的 {DISPLAY_FRAME_COUNT} 帧序列。")
    alphas = np.asarray(BODY_RENDER_ALPHAS, dtype=np.float32)
    if alphas.shape != selected.shape:
        raise RuntimeError("人体透明度与残影帧数不一致。")
    return alphas


def build_selected_mesh_sequences(
    *,
    inputs: HandballGhostInputs,
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
            # 两行共享 GT root translation，避免把平移误差伪装成切换跳变。
            translation_amass=translations_amass,
        )
    selected_roots_unity = translations_amass @ AMASS_TO_UNITY.T
    return (
        sequences,
        transform_faces_to_unity_winding(model.faces),
        np.asarray(selected_roots_unity, dtype=np.float32),
    )


def rotated_view_direction(yaw_offset_deg: float) -> np.ndarray:
    direction = np.asarray(presentation_view_direction_unity(), dtype=np.float64)
    angle = math.radians(float(yaw_offset_deg))
    rotation_y = np.asarray(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=np.float64,
    )
    value = rotation_y @ direction
    return value / np.linalg.norm(value)


def fit_fixed_camera(points: np.ndarray, view_direction: np.ndarray) -> CameraSpec:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("相机拟合需要非空有限点集。")
    direction = np.asarray(view_direction, dtype=np.float64)
    mins = np.min(values, axis=0)
    maxs = np.max(values, axis=0)
    target = (mins + maxs) * 0.5
    rotation_pose = camera_pose_look_at(target + direction, target)
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
    eye = target + direction * distance
    return CameraSpec(
        pose=camera_pose_look_at(eye, target),
        target=np.asarray(target, dtype=np.float64),
        yfov=float(CAMERA_YFOV),
        aspect_ratio=float(aspect),
    )


def project_points_to_pixels(points: np.ndarray, camera: CameraSpec) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.concatenate(
        [values, np.ones((values.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera_points = (
        np.linalg.inv(np.asarray(camera.pose, dtype=np.float64)) @ homogeneous.T
    ).T[:, :3]
    depth = -camera_points[:, 2]
    if np.any(depth <= 1e-8):
        raise ValueError("投影点必须位于相机前方。")
    tan_half_y = math.tan(float(camera.yfov) * 0.5)
    tan_half_x = tan_half_y * float(camera.aspect_ratio)
    normalized_x = camera_points[:, 0] / (depth * tan_half_x)
    normalized_y = camera_points[:, 1] / (depth * tan_half_y)
    x = (normalized_x + 1.0) * 0.5 * float(RENDER_WIDTH)
    y = (1.0 - normalized_y) * 0.5 * float(ROW_HEIGHT)
    return np.stack([x, y], axis=-1).astype(np.float32)


def build_display_geometry(
    *,
    inputs: HandballGhostInputs,
    sequences: dict[str, SmplMeshSequence],
    selected_roots_unity: np.ndarray,
) -> DisplayGeometry:
    view_direction = rotated_view_direction(inputs.camera_yaw_offset_deg)
    provisional_pose = camera_pose_look_at(
        view_direction,
        np.zeros((3,), dtype=np.float64),
    )
    camera_right = np.asarray(provisional_pose[:3, 0], dtype=np.float64)
    time_coordinates = np.arange(DISPLAY_FRAME_COUNT, dtype=np.float64)
    time_coordinates *= TIME_PITCH_METERS
    # 只压缩 t=-1→0 的展示间距。Binary 会形成清晰双轮廓，FLUID 则保持
    # 连续；其余时刻保留足够间距，避免 Handball 前倾躯干糊成一片。
    time_coordinates[TRANSITION_SLOT:] -= (
        TIME_PITCH_METERS - BOUNDARY_TIME_PITCH_METERS
    )
    time_coordinates -= 0.5 * (
        float(time_coordinates[0]) + float(time_coordinates[-1])
    )
    time_offsets = -time_coordinates[:, None] * camera_right[None, :]

    display_sequences = {
        row_name: SmplMeshSequence(
            vertices_world=(
                np.asarray(sequence.vertices_world, dtype=np.float64)
                + time_offsets[:, None, :]
            ).astype(np.float32),
            joints_world=(
                np.asarray(sequence.joints_world, dtype=np.float64)
                + time_offsets[:, None, :]
            ).astype(np.float32),
        )
        for row_name, sequence in sequences.items()
    }
    tracker_joint_indices = np.asarray(TRACKER_TO_JOINT, dtype=np.int64)
    display_trackers = {
        row_name: np.asarray(
            sequence.joints_world[:, tracker_joint_indices], dtype=np.float32
        )
        for row_name, sequence in display_sequences.items()
    }
    display_roots = (
        np.asarray(selected_roots_unity, dtype=np.float64) + time_offsets
    )
    fit_parts = [
        np.asarray(sequence.vertices_world, dtype=np.float64).reshape(-1, 3)
        for sequence in display_sequences.values()
    ]
    for row_trackers in display_trackers.values():
        fit_parts.append(
            row_trackers[:, inputs.changed_tracker_indices].reshape(-1, 3)
        )
    fit_points = np.concatenate(fit_parts, axis=0)
    camera = fit_fixed_camera(fit_points, view_direction)

    stage_anchors = np.stack(
        [
            np.mean(display_roots[:TRANSITION_SLOT], axis=0),
            np.mean(display_roots[TRANSITION_SLOT:], axis=0),
        ],
        axis=0,
    )
    stage_anchors[:, 1] = float(np.max(fit_points[:, 1]))
    boundary_anchor = np.mean(
        display_roots[TRANSITION_SLOT - 1 : TRANSITION_SLOT + 1], axis=0
    )[None]
    boundary_anchor[:, 1] = float(np.max(fit_points[:, 1]))
    # 离屏结果会水平镜像，所有屏幕标注使用相同像素变换。
    stage_centers_px = RENDER_WIDTH - project_points_to_pixels(
        stage_anchors, camera
    )[:, 0]
    boundary_center_px = float(
        RENDER_WIDTH - project_points_to_pixels(boundary_anchor, camera)[0, 0]
    )
    if np.any(np.diff(stage_centers_px) <= 0.0):
        raise RuntimeError("Tracker 阶段投影未保持从左到右。")

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
        tracker_available=np.asarray(
            inputs.hard.tracker_available[inputs.selected_indices], dtype=bool
        ),
        body_alphas=build_body_alphas(inputs.selected_indices),
        changed_tracker_indices=np.asarray(
            inputs.changed_tracker_indices, dtype=np.int64
        ),
        focus_tracker_index=int(inputs.focus_tracker_index),
        camera=camera,
        floor_y=float(floor_y),
        grid_center=grid_center,
        grid_size=float(grid_size),
        stage_label_centers_px=np.asarray(stage_centers_px, dtype=np.float32),
        boundary_center_px=float(boundary_center_px),
    )


# endregion


# region 离屏渲染与排版


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


def tracker_trajectory_color(
    tracker_index: int,
) -> tuple[float, float, float, float]:
    """返回脚部运动轨迹颜色；Tracker 球本身始终使用统一黄色。"""

    name = str(TRACKER_NAMES[int(tracker_index)])
    if name == "left_foot":
        return LEFT_FOOT_COLOR
    if name == "right_foot":
        return RIGHT_FOOT_COLOR
    raise ValueError(f"仅脚部 Tracker 支持轨迹颜色：{name}")


def select_available_tracker_positions(
    tracker_positions: np.ndarray,
    tracker_available: np.ndarray,
    tracker_index: int,
    *,
    excluded_slots: tuple[int, ...] = (),
) -> np.ndarray:
    """按逐帧 availability 选出单个 Tracker 的展示位置。

    ``tracker_positions`` 为残影舞台中的预测关节位置 ``[T, N, 3]``，
    ``tracker_available`` 为对应输入 Tracker 的有效状态 ``[T, N]``。
    球仍贴在预测人体上以展示重建结果，但显隐必须服从真实输入状态，避免把
    尚未接入的脚部 Tracker 错画到 t=0 左侧。``excluded_slots`` 用于将
    跳变端点从黄色普通球中排除，随后以 Binary/FLUID 强调色原位替换。
    """

    positions = np.asarray(tracker_positions)
    available = np.asarray(tracker_available)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            "tracker_positions 必须为 [T,N,3]，"
            f"实际为 {positions.shape}。"
        )
    if available.shape != positions.shape[:2]:
        raise ValueError(
            "tracker_available 必须与 tracker_positions 的 [T,N] 对齐，"
            f"实际为 {available.shape} 与 {positions.shape[:2]}。"
        )
    index = int(tracker_index)
    if not 0 <= index < positions.shape[1]:
        raise IndexError(f"tracker_index 越界：{index}")
    slot_mask = np.asarray(available[:, index], dtype=bool).copy()
    for slot in excluded_slots:
        slot_index = int(slot)
        if not 0 <= slot_index < positions.shape[0]:
            raise IndexError(f"excluded slot 越界：{slot_index}")
        slot_mask[slot_index] = False
    return np.asarray(positions[slot_mask, index], dtype=np.float32)


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
    row_trackers = geometry.tracker_positions[row_name]
    changed = set(geometry.changed_tracker_indices.tolist())
    focus = int(geometry.focus_tracker_index)
    for tracker_index in range(row_trackers.shape[1]):
        active_points = select_available_tracker_positions(
            row_trackers,
            geometry.tracker_available,
            tracker_index,
        )
        if active_points.shape[0] == 0:
            continue
        marker_active_points = select_available_tracker_positions(
            row_trackers,
            geometry.tracker_available,
            tracker_index,
            excluded_slots=(TRANSITION_SLOT,) if tracker_index == focus else (),
        )
        # 普通可用 Tracker 使用黄色；被圈出的右脚 t=0 球会在后面以当前
        # 方法的红/绿色原位替换。球比轨迹略靠近相机，保证端点清楚可见。
        marker_points = build_visible_tracker_glyph_points(
            marker_active_points,
            camera_position,
            camera_offset=0.090,
        )
        # 只有本次新增的脚部 Tracker 需要轨迹；原有头和双手只用球标出。
        # active_points 已应用 availability，因此彩色轨迹从 t=0 才开始。
        if tracker_index in changed:
            trajectory_points = build_visible_tracker_glyph_points(
                active_points,
                camera_position,
                camera_offset=0.070,
            )
            trajectory = create_polyline_tube(
                trajectory_points,
                radius=TRAJECTORY_RADIUS,
            )
            if trajectory is not None:
                scene.add(
                    pyrender.Mesh.from_trimesh(
                        trajectory,
                        material=create_material(
                            pyrender,
                            tracker_trajectory_color(tracker_index),
                            0.48,
                        ),
                        smooth=True,
                    )
                )
        marker_cloud = create_sphere_cloud(marker_points, radius=TRACKER_RADIUS)
        if marker_cloud is not None:
            scene.add(
                pyrender.Mesh.from_trimesh(
                    marker_cloud,
                    material=create_material(
                        pyrender,
                        AVAILABLE_TRACKER_COLOR,
                        0.48,
                    ),
                    smooth=True,
                )
            )

    boundary_line_points = build_visible_tracker_glyph_points(
        row_trackers[TRANSITION_SLOT - 1 : TRANSITION_SLOT + 1, focus],
        camera_position,
        camera_offset=0.070,
    )
    accent = HARD_ACCENT if row_name == "hard" else FLUID_ACCENT
    accent_material = create_material(pyrender, accent, 0.42)
    boundary_trajectory = create_polyline_tube(
        boundary_line_points,
        radius=BOUNDARY_TRAJECTORY_RADIUS,
    )
    if boundary_trajectory is not None:
        scene.add(
            pyrender.Mesh.from_trimesh(
                boundary_trajectory,
                material=accent_material,
                smooth=True,
            )
        )
    boundary_marker_points = build_visible_tracker_glyph_points(
        row_trackers[TRANSITION_SLOT - 1 : TRANSITION_SLOT + 1, focus],
        camera_position,
        camera_offset=0.090,
    )
    boundary_cloud = create_sphere_cloud(
        boundary_marker_points,
        radius=TRACKER_RADIUS,
    )
    if boundary_cloud is not None:
        scene.add(
            pyrender.Mesh.from_trimesh(
                boundary_cloud,
                material=accent_material,
                smooth=True,
            )
        )
    renderer = pyrender.OffscreenRenderer(RENDER_WIDTH, ROW_HEIGHT)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
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
    for y in range(y_start, y_end, 28):
        draw.line(
            (x, y, x, min(y + 13, y_end)),
            fill=(128, 137, 151, 125),
            width=4,
        )


def boundary_callout_box(
    *,
    geometry: DisplayGeometry,
    row_name: str,
    row_y: int,
) -> tuple[int, int, int, int]:
    points = geometry.tracker_positions[row_name][
        TRANSITION_SLOT - 1 : TRANSITION_SLOT + 1,
        geometry.focus_tracker_index,
    ]
    pixels = project_points_to_pixels(points, geometry.camera)
    pixels[:, 0] = float(RENDER_WIDTH) - pixels[:, 0]
    pixels[:, 0] += float(SIDE_LABEL_WIDTH)
    pixels[:, 1] += float(row_y)
    center = np.mean(pixels, axis=0)
    half_extent = np.max(np.abs(pixels - center[None]), axis=0)
    radius_x = int(np.clip(half_extent[0] + 72.0, 86.0, 230.0))
    radius_y = int(np.clip(half_extent[1] + 72.0, 86.0, 210.0))
    return (
        int(round(center[0])) - radius_x,
        int(round(center[1])) - radius_y,
        int(round(center[0])) + radius_x,
        int(round(center[1])) + radius_y,
    )


def draw_stage_header(
    *,
    draw: ImageDraw.ImageDraw,
    inputs: HandballGhostInputs,
    geometry: DisplayGeometry,
) -> None:
    centers = SIDE_LABEL_WIDTH + np.rint(
        geometry.stage_label_centers_px
    ).astype(np.int64)
    for label, center_x in zip(inputs.stage_labels, centers):
        box = (int(center_x - 170), 18, int(center_x + 170), 88)
        draw.rounded_rectangle(
            box,
            radius=23,
            fill=(255, 255, 255, 238),
            outline=(205, 211, 220, 245),
            width=3,
        )
        draw_centered_text(
            draw,
            box,
            label,
            load_font(34),
            (31, 41, 55, 255),
        )
    arrow_start = int(centers[0] + 195)
    arrow_end = int(centers[1] - 195)
    if arrow_end > arrow_start + 20:
        arrow_y = 53
        draw.line(
            (arrow_start, arrow_y, arrow_end, arrow_y),
            fill=(128, 137, 151, 230),
            width=5,
        )
        draw.polygon(
            (
                (arrow_end, arrow_y),
                (arrow_end - 16, arrow_y - 11),
                (arrow_end - 16, arrow_y + 11),
            ),
            fill=(128, 137, 151, 230),
        )
    changed_names = [
        str(TRACKER_NAMES[index]).replace("_", " ").title()
        for index in inputs.changed_tracker_indices.tolist()
    ]
    draw_centered_text(
        draw,
        (SIDE_LABEL_WIDTH, 94, OUTPUT_WIDTH, 140),
        "+ " + " + ".join(changed_names),
        load_font(24),
        (91, 99, 112, 255),
    )


def draw_tracker_legend(draw: ImageDraw.ImageDraw) -> None:
    """明确普通 Tracker 与跳变端点的颜色语义。"""

    box = (28, 18, SIDE_LABEL_WIDTH - 28, 134)
    draw.rounded_rectangle(
        box,
        radius=22,
        fill=(255, 255, 255, 238),
        outline=(205, 211, 220, 245),
        width=3,
    )
    marker_rgba = tuple(
        int(round(value * 255.0)) for value in AVAILABLE_TRACKER_COLOR
    )
    draw.ellipse(
        (54, 38, 84, 68),
        fill=marker_rgba,
        outline=(202, 139, 0, 255),
        width=2,
    )
    draw_centered_text(
        draw,
        (96, 24, SIDE_LABEL_WIDTH - 40, 82),
        "Available tracker",
        load_font(21),
        (31, 41, 55, 255),
    )
    hard_rgba = tuple(int(round(value * 255.0)) for value in HARD_ACCENT)
    fluid_rgba = tuple(int(round(value * 255.0)) for value in FLUID_ACCENT)
    draw.ellipse((54, 87, 78, 111), fill=hard_rgba)
    draw.ellipse((72, 87, 96, 111), fill=fluid_rgba)
    draw_centered_text(
        draw,
        (106, 73, SIDE_LABEL_WIDTH - 40, 127),
        "Transition pair",
        load_font(21),
        (31, 41, 55, 255),
    )


def draw_side_label(
    *,
    draw: ImageDraw.ImageDraw,
    row_y: int,
    label: str,
    step_cm: float,
    accent: tuple[float, float, float, float],
) -> None:
    box = (
        28,
        row_y + ROW_HEIGHT // 2 - 94,
        SIDE_LABEL_WIDTH - 28,
        row_y + ROW_HEIGHT // 2 + 6,
    )
    draw.rounded_rectangle(
        box,
        radius=24,
        fill=(255, 255, 255, 238),
        outline=(205, 211, 220, 245),
        width=3,
    )
    draw_centered_text(
        draw,
        box,
        label,
        load_font(30),
        (31, 41, 55, 255),
    )
    text_box = (
        28,
        row_y + ROW_HEIGHT // 2 + 26,
        SIDE_LABEL_WIDTH - 28,
        row_y + ROW_HEIGHT // 2 + 122,
    )
    draw.rounded_rectangle(
        text_box,
        radius=20,
        fill=(255, 255, 255, 225),
        outline=tuple(int(round(value * 255.0)) for value in accent),
        width=4,
    )
    draw_centered_text(
        draw,
        text_box,
        f"Right-foot step at t=0\n{step_cm:.2f} cm",
        load_font(23),
        (31, 41, 55, 255),
    )


def compose_output_png(
    *,
    inputs: HandballGhostInputs,
    geometry: DisplayGeometry,
    hard_row: np.ndarray,
    fluid_row: np.ndarray,
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

    draw_stage_header(draw=draw, inputs=inputs, geometry=geometry)
    draw_tracker_legend(draw)
    boundary_x = SIDE_LABEL_WIDTH + int(round(geometry.boundary_center_px))
    for row_y in (top_y, bottom_y):
        draw_dashed_vertical_line(
            draw,
            x=boundary_x,
            y_start=row_y + 12,
            y_end=row_y + ROW_HEIGHT - 12,
        )
    draw.rounded_rectangle(
        (boundary_x - 70, HEADER_HEIGHT + 8, boundary_x + 70, HEADER_HEIGHT + 58),
        radius=17,
        fill=(255, 255, 255, 232),
        outline=(180, 187, 198, 235),
        width=2,
    )
    draw_centered_text(
        draw,
        (boundary_x - 70, HEADER_HEIGHT + 8, boundary_x + 70, HEADER_HEIGHT + 58),
        "t = 0",
        load_font(24),
        (64, 73, 86, 255),
    )

    hard_step = inputs.hard_tracker_steps_cm[inputs.focus_tracker_index]
    fluid_step = inputs.fluid_tracker_steps_cm[inputs.focus_tracker_index]
    draw_side_label(
        draw=draw,
        row_y=top_y,
        label="Binary availability",
        step_cm=hard_step,
        accent=HARD_ACCENT,
    )
    draw_side_label(
        draw=draw,
        row_y=bottom_y,
        label="FLUID",
        step_cm=fluid_step,
        accent=FLUID_ACCENT,
    )
    for row_name, row_y, accent in (
        ("hard", top_y, HARD_ACCENT),
        ("fluid", bottom_y, FLUID_ACCENT),
    ):
        callout = boundary_callout_box(
            geometry=geometry,
            row_name=row_name,
            row_y=row_y,
        )
        draw.ellipse(
            callout,
            outline=tuple(int(round(value * 255.0)) for value in accent),
            width=9,
        )

    divider_y = HEADER_HEIGHT + ROW_HEIGHT + ROW_GAP // 2
    draw.line(
        (28, divider_y, OUTPUT_WIDTH - 28, divider_y),
        fill=(205, 211, 220, 225),
        width=3,
    )
    footer_top = OUTPUT_HEIGHT - FOOTER_HEIGHT
    draw.rectangle(
        (0, footer_top, OUTPUT_WIDTH, OUTPUT_HEIGHT),
        fill=(255, 255, 255, 238),
    )
    draw.line(
        (0, footer_top, OUTPUT_WIDTH, footer_top),
        fill=(205, 211, 220, 230),
        width=3,
    )
    footer_text = (
        f"{inputs.sequence_label}  |  switch at source frame "
        f"{inputs.transition_source_frame}  |  shared camera and GT root translation  "
        "|  dense sampling around t=0"
    )
    draw_centered_text(
        draw,
        (30, footer_top + 5, OUTPUT_WIDTH - 30, OUTPUT_HEIGHT - 5),
        footer_text,
        load_font(25),
        (31, 41, 55, 255),
    )

    output = Path(output_png).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("output_png 必须使用 .png 后缀。")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    with Image.open(output) as written:
        if written.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
            raise RuntimeError(f"4K Handball 图尺寸错误：{written.size}")
    print(f"[handball-ghosts] wrote: {output}", flush=True)
    return output


def write_output_report(
    *,
    inputs: HandballGhostInputs,
    output_png: Path,
) -> Path:
    report = {
        "experiment": "handball_binary_fluid_ghost_comparison",
        "case": inputs.case_name,
        "sequence": inputs.sequence_label,
        "source_path": str(inputs.source_path),
        "amass_path": str(inputs.amass_path),
        "transition_source_frame": int(inputs.transition_source_frame),
        "selected_source_frames": (
            inputs.frame_start + inputs.selected_indices
        ).astype(int).tolist(),
        "tracker_counts": [
            int(inputs.hard.tracker_available[inputs.transition_index - 1].sum()),
            int(inputs.hard.tracker_available[inputs.transition_index].sum()),
        ],
        "changed_trackers": [
            str(TRACKER_NAMES[index])
            for index in inputs.changed_tracker_indices.tolist()
        ],
        "focus_tracker": str(TRACKER_NAMES[inputs.focus_tracker_index]),
        "boundary_tracker_step_cm": {
            "hard": {
                str(TRACKER_NAMES[index]): float(value)
                for index, value in inputs.hard_tracker_steps_cm.items()
            },
            "fluid": {
                str(TRACKER_NAMES[index]): float(value)
                for index, value in inputs.fluid_tracker_steps_cm.items()
            },
        },
        "prereconnect_hard_fluid_max_abs": float(inputs.prereconnect_max_abs),
        "render": {
            "resolution": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
            "body_count_per_row": DISPLAY_FRAME_COUNT,
            "camera_yaw_offset_deg": float(inputs.camera_yaw_offset_deg),
            "output_png": str(Path(output_png).resolve()),
        },
    }
    output = Path(output_png).with_suffix(".json").resolve()
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[handball-ghosts] report: {output}", flush=True)
    return output


def render_case(
    *,
    inputs: HandballGhostInputs,
    amass_dir: Path,
    smpl_model_dir: Path,
    output_png: Path,
) -> Path:
    sequences, faces, roots = build_selected_mesh_sequences(
        inputs=inputs,
        amass_dir=amass_dir,
        smpl_model_dir=smpl_model_dir,
    )
    geometry = build_display_geometry(
        inputs=inputs,
        sequences=sequences,
        selected_roots_unity=roots,
    )
    hard_row = render_row(row_name="hard", geometry=geometry, faces=faces)
    fluid_row = render_row(row_name="fluid", geometry=geometry, faces=faces)
    output = compose_output_png(
        inputs=inputs,
        geometry=geometry,
        hard_row=hard_row,
        fluid_row=fluid_row,
        output_png=output_png,
    )
    write_output_report(inputs=inputs, output_png=output)
    return output


# endregion


def main(argv: list[str] | None = None) -> list[Path]:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    amass_dir = require_directory(args.amass_dir, "amass_dir")
    smpl_model_dir = require_directory(args.smpl_model_dir, "smpl_model_dir")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = (
        load_three_to_four_inputs(
            args.three_to_four_json,
            amass_dir=amass_dir,
        ),
        load_three_to_five_inputs(
            args.three_to_five_json,
            amass_dir=amass_dir,
        ),
    )
    outputs = []
    for inputs in cases:
        suffix = "3to4" if inputs.case_name == "three_to_four" else "3to5"
        output_png = output_dir / f"handball_{suffix}_binary_vs_fluid_ghosts_4k.png"
        outputs.append(
            render_case(
                inputs=inputs,
                amass_dir=amass_dir,
                smpl_model_dir=smpl_model_dir,
                output_png=output_png,
            )
        )
    return outputs


if __name__ == "__main__":
    main()
