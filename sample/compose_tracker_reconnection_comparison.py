from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from data_loaders.sensor_masking import TRACKER_NAMES
from sample.realtime_pose_smpl_rendering import (
    SmplMeshSequence,
    body_fbx_world_to_smpl_local_rotations,
    create_smplh_model,
    create_sphere_cloud,
    create_static_scene,
    load_font,
    rotation_matrices_to_axis_angle,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)
from sample.render_progressive_tracker_dropout_sequences import (
    ProgressiveSequenceResult,
    load_progressive_mesh_inputs,
)
from sample.render_realtime_pose_smpl_presentation import (
    build_follow_camera_poses,
    build_horizontal_pelvis_follow_offsets,
    build_visible_tracker_glyph_points,
    create_material,
    fit_fixed_presentation_camera,
)
from utils.video_io import Mp4FrameWriter


OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
FULL_CELL_WIDTH = OUTPUT_WIDTH // 2
FULL_CELL_HEIGHT = 510
FULL_HEADER_HEIGHT = 54
FULL_VIEW_HEIGHT = FULL_CELL_HEIGHT - FULL_HEADER_HEIGHT
INLINE_OVERLAY_X = 160
INLINE_OVERLAY_Y = 80
INLINE_PANEL_WIDTH = 800
INLINE_HEADER_HEIGHT = 60
INLINE_VIEW_HEIGHT = 800
FOOTER_HEIGHT = 60
DEFAULT_INTRO_FRAMES = 15
DEFAULT_PRE_FRAMES = 15
DEFAULT_POST_FRAMES = 15
DEFAULT_SLOWDOWN_FACTOR = 3
PRERECONNECTION_ATOL = 1e-6

GT_COLOR = (0x90 / 255.0, 0xA9 / 255.0, 0xC2 / 255.0, 1.0)
PREDICTION_COLOR = (0x35 / 255.0, 0xB8 / 255.0, 0xA6 / 255.0, 1.0)
TRACKER_COLOR = (1.0, 0.55, 0.05, 1.0)
PANEL_ACCENTS = {
    "GT": (144, 169, 194),
    "Core3": (56, 189, 248),
    "Hard": (55, 65, 81),
    "Soft": (45, 184, 166),
}
PANEL_LABELS = {
    "GT": "Ground truth",
    "Core3": "Always 3 trackers",
    "Hard": "Hard reconnection",
    "Soft": "Soft reconnection",
}


# region 数据契约


@dataclass(frozen=True)
class ComparisonInputs:
    reports: dict[str, dict]
    results: dict[str, ProgressiveSequenceResult]
    source_path: Path
    amass_path: Path
    source_relative_path: str
    reconnect_tracker: str
    transition_frame: int
    soft_blend_frames: int
    fps: int
    prereconnection_max_abs: dict[str, float]


@dataclass
class RenderContext:
    renderer: object
    scene: object
    camera_node: object
    camera_poses: np.ndarray

    def close(self) -> None:
        with suppress(Exception):
            self.renderer.delete()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用已有 hard/core3/soft NPZ 重建共享 GT 的 2×2 科研展示，"
            "并在重连窗口内中央叠加硬/软双列慢放。"
        )
    )
    paths = parser.add_argument_group("comparison paths")
    paths.add_argument("--hard_npz", required=True, type=Path)
    paths.add_argument("--core3_npz", required=True, type=Path)
    paths.add_argument("--soft_npz", required=True, type=Path)
    paths.add_argument("--amass_dir", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_path", required=True, type=Path)
    replay = parser.add_argument_group("inline transition detail")
    replay.add_argument("--intro_frames", default=DEFAULT_INTRO_FRAMES, type=int)
    replay.add_argument("--pre_frames", default=DEFAULT_PRE_FRAMES, type=int)
    replay.add_argument("--post_frames", default=DEFAULT_POST_FRAMES, type=int)
    replay.add_argument(
        "--slowdown_factor", default=DEFAULT_SLOWDOWN_FACTOR, type=int
    )
    replay.add_argument("--overwrite", action="store_true")
    return parser


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} 不存在：{resolved}")
    return resolved


def _require_directory(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} 不存在：{resolved}")
    return resolved


def _load_report(npz_path: Path) -> dict:
    report_path = npz_path.with_suffix(".json")
    if not report_path.is_file():
        raise FileNotFoundError(f"NPZ 缺少同名 JSON sidecar：{report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _load_result(npz_path: Path, report: dict) -> ProgressiveSequenceResult:
    required = (
        "target_rotations_world",
        "target_joints_world",
        "deployed_rotations_world",
        "deployed_joints_world",
        "deployed_root_yaw",
        "tracker_pos_world",
        "runtime_tracker_pos_world",
        "runtime_tracker_rot_world_6d",
        "tracker_blend_alpha",
        "tracker_available",
        "stage_indices",
    )
    with np.load(npz_path, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"{npz_path} 缺少字段：{missing}")
        arrays = {key: np.asarray(payload[key]) for key in required}
    frame_count = int(arrays["stage_indices"].shape[0])
    frame_start = int(report["frame_start"])
    runtime_tracker_positions = arrays["runtime_tracker_pos_world"]
    runtime_tracker_rotations = arrays["runtime_tracker_rot_world_6d"]
    return ProgressiveSequenceResult(
        frame_start=frame_start,
        frame_end_exclusive=frame_start + frame_count,
        tracker_available=np.asarray(arrays["tracker_available"], dtype=bool),
        stage_indices=np.asarray(arrays["stage_indices"], dtype=np.int64),
        target_rotations=np.asarray(arrays["target_rotations_world"], dtype=np.float32),
        target_positions=np.asarray(arrays["target_joints_world"], dtype=np.float32),
        deployed_rotations=np.asarray(
            arrays["deployed_rotations_world"], dtype=np.float32
        ),
        deployed_positions=np.asarray(arrays["deployed_joints_world"], dtype=np.float32),
        deployed_root_yaw=np.asarray(arrays["deployed_root_yaw"], dtype=np.float32),
        tracker_positions=np.asarray(arrays["tracker_pos_world"], dtype=np.float32),
        runtime_tracker_positions=(
            None
            if runtime_tracker_positions.shape[0] == 0
            else np.asarray(runtime_tracker_positions, dtype=np.float32)
        ),
        runtime_tracker_rotations_6d=(
            None
            if runtime_tracker_rotations.shape[0] == 0
            else np.asarray(runtime_tracker_rotations, dtype=np.float32)
        ),
        tracker_blend_alpha=np.asarray(
            arrays["tracker_blend_alpha"], dtype=np.float32
        ),
        activation_blend_frames=int(report["activation_blend"]["frames"]),
    )


def _find_reconnection(result: ProgressiveSequenceResult) -> tuple[int, str]:
    changes = np.flatnonzero(np.diff(result.stage_indices) != 0) + 1
    if changes.shape != (1,) or tuple(np.unique(result.stage_indices)) != (0, 1):
        raise ValueError("硬/软重连结果必须恰好包含一次 3→4 阶段切换。")
    frame = int(changes[0])
    before = result.tracker_available[frame - 1]
    after = result.tracker_available[frame]
    added = np.flatnonzero(after & ~before)
    removed = np.flatnonzero(before & ~after)
    if int(before.sum()) != 3 or int(after.sum()) != 4 or added.size != 1 or removed.size:
        raise ValueError("重连边界必须只增加一个 Tracker，并严格从三点切换到四点。")
    return frame, str(TRACKER_NAMES[int(added[0])])


def load_comparison_inputs(
    *, hard_npz: Path, core3_npz: Path, soft_npz: Path
) -> ComparisonInputs:
    paths = {
        "hard": _require_file(hard_npz, "hard_npz"),
        "core3": _require_file(core3_npz, "core3_npz"),
        "soft": _require_file(soft_npz, "soft_npz"),
    }
    reports = {name: _load_report(path) for name, path in paths.items()}
    results = {
        name: _load_result(path, reports[name]) for name, path in paths.items()
    }
    source_relative_paths = {
        str(report.get("source_relative_path", "")) for report in reports.values()
    }
    source_paths = {str(report.get("source_path", "")) for report in reports.values()}
    amass_paths = {str(report.get("amass_path", "")) for report in reports.values()}
    if len(source_relative_paths) != 1 or not next(iter(source_relative_paths)):
        raise ValueError("三路 NPZ 必须来自同一条 source_relative_path。")
    if len(source_paths) != 1 or len(amass_paths) != 1:
        raise ValueError("三路 NPZ 的 source_path/amass_path 不一致。")
    if reports["hard"].get("tracker_counts") != [3, 4]:
        raise ValueError("hard_npz 不是合法的 3→4 重连结果。")
    if reports["core3"].get("tracker_counts") != [3]:
        raise ValueError("core3_npz 必须全程保持核心三点。")
    if reports["soft"].get("tracker_counts") != [3, 4]:
        raise ValueError("soft_npz 不是合法的 3→4 重连结果。")
    if results["hard"].activation_blend_frames != 0:
        raise ValueError("hard_npz 必须使用 activation_blend_frames=0。")
    if results["soft"].activation_blend_frames <= 0:
        raise ValueError("soft_npz 必须使用正数 activation_blend_frames。")
    if tuple(np.unique(results["core3"].stage_indices)) != (0,):
        raise ValueError("core3_npz 的 stage_indices 必须全程为 0。")

    hard_transition, hard_tracker = _find_reconnection(results["hard"])
    soft_transition, soft_tracker = _find_reconnection(results["soft"])
    if hard_transition != soft_transition or hard_tracker != soft_tracker:
        raise ValueError("硬重连和软重连必须在同一帧恢复同一个 Tracker。")
    frame_counts = {name: result.frame_count for name, result in results.items()}
    if len(set(frame_counts.values())) != 1:
        raise ValueError(f"三路计分帧数不一致：{frame_counts}")
    hard_joints = results["hard"].deployed_positions
    core_joints = results["core3"].deployed_positions
    soft_joints = results["soft"].deployed_positions
    hard_core = float(
        np.max(np.abs(hard_joints[:hard_transition] - core_joints[:hard_transition]))
    )
    hard_soft = float(
        np.max(np.abs(hard_joints[:hard_transition] - soft_joints[:hard_transition]))
    )
    if max(hard_core, hard_soft) > PRERECONNECTION_ATOL:
        raise ValueError(
            "重连前三路轨迹不一致："
            f"hard-core3={hard_core:.3e}, hard-soft={hard_soft:.3e}。"
        )
    fps_values = {int(round(float(report["fps"]))) for report in reports.values()}
    if len(fps_values) != 1:
        raise ValueError("三路 sidecar 的 FPS 不一致。")
    return ComparisonInputs(
        reports=reports,
        results=results,
        source_path=_require_file(Path(next(iter(source_paths))), "source_path"),
        amass_path=_require_file(Path(next(iter(amass_paths))), "amass_path"),
        source_relative_path=next(iter(source_relative_paths)),
        reconnect_tracker=hard_tracker,
        transition_frame=hard_transition,
        soft_blend_frames=results["soft"].activation_blend_frames,
        fps=next(iter(fps_values)),
        prereconnection_max_abs={
            "hard_vs_core3": hard_core,
            "hard_vs_soft": hard_soft,
        },
    )


# endregion


# region SMPL 与共享相机


def build_comparison_mesh_sequences(
    *, inputs: ComparisonInputs, amass_dir: Path, smpl_model_dir: Path
) -> tuple[dict[str, SmplMeshSequence], np.ndarray]:
    hard = inputs.results["hard"]
    mesh_inputs = load_progressive_mesh_inputs(
        source_path=inputs.source_path,
        amass_path=inputs.amass_path,
        amass_dir=amass_dir,
        frame_start=hard.frame_start,
        frame_end_exclusive=hard.frame_end_exclusive,
    )
    model = create_smplh_model(
        model_dir=smpl_model_dir,
        gender=mesh_inputs.gender,
        batch_size=hard.frame_count,
    )
    sequences = {
        "GT": run_smplh_forward(
            model=model,
            pose_axis_angle=mesh_inputs.gt_pose_axis_angle,
            betas=mesh_inputs.betas,
            translation_amass=mesh_inputs.gt_translation_amass,
        )
    }
    for panel_name, result_name in (("Core3", "core3"), ("Hard", "hard"), ("Soft", "soft")):
        result = inputs.results[result_name]
        predicted_local = body_fbx_world_to_smpl_local_rotations(
            result.deployed_rotations,
            result.deployed_root_yaw,
            mesh_inputs.rest_local_rotations,
            mesh_inputs.parents,
        )
        predicted_pose = rotation_matrices_to_axis_angle(predicted_local[:, :22])
        sequences[panel_name] = run_smplh_forward(
            model=model,
            pose_axis_angle=predicted_pose,
            betas=mesh_inputs.betas,
            # 所有方法共享 GT 根平移，只比较姿态恢复，不混入根位移差异。
            translation_amass=mesh_inputs.gt_translation_amass,
        )
    return sequences, transform_faces_to_unity_winding(model.faces)


def _slice_sequences(
    sequences: dict[str, SmplMeshSequence], frame_slice: slice
) -> dict[str, SmplMeshSequence]:
    return {
        name: SmplMeshSequence(
            vertices_world=np.asarray(sequence.vertices_world[frame_slice]),
            joints_world=np.asarray(sequence.joints_world[frame_slice]),
        )
        for name, sequence in sequences.items()
    }


def create_render_context(
    *,
    sequences: dict[str, SmplMeshSequence],
    method_names: tuple[str, ...],
    tracker_positions: np.ndarray,
    viewport_width: int,
    viewport_height: int,
    camera_padding: float,
) -> RenderContext:
    import pyrender

    selected = [sequences[name] for name in method_names]
    # 把四种方法的顶点并到同一 envelope，只用于拟合相机；每个面板实际仍单独渲染。
    envelope = SmplMeshSequence(
        vertices_world=np.concatenate(
            [np.asarray(sequence.vertices_world) for sequence in selected], axis=1
        ),
        joints_world=np.asarray(sequences["GT"].joints_world),
    )
    follow_offsets = build_horizontal_pelvis_follow_offsets(
        sequences["GT"].joints_world
    ).astype(np.float64)
    camera_spec = fit_fixed_presentation_camera(
        sequences={"Envelope": envelope},
        tracker_pos_world=np.asarray(tracker_positions, dtype=np.float64),
        follow_offsets=follow_offsets,
        method_offsets=np.zeros((1, 3), dtype=np.float64),
        viewport_width=int(viewport_width),
        viewport_height=int(viewport_height),
        padding=float(camera_padding),
        method_order=("Envelope",),
        tracker_available_by_method=np.ones((1, 6), dtype=bool),
    )
    camera_poses = build_follow_camera_poses(camera_spec.pose, follow_offsets)
    floor_y = min(
        float(np.min(sequences[name].vertices_world[..., 1])) for name in method_names
    )
    horizontal = np.concatenate(
        [
            np.asarray(sequences[name].vertices_world[..., [0, 2]]).reshape(-1, 2)
            for name in method_names
        ],
        axis=0,
    )
    horizontal_min = np.min(horizontal, axis=0)
    horizontal_max = np.max(horizontal, axis=0)
    horizontal_center = (horizontal_min + horizontal_max) * 0.5
    grid_center = np.asarray(
        [horizontal_center[0], floor_y, horizontal_center[1]], dtype=np.float64
    )
    grid_size = max(6.0, float(np.max(horizontal_max - horizontal_min)) + 3.0)
    scene, camera_node = create_static_scene(
        camera_spec,
        floor_y=floor_y,
        grid_size=grid_size,
        grid_center=grid_center,
    )
    renderer = pyrender.OffscreenRenderer(int(viewport_width), int(viewport_height))
    return RenderContext(
        renderer=renderer,
        scene=scene,
        camera_node=camera_node,
        camera_poses=np.asarray(camera_poses, dtype=np.float64),
    )


def render_method_view(
    *,
    context: RenderContext,
    sequence: SmplMeshSequence,
    faces: np.ndarray,
    frame_index: int,
    method_name: str,
    tracker_positions: np.ndarray | None,
    tracker_available: np.ndarray | None,
) -> np.ndarray:
    import pyrender
    import trimesh

    context.scene.set_pose(
        context.camera_node, pose=np.asarray(context.camera_poses[frame_index])
    )
    nodes = []
    try:
        vertices = np.asarray(sequence.vertices_world[frame_index], dtype=np.float64)
        body = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        body_color = GT_COLOR if method_name == "GT" else PREDICTION_COLOR
        nodes.append(
            context.scene.add(
                pyrender.Mesh.from_trimesh(
                    body,
                    material=create_material(pyrender, body_color, 0.92),
                    smooth=True,
                )
            )
        )
        if tracker_positions is not None and tracker_available is not None:
            mask = np.asarray(tracker_available[frame_index], dtype=bool)
            if np.any(mask):
                points = build_visible_tracker_glyph_points(
                    np.asarray(tracker_positions[frame_index], dtype=np.float64),
                    np.asarray(context.camera_poses[frame_index])[:3, 3],
                )
                tracker_cloud = create_sphere_cloud(points[mask], radius=0.035)
                if tracker_cloud is not None:
                    nodes.append(
                        context.scene.add(
                            pyrender.Mesh.from_trimesh(
                                tracker_cloud,
                                material=create_material(pyrender, TRACKER_COLOR, 0.48),
                                smooth=True,
                            )
                        )
                    )
        color, _ = context.renderer.render(
            context.scene, flags=pyrender.RenderFlags.NONE
        )
        return np.asarray(color[..., :3], dtype=np.uint8)
    finally:
        for node in nodes:
            context.scene.remove_node(node)


# endregion


# region 画面排版


def _tracker_positions(result: ProgressiveSequenceResult) -> np.ndarray:
    # 橙色球代表物理 Tracker 的原始测量；Soft 插值只作为 runtime 内部条件，
    # 否则画面会误导为 Tracker 本身逐渐移动到了预测关节位置。
    return result.tracker_positions


def _frame_mpjpe_cm(result: ProgressiveSequenceResult, frame_index: int) -> float:
    return float(
        np.linalg.norm(
            result.deployed_positions[frame_index] - result.target_positions[frame_index],
            axis=-1,
        ).mean()
        * 100.0
    )


def compose_full_cell(
    *,
    viewport_rgb: np.ndarray,
    method_name: str,
    tracker_count: int | None,
    mpjpe_cm: float | None,
    soft_blend_frames: int,
) -> Image.Image:
    cell = Image.new("RGB", (FULL_CELL_WIDTH, FULL_CELL_HEIGHT), (246, 248, 251))
    cell.paste(Image.fromarray(viewport_rgb), (0, FULL_HEADER_HEIGHT))
    draw = ImageDraw.Draw(cell)
    accent = PANEL_ACCENTS[method_name]
    draw.rectangle((0, 0, FULL_CELL_WIDTH, FULL_HEADER_HEIGHT), fill=(255, 255, 255))
    draw.rectangle((0, 0, 8, FULL_HEADER_HEIGHT), fill=accent)
    label = PANEL_LABELS[method_name]
    if method_name == "Soft":
        label += f" ({soft_blend_frames}f blend)"
    draw.text((24, 13), label, font=load_font(22), fill=accent)
    if method_name == "GT":
        detail = "Shared reference motion"
    else:
        detail = f"{tracker_count} trackers   |   MPJPE {mpjpe_cm:.2f} cm"
    detail_box = draw.textbbox((0, 0), detail, font=load_font(16))
    draw.text(
        (FULL_CELL_WIDTH - int(detail_box[2] - detail_box[0]) - 22, 17),
        detail,
        font=load_font(16),
        fill=(75, 85, 99),
    )
    draw.rectangle(
        (0, 0, FULL_CELL_WIDTH - 1, FULL_CELL_HEIGHT - 1),
        outline=(207, 213, 221),
        width=2,
    )
    return cell


def draw_shared_footer(
    *,
    image: Image.Image,
    frame_index: int,
    frame_count: int,
    frame_start: int,
    transition_frame: int,
    fps: int,
    slow_motion: bool,
    slowdown_factor: int = 1,
) -> None:
    draw = ImageDraw.Draw(image)
    footer_top = OUTPUT_HEIGHT - FOOTER_HEIGHT
    draw.rectangle((0, footer_top, OUTPUT_WIDTH, OUTPUT_HEIGHT), fill=(255, 255, 255))
    draw.line((0, footer_top, OUTPUT_WIDTH, footer_top), fill=(207, 213, 221), width=2)
    timeline_left, timeline_right = 32, OUTPUT_WIDTH - 32
    timeline_y = footer_top + 12
    transition_x = timeline_left + int(
        round(transition_frame / frame_count * (timeline_right - timeline_left))
    )
    draw.rectangle(
        (timeline_left, timeline_y, transition_x, timeline_y + 7),
        fill=(56, 189, 248),
    )
    draw.rectangle(
        (transition_x, timeline_y, timeline_right, timeline_y + 7),
        fill=(45, 184, 166),
    )
    cursor_x = timeline_left + int(
        round(frame_index / max(1, frame_count - 1) * (timeline_right - timeline_left))
    )
    draw.ellipse(
        (cursor_x - 5, timeline_y - 4, cursor_x + 5, timeline_y + 11),
        fill=(31, 41, 55),
    )
    relative_seconds = (frame_index - transition_frame) / float(fps)
    phase = (
        f"TRANSITION DETAIL · {1.0 / float(slowdown_factor):.2f}×"
        if slow_motion
        else "FULL SEQUENCE · 1.00×"
    )
    detail = (
        f"source frame {frame_start + frame_index}   |   "
        f"t = {relative_seconds:+.2f} s   |   t = 0: tracker reconnects"
    )
    draw.text((32, footer_top + 30), phase, font=load_font(17), fill=(45, 184, 166))
    draw.text((310, footer_top + 30), detail, font=load_font(17), fill=(55, 65, 81))


def compose_full_frame(
    *,
    views: dict[str, np.ndarray],
    inputs: ComparisonInputs,
    frame_index: int,
) -> np.ndarray:
    canvas = Image.new("RGB", (OUTPUT_WIDTH, OUTPUT_HEIGHT), (246, 248, 251))
    cells = {
        "GT": compose_full_cell(
            viewport_rgb=views["GT"],
            method_name="GT",
            tracker_count=None,
            mpjpe_cm=None,
            soft_blend_frames=inputs.soft_blend_frames,
        ),
        "Core3": compose_full_cell(
            viewport_rgb=views["Core3"],
            method_name="Core3",
            tracker_count=3,
            mpjpe_cm=_frame_mpjpe_cm(inputs.results["core3"], frame_index),
            soft_blend_frames=inputs.soft_blend_frames,
        ),
        "Hard": compose_full_cell(
            viewport_rgb=views["Hard"],
            method_name="Hard",
            tracker_count=int(inputs.results["hard"].tracker_available[frame_index].sum()),
            mpjpe_cm=_frame_mpjpe_cm(inputs.results["hard"], frame_index),
            soft_blend_frames=inputs.soft_blend_frames,
        ),
        "Soft": compose_full_cell(
            viewport_rgb=views["Soft"],
            method_name="Soft",
            tracker_count=int(inputs.results["soft"].tracker_available[frame_index].sum()),
            mpjpe_cm=_frame_mpjpe_cm(inputs.results["soft"], frame_index),
            soft_blend_frames=inputs.soft_blend_frames,
        ),
    }
    canvas.paste(cells["GT"], (0, 0))
    canvas.paste(cells["Core3"], (FULL_CELL_WIDTH, 0))
    canvas.paste(cells["Hard"], (0, FULL_CELL_HEIGHT))
    canvas.paste(cells["Soft"], (FULL_CELL_WIDTH, FULL_CELL_HEIGHT))
    draw_shared_footer(
        image=canvas,
        frame_index=frame_index,
        frame_count=inputs.results["hard"].frame_count,
        frame_start=inputs.results["hard"].frame_start,
        transition_frame=inputs.transition_frame,
        fps=inputs.fps,
        slow_motion=False,
    )
    return np.asarray(canvas, dtype=np.uint8)


def _draw_panel_header_text(
    draw: ImageDraw.ImageDraw,
    *,
    panel_x: int,
    text: str,
) -> None:
    font = load_font(23)
    box = draw.textbbox((0, 0), text, font=font)
    text_width = int(box[2] - box[0])
    draw.text(
        (
            panel_x + (INLINE_PANEL_WIDTH - text_width) // 2,
            INLINE_OVERLAY_Y + 17,
        ),
        text,
        font=font,
        fill=(255, 255, 255),
    )


def compose_inline_transition_frame(
    *,
    full_frame_rgb: np.ndarray,
    hard_view: np.ndarray,
    soft_view: np.ndarray,
    inputs: ComparisonInputs,
    scored_frame_index: int,
    slowdown_factor: int,
) -> np.ndarray:
    image = Image.fromarray(full_frame_rgb).convert("RGB")
    content_box = (0, 0, OUTPUT_WIDTH, OUTPUT_HEIGHT - FOOTER_HEIGHT)
    content = image.crop(content_box)
    dim_color = Image.new("RGB", content.size, (31, 41, 55))
    image.paste(Image.blend(content, dim_color, 0.45), (0, 0))

    overlay_width = INLINE_PANEL_WIDTH * 2
    overlay_height = INLINE_HEADER_HEIGHT + INLINE_VIEW_HEIGHT
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (
            INLINE_OVERLAY_X - 10,
            INLINE_OVERLAY_Y - 10,
            INLINE_OVERLAY_X + overlay_width + 10,
            INLINE_OVERLAY_Y + overlay_height + 10,
        ),
        radius=10,
        fill=(18, 24, 33),
        outline=(255, 255, 255),
        width=2,
    )
    image.paste(
        Image.fromarray(hard_view),
        (INLINE_OVERLAY_X, INLINE_OVERLAY_Y + INLINE_HEADER_HEIGHT),
    )
    image.paste(
        Image.fromarray(soft_view),
        (
            INLINE_OVERLAY_X + INLINE_PANEL_WIDTH,
            INLINE_OVERLAY_Y + INLINE_HEADER_HEIGHT,
        ),
    )
    draw.rectangle(
        (
            INLINE_OVERLAY_X,
            INLINE_OVERLAY_Y,
            INLINE_OVERLAY_X + INLINE_PANEL_WIDTH,
            INLINE_OVERLAY_Y + INLINE_HEADER_HEIGHT,
        ),
        fill=(55, 65, 81),
    )
    draw.rectangle(
        (
            INLINE_OVERLAY_X + INLINE_PANEL_WIDTH,
            INLINE_OVERLAY_Y,
            INLINE_OVERLAY_X + overlay_width,
            INLINE_OVERLAY_Y + INLINE_HEADER_HEIGHT,
        ),
        fill=(45, 184, 166),
    )
    _draw_panel_header_text(
        draw,
        panel_x=INLINE_OVERLAY_X,
        text="HARD RECONNECTION · INSTANT",
    )
    _draw_panel_header_text(
        draw,
        panel_x=INLINE_OVERLAY_X + INLINE_PANEL_WIDTH,
        text=f"SOFT RECONNECTION · {inputs.soft_blend_frames}-FRAME BLEND",
    )
    draw.rectangle(
        (
            INLINE_OVERLAY_X + INLINE_PANEL_WIDTH - 2,
            INLINE_OVERLAY_Y + INLINE_HEADER_HEIGHT,
            INLINE_OVERLAY_X + INLINE_PANEL_WIDTH + 2,
            INLINE_OVERLAY_Y + overlay_height,
        ),
        fill=(207, 213, 221),
    )
    draw_shared_footer(
        image=image,
        frame_index=scored_frame_index,
        frame_count=inputs.results["hard"].frame_count,
        frame_start=inputs.results["hard"].frame_start,
        transition_frame=inputs.transition_frame,
        fps=inputs.fps,
        slow_motion=True,
        slowdown_factor=slowdown_factor,
    )
    return np.asarray(image, dtype=np.uint8)


# endregion


# region 渲染主流程


def render_comparison_video(
    *,
    inputs: ComparisonInputs,
    sequences: dict[str, SmplMeshSequence],
    faces: np.ndarray,
    output_path: Path,
    intro_frames: int,
    pre_frames: int,
    post_frames: int,
    slowdown_factor: int,
) -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    hard = inputs.results["hard"]
    replay_start = inputs.transition_frame - int(pre_frames)
    replay_end = inputs.transition_frame + int(post_frames)
    replay_slice = slice(replay_start, replay_end)
    replay_sequences = _slice_sequences(sequences, replay_slice)
    full_context = create_render_context(
        sequences=sequences,
        method_names=("GT", "Core3", "Hard", "Soft"),
        tracker_positions=hard.tracker_positions,
        viewport_width=FULL_CELL_WIDTH,
        viewport_height=FULL_VIEW_HEIGHT,
        camera_padding=1.28,
    )
    replay_context = create_render_context(
        sequences=replay_sequences,
        method_names=("Hard", "Soft"),
        tracker_positions=hard.tracker_positions[replay_slice],
        viewport_width=INLINE_PANEL_WIDTH,
        viewport_height=INLINE_VIEW_HEIGHT,
        camera_padding=1.16,
    )
    hard_replay_trackers = _tracker_positions(hard)[replay_slice]
    soft = inputs.results["soft"]
    soft_replay_trackers = _tracker_positions(soft)[replay_slice]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Mp4FrameWriter | None = None
    try:
        for frame_index in range(hard.frame_count):
            views = {
                "GT": render_method_view(
                    context=full_context,
                    sequence=sequences["GT"],
                    faces=faces,
                    frame_index=frame_index,
                    method_name="GT",
                    tracker_positions=None,
                    tracker_available=None,
                )
            }
            for panel_name, result_name in (("Core3", "core3"), ("Hard", "hard"), ("Soft", "soft")):
                result = inputs.results[result_name]
                views[panel_name] = render_method_view(
                    context=full_context,
                    sequence=sequences[panel_name],
                    faces=faces,
                    frame_index=frame_index,
                    method_name=panel_name,
                    tracker_positions=_tracker_positions(result),
                    tracker_available=result.tracker_available,
                )
            full_frame_rgb = compose_full_frame(
                views=views, inputs=inputs, frame_index=frame_index
            )
            if writer is None:
                writer = Mp4FrameWriter(output_path, full_frame_rgb, inputs.fps)
                for _ in range(int(intro_frames)):
                    writer.append(full_frame_rgb)

            if replay_start <= frame_index < replay_end:
                local_index = frame_index - replay_start
                hard_view = render_method_view(
                    context=replay_context,
                    sequence=replay_sequences["Hard"],
                    faces=faces,
                    frame_index=local_index,
                    method_name="Hard",
                    tracker_positions=hard_replay_trackers,
                    tracker_available=hard.tracker_available[replay_slice],
                )
                soft_view = render_method_view(
                    context=replay_context,
                    sequence=replay_sequences["Soft"],
                    faces=faces,
                    frame_index=local_index,
                    method_name="Soft",
                    tracker_positions=soft_replay_trackers,
                    tracker_available=soft.tracker_available[replay_slice],
                )
                frame_rgb = compose_inline_transition_frame(
                    full_frame_rgb=full_frame_rgb,
                    hard_view=hard_view,
                    soft_view=soft_view,
                    inputs=inputs,
                    scored_frame_index=frame_index,
                    slowdown_factor=slowdown_factor,
                )
                for _ in range(int(slowdown_factor)):
                    writer.append(frame_rgb)
            else:
                writer.append(full_frame_rgb)
            if frame_index % 15 == 0 or frame_index + 1 == hard.frame_count:
                print(
                    f"[reconnection-grid] timeline {frame_index + 1}/{hard.frame_count}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        full_context.close()
        replay_context.close()
    print(f"[reconnection-grid] wrote {output_path}", flush=True)


def compose_tracker_reconnection_comparison(
    *,
    hard_npz: Path,
    core3_npz: Path,
    soft_npz: Path,
    amass_dir: Path,
    smpl_model_dir: Path,
    output_path: Path,
    intro_frames: int = DEFAULT_INTRO_FRAMES,
    pre_frames: int = DEFAULT_PRE_FRAMES,
    post_frames: int = DEFAULT_POST_FRAMES,
    slowdown_factor: int = DEFAULT_SLOWDOWN_FACTOR,
    overwrite: bool = False,
) -> dict:
    amass_dir = _require_directory(amass_dir, "amass_dir")
    smpl_model_dir = _require_directory(smpl_model_dir, "smpl_model_dir")
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("output_path 必须使用 .mp4 后缀。")
    if output_path.exists() and not bool(overwrite):
        raise FileExistsError(f"output_path 已存在；如需覆盖请传 --overwrite：{output_path}")
    if int(intro_frames) < 0:
        raise ValueError("intro_frames 不能为负数。")
    if int(pre_frames) <= 0 or int(post_frames) <= 0:
        raise ValueError("pre_frames 和 post_frames 必须为正整数。")
    if int(slowdown_factor) <= 1:
        raise ValueError("slowdown_factor 必须是大于 1 的整数。")

    inputs = load_comparison_inputs(
        hard_npz=hard_npz, core3_npz=core3_npz, soft_npz=soft_npz
    )
    replay_start = inputs.transition_frame - int(pre_frames)
    replay_end = inputs.transition_frame + int(post_frames)
    if replay_start < 0 or replay_end > inputs.results["hard"].frame_count:
        raise ValueError(
            f"慢放窗口 [{replay_start}, {replay_end}) 越出计分区间。"
        )
    sequences, faces = build_comparison_mesh_sequences(
        inputs=inputs,
        amass_dir=amass_dir,
        smpl_model_dir=smpl_model_dir,
    )
    render_comparison_video(
        inputs=inputs,
        sequences=sequences,
        faces=faces,
        output_path=output_path,
        intro_frames=int(intro_frames),
        pre_frames=int(pre_frames),
        post_frames=int(post_frames),
        slowdown_factor=int(slowdown_factor),
    )
    replay_source_frames = int(pre_frames) + int(post_frames)
    expected_frames = (
        int(intro_frames)
        + inputs.results["hard"].frame_count
        - replay_source_frames
        + replay_source_frames * int(slowdown_factor)
    )
    actual_frames, duration_seconds = imageio_ffmpeg.count_frames_and_secs(
        str(output_path)
    )
    reader = imageio_ffmpeg.read_frames(str(output_path), pix_fmt="rgb24")
    try:
        video_metadata = dict(next(reader))
    finally:
        reader.close()
    if int(actual_frames) != expected_frames:
        raise RuntimeError(
            f"最终视频帧数错误：expected={expected_frames}, actual={actual_frames}。"
        )
    if tuple(video_metadata["size"]) != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        raise RuntimeError(f"最终视频分辨率错误：{video_metadata['size']}。")
    report = {
        "experiment": "tracker_reconnection_shared_gt_grid_with_inline_transition_detail",
        "source_relative_path": inputs.source_relative_path,
        "inputs": {
            "hard_npz": str(Path(hard_npz).expanduser().resolve()),
            "core3_npz": str(Path(core3_npz).expanduser().resolve()),
            "soft_npz": str(Path(soft_npz).expanduser().resolve()),
        },
        "layout": {
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "full_sequence": {
                "grid": [2, 2],
                "panel_order": ["GT", "always_core_three", "hard", "soft"],
                "shared_gt": True,
                "shared_camera": True,
            },
            "transition_detail": {
                "mode": "center_overlay",
                "panel_order": ["hard", "soft"],
                "overlay_xywh": [
                    INLINE_OVERLAY_X,
                    INLINE_OVERLAY_Y,
                    INLINE_PANEL_WIDTH * 2,
                    INLINE_HEADER_HEIGHT + INLINE_VIEW_HEIGHT,
                ],
                "background": "dimmed_full_grid",
                "gt_included": False,
                "core3_included": False,
                "shared_camera": True,
            },
        },
        "transition": {
            "reconnect_tracker": inputs.reconnect_tracker,
            "scored_frame_index": inputs.transition_frame,
            "source_frame": inputs.results["hard"].frame_start
            + inputs.transition_frame,
            "soft_blend_frames": inputs.soft_blend_frames,
        },
        "inline_slow_motion": {
            "pre_frames": int(pre_frames),
            "post_frames": int(post_frames),
            "scored_frame_range": [replay_start, replay_end],
            "slowdown_factor": int(slowdown_factor),
            "playback_speed": 1.0 / float(slowdown_factor),
            "uses_interpolation": False,
            "source_frames": replay_source_frames,
            "output_frames": replay_source_frames * int(slowdown_factor),
        },
        "sections": [
            {
                "name": "full_sequence_intro",
                "frames": int(intro_frames),
                "playback_speed": 1.0,
            },
            {
                "name": "full_before_transition",
                "frames": replay_start,
                "playback_speed": 1.0,
            },
            {
                "name": "inline_transition_detail",
                "frames": replay_source_frames * int(slowdown_factor),
                "playback_speed": 1.0 / float(slowdown_factor),
            },
            {
                "name": "full_after_transition",
                "frames": inputs.results["hard"].frame_count - replay_end,
                "playback_speed": 1.0,
            },
        ],
        "fairness_check": {
            "prereconnection_tolerance": PRERECONNECTION_ATOL,
            "deployed_joints_world_max_abs": inputs.prereconnection_max_abs,
        },
        "fps": inputs.fps,
        "frames": int(actual_frames),
        "duration_seconds": float(duration_seconds),
        "video_path": str(output_path),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[reconnection-grid] wrote {report_path}", flush=True)
    return report


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    return compose_tracker_reconnection_comparison(
        hard_npz=args.hard_npz,
        core3_npz=args.core3_npz,
        soft_npz=args.soft_npz,
        amass_dir=args.amass_dir,
        smpl_model_dir=args.smpl_model_dir,
        output_path=args.output_path,
        intro_frames=int(args.intro_frames),
        pre_frames=int(args.pre_frames),
        post_frames=int(args.post_frames),
        slowdown_factor=int(args.slowdown_factor),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()


# endregion
