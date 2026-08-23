from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from data_converter.amass_smpl_utils import SOURCE_BODY_JOINT_COUNT
from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    load_realtime_source,
)
from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from data_loaders.sensor_masking import (
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    STATIC_OPTIONAL_TRACKER_MASKS,
)
from eval.realtime_pose_metrics import compute_rpm_p2_mc_metrics
from sample.realtime_pose_smpl_rendering import (
    SmplMeshSequence,
    body_fbx_world_to_smpl_local_rotations,
    create_front_marker_mesh,
    create_smplh_model,
    create_sphere_cloud,
    create_static_scene,
    decode_png,
    encode_png,
    load_comparison_clip,
    load_font,
    normalize_vector,
    require_directory,
    rotation_matrices_to_axis_angle,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)
from sample.render_realtime_pose_smpl_presentation import (
    INTRO_FRAME_COUNT,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    TRACKER_COLOR,
    build_presentation_frame_schedule,
    build_presentation_layout,
    build_visible_tracker_glyph_points,
    create_material,
    draw_centered_text,
    rgba_color,
)
from sample.utils import load_checkpoint_model
from sample.visualize_realtime_pose_sequences import run_dit_sequence
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_realtime_pose_predictor
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import (
    add_base_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    parse_and_load_from_model,
    str2bool,
)
from utils.video_io import Mp4FrameWriter


METHOD_ORDER = ("3 trackers", "4 trackers", "5 trackers", "6 trackers")
METHOD_CONFIG_NAMES = (
    "core_only",
    "core_hip",
    "core_hip_right_foot",
    "all_six",
)
METHOD_COLORS = {
    "3 trackers": (0x9A / 255.0, 0xA6 / 255.0, 0xB2 / 255.0, 1.0),
    "4 trackers": (0x90 / 255.0, 0xA9 / 255.0, 0xC2 / 255.0, 1.0),
    "5 trackers": (0xC8 / 255.0, 0x92 / 255.0, 0x92 / 255.0, 1.0),
    "6 trackers": (0x59 / 255.0, 0xB9 / 255.0, 0xB7 / 255.0, 1.0),
}
METHOD_SENSOR_LABELS = {
    "3 trackers": "Head + wrists",
    "4 trackers": "+ Hip",
    "5 trackers": "+ Right foot",
    "6 trackers": "+ Left foot",
}
CHEST_MARKER_COLOR = (1.0, 0.88, 0.18, 1.0)

# 这条链路刻意保持嵌套：三点是 Head/双腕，之后依次增加 Hip、右脚和
# 左脚。这样相邻方法之间只变化一个 availability bit。
TRACKER_AVAILABLE_BY_METHOD = (
    STATIC_OPTIONAL_TRACKER_MASKS[0],
    STATIC_OPTIONAL_TRACKER_MASKS[1],
    STATIC_OPTIONAL_TRACKER_MASKS[6],
    STATIC_OPTIONAL_TRACKER_MASKS[7],
)


@dataclass(frozen=True)
class TrackerCountMetrics:
    mpjpe_cm: float
    mpjve_cm_per_s: float
    jitter_m_per_s3: float


# region CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one realtime-pose checkpoint with nested 3/4/5/6 tracker masks "
            "and render a shared-stage SMPL-H comparison."
        )
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    paths = parser.add_argument_group("tracker-count comparison paths")
    paths.add_argument("--comparison_npz", required=True, type=Path)
    paths.add_argument("--report_json", required=True, type=Path)
    paths.add_argument("--amass_npz", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--normalizer_dir", required=True)
    paths.add_argument("--normalize_input", default=True, type=str2bool)
    paths.add_argument("--output_mp4", required=True, type=Path)
    clip = parser.add_argument_group("tracker-count comparison clip")
    clip.add_argument("--source_frame_start", required=True, type=int)
    clip.add_argument("--source_frame_end_exclusive", required=True, type=int)
    return parser


# endregion


# region Tracker 配置与推理


def validate_tracker_count_configuration() -> None:
    """保证展示配置确实是 3→4→5→6 的逐点嵌套关系。"""

    masks = np.asarray(TRACKER_AVAILABLE_BY_METHOD, dtype=bool)
    if masks.shape != (len(METHOD_ORDER), 6):
        raise ValueError(
            f"TRACKER_AVAILABLE_BY_METHOD 应为 {(len(METHOD_ORDER), 6)}，"
            f"实际为 {masks.shape}"
        )
    if tuple(int(mask.sum()) for mask in masks) != (3, 4, 5, 6):
        raise ValueError("四种配置必须分别启用 3、4、5、6 个 Tracker。")
    if not all(
        np.all(masks[index] <= masks[index + 1])
        for index in range(len(masks) - 1)
    ):
        raise ValueError("3/4/5/6 点 availability mask 必须逐级包含。")


def build_active_tracker_points(
    tracker_frame: np.ndarray,
    method_offsets: np.ndarray,
    tracker_available_by_method: np.ndarray = TRACKER_AVAILABLE_BY_METHOD,
) -> tuple[np.ndarray, ...]:
    """按每路 availability 提取展示点，返回长度分别为 3、4、5、6 的世界坐标。"""

    trackers = np.asarray(tracker_frame, dtype=np.float64)
    offsets = np.asarray(method_offsets, dtype=np.float64)
    masks = np.asarray(tracker_available_by_method, dtype=bool)
    if trackers.shape != (6, 3):
        raise ValueError(f"tracker_frame 应为 [6,3]，实际为 {trackers.shape}")
    if offsets.shape != (len(METHOD_ORDER), 3):
        raise ValueError(
            f"method_offsets 应为 {(len(METHOD_ORDER), 3)}，实际为 {offsets.shape}"
        )
    if masks.shape != (len(METHOD_ORDER), 6):
        raise ValueError(
            "tracker_available_by_method 应为 "
            f"{(len(METHOD_ORDER), 6)}，实际为 {masks.shape}"
        )
    return tuple(
        (trackers[mask] + offsets[index]).astype(np.float32)
        for index, mask in enumerate(masks)
    )


def run_tracker_count_inference(
    *,
    source: dict[str, np.ndarray],
    predictor,
    dit,
    diffusion,
    device: torch.device,
    normalizer: RealtimePoseNormalizer,
    args,
    source_frame_start: int,
    source_frame_end_exclusive: int,
) -> dict[str, dict[str, np.ndarray]]:
    """用同一模型和噪声序列运行逐点嵌套的 3/4/5/6 点闭环推理。"""

    validate_tracker_count_configuration()
    start = int(source_frame_start)
    end = int(source_frame_end_exclusive)
    if start < REALTIME_POSE_EVAL_METRICS_START_FRAME:
        raise ValueError(
            f"source_frame_start 不能早于正式输出帧 "
            f"{REALTIME_POSE_EVAL_METRICS_START_FRAME}。"
        )
    if end <= start or end > int(source["tracker_pos_world"].shape[0]):
        raise ValueError(
            f"请求帧范围 [{start},{end}) 超出 source 长度 "
            f"{source['tracker_pos_world'].shape[0]}。"
        )
    world_rotations = compute_source_joint_rotations_world(source)
    selected = slice(start - REALTIME_POSE_EVAL_METRICS_START_FRAME, end - REALTIME_POSE_EVAL_METRICS_START_FRAME)
    results: dict[str, dict[str, np.ndarray]] = {}
    for method_name, config_name, tracker_mask in zip(
        METHOD_ORDER,
        METHOD_CONFIG_NAMES,
        TRACKER_AVAILABLE_BY_METHOD,
    ):
        print(
            f"[tracker-count] inference {method_name}: {config_name}",
            flush=True,
        )
        full_result = run_dit_sequence(
            source=source,
            world_rotations=world_rotations,
            predictor=predictor,
            dit=dit,
            diffusion=diffusion,
            device=device,
            normalizer=normalizer,
            tracker_available=np.asarray(tracker_mask, dtype=bool),
            args=args,
            last=end,
        )
        results[method_name] = {
            key: np.asarray(value[selected], dtype=np.float32)
            for key, value in full_result.items()
        }
    return results


# endregion


# region SMPL 与指标


def build_tracker_count_mesh_sequences(
    *,
    base_clip,
    results: dict[str, dict[str, np.ndarray]],
    smpl_model_dir: Path,
) -> tuple[dict[str, SmplMeshSequence], np.ndarray]:
    """三路共用 AMASS 身形和 GT 根平移，只把各自姿态送入 SMPL-H。"""

    model_dir = require_directory(smpl_model_dir, "smpl_model_dir")
    if tuple(results.keys()) != METHOD_ORDER:
        raise ValueError(
            f"results 必须按 {METHOD_ORDER} 排列，实际为 {tuple(results.keys())}"
        )
    model = create_smplh_model(
        model_dir=model_dir,
        gender=base_clip.gender,
        batch_size=base_clip.frame_count,
    )
    sequences: dict[str, SmplMeshSequence] = {}
    for method_name in METHOD_ORDER:
        result = results[method_name]
        local_rotations = body_fbx_world_to_smpl_local_rotations(
            result["rotations"],
            result["root_yaw"],
            base_clip.body_fbx_rest.rest_local_rotations,
            base_clip.body_fbx_rest.parents,
        )
        pose_axis_angle = rotation_matrices_to_axis_angle(
            local_rotations[:, :SOURCE_BODY_JOINT_COUNT]
        )
        sequences[method_name] = run_smplh_forward(
            model=model,
            pose_axis_angle=pose_axis_angle,
            betas=base_clip.betas,
            translation_amass=base_clip.gt_translation_amass,
        )
    return sequences, transform_faces_to_unity_winding(model.faces)


def compute_tracker_count_metrics(
    base_clip,
    results: dict[str, dict[str, np.ndarray]],
) -> dict[str, TrackerCountMetrics]:
    metrics = {}
    for method_name in METHOD_ORDER:
        result = results[method_name]
        values = compute_rpm_p2_mc_metrics(
            predicted_global_rotations=result["rotations"],
            target_global_rotations=base_clip.reference_rotations_world,
            predicted_joint_positions=result["positions"],
            target_joint_positions=base_clip.reference_joints_world,
            fps=float(base_clip.fps),
        )
        metrics[method_name] = TrackerCountMetrics(
            mpjpe_cm=float(values["mpjpe_cm"]),
            mpjve_cm_per_s=float(values["mpjve_cm_per_s"]),
            jitter_m_per_s3=float(values["pred_jitter_m_per_s3"]),
        )
    return metrics


def write_tracker_count_sidecars(
    *,
    output_mp4: Path,
    source_path: Path,
    base_clip,
    results: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, TrackerCountMetrics],
    args,
    dit_weight_source: str,
    sampling_steps: int,
) -> tuple[Path, Path]:
    """保存可重复渲染的姿态 NPZ 与记录 checkpoint/mask 的 JSON。"""

    output = Path(output_mp4).expanduser().resolve()
    npz_path = output.with_suffix(".npz")
    report_path = output.with_suffix(".json")
    arrays: dict[str, np.ndarray] = {
        "reference_joints_world": base_clip.reference_joints_world,
        "reference_rotations_world": base_clip.reference_rotations_world,
        "tracker_pos_world": base_clip.tracker_pos_world,
        "tracker_available_by_method": np.asarray(
            TRACKER_AVAILABLE_BY_METHOD,
            dtype=bool,
        ),
    }
    for method_name, config_name in zip(METHOD_ORDER, METHOD_CONFIG_NAMES):
        result = results[method_name]
        arrays[f"{config_name}_joints_world"] = result["positions"]
        arrays[f"{config_name}_rotations_world"] = result["rotations"]
        arrays[f"{config_name}_root_yaw"] = result["root_yaw"]
    np.savez_compressed(npz_path, **arrays)
    report = {
        "source_path": str(Path(source_path).resolve()),
        "frame_start": int(base_clip.source_frame_start),
        "frame_end_exclusive": int(base_clip.source_frame_end_exclusive),
        "frames": int(base_clip.frame_count),
        "fps": int(base_clip.fps),
        "dit_model_path": str(Path(args.dit_model_path).resolve()),
        "dit_weight_source": str(dit_weight_source),
        "predictor_model_path": str(Path(args.predictor_model_path).resolve()),
        "sampling_steps": int(sampling_steps),
        "sampling_noise_seed": int(args.seed),
        "method_config_names": dict(zip(METHOD_ORDER, METHOD_CONFIG_NAMES)),
        "tracker_available_by_method": {
            method_name: list(TRACKER_AVAILABLE_BY_METHOD[index])
            for index, method_name in enumerate(METHOD_ORDER)
        },
        "metrics": {
            method_name: {
                "mpjpe_cm": values.mpjpe_cm,
                "mpjve_cm_per_s": values.mpjve_cm_per_s,
                "pred_jitter_m_per_s3": values.jitter_m_per_s3,
            }
            for method_name, values in metrics.items()
        },
        "npz_path": str(npz_path),
        "video_path": str(output),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return npz_path, report_path


# endregion


# region 共享舞台渲染


def draw_method_labels(draw: ImageDraw.ImageDraw) -> None:
    centers = tuple(
        int(round(OUTPUT_WIDTH * (index + 0.5) / len(METHOD_ORDER)))
        for index in range(len(METHOD_ORDER))
    )
    for method_name, center_x in zip(METHOD_ORDER, centers):
        width = 350
        left = int(center_x - width * 0.5)
        right = int(center_x + width * 0.5)
        draw.rounded_rectangle(
            (left, 18, right, 82),
            radius=18,
            fill=(255, 255, 255, 230),
            outline=(210, 214, 220, 230),
            width=2,
        )
        color = rgba_color(METHOD_COLORS[method_name])
        draw.rounded_rectangle(
            (left + 16, 31, left + 40, 55),
            radius=7,
            fill=color,
        )
        draw.text(
            (left + 52, 25),
            method_name,
            font=load_font(19),
            fill=(31, 41, 55, 255),
        )
        draw.text(
            (left + 52, 52),
            METHOD_SENSOR_LABELS[method_name],
            font=load_font(13),
            fill=(76, 86, 99, 255),
        )


def compose_tracker_count_frame(
    *,
    viewport_rgb: np.ndarray,
    playback_label: str,
    metrics: dict[str, TrackerCountMetrics],
    show_intro_message: bool,
) -> Image.Image:
    image = Image.fromarray(np.asarray(viewport_rgb, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw_method_labels(draw)

    playback_width = 178 if playback_label == "0.5× replay" else 132
    draw.rounded_rectangle(
        (OUTPUT_WIDTH - playback_width - 28, 94, OUTPUT_WIDTH - 28, 134),
        radius=13,
        fill=(31, 41, 55, 218),
    )
    draw_centered_text(
        draw,
        (OUTPUT_WIDTH - playback_width - 28, 94, OUTPUT_WIDTH - 28, 134),
        playback_label,
        load_font(15),
        (255, 255, 255, 255),
    )
    if show_intro_message:
        draw.rounded_rectangle(
            (OUTPUT_WIDTH // 2 - 330, 98, OUTPUT_WIDTH // 2 + 330, 144),
            radius=15,
            fill=(255, 255, 255, 236),
            outline=(235, 142, 24, 245),
            width=2,
        )
        draw.ellipse(
            (OUTPUT_WIDTH // 2 - 296, 113, OUTPUT_WIDTH // 2 - 280, 129),
            fill=rgba_color(TRACKER_COLOR),
        )
        draw.text(
            (OUTPUT_WIDTH // 2 - 264, 108),
            "One 100k model | nested tracker masks | common diffusion noise",
            font=load_font(16),
            fill=(31, 41, 55, 255),
        )

    footer_top = OUTPUT_HEIGHT - 88
    draw.rectangle(
        (0, footer_top, OUTPUT_WIDTH, OUTPUT_HEIGHT),
        fill=(255, 255, 255, 234),
    )
    draw.line(
        (0, footer_top, OUTPUT_WIDTH, footer_top),
        fill=(202, 207, 214, 240),
        width=2,
    )
    mpjpe = " / ".join(
        f"{metrics[name].mpjpe_cm:.2f}" for name in METHOD_ORDER
    )
    mpjve = " / ".join(
        f"{metrics[name].mpjve_cm_per_s:.2f}" for name in METHOD_ORDER
    )
    footer = (
        "Same checkpoint + noise | Shared GT root translation     "
        f"MPJPE 3/4/5/6: {mpjpe} cm     MPJVE 3/4/5/6: {mpjve} cm/s"
    )
    draw_centered_text(
        draw,
        (22, footer_top + 4, OUTPUT_WIDTH - 22, OUTPUT_HEIGHT - 4),
        footer,
        load_font(18),
        (31, 41, 55, 255),
    )
    return Image.alpha_composite(image, overlay).convert("RGB")


def render_tracker_count_view(
    *,
    renderer,
    scene,
    frame_index: int,
    sequences: dict[str, SmplMeshSequence],
    rotations: dict[str, np.ndarray],
    tracker_pos_world: np.ndarray,
    faces: np.ndarray,
    method_offsets: np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    """一次渲染四路网格及各自真实启用的 3/4/5/6 个 Tracker。"""

    import pyrender
    import trimesh

    dynamic_nodes = []

    def add_trimesh(mesh, material, *, smooth: bool = True) -> None:
        node = scene.add(
            pyrender.Mesh.from_trimesh(mesh, material=material, smooth=smooth)
        )
        dynamic_nodes.append(node)

    try:
        for method_index, method_name in enumerate(METHOD_ORDER):
            offset = np.asarray(method_offsets[method_index], dtype=np.float64)
            sequence = sequences[method_name]
            vertices = np.asarray(
                sequence.vertices_world[frame_index], dtype=np.float64
            ) + offset
            joints = np.asarray(
                sequence.joints_world[frame_index], dtype=np.float64
            ) + offset
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
                frame_index, JOINT_INDEX["spine3"], :, 2
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

        active_points = build_active_tracker_points(
            tracker_pos_world[frame_index], method_offsets
        )
        tracker_points = np.concatenate(active_points, axis=0)
        visible_tracker_points = build_visible_tracker_glyph_points(
            tracker_points,
            np.asarray(camera_pose, dtype=np.float64)[:3, 3],
        )
        tracker_cloud = create_sphere_cloud(
            visible_tracker_points,
            radius=0.036,
        )
        add_trimesh(
            tracker_cloud,
            create_material(pyrender, TRACKER_COLOR, 0.48),
        )
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
        return np.asarray(color[..., :3], dtype=np.uint8)
    finally:
        for node in dynamic_nodes:
            scene.remove_node(node)


def render_tracker_count_video(
    *,
    output_path: Path,
    base_clip,
    results: dict[str, dict[str, np.ndarray]],
    sequences: dict[str, SmplMeshSequence],
    faces: np.ndarray,
    metrics: dict[str, TrackerCountMetrics],
) -> Path:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError("缺少 pyrender，无法执行 SMPL-H 离屏渲染。") from exc

    frame_count = int(base_clip.frame_count)
    layout = build_presentation_layout(
        sequences=sequences,
        tracker_pos_world=base_clip.tracker_pos_world,
        method_order=METHOD_ORDER,
        tracker_available_by_method=TRACKER_AVAILABLE_BY_METHOD,
        follow_method_name=METHOD_ORDER[0],
    )
    floor_y = min(
        float(np.min(sequence.vertices_world[..., 1]))
        for sequence in sequences.values()
    )
    scene, camera_node = create_static_scene(
        layout.base_camera,
        floor_y=floor_y,
        grid_size=layout.grid_size,
        grid_center=layout.grid_center,
    )
    renderer = pyrender.OffscreenRenderer(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    rotations = {
        method_name: results[method_name]["rotations"]
        for method_name in METHOD_ORDER
    }
    schedule = build_presentation_frame_schedule(frame_count)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: Mp4FrameWriter | None = None
    slow_frames: list[bytes] = []
    written_frame_count = 0
    try:
        scene.set_pose(camera_node, pose=layout.camera_poses[0])
        intro_view = render_tracker_count_view(
            renderer=renderer,
            scene=scene,
            frame_index=0,
            sequences=sequences,
            rotations=rotations,
            tracker_pos_world=base_clip.tracker_pos_world,
            faces=faces,
            method_offsets=layout.method_offsets,
            camera_pose=layout.camera_poses[0],
        )
        intro_frame = compose_tracker_count_frame(
            viewport_rgb=intro_view,
            playback_label="Input setup",
            metrics=metrics,
            show_intro_message=True,
        )
        intro_rgb = np.asarray(intro_frame, dtype=np.uint8)
        writer = Mp4FrameWriter(output_path=output, frame_rgb=intro_rgb, fps=base_clip.fps)
        for _ in range(INTRO_FRAME_COUNT):
            writer.append(intro_rgb)
            written_frame_count += 1

        for frame_index in range(frame_count):
            scene.set_pose(camera_node, pose=layout.camera_poses[frame_index])
            viewport = render_tracker_count_view(
                renderer=renderer,
                scene=scene,
                frame_index=frame_index,
                sequences=sequences,
                rotations=rotations,
                tracker_pos_world=base_clip.tracker_pos_world,
                faces=faces,
                method_offsets=layout.method_offsets,
                camera_pose=layout.camera_poses[frame_index],
            )
            normal_frame = compose_tracker_count_frame(
                viewport_rgb=viewport,
                playback_label="1.0×",
                metrics=metrics,
                show_intro_message=False,
            )
            slow_frame = compose_tracker_count_frame(
                viewport_rgb=viewport,
                playback_label="0.5× replay",
                metrics=metrics,
                show_intro_message=False,
            )
            writer.append(np.asarray(normal_frame, dtype=np.uint8))
            written_frame_count += 1
            slow_frames.append(encode_png(slow_frame))
            print(
                f"[tracker-count] rendered {frame_index + 1}/{frame_count} "
                f"(source frame {base_clip.source_frame_start + frame_index})",
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
        f"[tracker-count] stage spacing={layout.stage_spacing:.3f} m, "
        f"frames={written_frame_count}, fps={base_clip.fps}",
        flush=True,
    )
    print(f"[tracker-count] wrote: {output}", flush=True)
    return output


# endregion


def main(argv: list[str] | None = None) -> Path:
    parser = build_arg_parser()
    args = parse_and_load_from_model(parser, argv)
    fixseed(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    base_clip = load_comparison_clip(
        comparison_npz=args.comparison_npz,
        report_json=args.report_json,
        amass_npz=args.amass_npz,
        source_frame_start=int(args.source_frame_start),
        source_frame_end_exclusive=int(args.source_frame_end_exclusive),
        diffusion_variant="core_only",
    )
    report = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    source_path = Path(report["source_path"]).expanduser().resolve()
    source = load_realtime_source(source_path)
    dit, diffusion = create_model_and_diffusion(args)
    dit, dit_weight_source = load_checkpoint_model(
        dit,
        args.dit_model_path,
        device,
        use_ema=args.use_ema,
    )
    predictor = load_realtime_pose_predictor(args.predictor_model_path, device)
    normalizer = RealtimePoseNormalizer(
        args.normalizer_dir,
        disable=not bool(args.normalize_input),
    )
    results = run_tracker_count_inference(
        source=source,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        args=args,
        source_frame_start=int(args.source_frame_start),
        source_frame_end_exclusive=int(args.source_frame_end_exclusive),
    )
    metrics = compute_tracker_count_metrics(base_clip, results)
    output = Path(args.output_mp4).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    npz_path, report_path = write_tracker_count_sidecars(
        output_mp4=output,
        source_path=source_path,
        base_clip=base_clip,
        results=results,
        metrics=metrics,
        args=args,
        dit_weight_source=dit_weight_source,
        sampling_steps=int(diffusion.num_timesteps),
    )
    sequences, faces = build_tracker_count_mesh_sequences(
        base_clip=base_clip,
        results=results,
        smpl_model_dir=args.smpl_model_dir,
    )
    render_tracker_count_video(
        output_path=output,
        base_clip=base_clip,
        results=results,
        sequences=sequences,
        faces=faces,
        metrics=metrics,
    )
    print(f"[tracker-count] sidecars: {npz_path}, {report_path}", flush=True)
    return output


if __name__ == "__main__":
    main()
