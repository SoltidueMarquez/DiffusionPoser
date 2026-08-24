from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
import torch

from data_converter.amass_smpl_utils import (
    SOURCE_BODY_JOINT_COUNT,
    load_motion_source,
    normalize_gender,
)
from data_loaders.body_fbx_kinematics import load_body_fbx_rest
from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    load_realtime_source,
)
from data_loaders.sensor_masking import (
    ALL_SIX_AVAILABLE,
    CORE_THREE_AVAILABLE,
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
    REALTIME_POSE_TARGET_DIM,
    TRACKER_NAMES,
)
from eval.evaluate_realtime_pose_predictor import (
    PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
    evaluation_last_frame_exclusive,
)
from sample.evaluate_progressive_tracker_addition import (
    build_equal_quarter_tracker_schedule as build_addition_tracker_schedule,
)
from sample.evaluate_progressive_tracker_dropout import (
    MIN_PROGRESSIVE_SCORED_FRAMES,
    OPTIONAL_TRACKER_NAME_TO_INDEX,
    build_equal_quarter_tracker_schedule as build_dropout_tracker_schedule,
)
from sample.realtime_pose_longseq_evaluator import (
    compute_sequence_metrics_by_stage,
    create_eval_noise_generator,
    create_longseq_runtime,
)
from sample.realtime_pose_runtime import WorldPoseState
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
from sample.render_realtime_pose_smpl_presentation import (
    build_presentation_layout,
    build_visible_tracker_glyph_points,
    create_material,
)
from sample.utils import load_checkpoint_model
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


METHOD_ORDER = ("GT", "Dynamic")
METHOD_COLORS = {
    "GT": (0x90 / 255.0, 0xA9 / 255.0, 0xC2 / 255.0, 1.0),
    "Dynamic": (0x35 / 255.0, 0xB8 / 255.0, 0xA6 / 255.0, 1.0),
}
ACTIVE_TRACKER_COLOR = (1.0, 0.55, 0.05, 1.0)
INACTIVE_TRACKER_COLOR = (0.48, 0.51, 0.56, 1.0)
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
INTRO_FRAME_COUNT = 15


@dataclass(frozen=True)
class ProgressiveSequenceResult:
    """动态 schedule 的逐帧闭环输出，时间轴从正式计分帧 30 开始。"""

    frame_start: int
    frame_end_exclusive: int
    tracker_available: np.ndarray
    stage_indices: np.ndarray
    target_rotations: np.ndarray
    target_positions: np.ndarray
    deployed_rotations: np.ndarray
    deployed_positions: np.ndarray
    deployed_root_yaw: np.ndarray
    tracker_positions: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.stage_indices.shape[0])


@dataclass(frozen=True)
class ProgressiveMeshInputs:
    gt_pose_axis_angle: np.ndarray
    gt_translation_amass: np.ndarray
    betas: np.ndarray
    gender: str
    rest_local_rotations: np.ndarray
    parents: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "重跑指定长序列的动态 Tracker 6→3/3→6 协议，并输出 SMPL-H 视频、"
            "逐帧 NPZ 和连续性诊断 JSON。"
        )
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    paths = parser.add_argument_group("progressive render paths")
    paths.add_argument("--source_npz", nargs="+", required=True, type=Path)
    paths.add_argument("--source_dir", required=True, type=Path)
    paths.add_argument("--amass_dir", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--normalizer_dir", required=True, type=Path)
    paths.add_argument("--normalize_input", default=True, type=str2bool)
    protocol = parser.add_argument_group("progressive protocol")
    protocol.add_argument(
        "--direction",
        default="dropout",
        choices=("dropout", "addition"),
        help="dropout 运行 6→3；addition 运行 3→6。",
    )
    protocol.add_argument(
        "--drop_order",
        nargs=3,
        choices=tuple(OPTIONAL_TRACKER_NAME_TO_INDEX),
        default=("right_foot", "left_foot", "hip"),
        help="三个 optional Tracker 的移除顺序。",
    )
    protocol.add_argument(
        "--add_order",
        nargs=3,
        choices=tuple(OPTIONAL_TRACKER_NAME_TO_INDEX),
        default=("hip", "left_foot", "right_foot"),
        help="三个 optional Tracker 的添加顺序。",
    )
    protocol.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="正式计分区间最多渲染多少帧；0 表示整条序列。",
    )
    protocol.add_argument("--stride", default=1, type=int)
    protocol.add_argument("--skip_render", default=False, type=str2bool)
    return parser


def validate_drop_order(drop_order: tuple[str, ...]) -> tuple[int, ...]:
    names = tuple(str(name) for name in drop_order)
    if len(names) != 3 or set(names) != set(OPTIONAL_TRACKER_NAME_TO_INDEX):
        raise ValueError("drop_order 必须恰好包含 hip、left_foot、right_foot。")
    return tuple(OPTIONAL_TRACKER_NAME_TO_INDEX[name] for name in names)


def validate_add_order(add_order: tuple[str, ...]) -> tuple[int, ...]:
    names = tuple(str(name) for name in add_order)
    if len(names) != 3 or set(names) != set(OPTIONAL_TRACKER_NAME_TO_INDEX):
        raise ValueError("add_order 必须恰好包含 hip、left_foot、right_foot。")
    return tuple(OPTIONAL_TRACKER_NAME_TO_INDEX[name] for name in names)


def build_transition_schedule(
    *,
    scored_frame_count: int,
    direction: str,
    transition_indices: tuple[int, ...],
):
    """按展示方向选择正式评估使用的同一套等分 schedule 构造器。"""

    if direction == "dropout":
        return build_dropout_tracker_schedule(
            scored_frame_count=scored_frame_count,
            drop_order=transition_indices,
        )
    if direction == "addition":
        return build_addition_tracker_schedule(
            scored_frame_count=scored_frame_count,
            add_order=transition_indices,
        )
    raise ValueError(f"direction 必须为 dropout/addition，实际为 {direction}")


def warmup_tracker_available(direction: str) -> np.ndarray:
    """展示必须复用正式协议的预热 mask，否则第一阶段历史不可比较。"""

    if direction == "dropout":
        return np.asarray(ALL_SIX_AVAILABLE, dtype=bool)
    if direction == "addition":
        return np.asarray(CORE_THREE_AVAILABLE, dtype=bool)
    raise ValueError(f"direction 必须为 dropout/addition，实际为 {direction}")


def read_scalar_string(payload, key: str) -> str:
    if key not in payload.files:
        raise KeyError(f"source npz 缺少 {key}。")
    value = np.asarray(payload[key])
    if value.ndim != 0:
        raise ValueError(f"{key} 应为标量字符串，实际为 {value.shape}")
    return str(value.item())


def resolve_amass_path(
    source_path: Path,
    source_dir: Path,
    amass_dir: Path,
) -> tuple[Path, Path]:
    source = Path(source_path).expanduser().resolve()
    source_root = Path(source_dir).expanduser().resolve()
    amass_root = Path(amass_dir).expanduser().resolve()
    try:
        relative = source.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"source_npz 不在 source_dir 下：source={source}, root={source_root}"
        ) from exc
    amass_path = amass_root / relative
    if not amass_path.is_file():
        raise FileNotFoundError(f"找不到对应原始 AMASS：{amass_path}")
    return relative, amass_path


def run_progressive_sequence(
    *,
    source: dict[str, np.ndarray],
    predictor,
    dit,
    diffusion,
    device: torch.device,
    normalizer: RealtimePoseNormalizer,
    args,
    direction: str,
    transition_indices: tuple[int, ...],
) -> ProgressiveSequenceResult:
    """严格复用正式评估的预热、等分 schedule、闭环 runtime 和噪声规则。"""

    world_rotations = compute_source_joint_rotations_world(source)
    last = evaluation_last_frame_exclusive(len(world_rotations), int(args.max_frames))
    scored_frame_count = last - REALTIME_POSE_EVAL_METRICS_START_FRAME
    if scored_frame_count < MIN_PROGRESSIVE_SCORED_FRAMES:
        raise ValueError(
            f"动态展示至少需要 {MIN_PROGRESSIVE_SCORED_FRAMES} 个计分帧，"
            f"实际为 {scored_frame_count}。"
        )
    schedule = build_transition_schedule(
        scored_frame_count=scored_frame_count,
        direction=direction,
        transition_indices=transition_indices,
    )
    warmup_available = warmup_tracker_available(direction)
    runtime = create_longseq_runtime(
        source=source,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        args=args,
    )
    runtime.initialize_history(
        [
            WorldPoseState(
                joint_rotations_world=world_rotations[index],
                root_yaw_world=float(source["root_yaw"][index]),
                hip_height=float(source["pelvis_height"][index, 0]),
                root_position_world=source["root_pos_world"][index],
            )
            for index in range(1, 11)
        ],
        source["tracker_pos_world"][:11],
        source["tracker_rot_world_6d"][:11],
        source["root_pos_world"][:11, 1],
    )
    rotations: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    root_yaw: list[float] = []
    noise_generator = create_eval_noise_generator(args.seed, device)
    for current in range(PREDICTOR_EVAL_FIRST_GENERATED_FRAME, last):
        if current < REALTIME_POSE_EVAL_METRICS_START_FRAME:
            frame_tracker_available = warmup_available
        else:
            scored_offset = current - REALTIME_POSE_EVAL_METRICS_START_FRAME
            frame_tracker_available = schedule.tracker_available[scored_offset]
        noise = torch.randn(
            (1, REALTIME_POSE_TARGET_DIM),
            generator=noise_generator,
            device=device,
        )
        result = runtime.step(
            source["tracker_pos_world"][current],
            source["tracker_rot_world_6d"][current],
            frame_tracker_available,
            float(source["root_pos_world"][current, 1]),
            noise=noise,
        )
        if current >= REALTIME_POSE_EVAL_METRICS_START_FRAME:
            rotations.append(result.resolved_pose.joint_rotations_world)
            positions.append(result.resolved_pose.joints_world)
            root_yaw.append(result.resolved_pose.root_yaw_world)

    selected = slice(REALTIME_POSE_EVAL_METRICS_START_FRAME, last)
    return ProgressiveSequenceResult(
        frame_start=REALTIME_POSE_EVAL_METRICS_START_FRAME,
        frame_end_exclusive=last,
        tracker_available=np.asarray(schedule.tracker_available, dtype=bool),
        stage_indices=np.asarray(schedule.stage_indices, dtype=np.int64),
        target_rotations=np.asarray(world_rotations[selected], dtype=np.float32),
        target_positions=np.asarray(source["joints_world"][selected], dtype=np.float32),
        deployed_rotations=np.stack(rotations).astype(np.float32),
        deployed_positions=np.stack(positions).astype(np.float32),
        deployed_root_yaw=np.asarray(root_yaw, dtype=np.float32),
        tracker_positions=np.asarray(source["tracker_pos_world"][selected], dtype=np.float32),
    )


def load_progressive_mesh_inputs(
    *,
    source_path: Path,
    amass_path: Path,
    amass_dir: Path,
    frame_start: int,
    frame_end_exclusive: int,
) -> ProgressiveMeshInputs:
    with np.load(source_path, allow_pickle=False) as payload:
        rest_path = Path(read_scalar_string(payload, "body_fbx_rest_json")).resolve()
    if not rest_path.is_file():
        raise FileNotFoundError(f"body.fbx rest pose 不存在：{rest_path}")
    rest = load_body_fbx_rest(rest_path)
    motion = load_motion_source(
        path=amass_path,
        amass_dir=Path(amass_dir).resolve(),
        target_fps=float(REALTIME_POSE_FPS),
    )
    if frame_end_exclusive > int(motion.poses.shape[0]):
        raise ValueError(
            f"请求结束帧 {frame_end_exclusive} 超过 AMASS 重采样长度 "
            f"{motion.poses.shape[0]}。"
        )
    frame_count = frame_end_exclusive - frame_start
    gt_pose = motion.poses[
        frame_start:frame_end_exclusive,
        : SOURCE_BODY_JOINT_COUNT * 3,
    ].reshape(frame_count, SOURCE_BODY_JOINT_COUNT, 3)
    return ProgressiveMeshInputs(
        gt_pose_axis_angle=np.asarray(gt_pose, dtype=np.float32),
        gt_translation_amass=np.asarray(
            motion.trans[frame_start:frame_end_exclusive], dtype=np.float32
        ),
        betas=np.asarray(motion.betas, dtype=np.float32).reshape(-1),
        gender=normalize_gender(motion.gender),
        rest_local_rotations=np.asarray(rest.rest_local_rotations, dtype=np.float32),
        parents=np.asarray(rest.parents, dtype=np.int64),
    )


def build_progressive_mesh_sequences(
    *,
    result: ProgressiveSequenceResult,
    mesh_inputs: ProgressiveMeshInputs,
    smpl_model_dir: Path,
) -> tuple[dict[str, SmplMeshSequence], np.ndarray]:
    predicted_local = body_fbx_world_to_smpl_local_rotations(
        result.deployed_rotations,
        result.deployed_root_yaw,
        mesh_inputs.rest_local_rotations,
        mesh_inputs.parents,
    )
    predicted_pose = rotation_matrices_to_axis_angle(
        predicted_local[:, :SOURCE_BODY_JOINT_COUNT]
    )
    model = create_smplh_model(
        model_dir=Path(smpl_model_dir).expanduser().resolve(),
        gender=mesh_inputs.gender,
        batch_size=result.frame_count,
    )
    sequences = {
        "GT": run_smplh_forward(
            model=model,
            pose_axis_angle=mesh_inputs.gt_pose_axis_angle,
            betas=mesh_inputs.betas,
            translation_amass=mesh_inputs.gt_translation_amass,
        ),
        "Dynamic": run_smplh_forward(
            model=model,
            pose_axis_angle=predicted_pose,
            betas=mesh_inputs.betas,
            # 展示共用 GT 根平移，避免把位移差异误读成姿态断裂。
            translation_amass=mesh_inputs.gt_translation_amass,
        ),
    }
    return sequences, transform_faces_to_unity_winding(model.faces)


def compute_continuity_diagnostics(
    result: ProgressiveSequenceResult,
    *,
    direction: str = "dropout",
    local_radius: int = 5,
) -> list[dict[str, float | int | str]]:
    """量化三次 mask 切换对应的单帧关节步长，并与局部正常运动比较。"""

    predicted_steps = np.linalg.norm(
        np.diff(result.deployed_positions, axis=0), axis=-1
    ).mean(axis=-1)
    target_steps = np.linalg.norm(
        np.diff(result.target_positions, axis=0), axis=-1
    ).mean(axis=-1)
    boundaries = np.flatnonzero(np.diff(result.stage_indices) > 0) + 1
    reports: list[dict[str, float | int | str]] = []
    for boundary in boundaries.tolist():
        transition_index = boundary - 1
        left = max(0, transition_index - int(local_radius))
        right = min(predicted_steps.shape[0], transition_index + int(local_radius) + 1)
        neighbors = np.concatenate(
            [
                predicted_steps[left:transition_index],
                predicted_steps[transition_index + 1 : right],
            ]
        )
        local_median = float(np.median(neighbors)) if neighbors.size else 0.0
        predicted_step = float(predicted_steps[transition_index])
        entered_stage = int(result.stage_indices[boundary])
        previous_available = result.tracker_available[boundary - 1]
        current_available = result.tracker_available[boundary]
        if direction == "dropout":
            changed = previous_available & ~current_available
            tracker_key = "dropped_tracker"
        elif direction == "addition":
            changed = ~previous_available & current_available
            tracker_key = "added_tracker"
        else:
            raise ValueError(
                f"direction 必须为 dropout/addition，实际为 {direction}"
            )
        changed_indices = np.flatnonzero(changed)
        if changed_indices.shape != (1,):
            raise ValueError(
                f"切换边界应恰好改变一个 Tracker，实际改变 {changed_indices.tolist()}"
            )
        changed_index = int(changed_indices[0])
        reports.append(
            {
                "source_frame": int(result.frame_start + boundary),
                "entered_stage": entered_stage,
                "tracker_count": int(result.tracker_available[boundary].sum()),
                "transition": str(direction),
                "changed_tracker": str(TRACKER_NAMES[changed_index]),
                tracker_key: str(TRACKER_NAMES[changed_index]),
                "predicted_mean_joint_step_cm": predicted_step * 100.0,
                "gt_mean_joint_step_cm": float(target_steps[transition_index]) * 100.0,
                "predicted_step_to_local_median_ratio": (
                    predicted_step / local_median if local_median > 1e-8 else 0.0
                ),
                "predicted_mean_joint_speed_cm_per_s": (
                    predicted_step * float(REALTIME_POSE_FPS) * 100.0
                ),
            }
        )
    return reports


def compute_metrics(result: ProgressiveSequenceResult) -> tuple[dict, list[dict]]:
    overall, stages = compute_sequence_metrics_by_stage(
        predicted_global_rotations=result.deployed_rotations,
        target_global_rotations=result.target_rotations,
        predicted_joint_positions=result.deployed_positions,
        target_joint_positions=result.target_positions,
        stage_indices=result.stage_indices,
        stage_count=4,
        fps=float(REALTIME_POSE_FPS),
    )
    stage_reports = []
    for stage_index, values in enumerate(stages):
        selected = result.stage_indices == stage_index
        stage_reports.append(
            {
                "stage_index": stage_index,
                "tracker_count": int(result.tracker_available[selected][0].sum()),
                "frames": int(np.count_nonzero(selected)),
                "metrics": values,
            }
        )
    return overall, stage_reports


def sequence_stem(relative_path: Path) -> str:
    value = str(relative_path.with_suffix(""))
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def write_sidecars(
    *,
    output_mp4: Path,
    source_path: Path,
    relative_path: Path,
    amass_path: Path,
    result: ProgressiveSequenceResult,
    direction: str,
    transition_order: tuple[str, ...],
    args,
    dit_weight_source: str,
    sampling_steps: int,
) -> tuple[Path, Path, dict]:
    overall, stages = compute_metrics(result)
    boundaries = compute_continuity_diagnostics(result, direction=direction)
    npz_path = output_mp4.with_suffix(".npz")
    json_path = output_mp4.with_suffix(".json")
    np.savez_compressed(
        npz_path,
        target_rotations_world=result.target_rotations,
        target_joints_world=result.target_positions,
        deployed_rotations_world=result.deployed_rotations,
        deployed_joints_world=result.deployed_positions,
        deployed_root_yaw=result.deployed_root_yaw,
        tracker_pos_world=result.tracker_positions,
        tracker_available=result.tracker_available,
        stage_indices=result.stage_indices,
    )
    is_dropout = direction == "dropout"
    report = {
        "experiment": (
            "progressive_tracker_dropout_6_to_3_showcase"
            if is_dropout
            else "progressive_tracker_addition_3_to_6_showcase"
        ),
        "source_path": str(source_path),
        "source_relative_path": str(relative_path),
        "amass_path": str(amass_path),
        "frame_start": result.frame_start,
        "frame_end_exclusive": result.frame_end_exclusive,
        "frames": result.frame_count,
        "fps": float(REALTIME_POSE_FPS),
        ("drop_order" if is_dropout else "add_order"): list(transition_order),
        "tracker_counts": [6, 5, 4, 3] if is_dropout else [3, 4, 5, 6],
        "warmup_tracker_count": 6 if is_dropout else 3,
        "stage_policy": "equal_scored_quarters",
        "predictor_model_path": str(Path(args.predictor_model_path).resolve()),
        "dit_model_path": str(Path(args.dit_model_path).resolve()),
        "dit_weight_source": str(dit_weight_source),
        "sampling_steps": int(sampling_steps),
        "sampling_noise_seed": int(args.seed),
        "finite_outputs": bool(
            np.isfinite(result.deployed_rotations).all()
            and np.isfinite(result.deployed_positions).all()
        ),
        "overall_metrics": overall,
        "stage_metrics": stages,
        "switch_boundary_diagnostics": boundaries,
        "visualization_root_translation": "shared_ground_truth",
        "npz_path": str(npz_path),
        "video_path": str(output_mp4),
    }
    if not is_dropout:
        report["paired_drop_order"] = list(reversed(transition_order))
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return npz_path, json_path, report


def _rgba(color: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(int(round(float(value) * 255.0)) for value in color)


def compose_frame(
    *,
    viewport_rgb: np.ndarray,
    frame_index: int,
    result: ProgressiveSequenceResult,
    metrics: dict,
    direction: str,
    transition_order: tuple[str, ...],
) -> np.ndarray:
    image = Image.fromarray(np.asarray(viewport_rgb, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    stage = int(result.stage_indices[frame_index])
    tracker_count = int(result.tracker_available[frame_index].sum())
    stage_start = int(np.flatnonzero(result.stage_indices == stage)[0])
    changed = "none" if stage == 0 else ", ".join(transition_order[:stage])
    is_dropout = direction == "dropout"
    direction_label = "Dynamic 6→3" if is_dropout else "Dynamic 3→6"
    change_label = "Dropped" if is_dropout else "Added"

    draw.rounded_rectangle(
        (18, 16, OUTPUT_WIDTH - 18, 92),
        radius=16,
        fill=(255, 255, 255, 226),
        outline=(207, 213, 221, 240),
        width=2,
    )
    draw.text((42, 29), "GT", font=load_font(24), fill=_rgba(METHOD_COLORS["GT"]))
    draw.text(
        (OUTPUT_WIDTH // 2 + 30, 25),
        f"{direction_label}  |  Stage {stage + 1}/4  |  {tracker_count} trackers",
        font=load_font(23),
        fill=_rgba(METHOD_COLORS["Dynamic"]),
    )
    draw.text(
        (OUTPUT_WIDTH // 2 + 30, 57),
        f"{change_label}: {changed}",
        font=load_font(16),
        fill=(74, 83, 96, 255),
    )
    if frame_index == stage_start and stage > 0:
        draw.rounded_rectangle(
            (OUTPUT_WIDTH // 2 - 175, 108, OUTPUT_WIDTH // 2 + 175, 154),
            radius=12,
            fill=(255, 244, 214, 235),
            outline=(245, 158, 11, 240),
            width=2,
        )
        action = "removed" if is_dropout else "added"
        message = f"Tracker {action}: {transition_order[stage - 1]}"
        box = draw.textbbox((0, 0), message, font=load_font(19))
        draw.text(
            ((OUTPUT_WIDTH - (box[2] - box[0])) // 2, 119),
            message,
            font=load_font(19),
            fill=(138, 82, 4, 255),
        )

    footer_top = OUTPUT_HEIGHT - 84
    draw.rectangle(
        (0, footer_top, OUTPUT_WIDTH, OUTPUT_HEIGHT),
        fill=(255, 255, 255, 232),
    )
    timeline_left, timeline_right = 48, OUTPUT_WIDTH - 48
    timeline_y = footer_top + 18
    segment_colors = (
        (56, 189, 248, 255),
        (45, 212, 191, 255),
        (250, 204, 21, 255),
        (251, 146, 60, 255),
    )
    for stage_index in range(4):
        selected = np.flatnonzero(result.stage_indices == stage_index)
        left = timeline_left + int(
            round(selected[0] / result.frame_count * (timeline_right - timeline_left))
        )
        right = timeline_left + int(
            round((selected[-1] + 1) / result.frame_count * (timeline_right - timeline_left))
        )
        draw.rectangle((left, timeline_y, right, timeline_y + 9), fill=segment_colors[stage_index])
    cursor_x = timeline_left + int(
        round(frame_index / max(1, result.frame_count - 1) * (timeline_right - timeline_left))
    )
    draw.ellipse((cursor_x - 5, timeline_y - 4, cursor_x + 5, timeline_y + 13), fill=(20, 27, 38, 255))
    footer = (
        f"source frame {result.frame_start + frame_index}  |  "
        f"MPJRE {metrics['mpjre_deg']:.2f}°  MPJPE {metrics['mpjpe_cm']:.2f} cm  "
        f"MPJVE {metrics['mpjve_cm_per_s']:.2f} cm/s  "
        f"Jitter {metrics['pred_jitter_m_per_s3']:.2f} m/s³"
    )
    draw.text((48, footer_top + 41), footer, font=load_font(17), fill=(31, 41, 55, 255))
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def render_view(
    *,
    renderer,
    scene,
    frame_index: int,
    sequences: dict[str, SmplMeshSequence],
    faces: np.ndarray,
    method_offsets: np.ndarray,
    tracker_positions: np.ndarray,
    tracker_available: np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    import pyrender
    import trimesh

    nodes = []

    def add_mesh(mesh, material) -> None:
        node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
        nodes.append(node)

    try:
        for method_index, method_name in enumerate(METHOD_ORDER):
            vertices = (
                np.asarray(sequences[method_name].vertices_world[frame_index], dtype=np.float64)
                + np.asarray(method_offsets[method_index], dtype=np.float64)
            )
            body = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            add_mesh(body, create_material(pyrender, METHOD_COLORS[method_name], 0.92))

        dynamic_offset = np.asarray(method_offsets[1], dtype=np.float64)
        tracker_frame = np.asarray(tracker_positions[frame_index], dtype=np.float64)
        visible = build_visible_tracker_glyph_points(
            tracker_frame + dynamic_offset,
            np.asarray(camera_pose, dtype=np.float64)[:3, 3],
        )
        mask = np.asarray(tracker_available[frame_index], dtype=bool)
        active_cloud = create_sphere_cloud(visible[mask], radius=0.035)
        if active_cloud is not None:
            add_mesh(active_cloud, create_material(pyrender, ACTIVE_TRACKER_COLOR, 0.48))
        inactive_cloud = create_sphere_cloud(visible[~mask], radius=0.022)
        if inactive_cloud is not None:
            add_mesh(inactive_cloud, create_material(pyrender, INACTIVE_TRACKER_COLOR, 0.72))
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
        return np.asarray(color[..., :3], dtype=np.uint8)
    finally:
        for node in nodes:
            scene.remove_node(node)


def render_video(
    *,
    output_path: Path,
    result: ProgressiveSequenceResult,
    sequences: dict[str, SmplMeshSequence],
    faces: np.ndarray,
    metrics: dict,
    direction: str,
    transition_order: tuple[str, ...],
    stride: int,
) -> Path:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender
    except ImportError as exc:
        raise ImportError("缺少 pyrender，无法执行 SMPL-H 离屏渲染。") from exc

    # 相机拟合纳入动态路可能出现的全部六点，后续 mask 切换不会改变视野尺度。
    layout = build_presentation_layout(
        sequences=sequences,
        tracker_pos_world=result.tracker_positions,
        method_order=METHOD_ORDER,
        tracker_available_by_method=np.asarray(
            [[False] * 6, [True] * 6], dtype=bool
        ),
        follow_method_name="GT",
    )
    floor_y = min(float(np.min(value.vertices_world[..., 1])) for value in sequences.values())
    scene, camera_node = create_static_scene(
        layout.base_camera,
        floor_y=floor_y,
        grid_size=layout.grid_size,
        grid_center=layout.grid_center,
    )
    renderer = pyrender.OffscreenRenderer(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Mp4FrameWriter | None = None
    step = max(1, int(stride))
    output_fps = max(1, int(round(float(REALTIME_POSE_FPS) / step)))
    try:
        for written_index, frame_index in enumerate(range(0, result.frame_count, step)):
            scene.set_pose(camera_node, pose=layout.camera_poses[frame_index])
            viewport = render_view(
                renderer=renderer,
                scene=scene,
                frame_index=frame_index,
                sequences=sequences,
                faces=faces,
                method_offsets=layout.method_offsets,
                tracker_positions=result.tracker_positions,
                tracker_available=result.tracker_available,
                camera_pose=layout.camera_poses[frame_index],
            )
            frame_rgb = compose_frame(
                viewport_rgb=viewport,
                frame_index=frame_index,
                result=result,
                metrics=metrics,
                direction=direction,
                transition_order=transition_order,
            )
            if writer is None:
                writer = Mp4FrameWriter(output_path, frame_rgb, output_fps)
                for _ in range(INTRO_FRAME_COUNT):
                    writer.append(frame_rgb)
            writer.append(frame_rgb)
            if written_index % 30 == 0 or frame_index + step >= result.frame_count:
                print(
                    f"[progressive-render] {output_path.name}: "
                    f"{min(frame_index + step, result.frame_count)}/{result.frame_count}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        with suppress(Exception):
            renderer.delete()
    print(f"[progressive-render] wrote {output_path}", flush=True)
    return output_path


def render_source(
    *,
    source_path: Path,
    predictor,
    dit,
    diffusion,
    device: torch.device,
    normalizer: RealtimePoseNormalizer,
    args,
    direction: str,
    transition_order: tuple[str, ...],
    transition_indices: tuple[int, ...],
    dit_weight_source: str,
) -> dict[str, Path]:
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source_npz 不存在：{source_path}")
    relative_path, amass_path = resolve_amass_path(
        source_path, args.source_dir, args.amass_dir
    )
    print(f"[progressive-render] inference {relative_path}", flush=True)
    source = load_realtime_source(source_path)
    result = run_progressive_sequence(
        source=source,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        args=args,
        direction=direction,
        transition_indices=transition_indices,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    direction_token = "6to3" if direction == "dropout" else "3to6"
    output_mp4 = (
        output_dir
        / f"{sequence_stem(relative_path)}_progressive_{direction_token}.mp4"
    )
    npz_path, json_path, report = write_sidecars(
        output_mp4=output_mp4,
        source_path=source_path,
        relative_path=relative_path,
        amass_path=amass_path,
        result=result,
        direction=direction,
        transition_order=transition_order,
        args=args,
        dit_weight_source=dit_weight_source,
        sampling_steps=int(diffusion.num_timesteps),
    )
    if not bool(args.skip_render):
        mesh_inputs = load_progressive_mesh_inputs(
            source_path=source_path,
            amass_path=amass_path,
            amass_dir=args.amass_dir,
            frame_start=result.frame_start,
            frame_end_exclusive=result.frame_end_exclusive,
        )
        sequences, faces = build_progressive_mesh_sequences(
            result=result,
            mesh_inputs=mesh_inputs,
            smpl_model_dir=args.smpl_model_dir,
        )
        render_video(
            output_path=output_mp4,
            result=result,
            sequences=sequences,
            faces=faces,
            metrics=report["overall_metrics"],
            direction=direction,
            transition_order=transition_order,
            stride=int(args.stride),
        )
    return {"video": output_mp4, "npz": npz_path, "json": json_path}


def main(argv: list[str] | None = None) -> list[dict[str, Path]]:
    args = parse_and_load_from_model(build_arg_parser(), argv)
    if int(args.stride) <= 0:
        raise ValueError("--stride 必须为正整数。")
    if 0 < int(args.max_frames) < MIN_PROGRESSIVE_SCORED_FRAMES:
        raise ValueError(
            f"--max_frames 必须为 0 或至少 {MIN_PROGRESSIVE_SCORED_FRAMES}。"
        )
    direction = str(args.direction)
    if direction == "dropout":
        transition_order = tuple(str(name) for name in args.drop_order)
        transition_indices = validate_drop_order(transition_order)
    else:
        transition_order = tuple(str(name) for name in args.add_order)
        transition_indices = validate_add_order(transition_order)
    fixseed(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    dit, diffusion = create_model_and_diffusion(args)
    dit, dit_weight_source = load_checkpoint_model(
        dit, args.dit_model_path, device, use_ema=args.use_ema
    )
    predictor = load_realtime_pose_predictor(args.predictor_model_path, device)
    normalizer = RealtimePoseNormalizer(
        args.normalizer_dir, disable=not bool(args.normalize_input)
    )
    outputs = []
    for source_path in args.source_npz:
        outputs.append(
            render_source(
                source_path=source_path,
                predictor=predictor,
                dit=dit,
                diffusion=diffusion,
                device=device,
                normalizer=normalizer,
                args=args,
                direction=direction,
                transition_order=transition_order,
                transition_indices=transition_indices,
                dit_weight_source=dit_weight_source,
            )
        )
    return outputs


if __name__ == "__main__":
    main()
