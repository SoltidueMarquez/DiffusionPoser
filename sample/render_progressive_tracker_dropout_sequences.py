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
from scipy.spatial.transform import Rotation, Slerp
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
from data_loaders.realtime_pose_kinematics import (
    rotation_6d_forward_up_np,
    rotation_6d_to_matrix_np,
)
from data_loaders.sensor_masking import (
    ALL_SIX_AVAILABLE,
    CORE_THREE_AVAILABLE,
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
    REALTIME_POSE_TARGET_DIM,
    TRACKER_NAMES,
    TRACKER_TO_JOINT,
)
from eval.evaluate_realtime_pose_predictor import (
    PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
)
from sample.evaluate_progressive_tracker_addition import (
    build_equal_quarter_tracker_schedule as build_addition_tracker_schedule,
)
from sample.evaluate_progressive_tracker_dropout import (
    MIN_PROGRESSIVE_SCORED_FRAMES,
    OPTIONAL_TRACKER_NAME_TO_INDEX,
    build_equal_quarter_tracker_schedule as build_dropout_tracker_schedule,
)
from sample.evaluate_tracker_reconnection import (
    MIN_STAGE_FRAMES as MIN_RECONNECTION_STAGE_FRAMES,
    RECONNECT_TRACKER_NAME_TO_INDEX,
    build_tracker_reconnection_schedule,
)
from sample.realtime_pose_longseq_evaluator import (
    TrackerSequenceSchedule,
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
# 两路人物在 720p 画面中需要比通用三路展示留出更多边缘，避免脚部动作和
# Tracker 图标贴近画面边界；只改变可视化相机，不参与推理或指标计算。
PROGRESSIVE_CAMERA_FIT_PADDING = 1.35
RECONNECT_ERROR_FRAME_OFFSETS = (0, 4, 9, 19, 29)
RECONNECT_ERROR_THRESHOLDS_CM = (5.0, 2.0, 1.0)


@dataclass(frozen=True)
class ProgressiveSequenceResult:
    """动态 schedule 的逐帧闭环输出，时间轴从指定 source frame 开始。"""

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
    runtime_tracker_positions: np.ndarray | None = None
    runtime_tracker_rotations_6d: np.ndarray | None = None
    tracker_blend_alpha: np.ndarray | None = None
    activation_blend_frames: int = 0

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
            "重跑指定长序列的动态 Tracker 6→3/3→6、固定三点或 3→4 重连协议，"
            "并输出 SMPL-H 视频、逐帧 NPZ 和连续性诊断 JSON。"
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
        choices=("dropout", "addition", "core_only", "reconnection"),
        help=(
            "dropout 运行 6→3；addition 运行 3→6；"
            "core_only 全程保持核心三点；reconnection 只运行一次 3→4 重连。"
        ),
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
        "--reconnect_tracker",
        default="hip",
        choices=tuple(RECONNECT_TRACKER_NAME_TO_INDEX),
        help="3→4 重连展示中作为第四点恢复的 Tracker。",
    )
    protocol.add_argument(
        "--reconnect_after_frames",
        default=0,
        type=int,
        help=(
            "3→4 展示在多少个计分帧后恢复第四点；0 表示保持默认等分边界。"
            "该参数只改变单序列展示，不改变正式 150/150 评估协议。"
        ),
    )
    protocol.add_argument(
        "--source_frame_start",
        default=REALTIME_POSE_EVAL_METRICS_START_FRAME,
        type=int,
        help=(
            "动态 schedule 在哪个 30 Hz source frame 开始；此前帧继续使用预热 "
            "Tracker 配置并参与闭环历史。"
        ),
    )
    protocol.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="从 source_frame_start 起最多渲染多少帧；0 表示直到序列结束。",
    )
    protocol.add_argument(
        "--activation_blend_frames",
        default=0,
        type=int,
        help=(
            "3→6 添加或 3→4 重连中新 Tracker 测量的渐入帧数；"
            "位置使用 LERP、旋转使用 SLERP，0 保持硬切换。"
        ),
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


def validate_reconnect_tracker(reconnect_tracker: str) -> int:
    name = str(reconnect_tracker)
    if name not in RECONNECT_TRACKER_NAME_TO_INDEX:
        raise ValueError(f"未知重连 Tracker：{name}")
    return int(RECONNECT_TRACKER_NAME_TO_INDEX[name])


def build_transition_schedule(
    *,
    scored_frame_count: int,
    direction: str,
    transition_indices: tuple[int, ...],
    reconnect_after_frames: int = 0,
):
    """默认复用正式评估 schedule；仅重连展示可显式推迟单次边界。"""

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
    if direction == "reconnection":
        if len(transition_indices) != 1:
            raise ValueError("reconnection 必须恰好指定一个重连 Tracker。")
        frame_count = int(scored_frame_count)
        reconnect_boundary = int(reconnect_after_frames)
        if reconnect_boundary < 0:
            raise ValueError("reconnect_after_frames 不能为负数。")
        if reconnect_boundary == 0:
            if frame_count % 2 != 0:
                raise ValueError("默认 3→4 重连展示的计分帧数必须为偶数。")
            return build_tracker_reconnection_schedule(
                scored_frame_count=frame_count,
                reconnect_tracker_index=int(transition_indices[0]),
                stage_frames=frame_count // 2,
            )
        remaining_frames = frame_count - reconnect_boundary
        if min(reconnect_boundary, remaining_frames) < MIN_RECONNECTION_STAGE_FRAMES:
            raise ValueError(
                "自定义重连边界要求前后阶段均至少包含 "
                f"{MIN_RECONNECTION_STAGE_FRAMES} 帧，实际为 "
                f"{reconnect_boundary}/{remaining_frames}。"
            )
        # 自定义边界只服务于定性展示，因此在这里直接构造非等长的 3→4 mask，
        # 不放宽 evaluate_tracker_reconnection 中正式 150/150 协议的约束。
        tracker_available = np.broadcast_to(
            np.asarray(CORE_THREE_AVAILABLE, dtype=bool)[None],
            (frame_count, len(CORE_THREE_AVAILABLE)),
        ).copy()
        tracker_available[reconnect_boundary:, int(transition_indices[0])] = True
        stage_indices = np.zeros((frame_count,), dtype=np.int64)
        stage_indices[reconnect_boundary:] = 1
        return TrackerSequenceSchedule(
            tracker_available=tracker_available,
            stage_indices=stage_indices,
        )
    if direction == "core_only":
        if transition_indices:
            raise ValueError("core_only 不接受 Tracker 切换顺序。")
        frame_count = int(scored_frame_count)
        return TrackerSequenceSchedule(
            tracker_available=np.broadcast_to(
                np.asarray(CORE_THREE_AVAILABLE, dtype=bool)[None],
                (frame_count, len(CORE_THREE_AVAILABLE)),
            ).copy(),
            stage_indices=np.zeros((frame_count,), dtype=np.int64),
        )
    raise ValueError(
        "direction 必须为 dropout/addition/core_only/reconnection，"
        f"实际为 {direction}"
    )


def warmup_tracker_available(direction: str) -> np.ndarray:
    """展示必须复用正式协议的预热 mask，否则第一阶段历史不可比较。"""

    if direction == "dropout":
        return np.asarray(ALL_SIX_AVAILABLE, dtype=bool)
    if direction in ("addition", "core_only", "reconnection"):
        return np.asarray(CORE_THREE_AVAILABLE, dtype=bool)
    raise ValueError(
        "direction 必须为 dropout/addition/core_only/reconnection，"
        f"实际为 {direction}"
    )


def smoothstep_activation_alpha(frame_offset: int, frame_count: int) -> float:
    """返回新 Tracker 第 `frame_offset` 帧的 smoothstep 渐入权重。"""

    if int(frame_count) <= 0:
        raise ValueError("frame_count 必须为正整数。")
    if int(frame_offset) < 0:
        raise ValueError("frame_offset 不能为负数。")
    unit = float(np.clip((int(frame_offset) + 1) / int(frame_count), 0.0, 1.0))
    return unit * unit * (3.0 - 2.0 * unit)


def interpolate_tracker_measurement(
    *,
    anchor_position: np.ndarray,
    anchor_rotation: np.ndarray,
    measured_position: np.ndarray,
    measured_rotation_6d: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """从切换前部署姿态平滑过渡到当前真实 Tracker 测量。"""

    weight = float(alpha)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"alpha 必须位于 [0,1]，实际为 {weight}")
    anchor_position = np.asarray(anchor_position, dtype=np.float64)
    measured_position = np.asarray(measured_position, dtype=np.float64)
    anchor_rotation = np.asarray(anchor_rotation, dtype=np.float64)
    measured_rotation = rotation_6d_to_matrix_np(measured_rotation_6d)
    if anchor_position.shape != (3,) or measured_position.shape != (3,):
        raise ValueError(
            f"Tracker 位置应为 [3]，实际为 {anchor_position.shape}/{measured_position.shape}"
        )
    if anchor_rotation.shape != (3, 3) or measured_rotation.shape != (3, 3):
        raise ValueError(
            f"Tracker 旋转应为 [3,3]，实际为 {anchor_rotation.shape}/{measured_rotation.shape}"
        )
    position = (1.0 - weight) * anchor_position + weight * measured_position
    key_rotations = Rotation.from_matrix(
        np.stack([anchor_rotation, measured_rotation], axis=0)
    )
    rotation = Slerp([0.0, 1.0], key_rotations)([weight]).as_matrix()[0]
    return (
        position.astype(np.float32),
        rotation_6d_forward_up_np(rotation).astype(np.float32),
    )


def progressive_output_filename(
    relative_path: Path,
    *,
    direction: str,
    activation_blend_frames: int,
    transition_order: tuple[str, ...] | None = None,
    stage_frames: int | None = None,
) -> str:
    """soft-start 使用独立文件名，避免覆盖原来的硬切换展示。"""

    blend_frames = int(activation_blend_frames)
    blend_token = f"_soft{blend_frames}f" if blend_frames > 0 else ""
    if direction == "core_only":
        frame_count = int(stage_frames or 0)
        if frame_count <= 0:
            raise ValueError("core_only 输出文件名需要正数计分帧数。")
        return f"{sequence_stem(relative_path)}_core3_{frame_count}f.mp4"
    if direction == "reconnection":
        order = tuple(transition_order or ())
        if len(order) != 1:
            raise ValueError("reconnection 输出文件名需要一个重连 Tracker。")
        hold_frames = int(stage_frames or 0)
        if hold_frames <= 0:
            raise ValueError("reconnection 输出文件名需要正数 stage_frames。")
        return (
            f"{sequence_stem(relative_path)}_reconnect_{order[0]}_"
            f"after{hold_frames}f{blend_token}.mp4"
        )

    direction_token = "6to3" if direction == "dropout" else "3to6"
    return (
        f"{sequence_stem(relative_path)}_progressive_"
        f"{direction_token}{blend_token}.mp4"
    )


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
    """复用正式评估的预热、等分 schedule、闭环 runtime 和噪声规则。"""

    blend_frames = int(getattr(args, "activation_blend_frames", 0))
    world_rotations = compute_source_joint_rotations_world(source)
    source_frame_start = int(
        getattr(args, "source_frame_start", REALTIME_POSE_EVAL_METRICS_START_FRAME)
    )
    max_frames = int(args.max_frames)
    last = (
        len(world_rotations)
        if max_frames <= 0
        else min(len(world_rotations), source_frame_start + max_frames)
    )
    scored_frame_count = last - source_frame_start
    minimum_scored_frames = (
        2 * MIN_RECONNECTION_STAGE_FRAMES
        if direction == "reconnection"
        else MIN_PROGRESSIVE_SCORED_FRAMES
    )
    if scored_frame_count < minimum_scored_frames:
        raise ValueError(
            f"动态展示至少需要 {minimum_scored_frames} 个计分帧，"
            f"实际为 {scored_frame_count}。"
        )
    schedule = build_transition_schedule(
        scored_frame_count=scored_frame_count,
        direction=direction,
        transition_indices=transition_indices,
        reconnect_after_frames=int(getattr(args, "reconnect_after_frames", 0)),
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
    runtime_tracker_positions: list[np.ndarray] = []
    runtime_tracker_rotations_6d: list[np.ndarray] = []
    tracker_blend_alpha: list[np.ndarray] = []
    noise_generator = create_eval_noise_generator(args.seed, device)
    previous_available = np.asarray(warmup_available, dtype=bool).copy()
    previous_result = None
    # tracker_index -> (起始帧、切换前部署位置、切换前部署旋转)
    activation_ramps: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
    for current in range(PREDICTOR_EVAL_FIRST_GENERATED_FRAME, last):
        if current < source_frame_start:
            frame_tracker_available = warmup_available
        else:
            scored_offset = current - source_frame_start
            frame_tracker_available = schedule.tracker_available[scored_offset]
        frame_tracker_available = np.asarray(frame_tracker_available, dtype=bool)
        frame_tracker_positions = np.asarray(
            source["tracker_pos_world"][current], dtype=np.float32
        ).copy()
        frame_tracker_rotations_6d = np.asarray(
            source["tracker_rot_world_6d"][current], dtype=np.float32
        ).copy()
        frame_blend_alpha = frame_tracker_available.astype(np.float32)

        newly_added = ~previous_available & frame_tracker_available
        if blend_frames > 0 and np.any(newly_added):
            if previous_result is None:
                raise RuntimeError("Tracker 渐入缺少切换前一帧的部署姿态。")
            for tracker_index in np.flatnonzero(newly_added).tolist():
                joint_index = int(TRACKER_TO_JOINT[tracker_index])
                activation_ramps[tracker_index] = (
                    current,
                    np.asarray(
                        previous_result.resolved_pose.joints_world[joint_index],
                        dtype=np.float32,
                    ).copy(),
                    np.asarray(
                        previous_result.resolved_pose.joint_rotations_world[joint_index],
                        dtype=np.float32,
                    ).copy(),
                )

        finished_ramps: list[int] = []
        for tracker_index, (start_frame, anchor_position, anchor_rotation) in (
            activation_ramps.items()
        ):
            if not frame_tracker_available[tracker_index]:
                finished_ramps.append(tracker_index)
                continue
            frame_offset = current - start_frame
            if frame_offset >= blend_frames:
                finished_ramps.append(tracker_index)
                continue
            alpha = smoothstep_activation_alpha(frame_offset, blend_frames)
            blended_position, blended_rotation_6d = interpolate_tracker_measurement(
                anchor_position=anchor_position,
                anchor_rotation=anchor_rotation,
                measured_position=frame_tracker_positions[tracker_index],
                measured_rotation_6d=frame_tracker_rotations_6d[tracker_index],
                alpha=alpha,
            )
            frame_tracker_positions[tracker_index] = blended_position
            frame_tracker_rotations_6d[tracker_index] = blended_rotation_6d
            frame_blend_alpha[tracker_index] = alpha
        for tracker_index in finished_ramps:
            del activation_ramps[tracker_index]

        noise = torch.randn(
            (1, REALTIME_POSE_TARGET_DIM),
            generator=noise_generator,
            device=device,
        )
        result = runtime.step(
            frame_tracker_positions,
            frame_tracker_rotations_6d,
            frame_tracker_available,
            float(source["root_pos_world"][current, 1]),
            noise=noise,
        )
        if current >= source_frame_start:
            rotations.append(result.resolved_pose.joint_rotations_world)
            positions.append(result.resolved_pose.joints_world)
            root_yaw.append(result.resolved_pose.root_yaw_world)
            runtime_tracker_positions.append(frame_tracker_positions)
            runtime_tracker_rotations_6d.append(frame_tracker_rotations_6d)
            tracker_blend_alpha.append(frame_blend_alpha)
        previous_available = frame_tracker_available.copy()
        previous_result = result

    selected = slice(source_frame_start, last)
    return ProgressiveSequenceResult(
        frame_start=source_frame_start,
        frame_end_exclusive=last,
        tracker_available=np.asarray(schedule.tracker_available, dtype=bool),
        stage_indices=np.asarray(schedule.stage_indices, dtype=np.int64),
        target_rotations=np.asarray(world_rotations[selected], dtype=np.float32),
        target_positions=np.asarray(source["joints_world"][selected], dtype=np.float32),
        deployed_rotations=np.stack(rotations).astype(np.float32),
        deployed_positions=np.stack(positions).astype(np.float32),
        deployed_root_yaw=np.asarray(root_yaw, dtype=np.float32),
        tracker_positions=np.asarray(source["tracker_pos_world"][selected], dtype=np.float32),
        runtime_tracker_positions=np.stack(runtime_tracker_positions).astype(np.float32),
        runtime_tracker_rotations_6d=np.stack(runtime_tracker_rotations_6d).astype(
            np.float32
        ),
        tracker_blend_alpha=np.stack(tracker_blend_alpha).astype(np.float32),
        activation_blend_frames=blend_frames,
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
    """量化各 mask 切换的单帧关节步长，并与局部正常运动比较。"""

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
        elif direction in ("addition", "reconnection"):
            changed = ~previous_available & current_available
            tracker_key = (
                "reconnected_tracker"
                if direction == "reconnection"
                else "added_tracker"
            )
        else:
            raise ValueError(
                "direction 必须为 dropout/addition/reconnection，"
                f"实际为 {direction}"
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


def compute_reconnection_diagnostics(
    result: ProgressiveSequenceResult,
) -> dict[str, object]:
    """量化第四点从不可用到恢复后的逐帧位置收敛速度。"""

    boundaries = np.flatnonzero(np.diff(result.stage_indices) > 0) + 1
    if boundaries.shape != (1,):
        raise ValueError(
            f"3→4 重连应恰好包含一个阶段边界，实际为 {boundaries.tolist()}。"
        )
    boundary = int(boundaries[0])
    previous_available = np.asarray(result.tracker_available[boundary - 1], dtype=bool)
    current_available = np.asarray(result.tracker_available[boundary], dtype=bool)
    changed_indices = np.flatnonzero(~previous_available & current_available)
    if changed_indices.shape != (1,):
        raise ValueError(
            "3→4 重连边界应恰好恢复一个 Tracker，"
            f"实际为 {changed_indices.tolist()}。"
        )
    tracker_index = int(changed_indices[0])
    joint_index = int(TRACKER_TO_JOINT[tracker_index])
    # 与真实 Tracker 测量比较，而不是与可选 soft-start 的运行时插值目标比较，
    # 这样硬重连和渐入展示的恢复速度保持同一物理参照。
    position_error_cm = (
        np.linalg.norm(
            result.deployed_positions[:, joint_index]
            - result.tracker_positions[:, tracker_index],
            axis=-1,
        )
        * 100.0
    )
    post_error_cm = position_error_cm[boundary:]
    sampled_errors: dict[str, float | None] = {}
    for frame_offset in RECONNECT_ERROR_FRAME_OFFSETS:
        sampled_errors[str(frame_offset)] = (
            float(post_error_cm[frame_offset])
            if frame_offset < post_error_cm.shape[0]
            else None
        )

    frames_to_threshold: dict[str, int | None] = {}
    for threshold_cm in RECONNECT_ERROR_THRESHOLDS_CM:
        reached = np.flatnonzero(post_error_cm <= float(threshold_cm))
        # 使用 1-based 帧数：1 表示第一个重连帧已经进入阈值。
        frames_to_threshold[f"{threshold_cm:g}"] = (
            int(reached[0]) + 1 if reached.size else None
        )
    return {
        "reconnect_tracker": str(TRACKER_NAMES[tracker_index]),
        "reconnect_source_frame": int(result.frame_start + boundary),
        "pre_reconnect_position_error_cm": float(position_error_cm[boundary - 1]),
        "post_reconnect_position_error_cm_by_frame_offset": sampled_errors,
        "frames_to_position_error_threshold_cm": frames_to_threshold,
    }


def compute_metrics(result: ProgressiveSequenceResult) -> tuple[dict, list[dict]]:
    stage_count = int(np.max(result.stage_indices)) + 1
    overall, stages = compute_sequence_metrics_by_stage(
        predicted_global_rotations=result.deployed_rotations,
        target_global_rotations=result.target_rotations,
        predicted_joint_positions=result.deployed_positions,
        target_joint_positions=result.target_positions,
        stage_indices=result.stage_indices,
        stage_count=stage_count,
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
    runtime_positions = (
        result.tracker_positions
        if result.runtime_tracker_positions is None
        else result.runtime_tracker_positions
    )
    runtime_rotations_6d = result.runtime_tracker_rotations_6d
    blend_alpha = (
        result.tracker_available.astype(np.float32)
        if result.tracker_blend_alpha is None
        else result.tracker_blend_alpha
    )
    np.savez_compressed(
        npz_path,
        target_rotations_world=result.target_rotations,
        target_joints_world=result.target_positions,
        deployed_rotations_world=result.deployed_rotations,
        deployed_joints_world=result.deployed_positions,
        deployed_root_yaw=result.deployed_root_yaw,
        tracker_pos_world=result.tracker_positions,
        runtime_tracker_pos_world=runtime_positions,
        runtime_tracker_rot_world_6d=(
            np.empty((0, 6, 6), dtype=np.float32)
            if runtime_rotations_6d is None
            else runtime_rotations_6d
        ),
        tracker_blend_alpha=blend_alpha,
        tracker_available=result.tracker_available,
        stage_indices=result.stage_indices,
    )
    stage_count = int(np.max(result.stage_indices)) + 1
    tracker_counts = [
        int(result.tracker_available[result.stage_indices == stage_index][0].sum())
        for stage_index in range(stage_count)
    ]
    is_dropout = direction == "dropout"
    is_reconnection = direction == "reconnection"
    is_core_only = direction == "core_only"
    if is_dropout:
        experiment = "progressive_tracker_dropout_6_to_3_showcase"
        transition_metadata = {"drop_order": list(transition_order)}
        stage_policy = "equal_scored_quarters"
    elif is_reconnection:
        experiment = "tracker_reconnection_3_to_4_showcase"
        reconnect_after_frames = int(np.count_nonzero(result.stage_indices == 0))
        transition_metadata = {
            "reconnect_tracker": str(transition_order[0]),
            "reconnect_after_frames": reconnect_after_frames,
        }
        stage_policy = (
            "custom_reconnect_boundary"
            if int(getattr(args, "reconnect_after_frames", 0)) > 0
            else "equal_scored_halves"
        )
    elif is_core_only:
        experiment = "tracker_core_only_showcase"
        transition_metadata = {}
        stage_policy = "constant_core_three"
    else:
        experiment = "progressive_tracker_addition_3_to_6_showcase"
        transition_metadata = {"add_order": list(transition_order)}
        stage_policy = "equal_scored_quarters"
    report = {
        "experiment": experiment,
        "source_path": str(source_path),
        "source_relative_path": str(relative_path),
        "amass_path": str(amass_path),
        "frame_start": result.frame_start,
        "frame_end_exclusive": result.frame_end_exclusive,
        "frames": result.frame_count,
        "fps": float(REALTIME_POSE_FPS),
        **transition_metadata,
        "tracker_counts": tracker_counts,
        "warmup_tracker_count": 6 if is_dropout else 3,
        "stage_policy": stage_policy,
        "activation_blend": {
            "frames": int(result.activation_blend_frames),
            "curve": (
                "smoothstep" if int(result.activation_blend_frames) > 0 else "none"
            ),
            "position": "lerp",
            "rotation": "slerp",
            "anchor": "previous_deployed_tracker_joint",
            "tracker_available_remains_binary": True,
        },
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
        "visualization_camera_fit_padding": PROGRESSIVE_CAMERA_FIT_PADDING,
        "npz_path": str(npz_path),
        "video_path": str(output_mp4),
    }
    if is_reconnection:
        report["reconnection_diagnostics"] = compute_reconnection_diagnostics(result)
    elif direction == "addition":
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
    stage_count = int(np.max(result.stage_indices)) + 1
    tracker_count = int(result.tracker_available[frame_index].sum())
    stage_start = int(np.flatnonzero(result.stage_indices == stage)[0])
    changed = "none" if stage == 0 else ", ".join(transition_order[:stage])
    is_dropout = direction == "dropout"
    is_reconnection = direction == "reconnection"
    is_core_only = direction == "core_only"
    if is_dropout:
        direction_label = "Dynamic 6→3"
        change_label = "Dropped"
    elif is_reconnection:
        direction_label = "Reconnect 3→4"
        change_label = "Reconnected"
    elif is_core_only:
        direction_label = "Always 3 trackers"
        change_label = "Active"
        changed = "head + wrists"
    else:
        direction_label = "Dynamic 3→6"
        change_label = "Added"
    if int(result.activation_blend_frames) > 0:
        direction_label += f" Soft-start {int(result.activation_blend_frames)}f"

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
        (
            f"{direction_label}  |  Stage {stage + 1}/{stage_count}  |  "
            f"{tracker_count} trackers"
        ),
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
        action = (
            "removed"
            if is_dropout
            else "reconnected" if is_reconnection else "added"
        )
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
    for stage_index in range(stage_count):
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
    # 可视化始终显示物理 Tracker 的原始测量；runtime 插值仍用于 Soft 推理，
    # 但不再用橙色球展示，以免混淆测量位置与模型的有效输入。
    displayed_tracker_positions = result.tracker_positions
    layout = build_presentation_layout(
        sequences=sequences,
        tracker_pos_world=displayed_tracker_positions,
        method_order=METHOD_ORDER,
        tracker_available_by_method=np.asarray(
            [[False] * 6, [True] * 6], dtype=bool
        ),
        follow_method_name="GT",
        camera_fit_padding=PROGRESSIVE_CAMERA_FIT_PADDING,
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
                tracker_positions=displayed_tracker_positions,
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
    stage_frames = int(np.count_nonzero(result.stage_indices == 0))
    output_mp4 = output_dir / progressive_output_filename(
        relative_path,
        direction=direction,
        activation_blend_frames=int(args.activation_blend_frames),
        transition_order=transition_order,
        stage_frames=stage_frames,
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
    if int(args.activation_blend_frames) < 0:
        raise ValueError("--activation_blend_frames 不能为负数。")
    reconnect_after_frames = int(args.reconnect_after_frames)
    if reconnect_after_frames < 0:
        raise ValueError("--reconnect_after_frames 不能为负数。")
    direction = str(args.direction)
    source_frame_start = int(args.source_frame_start)
    if source_frame_start < REALTIME_POSE_EVAL_METRICS_START_FRAME:
        raise ValueError(
            "--source_frame_start 不能早于正式可重建帧 "
            f"{REALTIME_POSE_EVAL_METRICS_START_FRAME}。"
        )
    max_frames = int(args.max_frames)
    if direction == "reconnection":
        minimum_frames = 2 * MIN_RECONNECTION_STAGE_FRAMES
        if max_frames < minimum_frames:
            raise ValueError(
                "3→4 重连展示要求 --max_frames 为不小于 "
                f"{minimum_frames} 的正数。"
            )
        if reconnect_after_frames == 0 and max_frames % 2 != 0:
            raise ValueError("默认 3→4 重连展示要求 --max_frames 为偶数。")
        if reconnect_after_frames > 0 and min(
            reconnect_after_frames, max_frames - reconnect_after_frames
        ) < MIN_RECONNECTION_STAGE_FRAMES:
            raise ValueError(
                "--reconnect_after_frames 要求重连前后均至少保留 "
                f"{MIN_RECONNECTION_STAGE_FRAMES} 帧。"
            )
    elif 0 < max_frames < MIN_PROGRESSIVE_SCORED_FRAMES:
        raise ValueError(
            f"--max_frames 必须为 0 或至少 {MIN_PROGRESSIVE_SCORED_FRAMES}。"
        )
    if direction != "reconnection" and reconnect_after_frames > 0:
        raise ValueError("--reconnect_after_frames 只用于 direction=reconnection。")
    if direction not in ("addition", "reconnection") and int(
        args.activation_blend_frames
    ) > 0:
        raise ValueError(
            "--activation_blend_frames 目前只用于 addition/reconnection 展示。"
        )
    if direction == "dropout":
        transition_order = tuple(str(name) for name in args.drop_order)
        transition_indices = validate_drop_order(transition_order)
    elif direction == "addition":
        transition_order = tuple(str(name) for name in args.add_order)
        transition_indices = validate_add_order(transition_order)
    elif direction == "reconnection":
        reconnect_tracker = str(args.reconnect_tracker)
        transition_order = (reconnect_tracker,)
        transition_indices = (validate_reconnect_tracker(reconnect_tracker),)
    else:
        transition_order = ()
        transition_indices = ()
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
