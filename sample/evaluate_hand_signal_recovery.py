from __future__ import annotations

import argparse
import json
import time
from pathlib import Path, PurePosixPath

import numpy as np
import torch

from data_loaders.build_realtime_longseq_eval_set import (
    DEFAULT_SOURCE_DIR,
    DEFAULT_SPLIT_DIR,
    read_longseq_source_entries,
    resolve_source_entry_path,
)
from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    load_realtime_source,
    normalize_split_key,
)
from data_loaders.sensor_masking import (
    CORE_THREE_AVAILABLE,
    HAND_TRACKER_INDICES,
    LEFT_HAND_TRACKER_INDEX,
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
    REALTIME_POSE_MODEL_TOKEN_LENGTH,
    REALTIME_POSE_TARGET_DIM,
    RIGHT_HAND_TRACKER_INDEX,
    TRACKER_COUNT,
)
from eval.evaluate_realtime_pose_predictor import (
    PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
    evaluation_last_frame_exclusive,
)
from eval.realtime_pose_metrics import (
    aggregate_rpm_p2_mc_metrics,
    aggregate_rpm_transition_jerk_segments,
    compute_rpm_p2_mc_metrics,
    extract_rpm_transition_jerk_segments,
)
from sample.realtime_pose_longseq_evaluator import (
    _append_tracker_errors,
    create_eval_noise_generator,
    create_longseq_runtime,
)
from sample.realtime_pose_runtime import WorldPoseState, decode_and_resolve_pose
from sample.tracker_activation_blending import (
    TrackerActivationRamps,
    apply_tracker_activation_blend,
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


DEFAULT_TRANSITION_SECONDS = 0.5
DEFAULT_GAP_FRAME_OFFSET = 1
DEFAULT_ACTIVATION_BLEND_FRAMES = 10
RPM_P2_HAND_GAP_MASKER = "seg_hands_idp"
HAND_GAP_NAMES = ("left_hand", "right_hand")
HAND_GAP_TRACKER_INDICES = (
    LEFT_HAND_TRACKER_INDEX,
    RIGHT_HAND_TRACKER_INDEX,
)


# region 参数与 RPM-P2 gap 映射


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "在 RPM-P2 官方 hand_tracking gap 上评估左右手断线与恢复；"
            "这是当前模型未见过的核心 Tracker 缺失诊断。"
        )
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    group = parser.add_argument_group("RPM-P2 hand signal recovery")
    group.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR)
    group.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    group.add_argument("--split", default="test")
    group.add_argument("--normalizer_dir", required=True)
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument("--gap_config", required=True, type=Path)
    group.add_argument("--limit", default=0, type=int)
    group.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="从 P2 正式计分帧开始最多运行的帧数；0 表示整条序列。",
    )
    group.add_argument(
        "--transition_seconds",
        default=DEFAULT_TRANSITION_SECONDS,
        type=float,
        help=(
            "PJ/AUJ 的过渡窗口。默认 0.5 秒以复现 RPM 发布代码；"
            "论文正文写作 1 秒。"
        ),
    )
    group.add_argument(
        "--gap_frame_offset",
        default=DEFAULT_GAP_FRAME_OFFSET,
        type=int,
        help=(
            "官方 gap 基于删除首帧后的 54D tracking feature；"
            "映射回本项目 source 默认整体加 1 帧。"
        ),
    )
    group.add_argument(
        "--activation_blend_frames",
        default=DEFAULT_ACTIVATION_BLEND_FRAMES,
        type=int,
        help=(
            "手 Tracker 重连后复用项目 soft-start 的渐入帧数；"
            "位置 LERP、旋转 SLERP、权重 smoothstep，0 表示硬重连。"
        ),
    )
    group.add_argument("--output_json", required=True)
    return parser


def load_rpm_p2_hand_gap_config(path: str | Path) -> tuple[dict, dict]:
    gap_path = Path(path).expanduser().resolve()
    if not gap_path.is_file():
        raise FileNotFoundError(f"RPM-P2 hand gap config 不存在：{gap_path}")
    payload = json.loads(gap_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    gaps = payload.get("gaps")
    if not isinstance(metadata, dict) or not isinstance(gaps, dict) or not gaps:
        raise ValueError("RPM-P2 hand gap config 必须包含非空 metadata/gaps。")
    if str(metadata.get("masker")) != RPM_P2_HAND_GAP_MASKER:
        raise ValueError(
            "RPM-P2 hand gap config 的 masker 必须为 "
            f"{RPM_P2_HAND_GAP_MASKER}。"
        )
    normalized: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {}
    for sequence_key, hand_gaps in gaps.items():
        if not isinstance(hand_gaps, list) or len(hand_gaps) != len(HAND_GAP_NAMES):
            raise ValueError(f"{sequence_key} 必须恰好包含左/右手两组 gap。")
        normalized_hands = []
        for values in hand_gaps:
            normalized_intervals = []
            previous_end = -1
            for interval in values:
                if not isinstance(interval, list) or len(interval) != 2:
                    raise ValueError(f"{sequence_key} 包含非法 gap：{interval}")
                start, end = (int(interval[0]), int(interval[1]))
                if start < 0 or end <= start or start < previous_end:
                    raise ValueError(f"{sequence_key} 包含非法或重叠 gap：{interval}")
                normalized_intervals.append((start, end))
                previous_end = end
            normalized_hands.append(tuple(normalized_intervals))
        normalized[str(sequence_key)] = tuple(normalized_hands)
    return dict(metadata), normalized


def build_gap_key_by_source(
    split_file: str | Path,
    gap_keys: set[str] | list[str] | tuple[str, ...],
) -> dict[str, str]:
    """按 RPM prepare_data 的“数据集内 1-based 行号”恢复 gap key。"""

    path = Path(split_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RPM-P2 split 文件不存在：{path}")
    available_gap_keys = {str(key) for key in gap_keys}
    dataset_names = {
        key.rsplit("-", 1)[0]
        for key in available_gap_keys
        if key.rsplit("-", 1)[-1].isdigit()
    }
    counters = {dataset_name: 0 for dataset_name in dataset_names}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        source_key = normalize_split_key(raw_line)
        dataset_name = PurePosixPath(source_key).parts[0]
        if dataset_name not in counters:
            raise ValueError(
                f"split 中的数据集 {dataset_name} 在 gap config 中没有对应 key。"
            )
        counters[dataset_name] += 1
        gap_key = f"{dataset_name}-{counters[dataset_name]}"
        if gap_key not in available_gap_keys:
            raise ValueError(f"gap config 缺少 {source_key} 对应的 {gap_key}。")
        result[source_key] = gap_key
    if set(result.values()) != available_gap_keys:
        missing = sorted(available_gap_keys.difference(result.values()))
        raise ValueError(f"split 未覆盖全部 gap key；缺少示例：{missing[:3]}")
    return result


def build_hand_tracker_availability(
    *,
    source_frame_count: int,
    hand_gap_intervals: tuple[tuple[tuple[int, int], ...], ...],
    gap_frame_offset: int = DEFAULT_GAP_FRAME_OFFSET,
) -> np.ndarray:
    """把官方 54D feature gap 映射为本项目 source `[T,6]` availability。"""

    frame_count = int(source_frame_count)
    if frame_count <= 0:
        raise ValueError("source_frame_count 必须为正数。")
    if len(hand_gap_intervals) != len(HAND_GAP_TRACKER_INDICES):
        raise ValueError("hand_gap_intervals 必须恰好包含左/右手两组。")
    offset = int(gap_frame_offset)
    available = np.broadcast_to(
        np.asarray(CORE_THREE_AVAILABLE, dtype=bool)[None],
        (frame_count, TRACKER_COUNT),
    ).copy()
    for tracker_index, intervals in zip(
        HAND_GAP_TRACKER_INDICES, hand_gap_intervals
    ):
        for raw_start, raw_end in intervals:
            start = int(raw_start) + offset
            end = int(raw_end) + offset
            if start < 0 or end > frame_count or end <= start:
                raise ValueError(
                    "gap 映射后超出 source："
                    f"raw=({raw_start},{raw_end}), mapped=({start},{end}), "
                    f"frames={frame_count}。"
                )
            available[start:end, int(tracker_index)] = False
    if not available[:, 0].all():
        raise RuntimeError("RPM-P2 hand tracking 协议不得移除 Head。")
    return available


def gaps_in_scored_coordinates(
    hand_gap_intervals: tuple[tuple[tuple[int, int], ...], ...],
    *,
    gap_frame_offset: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """把官方 feature gap 转为 source 正式计分数组内的帧坐标。"""

    scored_offset = int(gap_frame_offset) - REALTIME_POSE_EVAL_METRICS_START_FRAME
    return tuple(
        tuple(
            (int(start) + scored_offset, int(end) + scored_offset)
            for start, end in intervals
        )
        for intervals in hand_gap_intervals
    )


# endregion


def evaluate_hand_recovery_sequences(
    *,
    entries: list[dict],
    gap_key_by_source: dict[str, str],
    gaps_by_key: dict,
    predictor,
    dit,
    diffusion,
    device: torch.device,
    normalizer,
    args,
) -> dict:
    """在共享模型上逐条运行官方 P2 左/右手 gap，并汇总普通与过渡指标。"""

    if not entries:
        raise RuntimeError("没有可用于 RPM-P2 hand recovery 的 source sequence。")
    args.allow_missing_core_trackers = True
    raw_sequence_metrics: list[dict[str, float | None]] = []
    deployed_sequence_metrics: list[dict[str, float | None]] = []
    raw_transition_segments: list[dict[str, np.ndarray]] = []
    deployed_transition_segments: list[dict[str, np.ndarray]] = []
    target_transition_segments: list[dict[str, np.ndarray]] = []
    tracker_position_errors: list[float] = []
    tracker_rotation_errors: list[float] = []
    runtime_latencies_ms: list[float] = []
    per_sequence: list[dict] = []
    generated_frame_count = 0
    evaluated_frame_count = 0
    lost_frame_counts = np.zeros((len(HAND_GAP_NAMES),), dtype=np.int64)
    gap_counts = np.zeros((len(HAND_GAP_NAMES),), dtype=np.int64)
    reconnection_counts = np.zeros((len(HAND_GAP_NAMES),), dtype=np.int64)
    noise_generator = create_eval_noise_generator(args.seed, device)
    blend_frames = int(args.activation_blend_frames)

    for sequence_index, entry in enumerate(entries, start=1):
        source_key = normalize_split_key(str(entry["source_relative_path"]))
        if source_key not in gap_key_by_source:
            raise KeyError(f"source 没有 RPM-P2 gap 映射：{source_key}")
        gap_key = gap_key_by_source[source_key]
        hand_gaps = gaps_by_key[gap_key]
        source = load_realtime_source(resolve_source_entry_path(entry))
        world_rotations = compute_source_joint_rotations_world(source)
        frame_count = len(world_rotations)
        tracker_available = build_hand_tracker_availability(
            source_frame_count=frame_count,
            hand_gap_intervals=hand_gaps,
            gap_frame_offset=int(args.gap_frame_offset),
        )
        last = evaluation_last_frame_exclusive(frame_count, int(args.max_frames))
        if last <= REALTIME_POSE_EVAL_METRICS_START_FRAME:
            continue

        pose_history = [
            WorldPoseState(
                joint_rotations_world=world_rotations[index],
                root_yaw_world=float(source["root_yaw"][index]),
                hip_height=float(source["pelvis_height"][index, 0]),
                root_position_world=source["root_pos_world"][index],
            )
            for index in range(1, 11)
        ]
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
            pose_history,
            source["tracker_pos_world"][:11],
            source["tracker_rot_world_6d"][:11],
            source["root_pos_world"][:11, 1],
            tracker_available_history=tracker_available[:11],
        )
        previous_available = tracker_available[
            PREDICTOR_EVAL_FIRST_GENERATED_FRAME - 1
        ].copy()
        previous_result = None
        activation_ramps: TrackerActivationRamps = {}
        sequence_reconnection_counts = np.zeros(
            (len(HAND_GAP_NAMES),), dtype=np.int64
        )

        scored_raw_rotations: list[np.ndarray] = []
        scored_deployed_rotations: list[np.ndarray] = []
        scored_target_rotations: list[np.ndarray] = []
        scored_raw_positions: list[np.ndarray] = []
        scored_deployed_positions: list[np.ndarray] = []
        scored_target_positions: list[np.ndarray] = []
        for current in range(PREDICTOR_EVAL_FIRST_GENERATED_FRAME, last):
            scored = current >= REALTIME_POSE_EVAL_METRICS_START_FRAME
            previous_root_yaw = runtime.pose_history[-1].root_yaw_world
            noise = torch.randn(
                (1, REALTIME_POSE_TARGET_DIM),
                generator=noise_generator,
                device=device,
            )
            if scored and device.type == "cuda":
                torch.cuda.synchronize(device)
            started_at = time.perf_counter() if scored else 0.0
            frame_available = tracker_available[current]
            (
                runtime_tracker_positions,
                runtime_tracker_rotations_6d,
                _,
                newly_added,
            ) = apply_tracker_activation_blend(
                current_frame=current,
                blend_frames=blend_frames,
                previous_available=previous_available,
                current_available=frame_available,
                measured_positions=source["tracker_pos_world"][current],
                measured_rotations_6d=source["tracker_rot_world_6d"][current],
                previous_joint_positions=(
                    source["joints_world"][current - 1]
                    if previous_result is None
                    else previous_result.resolved_pose.joints_world
                ),
                previous_joint_rotations=(
                    world_rotations[current - 1]
                    if previous_result is None
                    else previous_result.resolved_pose.joint_rotations_world
                ),
                activation_ramps=activation_ramps,
            )
            for hand_index, tracker_index in enumerate(HAND_GAP_TRACKER_INDICES):
                if int(tracker_index) in newly_added:
                    sequence_reconnection_counts[hand_index] += 1
            result = runtime.step(
                runtime_tracker_positions,
                runtime_tracker_rotations_6d,
                frame_available,
                float(source["root_pos_world"][current, 1]),
                noise=noise,
            )
            generated_frame_count += 1
            previous_available = frame_available.copy()
            previous_result = result
            if not scored:
                continue
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime_latencies_ms.append((time.perf_counter() - started_at) * 1000.0)
            raw_resolved = decode_and_resolve_pose(
                result.raw_pred_pose,
                result.current_tracker_raw,
                result.current_head_yaw_world,
                source["tracker_pos_world"][current, 0],
                float(source["root_pos_world"][current, 1]),
                source["joint_offsets_parent"],
                source["joint_rest_local_rotations_6d"],
                previous_root_yaw,
            )
            scored_raw_rotations.append(raw_resolved.joint_rotations_world)
            scored_deployed_rotations.append(result.resolved_pose.joint_rotations_world)
            scored_target_rotations.append(world_rotations[current])
            scored_raw_positions.append(raw_resolved.joints_world)
            scored_deployed_positions.append(result.resolved_pose.joints_world)
            scored_target_positions.append(source["joints_world"][current])
            _append_tracker_errors(
                tracker_available=frame_available,
                predicted_joint_positions=result.resolved_pose.joints_world,
                predicted_joint_rotations=result.resolved_pose.joint_rotations_world,
                measured_tracker_positions=source["tracker_pos_world"][current],
                measured_tracker_rotations_6d=source["tracker_rot_world_6d"][current],
                position_errors=tracker_position_errors,
                rotation_errors=tracker_rotation_errors,
            )
            evaluated_frame_count += 1

        raw_rotations = np.stack(scored_raw_rotations)
        deployed_rotations = np.stack(scored_deployed_rotations)
        target_rotations = np.stack(scored_target_rotations)
        raw_positions = np.stack(scored_raw_positions)
        deployed_positions = np.stack(scored_deployed_positions)
        target_positions = np.stack(scored_target_positions)
        common = {
            "target_global_rotations": target_rotations,
            "target_joint_positions": target_positions,
            "fps": REALTIME_POSE_FPS,
        }
        raw_metrics = compute_rpm_p2_mc_metrics(
            predicted_global_rotations=raw_rotations,
            predicted_joint_positions=raw_positions,
            **common,
        )
        deployed_metrics = compute_rpm_p2_mc_metrics(
            predicted_global_rotations=deployed_rotations,
            predicted_joint_positions=deployed_positions,
            **common,
        )
        raw_sequence_metrics.append(raw_metrics)
        deployed_sequence_metrics.append(deployed_metrics)
        scored_gaps = gaps_in_scored_coordinates(
            hand_gaps,
            gap_frame_offset=int(args.gap_frame_offset),
        )
        transition_common = {
            "hand_gap_intervals": scored_gaps,
            "fps": REALTIME_POSE_FPS,
            "transition_seconds": float(args.transition_seconds),
        }
        raw_transition_segments.append(
            extract_rpm_transition_jerk_segments(
                joint_positions=raw_positions,
                **transition_common,
            )
        )
        deployed_transition_segments.append(
            extract_rpm_transition_jerk_segments(
                joint_positions=deployed_positions,
                **transition_common,
            )
        )
        target_transition_segments.append(
            extract_rpm_transition_jerk_segments(
                joint_positions=target_positions,
                **transition_common,
            )
        )
        scored_available = tracker_available[
            REALTIME_POSE_EVAL_METRICS_START_FRAME:last
        ]
        sequence_lost_counts = [
            int(np.count_nonzero(~scored_available[:, tracker_index]))
            for tracker_index in HAND_GAP_TRACKER_INDICES
        ]
        lost_frame_counts += np.asarray(sequence_lost_counts, dtype=np.int64)
        gap_counts += np.asarray([len(values) for values in hand_gaps], dtype=np.int64)
        reconnection_counts += sequence_reconnection_counts
        per_sequence.append(
            {
                "sequence_id": str(entry.get("sequence_id", source_key)),
                "source_relative_path": str(entry["source_relative_path"]),
                "gap_key": gap_key,
                "evaluated_frames": int(last - REALTIME_POSE_EVAL_METRICS_START_FRAME),
                "gap_counts": dict(zip(HAND_GAP_NAMES, map(int, map(len, hand_gaps)))),
                "lost_scored_frames": dict(
                    zip(HAND_GAP_NAMES, sequence_lost_counts)
                ),
                "smooth_reconnection_counts": dict(
                    zip(
                        HAND_GAP_NAMES,
                        sequence_reconnection_counts.astype(int).tolist(),
                    )
                ),
                "dit_raw": raw_metrics,
                "dit_deployed": deployed_metrics,
            }
        )
        print(
            "[hand-signal-recovery] "
            f"sequence {sequence_index}/{len(entries)} "
            f"{entry.get('sequence_id', source_key)} "
            f"MPJPE={deployed_metrics['mpjpe_cm']:.3f}cm",
            flush=True,
        )

    if evaluated_frame_count <= 0:
        raise RuntimeError("RPM-P2 hand recovery 没有可计分帧。")
    latency = np.asarray(runtime_latencies_ms, dtype=np.float64)
    return {
        "generated_frames": int(generated_frame_count),
        "evaluated_frames": int(evaluated_frame_count),
        "dit_raw": aggregate_rpm_p2_mc_metrics(raw_sequence_metrics),
        "dit_deployed": aggregate_rpm_p2_mc_metrics(deployed_sequence_metrics),
        "transition_dit_raw": aggregate_rpm_transition_jerk_segments(
            predicted_segments=raw_transition_segments,
            target_segments=target_transition_segments,
        ),
        "transition_dit_deployed": aggregate_rpm_transition_jerk_segments(
            predicted_segments=deployed_transition_segments,
            target_segments=target_transition_segments,
        ),
        "gap_counts": dict(zip(HAND_GAP_NAMES, gap_counts.astype(int).tolist())),
        "lost_scored_frames": dict(
            zip(HAND_GAP_NAMES, lost_frame_counts.astype(int).tolist())
        ),
        "smooth_reconnection": {
            "activation_blend_frames": blend_frames,
            "event_counts": dict(
                zip(HAND_GAP_NAMES, reconnection_counts.astype(int).tolist())
            ),
        },
        "tracker_error": {
            "position_cm": float(np.mean(tracker_position_errors) * 100.0),
            "rotation_deg": float(np.degrees(np.mean(tracker_rotation_errors))),
        },
        "runtime_latency_ms": {
            "mean": float(np.mean(latency)),
            "p50": float(np.percentile(latency, 50.0)),
            "p90": float(np.percentile(latency, 90.0)),
        },
        "per_sequence": per_sequence,
    }


def main(argv: list[str] | None = None) -> dict:
    args = parse_and_load_from_model(build_arg_parser(), argv)
    # 早期 f0_past10 checkpoint 的 args.json 把条件帧数 `seq_len=11` 误写进了
    # DiT token 上限。当前 DiT 的真实 token 契约始终是 past 10 + current 1 +
    # future 10，因此实验入口按当前固定契约覆盖这项非权重配置。
    if str(args.model_arch) == "current_dit":
        args.max_seq_len = REALTIME_POSE_MODEL_TOKEN_LENGTH
    if str(args.split) != "test":
        raise ValueError("RPM-P2 官方 hand_tracking gap 只定义在 test split。")
    if (
        int(args.limit) < 0
        or int(args.max_frames) < 0
        or int(args.activation_blend_frames) < 0
    ):
        raise ValueError(
            "--limit、--max_frames 与 --activation_blend_frames 必须大于等于 0。"
        )
    if float(args.transition_seconds) <= 0.0:
        raise ValueError("--transition_seconds 必须为正数。")
    fixseed(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if args.cuda and torch.cuda.is_available() else "cpu"
    )
    metadata, gaps_by_key = load_rpm_p2_hand_gap_config(args.gap_config)
    split_file = Path(args.split_dir).expanduser().resolve() / "test.txt"
    gap_key_by_source = build_gap_key_by_source(split_file, set(gaps_by_key))
    entries = read_longseq_source_entries(
        args.source_dir,
        args.split_dir,
        split="test",
        min_frames=REALTIME_POSE_EVAL_METRICS_START_FRAME + 1,
        include_mirror=False,
    )
    if int(args.limit) > 0:
        entries = entries[: int(args.limit)]

    dit, diffusion = create_model_and_diffusion(args)
    dit, dit_weight_source = load_checkpoint_model(
        dit, args.dit_model_path, device, use_ema=args.use_ema
    )
    predictor = load_realtime_pose_predictor(args.predictor_model_path, device)
    normalizer = RealtimePoseNormalizer(
        args.normalizer_dir, disable=not bool(args.normalize_input)
    )
    report = evaluate_hand_recovery_sequences(
        entries=entries,
        gap_key_by_source=gap_key_by_source,
        gaps_by_key=gaps_by_key,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        args=args,
    )
    payload = {
        "experiment": "rpm_p2_hand_signal_loss_and_recovery",
        "diagnostic_scope": (
            "out_of_training_distribution: current Predictor/DiT were not trained "
            "with missing core hand trackers"
        ),
        "predictor_model_path": str(Path(args.predictor_model_path).resolve()),
        "dit_model_path": str(Path(args.dit_model_path).resolve()),
        "dit_weight_source": dit_weight_source,
        "normalizer_dir": str(Path(args.normalizer_dir).resolve()),
        "source_dir": str(Path(args.source_dir).resolve()),
        "split_dir": str(Path(args.split_dir).resolve()),
        "split": "test",
        "source_sequence_count": len(entries),
        "source_fps": float(REALTIME_POSE_FPS),
        "initial_context_frames": PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
        "metrics_start_frame": REALTIME_POSE_EVAL_METRICS_START_FRAME,
        "sampler": "projected_ddim",
        "diffusion_training_steps": int(args.diffusion_steps),
        "timestep_respacing": str(args.ts_respace),
        "sampling_steps": int(diffusion.num_timesteps),
        "sampling_noise_policy": "single deterministic stream",
        "sampling_noise_seed": int(args.seed),
        "gap_protocol": {
            "config_path": str(Path(args.gap_config).expanduser().resolve()),
            "metadata": metadata,
            "gap_frame_offset": int(args.gap_frame_offset),
            "tracker_policy": (
                "Head always available; left/right wrist gaps are independent; "
                "Hip and feet remain unavailable"
            ),
            "predictor_missing_signal_policy": (
                "zero in normalized feature space; absolute channels use current "
                "availability and velocity channels require both adjacent frames"
            ),
            "reconnection_policy": {
                "name": "project_soft_activation_blend",
                "activation_blend_frames": int(args.activation_blend_frames),
                "position": "LERP from previous deployed wrist to measurement",
                "rotation": "SLERP from previous deployed wrist to measurement",
                "weight": "smoothstep",
            },
            "transition_seconds": float(args.transition_seconds),
            "transition_window_source": (
                "official released evaluator default is 0.5 s; paper text states 1 s"
            ),
        },
        "report": report,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[hand-signal-recovery] wrote {output}", flush=True)
    return payload


if __name__ == "__main__":
    main()
