from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_loaders.build_realtime_longseq_eval_set import (
    DEFAULT_SOURCE_DIR,
    DEFAULT_SPLIT_DIR,
    read_longseq_source_entries,
)
from data_loaders.sensor_masking import (
    CORE_THREE_AVAILABLE,
    HIP_TRACKER_INDEX,
    LEFT_FOOT_TRACKER_INDEX,
    OPTIONAL_TRACKER_INDICES,
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
    RIGHT_FOOT_TRACKER_INDEX,
)
from eval.evaluate_realtime_pose_predictor import (
    PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
)
from sample.realtime_pose_longseq_evaluator import (
    TrackerEvaluationProtocol,
    TrackerSequenceSchedule,
    TrackerStageDefinition,
    evaluate_longseq_protocols,
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


DEFAULT_STAGE_FRAMES = 150
MIN_STAGE_FRAMES = 4
RECONNECT_TRACKER_NAME_TO_INDEX = {
    "hip": HIP_TRACKER_INDEX,
    "left_foot": LEFT_FOOT_TRACKER_INDEX,
    "right_foot": RIGHT_FOOT_TRACKER_INDEX,
}
RECONNECT_TRACKER_NAMES = tuple(RECONNECT_TRACKER_NAME_TO_INDEX)


# region 参数与协议构造


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "评估核心三点长期闭环后恢复一个 optional Tracker 的 3→4 表现。"
        )
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    group = parser.add_argument_group("tracker reconnection")
    group.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR)
    group.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    group.add_argument("--split", default="test")
    group.add_argument("--normalizer_dir", required=True)
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument("--limit", default=0, type=int)
    group.add_argument(
        "--stage_frames",
        default=DEFAULT_STAGE_FRAMES,
        type=int,
        help="三点掉线阶段和四点恢复阶段各自的计分帧数。",
    )
    group.add_argument(
        "--reconnect_trackers",
        nargs="+",
        choices=("all", *RECONNECT_TRACKER_NAMES),
        default=["all"],
        help="选择作为第四点恢复的 optional Tracker；默认分别运行三种。",
    )
    group.add_argument("--output_json", required=True)
    return parser


def validate_stage_frames(stage_frames: int) -> int:
    value = int(stage_frames)
    if value < MIN_STAGE_FRAMES:
        raise ValueError(
            f"--stage_frames 至少为 {MIN_STAGE_FRAMES}，确保每阶段可计算 Jitter。"
        )
    return value


def resolve_reconnect_tracker_names(requested: list[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in requested)
    if "all" in values:
        if values != ("all",):
            raise ValueError("--reconnect_trackers all 不能与具体 Tracker 同时使用。")
        return RECONNECT_TRACKER_NAMES
    if not values:
        raise ValueError("至少需要一个重连 Tracker。")
    if len(set(values)) != len(values):
        raise ValueError("--reconnect_trackers 不能包含重复 Tracker。")
    return values


def build_tracker_reconnection_schedule(
    *,
    scored_frame_count: int,
    reconnect_tracker_index: int,
    stage_frames: int,
) -> TrackerSequenceSchedule:
    """构造固定等长的两阶段 3→4 mask；每条序列只发生一次重连。"""

    frames_per_stage = validate_stage_frames(stage_frames)
    frame_count = int(scored_frame_count)
    expected_frame_count = 2 * frames_per_stage
    if frame_count != expected_frame_count:
        raise ValueError(
            "3→4 重连协议要求计分帧数恰好为 "
            f"2 * stage_frames={expected_frame_count}，实际为 {frame_count}。"
        )
    tracker_index = int(reconnect_tracker_index)
    if tracker_index not in OPTIONAL_TRACKER_INDICES:
        raise ValueError(
            f"重连 Tracker 必须是 optional Tracker，实际 index={tracker_index}。"
        )

    # [T,6]：掉线阶段只提供核心三点，边界之后仅恢复指定的第四点。
    tracker_available = np.broadcast_to(
        np.asarray(CORE_THREE_AVAILABLE, dtype=bool)[None],
        (frame_count, len(CORE_THREE_AVAILABLE)),
    ).copy()
    tracker_available[frames_per_stage:, tracker_index] = True
    stage_indices = np.zeros((frame_count,), dtype=np.int64)
    stage_indices[frames_per_stage:] = 1
    return TrackerSequenceSchedule(
        tracker_available=tracker_available,
        stage_indices=stage_indices,
    )


def build_tracker_reconnection_protocols(
    reconnect_tracker_names: tuple[str, ...] | list[str] | None = None,
    *,
    stage_frames: int = DEFAULT_STAGE_FRAMES,
) -> tuple[TrackerEvaluationProtocol, ...]:
    frames_per_stage = validate_stage_frames(stage_frames)
    selected = (
        RECONNECT_TRACKER_NAMES
        if reconnect_tracker_names is None
        else tuple(str(name) for name in reconnect_tracker_names)
    )
    protocols: list[TrackerEvaluationProtocol] = []
    for tracker_name in selected:
        if tracker_name not in RECONNECT_TRACKER_NAME_TO_INDEX:
            raise ValueError(f"未知重连 Tracker：{tracker_name}")
        tracker_index = int(RECONNECT_TRACKER_NAME_TO_INDEX[tracker_name])
        reconnected_available = np.asarray(CORE_THREE_AVAILABLE, dtype=bool).copy()
        reconnected_available[tracker_index] = True

        def build_schedule(
            scored_frame_count: int,
            *,
            current_tracker_index: int = tracker_index,
            current_stage_frames: int = frames_per_stage,
        ) -> TrackerSequenceSchedule:
            return build_tracker_reconnection_schedule(
                scored_frame_count=scored_frame_count,
                reconnect_tracker_index=current_tracker_index,
                stage_frames=current_stage_frames,
            )

        protocols.append(
            TrackerEvaluationProtocol(
                name=f"reconnect_{tracker_name}",
                # 预热也保持三点，避免重连前历史继承 optional Tracker 信息。
                warmup_tracker_available=tuple(CORE_THREE_AVAILABLE),
                stages=(
                    TrackerStageDefinition(
                        index=0,
                        name="trackers_3_disconnected",
                        tracker_available=tuple(CORE_THREE_AVAILABLE),
                        metadata={"reconnect_tracker": tracker_name},
                    ),
                    TrackerStageDefinition(
                        index=1,
                        name=f"trackers_4_{tracker_name}_reconnected",
                        tracker_available=tuple(
                            bool(value) for value in reconnected_available.tolist()
                        ),
                        metadata={"reconnect_tracker": tracker_name},
                    ),
                ),
                schedule_builder=build_schedule,
                include_stage_metrics=True,
                metadata={
                    "reconnect_tracker": tracker_name,
                    "stage_frames": frames_per_stage,
                },
            )
        )
    if not protocols:
        raise ValueError("至少需要一个 Tracker 重连协议。")
    return tuple(protocols)


# endregion


def format_tracker_reconnection_report(report: dict) -> dict:
    return {
        "name": report["name"],
        "reconnect_tracker": str(report["reconnect_tracker"]),
        "stage_frames": int(report["stage_frames"]),
        "overall": {
            "generated_frames": int(report["generated_frames"]),
            "evaluated_frames": int(report["evaluated_frames"]),
            "dit_raw": report["dit_raw"],
            "dit_deployed": report["dit_deployed"],
        },
        "stages": report["stages"],
        "tracker_error": report["tracker_error"],
        "runtime_latency_ms": report["runtime_latency_ms"],
    }


def main(argv: list[str] | None = None) -> dict:
    args = parse_and_load_from_model(build_arg_parser(), argv)
    stage_frames = validate_stage_frames(args.stage_frames)
    scored_frame_count = 2 * stage_frames
    # 共享长序列内核通过 max_frames 裁剪正式计分区；这里从唯一的阶段参数派生，
    # 避免调用者传入相互矛盾的总时长和阶段时长。
    args.max_frames = scored_frame_count
    selected_tracker_names = resolve_reconnect_tracker_names(
        list(args.reconnect_trackers)
    )

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
    minimum_source_frames = (
        REALTIME_POSE_EVAL_METRICS_START_FRAME + scored_frame_count
    )
    entries = read_longseq_source_entries(
        args.source_dir,
        args.split_dir,
        split=args.split,
        min_frames=minimum_source_frames,
        include_mirror=False,
    )
    if args.limit > 0:
        entries = entries[: args.limit]

    protocols = build_tracker_reconnection_protocols(
        selected_tracker_names,
        stage_frames=stage_frames,
    )
    shared_reports = evaluate_longseq_protocols(
        protocols=protocols,
        entries=entries,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        args=args,
    )
    reports = [
        format_tracker_reconnection_report(report) for report in shared_reports
    ]
    payload = {
        "experiment": "tracker_reconnection_3_to_4",
        "predictor_model_path": str(Path(args.predictor_model_path).resolve()),
        "dit_model_path": str(Path(args.dit_model_path).resolve()),
        "dit_weight_source": dit_weight_source,
        "sampler": "projected_ddim",
        "diffusion_training_steps": int(args.diffusion_steps),
        "timestep_respacing": str(args.ts_respace),
        "sampling_steps": int(diffusion.num_timesteps),
        "split": str(args.split),
        "source_sequence_count": len(entries),
        "source_fps": float(REALTIME_POSE_FPS),
        "initial_context_frames": PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
        "metrics_start_frame": REALTIME_POSE_EVAL_METRICS_START_FRAME,
        "history_policy": (
            "closed-loop starts at frame 11 with core three; official metrics "
            "start at frame 30"
        ),
        "tracker_policy": (
            "core three during warmup and stage 0; one optional Tracker hard "
            "reconnects at the stage boundary"
        ),
        "sampling_noise_policy": "common_random_numbers",
        "sampling_noise_seed": int(args.seed),
        "schedule": {
            "stage_policy": "fixed_equal_halves",
            "stage_frames": stage_frames,
            "stage_seconds": stage_frames / float(REALTIME_POSE_FPS),
            "scored_frames": scored_frame_count,
            "tracker_counts": [3, 4],
            "warmup_tracker_available": list(CORE_THREE_AVAILABLE),
            "warmup_tracker_count": int(sum(CORE_THREE_AVAILABLE)),
            "reconnect_trackers": list(selected_tracker_names),
            "minimum_source_frames": minimum_source_frames,
        },
        "reconnection_protocols": reports,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[tracker-reconnection] wrote {output}")
    return payload


if __name__ == "__main__":
    main()
