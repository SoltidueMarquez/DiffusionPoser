from __future__ import annotations

import argparse
import json
from itertools import combinations
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
    OPTIONAL_TRACKER_INDICES,
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
)
from eval.evaluate_realtime_pose_predictor import (
    PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
)
from sample.evaluate_tracker_reconnection import (
    DEFAULT_STAGE_FRAMES,
    RECONNECT_TRACKER_NAME_TO_INDEX,
    RECONNECT_TRACKER_NAMES,
    validate_stage_frames,
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


RECONNECT_TRACKER_PAIRS = tuple(combinations(RECONNECT_TRACKER_NAMES, 2))
RECONNECT_PAIR_NAMES = tuple("_and_".join(pair) for pair in RECONNECT_TRACKER_PAIRS)
RECONNECT_PAIR_NAME_TO_TRACKERS = dict(
    zip(RECONNECT_PAIR_NAMES, RECONNECT_TRACKER_PAIRS)
)


# region 参数与协议构造


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "评估核心三点长期闭环后同帧恢复两个 optional Tracker 的 3→5 表现。"
        )
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    group = parser.add_argument_group("tracker pair reconnection")
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
        help="三点阶段和五点恢复阶段各自的计分帧数。",
    )
    group.add_argument(
        "--reconnect_pairs",
        nargs="+",
        choices=("all", *RECONNECT_PAIR_NAMES),
        default=["all"],
        help="选择同帧恢复的 optional Tracker 组合；默认分别运行全部三种。",
    )
    group.add_argument("--output_json", required=True)
    return parser


def resolve_reconnect_pair_names(requested: list[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in requested)
    if "all" in values:
        if values != ("all",):
            raise ValueError("--reconnect_pairs all 不能与具体组合同时使用。")
        return RECONNECT_PAIR_NAMES
    if not values:
        raise ValueError("至少需要一个 Tracker 重连组合。")
    if len(set(values)) != len(values):
        raise ValueError("--reconnect_pairs 不能包含重复组合。")
    return values


def build_tracker_pair_reconnection_schedule(
    *,
    scored_frame_count: int,
    reconnect_tracker_indices: tuple[int, int],
    stage_frames: int,
) -> TrackerSequenceSchedule:
    """构造固定等长的两阶段 3→5 mask；两个 Tracker 在同一边界恢复。"""

    frames_per_stage = validate_stage_frames(stage_frames)
    frame_count = int(scored_frame_count)
    expected_frame_count = 2 * frames_per_stage
    if frame_count != expected_frame_count:
        raise ValueError(
            "3→5 重连协议要求计分帧数恰好为 "
            f"2 * stage_frames={expected_frame_count}，实际为 {frame_count}。"
        )

    tracker_indices = tuple(int(index) for index in reconnect_tracker_indices)
    if len(tracker_indices) != 2 or len(set(tracker_indices)) != 2:
        raise ValueError("3→5 重连必须指定两个不同的 optional Tracker。")
    if not set(tracker_indices).issubset(OPTIONAL_TRACKER_INDICES):
        raise ValueError(
            "重连组合只能包含 optional Tracker，"
            f"实际 indices={tracker_indices}。"
        )

    # [T,6]：前半段只有核心三点，后半段在同一帧把两个可选点都切为 available。
    tracker_available = np.broadcast_to(
        np.asarray(CORE_THREE_AVAILABLE, dtype=bool)[None],
        (frame_count, len(CORE_THREE_AVAILABLE)),
    ).copy()
    tracker_available[frames_per_stage:, list(tracker_indices)] = True
    stage_indices = np.zeros((frame_count,), dtype=np.int64)
    stage_indices[frames_per_stage:] = 1
    return TrackerSequenceSchedule(
        tracker_available=tracker_available,
        stage_indices=stage_indices,
    )


def build_tracker_pair_reconnection_protocols(
    reconnect_pair_names: tuple[str, ...] | list[str] | None = None,
    *,
    stage_frames: int = DEFAULT_STAGE_FRAMES,
) -> tuple[TrackerEvaluationProtocol, ...]:
    frames_per_stage = validate_stage_frames(stage_frames)
    selected = (
        RECONNECT_PAIR_NAMES
        if reconnect_pair_names is None
        else tuple(str(name) for name in reconnect_pair_names)
    )
    protocols: list[TrackerEvaluationProtocol] = []
    for pair_name in selected:
        if pair_name not in RECONNECT_PAIR_NAME_TO_TRACKERS:
            raise ValueError(f"未知 Tracker 重连组合：{pair_name}")
        tracker_names = RECONNECT_PAIR_NAME_TO_TRACKERS[pair_name]
        tracker_indices = tuple(
            int(RECONNECT_TRACKER_NAME_TO_INDEX[name]) for name in tracker_names
        )
        reconnected_available = np.asarray(CORE_THREE_AVAILABLE, dtype=bool).copy()
        reconnected_available[list(tracker_indices)] = True

        def build_schedule(
            scored_frame_count: int,
            *,
            current_tracker_indices: tuple[int, int] = tracker_indices,
            current_stage_frames: int = frames_per_stage,
        ) -> TrackerSequenceSchedule:
            return build_tracker_pair_reconnection_schedule(
                scored_frame_count=scored_frame_count,
                reconnect_tracker_indices=current_tracker_indices,
                stage_frames=current_stage_frames,
            )

        reconnect_trackers = list(tracker_names)
        protocols.append(
            TrackerEvaluationProtocol(
                name=f"reconnect_{pair_name}",
                # 三点预热保证切换前闭环历史不包含任何 optional Tracker 信息。
                warmup_tracker_available=tuple(CORE_THREE_AVAILABLE),
                stages=(
                    TrackerStageDefinition(
                        index=0,
                        name="trackers_3_disconnected",
                        tracker_available=tuple(CORE_THREE_AVAILABLE),
                        metadata={"reconnect_trackers": reconnect_trackers},
                    ),
                    TrackerStageDefinition(
                        index=1,
                        name=f"trackers_5_{pair_name}_reconnected",
                        tracker_available=tuple(
                            bool(value) for value in reconnected_available.tolist()
                        ),
                        metadata={"reconnect_trackers": reconnect_trackers},
                    ),
                ),
                schedule_builder=build_schedule,
                include_stage_metrics=True,
                metadata={
                    "reconnect_trackers": reconnect_trackers,
                    "stage_frames": frames_per_stage,
                },
            )
        )
    if not protocols:
        raise ValueError("至少需要一个 Tracker 双点重连协议。")
    return tuple(protocols)


# endregion


def format_tracker_pair_reconnection_report(report: dict) -> dict:
    return {
        "name": report["name"],
        "reconnect_trackers": list(report["reconnect_trackers"]),
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
    # 总计分长度只由两个等长阶段推导，避免额外 max_frames 与协议冲突。
    args.max_frames = scored_frame_count
    selected_pair_names = resolve_reconnect_pair_names(list(args.reconnect_pairs))

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

    protocols = build_tracker_pair_reconnection_protocols(
        selected_pair_names,
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
        format_tracker_pair_reconnection_report(report)
        for report in shared_reports
    ]
    payload = {
        "experiment": "tracker_reconnection_3_to_5",
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
            "core three during warmup and stage 0; two optional Trackers hard "
            "reconnect at the same stage boundary"
        ),
        "sampling_noise_policy": "common_random_numbers",
        "sampling_noise_seed": int(args.seed),
        "schedule": {
            "stage_policy": "fixed_equal_halves",
            "stage_frames": stage_frames,
            "stage_seconds": stage_frames / float(REALTIME_POSE_FPS),
            "scored_frames": scored_frame_count,
            "tracker_counts": [3, 5],
            "warmup_tracker_available": list(CORE_THREE_AVAILABLE),
            "warmup_tracker_count": int(sum(CORE_THREE_AVAILABLE)),
            "reconnect_pairs": [
                list(RECONNECT_PAIR_NAME_TO_TRACKERS[name])
                for name in selected_pair_names
            ],
            "minimum_source_frames": minimum_source_frames,
        },
        "reconnection_protocols": reports,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[tracker-pair-reconnection] wrote {output}")
    return payload


if __name__ == "__main__":
    main()
