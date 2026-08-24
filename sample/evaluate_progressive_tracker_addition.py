from __future__ import annotations

import argparse
import json
from itertools import permutations
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


PROGRESSIVE_STAGE_COUNT = 4
MIN_PROGRESSIVE_SCORED_FRAMES = 4 * PROGRESSIVE_STAGE_COUNT
FINAL_METRIC_KEYS = (
    "mpjre_deg",
    "mpjpe_cm",
    "mpjve_cm_per_s",
    "pred_jitter_m_per_s3",
)
OPTIONAL_TRACKER_NAME_TO_INDEX = {
    "hip": HIP_TRACKER_INDEX,
    "left_foot": LEFT_FOOT_TRACKER_INDEX,
    "right_foot": RIGHT_FOOT_TRACKER_INDEX,
}
PROGRESSIVE_ADD_ORDERS = tuple(
    tuple(order) for order in permutations(OPTIONAL_TRACKER_NAME_TO_INDEX)
)
PROGRESSIVE_ADD_ORDER_NAMES = tuple(
    "_then_".join(order) for order in PROGRESSIVE_ADD_ORDERS
)


# region 参数与协议构造


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "评估同一长序列中 Tracker 从三点逐阶段增加到六点的闭环表现。"
        )
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    group = parser.add_argument_group("progressive tracker addition")
    group.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR)
    group.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    group.add_argument("--split", default="test")
    group.add_argument("--normalizer_dir", required=True)
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument("--limit", default=0, type=int)
    group.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="P2 预热完成后每条序列最多计分的帧数；0 表示直到序列结束。",
    )
    group.add_argument(
        "--add_orders",
        nargs="+",
        choices=("all", *PROGRESSIVE_ADD_ORDER_NAMES),
        default=["all"],
        help="选择 optional Tracker 添加顺序；默认运行全部六种排列。",
    )
    group.add_argument("--output_json", required=True)
    return parser


def resolve_add_order_names(requested: list[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in requested)
    if "all" in values:
        if values != ("all",):
            raise ValueError("--add_orders all 不能与具体添加顺序同时使用。")
        return PROGRESSIVE_ADD_ORDER_NAMES
    if not values:
        raise ValueError("至少需要一个 Tracker 添加顺序。")
    if len(set(values)) != len(values):
        raise ValueError("--add_orders 不能包含重复顺序。")
    return values


def build_progressive_addition_protocols(
    add_order_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[TrackerEvaluationProtocol, ...]:
    selected = (
        PROGRESSIVE_ADD_ORDER_NAMES
        if add_order_names is None
        else tuple(add_order_names)
    )
    order_lookup = dict(zip(PROGRESSIVE_ADD_ORDER_NAMES, PROGRESSIVE_ADD_ORDERS))
    protocols: list[TrackerEvaluationProtocol] = []
    for order_name in selected:
        if order_name not in order_lookup:
            raise ValueError(f"未知 Tracker 添加顺序：{order_name}")
        add_order = order_lookup[order_name]
        add_indices = tuple(
            OPTIONAL_TRACKER_NAME_TO_INDEX[name] for name in add_order
        )
        stages = build_progressive_stage_definitions(add_order)

        def build_schedule(
            scored_frame_count: int,
            *,
            current_add_indices: tuple[int, ...] = add_indices,
        ) -> TrackerSequenceSchedule:
            return build_equal_quarter_tracker_schedule(
                scored_frame_count=scored_frame_count,
                add_order=current_add_indices,
            )

        protocols.append(
            TrackerEvaluationProtocol(
                name=f"add_{order_name}",
                # 三点预热使第一计分阶段的闭环历史也来自三点条件，避免继承
                # 六点历史而高估 3 Tracker 阶段的表现。
                warmup_tracker_available=tuple(CORE_THREE_AVAILABLE),
                stages=stages,
                schedule_builder=build_schedule,
                include_stage_metrics=True,
                metadata={
                    "add_order": list(add_order),
                    "paired_drop_order": list(reversed(add_order)),
                    "frame_masks_seen_during_training": True,
                    "temporal_schedule_seen_during_training": False,
                },
            )
        )
    if not protocols:
        raise ValueError("至少需要一个动态 Tracker 协议。")
    return tuple(protocols)


def build_progressive_stage_definitions(
    add_order: tuple[str, ...],
) -> tuple[TrackerStageDefinition, ...]:
    if len(add_order) != len(OPTIONAL_TRACKER_INDICES) or set(add_order) != set(
        OPTIONAL_TRACKER_NAME_TO_INDEX
    ):
        raise ValueError("add_order 必须恰好包含 Hip、Left Foot 和 Right Foot。")
    available = np.asarray(CORE_THREE_AVAILABLE, dtype=bool).copy()
    stages: list[TrackerStageDefinition] = []
    for stage_index in range(PROGRESSIVE_STAGE_COUNT):
        if stage_index > 0:
            added_name = add_order[stage_index - 1]
            available[OPTIONAL_TRACKER_NAME_TO_INDEX[added_name]] = True
        stages.append(
            TrackerStageDefinition(
                index=stage_index,
                name=f"trackers_{3 + stage_index}",
                tracker_available=tuple(bool(value) for value in available.tolist()),
                metadata={"added_trackers": list(add_order[:stage_index])},
            )
        )
    return tuple(stages)


def build_equal_quarter_tracker_schedule(
    *,
    scored_frame_count: int,
    add_order: tuple[int, ...],
) -> TrackerSequenceSchedule:
    """把正式计分帧等分四段，依次添加一个 optional Tracker。"""

    frame_count = int(scored_frame_count)
    if frame_count < MIN_PROGRESSIVE_SCORED_FRAMES:
        raise ValueError(
            "动态 3→6 实验每条序列至少需要 "
            f"{MIN_PROGRESSIVE_SCORED_FRAMES} 个计分帧，实际为 {frame_count}。"
        )
    normalized_order = tuple(int(index) for index in add_order)
    if len(normalized_order) != len(OPTIONAL_TRACKER_INDICES) or set(
        normalized_order
    ) != set(OPTIONAL_TRACKER_INDICES):
        raise ValueError("add_order 必须是三个 optional Tracker index 的排列。")

    scored_offsets = np.arange(frame_count, dtype=np.int64)
    stage_indices = np.minimum(
        PROGRESSIVE_STAGE_COUNT - 1,
        scored_offsets * PROGRESSIVE_STAGE_COUNT // frame_count,
    )
    stage_masks = []
    available = np.asarray(CORE_THREE_AVAILABLE, dtype=bool).copy()
    for stage_index in range(PROGRESSIVE_STAGE_COUNT):
        if stage_index > 0:
            available[normalized_order[stage_index - 1]] = True
        stage_masks.append(available.copy())
    masks = np.stack(stage_masks, axis=0)
    return TrackerSequenceSchedule(
        tracker_available=masks[stage_indices],
        stage_indices=stage_indices,
    )


# endregion

# region 汇总与输出


def format_progressive_order_report(report: dict) -> dict:
    """把共享内核结果整理成 3→6 实验的 JSON 层次。"""

    return {
        "name": report["name"],
        "add_order": list(report["add_order"]),
        "paired_drop_order": list(report["paired_drop_order"]),
        "frame_masks_seen_during_training": bool(
            report["frame_masks_seen_during_training"]
        ),
        "temporal_schedule_seen_during_training": bool(
            report["temporal_schedule_seen_during_training"]
        ),
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


def summarize_progressive_orders(reports: list[dict]) -> dict:
    metrics = [report["overall"]["dit_deployed"] for report in reports]
    return _summarize_metric_dicts(metrics)


def summarize_progressive_stages(reports: list[dict]) -> list[dict]:
    summaries = []
    for stage_index in range(PROGRESSIVE_STAGE_COUNT):
        stage_reports = [report["stages"][stage_index] for report in reports]
        tracker_count = 3 + stage_index
        if any(int(stage["tracker_count"]) != tracker_count for stage in stage_reports):
            raise RuntimeError("不同添加顺序的阶段 Tracker 数量不一致。")
        summaries.append(
            {
                "stage_index": stage_index,
                "tracker_count": tracker_count,
                "dit_deployed": _summarize_metric_dicts(
                    [stage["dit_deployed"] for stage in stage_reports]
                ),
            }
        )
    return summaries


def ground_truth_jitter_reference(reports: list[dict]) -> float:
    values = np.asarray(
        [
            report["overall"]["dit_deployed"]["gt_jitter_m_per_s3"]
            for report in reports
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or not np.allclose(
        values, values[0], atol=1e-9, rtol=1e-9
    ):
        raise RuntimeError("六种添加顺序的 GT Jitter reference 必须一致。")
    return float(values[0])


def _summarize_metric_dicts(metrics: list[dict]) -> dict:
    if not metrics:
        raise ValueError("指标汇总至少需要一组结果。")
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for key in FINAL_METRIC_KEYS:
        values = np.asarray([value[key] for value in metrics], dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"指标 {key} 包含无效值，无法汇总。")
        mean[key] = float(np.mean(values))
        # 六种排列是完整总体而非抽样，因此使用总体标准差 ddof=0。
        std[key] = float(np.std(values, ddof=0))
    return {"mean": mean, "std_across_add_orders": std}


# endregion


def main(argv: list[str] | None = None) -> dict:
    parser = build_arg_parser()
    args = parse_and_load_from_model(parser, argv)
    if 0 < int(args.max_frames) < MIN_PROGRESSIVE_SCORED_FRAMES:
        raise ValueError(
            f"--max_frames 必须为 0 或至少 {MIN_PROGRESSIVE_SCORED_FRAMES}。"
        )
    selected_order_names = resolve_add_order_names(list(args.add_orders))
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
    entries = read_longseq_source_entries(
        args.source_dir,
        args.split_dir,
        split=args.split,
        min_frames=(
            REALTIME_POSE_EVAL_METRICS_START_FRAME
            + MIN_PROGRESSIVE_SCORED_FRAMES
        ),
        include_mirror=False,
    )
    if args.limit > 0:
        entries = entries[: args.limit]
    protocols = build_progressive_addition_protocols(selected_order_names)
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
    reports = [format_progressive_order_report(report) for report in shared_reports]
    payload = {
        "experiment": "progressive_tracker_addition_3_to_6",
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
            "closed-loop starts at frame 11; official metrics start at frame 30"
        ),
        "tracker_policy": (
            "core three during warmup; scored current-frame masks follow equal quarters"
        ),
        "sampling_noise_policy": "common_random_numbers",
        "sampling_noise_seed": int(args.seed),
        "schedule": {
            "stage_policy": "equal_scored_quarters",
            "warmup_tracker_available": list(CORE_THREE_AVAILABLE),
            "warmup_tracker_count": int(sum(CORE_THREE_AVAILABLE)),
            "tracker_counts": [3, 4, 5, 6],
            "minimum_scored_frames": MIN_PROGRESSIVE_SCORED_FRAMES,
            "add_orders": [
                list(protocol.metadata["add_order"]) for protocol in protocols
            ],
        },
        "progressive_orders": reports,
        "final_metrics": summarize_progressive_orders(reports),
        "progressive_stage_summary": summarize_progressive_stages(reports),
        "ground_truth_jitter_reference": ground_truth_jitter_reference(reports),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[progressive-addition] wrote {output}")
    return payload


if __name__ == "__main__":
    main()
