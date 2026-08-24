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
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
    STATIC_OPTIONAL_TRACKER_MASKS,
)
from eval.evaluate_realtime_pose_predictor import (
    PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
    evaluate_predictor_entries,
)
from sample.realtime_pose_longseq_evaluator import (
    build_static_tracker_protocol,
    create_eval_noise_generator,
    evaluate_longseq_protocol,
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


TRACKER_CONFIG_NAMES = (
    "core_only",
    "core_hip",
    "core_left_foot",
    "core_right_foot",
    "core_both_feet",
    "core_hip_left_foot",
    "core_hip_right_foot",
    "all_six",
)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="评估 Predictor + 单帧 DiT 的 8 种静态 Tracker 配置。"
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    group = parser.add_argument_group("long sequence")
    group.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR)
    group.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    group.add_argument("--split", default="test")
    group.add_argument("--normalizer_dir", required=True)
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument("--limit", default=0, type=int)
    group.add_argument(
        "--tracker_configs",
        nargs="+",
        choices=TRACKER_CONFIG_NAMES,
        default=list(TRACKER_CONFIG_NAMES),
        help="选择要独立闭环评估的 Tracker 配置；默认运行全部 8 种。",
    )
    group.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="P2 预热完成后每条序列最多计分的帧数；0 表示直到序列结束。",
    )
    group.add_argument("--output_json", required=True)
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_arg_parser()
    args = parse_and_load_from_model(parser, argv)
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
        min_frames=REALTIME_POSE_EVAL_METRICS_START_FRAME + 1,
        include_mirror=False,
    )
    if args.limit > 0:
        entries = entries[: args.limit]
    predictor_only = evaluate_predictor_entries(
        entries=entries,
        predictor=predictor,
        device=device,
        normalizer=normalizer,
        max_frames=args.max_frames,
    )
    tracker_masks = dict(zip(TRACKER_CONFIG_NAMES, STATIC_OPTIONAL_TRACKER_MASKS))
    protocols = [
        build_static_tracker_protocol(
            config_name, np.asarray(tracker_masks[config_name], dtype=bool)
        )
        for config_name in args.tracker_configs
    ]
    reports = evaluate_longseq_protocols(
        protocols=protocols,
        entries=entries,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        args=args,
    )
    payload = {
        "predictor_model_path": str(Path(args.predictor_model_path).resolve()),
        "dit_model_path": str(Path(args.dit_model_path).resolve()),
        "dit_weight_source": dit_weight_source,
        "sampler": "projected_ddim",
        "diffusion_training_steps": int(args.diffusion_steps),
        "timestep_respacing": str(args.ts_respace),
        "sampling_steps": int(diffusion.num_timesteps),
        "source_fps": float(REALTIME_POSE_FPS),
        "initial_context_frames": PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
        "metrics_start_frame": REALTIME_POSE_EVAL_METRICS_START_FRAME,
        "history_policy": "closed-loop starts at frame 11; official metrics start at frame 30",
        "tracker_policy": "current frame only; no future tracker",
        "sampling_noise_policy": "common_random_numbers",
        "sampling_noise_seed": int(args.seed),
        "requested_tracker_configs": list(args.tracker_configs),
        "predictor_only": predictor_only,
        "configurations": reports,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[longseq] wrote {output}")
    return payload


def evaluate_tracker_configuration(
    *,
    config_name: str,
    tracker_available: np.ndarray,
    entries: list[dict],
    predictor,
    dit,
    diffusion,
    device: torch.device,
    normalizer,
    args,
) -> dict:
    protocol = build_static_tracker_protocol(config_name, tracker_available)
    return evaluate_longseq_protocol(
        protocol=protocol,
        entries=entries,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        args=args,
    )


if __name__ == "__main__":
    main()
