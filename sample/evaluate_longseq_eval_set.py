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
    resolve_source_entry_path,
)
from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    load_realtime_source,
)
from data_loaders.sensor_masking import (
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
    STATIC_OPTIONAL_TRACKER_MASKS,
)
from eval.evaluate_realtime_pose_predictor import (
    PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
    evaluation_last_frame_exclusive,
    evaluate_predictor_entries,
    pose_rotation_error_deg,
)
from eval.realtime_pose_metrics import (
    aggregate_rpm_p2_mc_metrics,
    compute_rpm_p2_mc_metrics,
)
from sample.realtime_pose_runtime import (
    RealtimePoseRuntime,
    WorldPoseState,
    decode_and_resolve_pose,
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
    reports = []
    for config_name in args.tracker_configs:
        mask = tracker_masks[config_name]
        reports.append(
            evaluate_tracker_configuration(
                config_name=config_name,
                tracker_available=np.asarray(mask, dtype=bool),
                entries=entries,
                predictor=predictor,
                dit=dit,
                diffusion=diffusion,
                device=device,
                normalizer=normalizer,
                args=args,
            )
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
    generated_frame_count = 0
    frame_count = 0
    raw_sequence_metrics: list[dict[str, float | None]] = []
    deployed_sequence_metrics: list[dict[str, float | None]] = []
    for entry in entries:
        source = load_realtime_source(resolve_source_entry_path(entry))
        world_rotations = compute_source_joint_rotations_world(source)
        last = evaluation_last_frame_exclusive(
            len(world_rotations), int(args.max_frames)
        )
        if last <= REALTIME_POSE_EVAL_METRICS_START_FRAME:
            continue
        scored_raw_rotations: list[np.ndarray] = []
        scored_deployed_rotations: list[np.ndarray] = []
        scored_target_rotations: list[np.ndarray] = []
        scored_raw_positions: list[np.ndarray] = []
        scored_deployed_positions: list[np.ndarray] = []
        scored_target_positions: list[np.ndarray] = []
        pose_history = [
            WorldPoseState(
                joint_rotations_world=world_rotations[index],
                root_yaw_world=float(source["root_yaw"][index]),
                hip_height=float(source["pelvis_height"][index, 0]),
                root_position_world=source["root_pos_world"][index],
            )
            for index in range(1, 11)
        ]
        runtime = RealtimePoseRuntime(
            predictor,
            dit,
            diffusion,
            device,
            source["joint_offsets_parent"],
            source["joint_rest_local_rotations_6d"],
            normalizer,
            fabrik_iterations=args.fabrik_iterations,
            ik_direction_only_quality=args.ik_direction_only_quality,
            ik_residual_scale=args.ik_residual_scale,
            ik_position_solved_quality=args.ik_position_solved_quality,
        )
        runtime.initialize_history(
            pose_history,
            source["tracker_pos_world"][:11],
            source["tracker_rot_world_6d"][:11],
            source["root_pos_world"][:11, 1],
        )
        for current in range(PREDICTOR_EVAL_FIRST_GENERATED_FRAME, last):
            previous_root_yaw = runtime.pose_history[-1].root_yaw_world
            result = runtime.step(
                source["tracker_pos_world"][current],
                source["tracker_rot_world_6d"][current],
                tracker_available,
                float(source["root_pos_world"][current, 1]),
            )
            generated_frame_count += 1
            if current < REALTIME_POSE_EVAL_METRICS_START_FRAME:
                continue
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
            scored_deployed_rotations.append(
                result.resolved_pose.joint_rotations_world
            )
            scored_target_rotations.append(world_rotations[current])
            scored_raw_positions.append(raw_resolved.joints_world)
            scored_deployed_positions.append(result.resolved_pose.joints_world)
            scored_target_positions.append(source["joints_world"][current])
            frame_count += 1
        if scored_target_rotations:
            target_rotations = np.stack(scored_target_rotations, axis=0)
            target_positions = np.stack(scored_target_positions, axis=0)
            common = {
                "target_global_rotations": target_rotations,
                "target_joint_positions": target_positions,
                "fps": REALTIME_POSE_FPS,
            }
            raw_sequence_metrics.append(
                compute_rpm_p2_mc_metrics(
                    predicted_global_rotations=np.stack(
                        scored_raw_rotations, axis=0
                    ),
                    predicted_joint_positions=np.stack(
                        scored_raw_positions, axis=0
                    ),
                    **common,
                )
            )
            deployed_sequence_metrics.append(
                compute_rpm_p2_mc_metrics(
                    predicted_global_rotations=np.stack(
                        scored_deployed_rotations, axis=0
                    ),
                    predicted_joint_positions=np.stack(
                        scored_deployed_positions, axis=0
                    ),
                    **common,
                )
            )
    if frame_count <= 0:
        raise RuntimeError(f"Tracker 配置 {config_name} 没有可计分帧。")
    report = {
        "name": config_name,
        "tracker_available": tracker_available.tolist(),
        "seen_during_training": config_name in {"core_only", "all_six"},
        "generated_frames": generated_frame_count,
        "evaluated_frames": frame_count,
        "dit_raw": aggregate_rpm_p2_mc_metrics(raw_sequence_metrics),
        "dit_deployed": aggregate_rpm_p2_mc_metrics(
            deployed_sequence_metrics
        ),
    }
    print(f"[longseq] {config_name}: {report['dit_deployed']}", flush=True)
    return report


if __name__ == "__main__":
    main()
