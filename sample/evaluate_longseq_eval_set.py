from __future__ import annotations

import argparse
import json
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_loaders.build_realtime_longseq_eval_set import (
    DEFAULT_LONGSEQ_EVAL_ROOT,
    build_sequence_output_dir_name,
    read_longseq_manifest,
    resolve_longseq_eval_dir,
    resolve_manifest_source_path,
    sanitize_path_token,
)
from data_loaders.longseq_eval_dropout import (
    DROPOUT_PRESET_NONE,
    LongseqDropoutConfig,
    add_longseq_dropout_options,
    apply_longseq_dropout_to_source,
    build_longseq_dropout_config,
)
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, REALTIME_POSE_SEQ_LEN, get_schema_spec
from eval.evaluate_realtime_pose_rollout import evaluate_rollout_file, summarize
from sample.evaluate_unity_stream_source import (
    HISTORY_POSE_SOURCE_CHOICES,
    HISTORY_POSE_SOURCE_REFERENCE,
    WARMUP_TARGET_SOURCE_CHOICES,
    WARMUP_TARGET_SOURCE_FIRST_FRAME,
    build_long_sequence_payload,
    load_v2_source_with_sensor_valid,
    repeat_source_sequence,
    save_long_sequence_result,
    validate_v2_runtime_args,
)
from sample.render_realtime_pose_comparison import render_realtime_pose_comparison
from sample.simulate_unity_stream import (
    DEFAULT_TRACKER_IK_BLEND,
    DEFAULT_TRACKER_IK_DELTA_LIMIT,
    DEFAULT_TRACKER_IK_ITERATIONS,
    DEFAULT_TRACKER_IK_LR,
    DEFAULT_TRACKER_IK_TARGET_SMOOTHING,
)
from sample.utils import load_checkpoint_model
from utils import dist_util
from utils.model_util import create_model_and_diffusion
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import (
    add_base_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    parse_and_load_from_model,
    str2bool,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a fixed realtime_pose longseq eval set.")
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)

    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    longseq = parser.add_argument_group("longseq_eval")
    longseq.add_argument("--eval_root", default=DEFAULT_LONGSEQ_EVAL_ROOT, type=str)
    longseq.add_argument("--eval_set", default="latest", type=str)
    longseq.add_argument("--normalizer_dir", default="dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_60hz", type=str)
    longseq.add_argument("--normalize_input", default=True, type=str2bool)
    longseq.add_argument("--input_feats", default=schema.feature_dim, type=int)
    longseq.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
    longseq.add_argument("--loop_count", default=1, type=int)
    longseq.add_argument("--limit", default=0, type=int)
    longseq.add_argument("--initial_root_yaw", default=None, type=float)
    longseq.add_argument(
        "--history_pose_source",
        default=HISTORY_POSE_SOURCE_REFERENCE,
        choices=HISTORY_POSE_SOURCE_CHOICES,
        type=str,
    )
    longseq.add_argument(
        "--warmup_target_source",
        default=WARMUP_TARGET_SOURCE_FIRST_FRAME,
        choices=WARMUP_TARGET_SOURCE_CHOICES,
        type=str,
    )

    render = parser.add_argument_group("render")
    render.add_argument("--render_mp4", default=True, type=str2bool)
    render.add_argument("--render_fps", default=30, type=int)
    render.add_argument("--render_stride", default=1, type=int)
    render.add_argument("--render_camera_mode", default="follow", choices=["global", "follow"], type=str)
    render.add_argument("--render_layout", default="overlay", choices=["split", "overlay"], type=str)
    render.add_argument("--render_local_radius", default=1.25, type=float)

    stream = parser.add_argument_group("stream")
    stream.add_argument("--root_correction", default=True, action=BooleanOptionalAction)
    stream.add_argument("--tracker_ik", default=True, action=BooleanOptionalAction)
    stream.add_argument("--tracker_ik_iterations", default=DEFAULT_TRACKER_IK_ITERATIONS, type=int)
    stream.add_argument("--tracker_ik_lr", default=DEFAULT_TRACKER_IK_LR, type=float)
    stream.add_argument("--tracker_ik_blend", default=DEFAULT_TRACKER_IK_BLEND, type=float)
    stream.add_argument("--tracker_ik_target_smoothing", default=DEFAULT_TRACKER_IK_TARGET_SMOOTHING, type=float)
    stream.add_argument("--tracker_ik_delta_limit", default=DEFAULT_TRACKER_IK_DELTA_LIMIT, type=float)
    add_longseq_dropout_options(parser)
    return parser


def evaluate_longseq_entries(
    entries: list[dict[str, Any]],
    eval_set_dir: Path,
    output_dir: Path,
    model,
    diffusion,
    device: torch.device,
    normalizer: RealtimePoseNormalizer | None,
    use_ddim: bool,
    model_path: str | Path = "",
    weights: str = "",
    loop_count: int = 1,
    limit: int = 0,
    initial_root_yaw: float | None = None,
    history_pose_source: str = HISTORY_POSE_SOURCE_REFERENCE,
    warmup_target_source: str = WARMUP_TARGET_SOURCE_FIRST_FRAME,
    root_correction: bool = True,
    tracker_ik: bool = True,
    tracker_ik_iterations: int = DEFAULT_TRACKER_IK_ITERATIONS,
    tracker_ik_lr: float = DEFAULT_TRACKER_IK_LR,
    tracker_ik_blend: float = DEFAULT_TRACKER_IK_BLEND,
    tracker_ik_target_smoothing: float = DEFAULT_TRACKER_IK_TARGET_SMOOTHING,
    tracker_ik_delta_limit: float = DEFAULT_TRACKER_IK_DELTA_LIMIT,
    ik_init_mode: str = "random",
    ik_init_timestep: int = -1,
    ik_init_iterations: int = 16,
    ik_init_lr: float = 0.03,
    ik_init_pos_weight: float = 1.0,
    ik_init_rot_weight: float = 0.2,
    ik_init_reg_weight: float = 0.01,
    ik_init_delta_limit: float = 0.15,
    dropout_config: LongseqDropoutConfig | None = None,
    render_mp4: bool = True,
    render_fps: int = 30,
    render_stride: int = 1,
    render_camera_mode: str = "follow",
    render_layout: str = "overlay",
    render_local_radius: float = 1.25,
) -> dict[str, Any]:
    eval_set_dir = Path(eval_set_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_entries = entries[: int(limit)] if int(limit) > 0 else entries
    if not selected_entries:
        raise RuntimeError("No longseq entries to evaluate.")
    dropout_config = dropout_config or LongseqDropoutConfig()

    results = []
    for entry in selected_entries:
        sequence_id = str(entry["sequence_id"])
        sequence_dir = output_dir / build_sequence_output_dir_name(entry)
        sequence_dir.mkdir(parents=True, exist_ok=True)

        source_path = resolve_manifest_source_path(eval_set_dir=eval_set_dir, entry=entry)
        source = load_v2_source_with_sensor_valid(source_path)
        source, dropout_metadata = apply_longseq_dropout_to_source(
            source=source,
            sequence_id=sequence_id,
            config=dropout_config,
        )
        source = repeat_source_sequence(source, loop_count=int(loop_count))
        payload = build_long_sequence_payload(
            model=model,
            diffusion=diffusion,
            source=source,
            device=device,
            use_ddim=bool(use_ddim),
            normalizer=normalizer,
            initial_root_yaw=initial_root_yaw,
            history_pose_source=history_pose_source,
            warmup_target_source=warmup_target_source,
            root_correction=bool(root_correction),
            tracker_ik=bool(tracker_ik),
            tracker_ik_iterations=int(tracker_ik_iterations),
            tracker_ik_lr=float(tracker_ik_lr),
            tracker_ik_blend=float(tracker_ik_blend),
            tracker_ik_target_smoothing=float(tracker_ik_target_smoothing),
            tracker_ik_delta_limit=float(tracker_ik_delta_limit),
            ik_init_mode=ik_init_mode,
            ik_init_timestep=int(ik_init_timestep),
            ik_init_iterations=int(ik_init_iterations),
            ik_init_lr=float(ik_init_lr),
            ik_init_pos_weight=float(ik_init_pos_weight),
            ik_init_rot_weight=float(ik_init_rot_weight),
            ik_init_reg_weight=float(ik_init_reg_weight),
            ik_init_delta_limit=float(ik_init_delta_limit),
        )
        metadata = dict(payload["metadata"].item())
        metadata.update(
            {
                "sequence_id": sequence_id,
                "eval_set_dir": str(eval_set_dir),
                "eval_set_source_path": str(source_path),
                "source_relative_path": str(entry.get("source_relative_path", "")),
                "model_path": str(model_path),
                "weights": str(weights),
                "loop_count": int(loop_count),
                **dropout_metadata,
            }
        )
        payload["metadata"] = np.asarray(metadata, dtype=object)

        result_path = sequence_dir / "unity_stream_long_sequence_result.npz"
        summary_path = sequence_dir / "unity_stream_eval_summary.json"
        save_long_sequence_result(result_path, payload)
        result = evaluate_rollout_file(result_path)
        result.update(
            {
                "sequence_id": sequence_id,
                "source_relative_path": str(entry.get("source_relative_path", "")),
                "num_frames": int(entry["num_frames"]),
                "result_path": str(result_path),
                "summary_path": str(summary_path),
                "valid_tracker_ratio": float(dropout_metadata["valid_tracker_ratio"]),
                "min_valid_trackers": int(dropout_metadata["min_valid_trackers"]),
                "dropout_preset": dropout_config.preset,
                "tracker_mask_policy": dropout_config.tracker_mask_policy,
            }
        )
        if bool(render_mp4):
            mp4_path = sequence_dir / "unity_stream_comparison.mp4"
            render_realtime_pose_comparison(
                output_path=mp4_path,
                reference_joints=payload["reference_joints_world"],
                predicted_joints=payload["predicted_joints_world"],
                tracker_pos_world=payload["tracker_pos_world"],
                sensor_valid=payload["sensor_valid"],
                eval_frame_mask=payload["eval_frame_mask"],
                root_yaw_reference=payload["root_yaw_reference"],
                root_yaw_predicted=payload["root_yaw_predicted"],
                fps=int(render_fps),
                stride=int(render_stride),
                camera_mode=str(render_camera_mode),
                layout=str(render_layout),
                local_radius=float(render_local_radius),
            )
            result["mp4_path"] = str(mp4_path)
        write_sequence_summary(summary_path=summary_path, result=result)
        results.append(result)

    aggregate = summarize(results)
    summary_payload = {
        "summary": aggregate,
        "files": results,
        "metadata": {
            "kind": "longseq_eval_rollout",
            "eval_set_dir": str(eval_set_dir),
            "output_dir": str(output_dir),
            "model_path": str(model_path),
            "weights": str(weights),
            "sequence_count": len(results),
            "dropout_config": dropout_config.to_dict(),
        },
    }
    aggregate_path = output_dir / "longseq_eval_summary.json"
    with aggregate_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(summary_payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    summary_payload["summary_path"] = str(aggregate_path)
    return summary_payload


def write_sequence_summary(summary_path: Path, result: dict[str, Any]) -> None:
    compact = {key: value for key, value in result.items() if key != "path" and not isinstance(value, list)}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump({"summary": compact, "files": [result]}, file, indent=2, ensure_ascii=False)
        file.write("\n")


def build_default_output_dir(
    eval_set_dir: Path,
    model_path: str | Path,
    weights: str,
    dropout_config: LongseqDropoutConfig | None = None,
) -> Path:
    checkpoint_tag = sanitize_path_token(Path(model_path).stem)
    if weights:
        checkpoint_tag = f"{checkpoint_tag}_{sanitize_path_token(weights)}"
    if dropout_config is not None and dropout_config.preset != DROPOUT_PRESET_NONE:
        checkpoint_tag = f"{checkpoint_tag}_{sanitize_path_token(dropout_config.preset)}"
    return Path("output") / "longseq_eval" / eval_set_dir.name / checkpoint_tag


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parse_and_load_from_model(parser, argv=argv)
    validate_v2_runtime_args(args)

    eval_set_dir = resolve_longseq_eval_dir(eval_root=args.eval_root, eval_set=args.eval_set)
    entries = read_longseq_manifest(eval_set_dir)
    dropout_config = build_longseq_dropout_config(args)
    normalizer = RealtimePoseNormalizer(args.normalizer_dir, schema_name=REALTIME_POSE_SCHEMA_NAME)
    if not bool(args.normalize_input):
        normalizer = None

    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    model, diffusion = create_model_and_diffusion(args)
    model, weights = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)

    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else build_default_output_dir(
        eval_set_dir=eval_set_dir,
        model_path=args.model_path,
        weights=weights,
        dropout_config=dropout_config,
    ).resolve()
    summary = evaluate_longseq_entries(
        entries=entries,
        eval_set_dir=eval_set_dir,
        output_dir=output_dir,
        model=model,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        use_ddim=str(args.ts_respace).startswith("ddim"),
        model_path=args.model_path,
        weights=weights,
        loop_count=int(args.loop_count),
        limit=int(args.limit),
        initial_root_yaw=args.initial_root_yaw,
        history_pose_source=str(args.history_pose_source),
        warmup_target_source=str(args.warmup_target_source),
        root_correction=bool(args.root_correction),
        tracker_ik=bool(args.tracker_ik),
        tracker_ik_iterations=int(args.tracker_ik_iterations),
        tracker_ik_lr=float(args.tracker_ik_lr),
        tracker_ik_blend=float(args.tracker_ik_blend),
        tracker_ik_target_smoothing=float(args.tracker_ik_target_smoothing),
        tracker_ik_delta_limit=float(args.tracker_ik_delta_limit),
        ik_init_mode=args.ik_init_mode,
        ik_init_timestep=int(args.ik_init_timestep),
        ik_init_iterations=int(args.ik_init_iterations),
        ik_init_lr=float(args.ik_init_lr),
        ik_init_pos_weight=float(args.ik_init_pos_weight),
        ik_init_rot_weight=float(args.ik_init_rot_weight),
        ik_init_reg_weight=float(args.ik_init_reg_weight),
        ik_init_delta_limit=float(args.ik_init_delta_limit),
        dropout_config=dropout_config,
        render_mp4=bool(args.render_mp4),
        render_fps=int(args.render_fps),
        render_stride=int(args.render_stride),
        render_camera_mode=str(args.render_camera_mode),
        render_layout=str(args.render_layout),
        render_local_radius=float(args.render_local_radius),
    )
    print(
        "[evaluate_longseq_eval_set] "
        f"sequences={summary['metadata']['sequence_count']} output={output_dir} summary={summary['summary_path']}"
    )
    return summary


if __name__ == "__main__":
    main()
