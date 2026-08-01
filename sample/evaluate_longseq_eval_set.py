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
from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    load_realtime_source,
)
from data_loaders.realtime_pose_geometry import build_pose_target_np, extract_forward_yaw_np
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_TARGET_DIM,
)
from data_loaders.tracker_timeline import TrackerTimeline, build_tracker_timeline, classify_tracker_window
from eval.evaluate_realtime_pose import public_result
from eval.evaluate_realtime_pose_rollout import evaluate_rollout_file, summarize_rollouts
from sample.realtime_pose_runtime import (
    WorldPoseState,
    build_online_conditioning,
    decode_and_resolve_pose,
    sample_online_target,
)
from sample.render_realtime_pose_comparison import render_realtime_pose_comparison
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
    parser = argparse.ArgumentParser(description="在固定长序列集合上执行 140D 自回归评估。")
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)

    longseq = parser.add_argument_group("longseq_eval")
    longseq.add_argument("--eval_root", default=DEFAULT_LONGSEQ_EVAL_ROOT, type=str)
    longseq.add_argument("--eval_set", default="latest", type=str)
    longseq.add_argument(
        "--normalizer_dir",
        default="dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
        type=str,
    )
    longseq.add_argument("--normalize_input", default=True, type=str2bool)
    longseq.add_argument("--input_feats", default=REALTIME_POSE_TARGET_DIM, type=int)
    longseq.add_argument("--limit", default=0, type=int)
    longseq.add_argument("--timeline_seed", default=10, type=int)

    render = parser.add_argument_group("render")
    render.add_argument("--render_mp4", default=False, action=BooleanOptionalAction)
    render.add_argument("--render_fps", default=30, type=int)
    render.add_argument("--render_stride", default=1, type=int)
    render.add_argument("--render_camera_mode", default="follow", choices=["global", "follow"], type=str)
    render.add_argument("--render_layout", default="overlay", choices=["split", "overlay"], type=str)
    render.add_argument("--render_local_radius", default=1.25, type=float)
    return parser


def build_initial_pose_history(
    source: dict[str, np.ndarray],
    joint_rotations_world: np.ndarray,
) -> list[WorldPoseState]:
    """前 60 帧只用于启动自回归，不进入评估结果。"""

    return [
        WorldPoseState(
            joint_rotations_world=joint_rotations_world[frame_index].copy(),
            root_yaw_world=float(source["root_yaw"][frame_index]),
            hip_height=float(source["pelvis_height"][frame_index, 0]),
            root_position_world=source["root_pos_world"][frame_index].copy(),
        )
        for frame_index in range(REALTIME_POSE_HISTORY_LENGTH)
    ]


def rollout_long_sequence_source(
    model,
    diffusion,
    source: dict[str, np.ndarray],
    timeline: TrackerTimeline,
    device: torch.device,
    normalizer: RealtimePoseNormalizer | None,
) -> dict[str, np.ndarray]:
    """对一个完整 source 做逐帧闭环采样，输出形状统一为 `[1,T,...]`。"""

    frame_count = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
    if frame_count <= REALTIME_POSE_HISTORY_LENGTH:
        raise ValueError(f"长序列至少需要 61 帧，实际为 {frame_count}")

    joint_rotations_world = compute_source_joint_rotations_world(source)
    head_rotations_world = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"][:, 0])
    head_yaws = extract_forward_yaw_np(head_rotations_world, initial_yaw=0.0)
    pose_history = build_initial_pose_history(source, joint_rotations_world)

    reference_targets: list[np.ndarray] = []
    predicted_targets: list[np.ndarray] = []
    reference_local_delta: list[np.ndarray] = []
    predicted_local_delta: list[np.ndarray] = []
    reference_joints: list[np.ndarray] = []
    predicted_joints: list[np.ndarray] = []
    reference_roots: list[np.ndarray] = []
    predicted_roots: list[np.ndarray] = []
    reference_root_yaws: list[float] = []
    predicted_root_yaws: list[float] = []
    reference_hip_heights: list[float] = []
    predicted_hip_heights: list[float] = []
    known_masks: list[np.ndarray] = []
    tracker_positions: list[np.ndarray] = []
    configured_values: list[np.ndarray] = []
    measured_values: list[np.ndarray] = []
    missing_ages: list[np.ndarray] = []
    scenarios: list[str] = []
    known_errors: list[float] = []

    for absolute_frame in range(REALTIME_POSE_HISTORY_LENGTH, frame_count):
        window_start = absolute_frame - REALTIME_POSE_HISTORY_LENGTH
        frame_slice = slice(window_start, absolute_frame + 1)
        timeline_window = timeline.window(window_start)
        previous_head_yaw = float(head_yaws[window_start - 1]) if window_start > 0 else 0.0
        floor_y = float(source["root_pos_world"][absolute_frame, 1])
        conditioning = build_online_conditioning(
            pose_history_world=pose_history,
            tracker_pos_world=source["tracker_pos_world"][frame_slice],
            tracker_rot_world_6d=source["tracker_rot_world_6d"][frame_slice],
            configured=timeline_window.configured,
            measured_valid=timeline_window.measured_valid,
            missing_age=timeline_window.missing_age,
            floor_y=floor_y,
            normalizer=normalizer,
            initial_head_yaw=previous_head_yaw,
        )
        predicted_target = sample_online_target(
            model=model,
            diffusion=diffusion,
            conditioning=conditioning,
            device=device,
            normalizer=normalizer,
        )
        current_head_yaw = float(conditioning["current_head_yaw_world"])
        current_head_position = np.asarray(conditioning["current_head_position_world"], dtype=np.float32)
        resolved = decode_and_resolve_pose(
            target_raw=predicted_target,
            tracker_current_raw=conditioning["tracker_window_raw"][-1],
            current_head_yaw_world=current_head_yaw,
            current_head_position_world=current_head_position,
            floor_y=floor_y,
            joint_offsets_parent=source["joint_offsets_parent"],
            joint_rest_local_rotations_6d=source["joint_rest_local_rotations_6d"],
        )
        reference_target = build_pose_target_np(
            joint_rotations_world[absolute_frame : absolute_frame + 1],
            source["root_yaw"][absolute_frame : absolute_frame + 1],
            current_head_yaw,
        )[0]
        scenario = classify_tracker_window(
            configured=timeline_window.configured,
            measured_valid=timeline_window.measured_valid,
        )
        if scenario is None:
            raise ValueError(f"绝对帧 {absolute_frame} 的 Tracker 窗口无法归入五类场景")

        reference_targets.append(reference_target)
        predicted_targets.append(predicted_target)
        reference_local_delta.append(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY][absolute_frame])
        predicted_local_delta.append(resolved.body_local_delta_6d)
        reference_joints.append(source["joints_world"][absolute_frame])
        predicted_joints.append(resolved.joints_world)
        reference_roots.append(source["root_pos_world"][absolute_frame])
        predicted_roots.append(resolved.root_position_world)
        reference_root_yaws.append(float(source["root_yaw"][absolute_frame]))
        predicted_root_yaws.append(resolved.root_yaw_world)
        reference_hip_heights.append(float(source["pelvis_height"][absolute_frame, 0]))
        predicted_hip_heights.append(resolved.hip_height)
        known_masks.append(np.asarray(conditioning["known_mask"], dtype=bool))
        tracker_positions.append(source["tracker_pos_world"][absolute_frame])
        configured_values.append(timeline.configured[absolute_frame])
        measured_values.append(timeline.measured_valid[absolute_frame])
        missing_ages.append(timeline.missing_age[absolute_frame])
        scenarios.append(scenario)
        known_errors.append(resolved.known_rotation_max_error)

        # 这里只回灌模型输出；GT pose 在第 60 帧之后不会再次进入历史。
        pose_history = [*pose_history[1:], resolved.as_world_state()]

    evaluated_frames = frame_count - REALTIME_POSE_HISTORY_LENGTH
    return {
        "fps": np.float32(60.0),
        "absolute_frame_index": np.arange(
            REALTIME_POSE_HISTORY_LENGTH,
            frame_count,
            dtype=np.int64,
        ),
        "reference_target_raw": np.asarray(reference_targets, dtype=np.float32)[None],
        "reconstructed_target_raw": np.asarray(predicted_targets, dtype=np.float32)[None],
        "reference_body_local_delta_6d": np.asarray(reference_local_delta, dtype=np.float32)[None],
        "predicted_body_local_delta_6d": np.asarray(predicted_local_delta, dtype=np.float32)[None],
        "reference_joints_world": np.asarray(reference_joints, dtype=np.float32)[None],
        "predicted_joints_world": np.asarray(predicted_joints, dtype=np.float32)[None],
        "reference_root_position_world": np.asarray(reference_roots, dtype=np.float32)[None],
        "predicted_root_position_world": np.asarray(predicted_roots, dtype=np.float32)[None],
        "reference_root_yaw_world": np.asarray(reference_root_yaws, dtype=np.float32)[None],
        "predicted_root_yaw_world": np.asarray(predicted_root_yaws, dtype=np.float32)[None],
        "reference_hip_height": np.asarray(reference_hip_heights, dtype=np.float32)[None],
        "predicted_hip_height": np.asarray(predicted_hip_heights, dtype=np.float32)[None],
        "known_mask": np.asarray(known_masks, dtype=bool)[None],
        "tracker_pos_world": np.asarray(tracker_positions, dtype=np.float32)[None],
        "configured": np.asarray(configured_values, dtype=bool)[None],
        "measured_valid": np.asarray(measured_values, dtype=bool)[None],
        "missing_age": np.asarray(missing_ages, dtype=np.int64)[None],
        "scenario": np.asarray(scenarios)[None],
        "eval_frame_mask": np.ones((1, evaluated_frames), dtype=bool),
        "known_rotation_max_error": np.asarray(known_errors, dtype=np.float32)[None],
    }


def evaluate_longseq_entries(
    entries: list[dict[str, Any]],
    eval_set_dir: Path,
    output_dir: Path,
    model,
    diffusion,
    device: torch.device,
    normalizer: RealtimePoseNormalizer | None,
    model_path: str | Path = "",
    weights: str = "",
    limit: int = 0,
    timeline_seed: int = 10,
    render_mp4: bool = False,
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
        raise RuntimeError("长序列评估集合为空。")

    results = []
    for entry in selected_entries:
        sequence_id = str(entry["sequence_id"])
        source_path = resolve_manifest_source_path(eval_set_dir=eval_set_dir, entry=entry)
        source = load_realtime_source(source_path)
        frame_count = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
        timeline = build_tracker_timeline(
            source_id=sequence_id,
            frame_count=frame_count,
            global_seed=int(timeline_seed),
        )
        print(f"[longseq] {sequence_id}: {frame_count} frames")
        payload = rollout_long_sequence_source(
            model=model,
            diffusion=diffusion,
            source=source,
            timeline=timeline,
            device=device,
            normalizer=normalizer,
        )

        sequence_dir = output_dir / build_sequence_output_dir_name(entry)
        sequence_dir.mkdir(parents=True, exist_ok=True)
        result_path = sequence_dir / "rollout_result.npz"
        np.savez(result_path, **payload)
        result = evaluate_rollout_file(result_path)
        result.update(
            {
                "sequence_id": sequence_id,
                "source_relative_path": str(entry.get("source_relative_path", "")),
                "num_frames": frame_count,
                "evaluated_frames": frame_count - REALTIME_POSE_HISTORY_LENGTH,
                "result_path": str(result_path),
            }
        )

        if render_mp4:
            mp4_path = sequence_dir / "comparison.mp4"
            render_realtime_pose_comparison(
                output_path=mp4_path,
                reference_joints=payload["reference_joints_world"],
                predicted_joints=payload["predicted_joints_world"],
                tracker_pos_world=payload["tracker_pos_world"],
                sensor_valid=payload["measured_valid"],
                eval_frame_mask=payload["eval_frame_mask"],
                root_yaw_reference=payload["reference_root_yaw_world"],
                root_yaw_predicted=payload["predicted_root_yaw_world"],
                fps=int(render_fps),
                stride=int(render_stride),
                camera_mode=str(render_camera_mode),
                layout=str(render_layout),
                local_radius=float(render_local_radius),
            )
            result["mp4_path"] = str(mp4_path)

        summary_path = sequence_dir / "rollout_eval_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump({"summary": public_result(result)}, file, ensure_ascii=False, indent=2)
        result["summary_path"] = str(summary_path)
        results.append(result)

    aggregate = summarize_rollouts(results)
    summary_payload = {
        "summary": aggregate,
        "files": [public_result(result) for result in results],
        "metadata": {
            "kind": "realtime_pose_140d_longseq_rollout",
            "eval_set_dir": str(eval_set_dir),
            "output_dir": str(output_dir),
            "model_path": str(model_path),
            "weights": str(weights),
            "timeline_seed": int(timeline_seed),
            "sequence_count": len(results),
        },
    }
    aggregate_path = output_dir / "longseq_eval_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as file:
        json.dump(summary_payload, file, ensure_ascii=False, indent=2)
    summary_payload["summary_path"] = str(aggregate_path)
    return summary_payload


def build_default_output_dir(eval_set_dir: Path, model_path: str | Path, weights: str) -> Path:
    checkpoint_tag = sanitize_path_token(Path(model_path).stem)
    if weights:
        checkpoint_tag = f"{checkpoint_tag}_{sanitize_path_token(weights)}"
    return Path("output") / "longseq_eval" / eval_set_dir.name / checkpoint_tag


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_and_load_from_model(build_arg_parser(), argv=argv)
    eval_set_dir = resolve_longseq_eval_dir(eval_root=args.eval_root, eval_set=args.eval_set)
    entries = read_longseq_manifest(eval_set_dir)
    normalizer = (
        RealtimePoseNormalizer(args.normalizer_dir)
        if bool(args.normalize_input)
        else None
    )

    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    model, diffusion = create_model_and_diffusion(args)
    model, weights = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    output_dir = (
        Path(args.output_dir).resolve()
        if str(args.output_dir).strip()
        else build_default_output_dir(eval_set_dir, args.model_path, weights).resolve()
    )
    summary = evaluate_longseq_entries(
        entries=entries,
        eval_set_dir=eval_set_dir,
        output_dir=output_dir,
        model=model,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        model_path=args.model_path,
        weights=weights,
        limit=int(args.limit),
        timeline_seed=int(args.timeline_seed),
        render_mp4=bool(args.render_mp4),
        render_fps=int(args.render_fps),
        render_stride=int(args.render_stride),
        render_camera_mode=str(args.render_camera_mode),
        render_layout=str(args.render_layout),
        render_local_radius=float(args.render_local_radius),
    )
    print(f"[evaluate_longseq_eval_set] wrote {summary['summary_path']}")
    return summary


if __name__ == "__main__":
    main()
