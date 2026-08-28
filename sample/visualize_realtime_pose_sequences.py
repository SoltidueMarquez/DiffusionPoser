from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import torch

from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    load_realtime_source,
)
from data_loaders.realtime_pose_geometry import (
    assemble_current_tracker_features_np,
    build_tracker_measurements_np,
)
from data_loaders.realtime_pose_predictor_features import (
    build_predictor_step_features_np,
    pose_head_to_world_rotations_np,
)
from data_loaders.sensor_masking import (
    CORE_THREE_AVAILABLE,
    HEAD_TRACKER_INDEX,
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
    REALTIME_POSE_TARGET_DIM,
    STATIC_OPTIONAL_TRACKER_MASKS,
)
from eval.evaluate_realtime_pose_predictor import (
    PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
    evaluation_last_frame_exclusive,
)
from eval.realtime_pose_metrics import compute_rpm_p2_mc_metrics
from sample.evaluate_longseq_eval_set import create_eval_noise_generator
from sample.realtime_pose_runtime import (
    RealtimePoseRuntime,
    WorldPoseState,
    decode_and_resolve_pose,
)
from sample.render_realtime_pose_comparison import (
    render_realtime_pose_four_way_comparison,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="串行生成 GT、Predictor、core-only DiT 与 all-six DiT 四路对比视频。"
    )
    add_base_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    group = parser.add_argument_group("visualization sequences")
    group.add_argument("--source_npz", nargs="+", required=True)
    group.add_argument("--normalizer_dir", required=True)
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="从正式计分帧开始最多可视化多少帧；0 表示直到序列结束。",
    )
    group.add_argument("--stride", default=1, type=int)
    group.add_argument(
        "--camera_mode", default="follow", choices=("global", "follow")
    )
    return parser


def _build_current_tracker_raw(
    source: dict[str, np.ndarray],
    current: int,
    head_yaw_world: float,
) -> np.ndarray:
    measurements = build_tracker_measurements_np(
        source["tracker_pos_world"][current : current + 1],
        source["tracker_rot_world_6d"][current : current + 1],
        source["tracker_pos_world"][current, HEAD_TRACKER_INDEX],
        float(source["root_pos_world"][current, 1]),
        head_yaw_world,
    )[0]
    return assemble_current_tracker_features_np(
        measurements,
        np.asarray(CORE_THREE_AVAILABLE, dtype=bool),
    )


def run_predictor_sequence(
    *,
    source: dict[str, np.ndarray],
    world_rotations: np.ndarray,
    predictor,
    device: torch.device,
    normalizer: RealtimePoseNormalizer,
    last: int,
) -> dict[str, np.ndarray]:
    """按正式评估相同的闭环策略生成 Predictor-only 序列。"""

    predicted_rotations: list[np.ndarray] = []
    predicted_positions: list[np.ndarray] = []
    predicted_root_yaw: list[float] = []
    motion_world = world_rotations[1:11].copy()
    previous_root_yaw = float(source["root_yaw"][10])
    for current in range(PREDICTOR_EVAL_FIRST_GENERATED_FRAME, last):
        features = build_predictor_step_features_np(
            motion_world,
            source["tracker_pos_world"][current - 11 : current + 1],
            source["tracker_rot_world_6d"][current - 11 : current + 1],
            float(source["root_pos_world"][current, 1]),
        )
        motion = normalizer.normalize_pose(features.motion_context)
        sparse = normalizer.normalize_predictor_sparse(
            features.core_tracker_context
        )
        with torch.no_grad():
            normalized = predictor(
                torch.as_tensor(motion, device=device)[None],
                torch.as_tensor(sparse, device=device)[None],
            )[0]
        horizon = np.asarray(
            normalizer.inverse_pose(normalized).cpu(), dtype=np.float32
        )
        current_world = pose_head_to_world_rotations_np(
            horizon[0], features.current_head_yaw_world
        )
        motion_world = np.concatenate(
            [motion_world[1:], current_world[None]], axis=0
        )
        resolved = decode_and_resolve_pose(
            horizon[0],
            _build_current_tracker_raw(
                source, current, features.current_head_yaw_world
            ),
            features.current_head_yaw_world,
            source["tracker_pos_world"][current, HEAD_TRACKER_INDEX],
            float(source["root_pos_world"][current, 1]),
            source["joint_offsets_parent"],
            source["joint_rest_local_rotations_6d"],
            previous_root_yaw,
        )
        previous_root_yaw = resolved.root_yaw_world
        if current >= REALTIME_POSE_EVAL_METRICS_START_FRAME:
            predicted_rotations.append(resolved.joint_rotations_world)
            predicted_positions.append(resolved.joints_world)
            predicted_root_yaw.append(resolved.root_yaw_world)
    return {
        "rotations": np.stack(predicted_rotations).astype(np.float32),
        "positions": np.stack(predicted_positions).astype(np.float32),
        "root_yaw": np.asarray(predicted_root_yaw, dtype=np.float32),
    }


def run_dit_sequence(
    *,
    source: dict[str, np.ndarray],
    world_rotations: np.ndarray,
    predictor,
    dit,
    diffusion,
    device: torch.device,
    normalizer: RealtimePoseNormalizer,
    tracker_available: np.ndarray,
    args,
    last: int,
) -> dict[str, np.ndarray]:
    """用固定噪声序列闭环生成一种 Tracker 配置下的 DiT 结果。"""

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
        ik_gap_low=args.ik_gap_low,
        ik_gap_high=args.ik_gap_high,
        ik_direction_support=args.ik_direction_support,
        ik_untracked_strength=args.ik_untracked_strength,
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
    predicted_rotations: list[np.ndarray] = []
    predicted_positions: list[np.ndarray] = []
    predicted_root_yaw: list[float] = []
    predicted_ik_confidence: list[np.ndarray] = []
    noise_generator = create_eval_noise_generator(args.seed, device)
    for current in range(PREDICTOR_EVAL_FIRST_GENERATED_FRAME, last):
        noise = torch.randn(
            (1, REALTIME_POSE_TARGET_DIM),
            generator=noise_generator,
            device=device,
        )
        result = runtime.step(
            source["tracker_pos_world"][current],
            source["tracker_rot_world_6d"][current],
            tracker_available,
            float(source["root_pos_world"][current, 1]),
            noise=noise,
        )
        if current >= REALTIME_POSE_EVAL_METRICS_START_FRAME:
            predicted_rotations.append(
                result.resolved_pose.joint_rotations_world
            )
            predicted_positions.append(result.resolved_pose.joints_world)
            predicted_root_yaw.append(result.resolved_pose.root_yaw_world)
            # 保留 Runtime 实际用于当前 Tracker 配置的逐关节观测置信度，
            # 让静态论文图能够直接展示模型条件，而不是另外构造启发式热力图。
            predicted_ik_confidence.append(result.ik_confidence)
    return {
        "rotations": np.stack(predicted_rotations).astype(np.float32),
        "positions": np.stack(predicted_positions).astype(np.float32),
        "root_yaw": np.asarray(predicted_root_yaw, dtype=np.float32),
        "ik_confidence": np.stack(predicted_ik_confidence).astype(np.float32),
    }


def sequence_output_stem(source_path: Path) -> str:
    """保留数据集与动作名，避免不同数据集的同名序列互相覆盖。"""

    parts = source_path.with_suffix("").parts[-3:]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", "_".join(parts)).strip("_")


def visualize_source(
    *,
    source_path: Path,
    output_dir: Path,
    predictor,
    dit,
    diffusion,
    dit_weight_source: str,
    device: torch.device,
    normalizer: RealtimePoseNormalizer,
    args,
) -> dict[str, Path]:
    source = load_realtime_source(source_path)
    world_rotations = compute_source_joint_rotations_world(source)
    last = evaluation_last_frame_exclusive(
        len(world_rotations), int(args.max_frames)
    )
    if last <= REALTIME_POSE_EVAL_METRICS_START_FRAME:
        raise ValueError(f"序列没有正式可视化帧：{source_path}")

    print(f"[visualize] predictor-only: {source_path.name}", flush=True)
    predictor_result = run_predictor_sequence(
        source=source,
        world_rotations=world_rotations,
        predictor=predictor,
        device=device,
        normalizer=normalizer,
        last=last,
    )
    print(f"[visualize] DiT core-only: {source_path.name}", flush=True)
    core_result = run_dit_sequence(
        source=source,
        world_rotations=world_rotations,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        tracker_available=np.asarray(
            STATIC_OPTIONAL_TRACKER_MASKS[0], dtype=bool
        ),
        args=args,
        last=last,
    )
    print(f"[visualize] DiT all-six: {source_path.name}", flush=True)
    all_six_result = run_dit_sequence(
        source=source,
        world_rotations=world_rotations,
        predictor=predictor,
        dit=dit,
        diffusion=diffusion,
        device=device,
        normalizer=normalizer,
        tracker_available=np.asarray(
            STATIC_OPTIONAL_TRACKER_MASKS[-1], dtype=bool
        ),
        args=args,
        last=last,
    )

    start = REALTIME_POSE_EVAL_METRICS_START_FRAME
    reference_rotations = world_rotations[start:last].astype(np.float32)
    reference_positions = source["joints_world"][start:last].astype(np.float32)
    tracker_positions = source["tracker_pos_world"][start:last].astype(np.float32)
    methods = {
        "predictor_only": predictor_result,
        "core_only": core_result,
        "all_six": all_six_result,
    }
    metrics = {
        name: compute_rpm_p2_mc_metrics(
            predicted_global_rotations=result["rotations"],
            target_global_rotations=reference_rotations,
            predicted_joint_positions=result["positions"],
            target_joint_positions=reference_positions,
            fps=REALTIME_POSE_FPS,
        )
        for name, result in methods.items()
    }

    stem = sequence_output_stem(source_path)
    npz_path = output_dir / f"{stem}_four_way.npz"
    report_path = output_dir / f"{stem}_four_way.json"
    video_path = output_dir / f"{stem}_four_way.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        reference_joints_world=reference_positions,
        reference_rotations_world=reference_rotations,
        predictor_only_joints_world=predictor_result["positions"],
        predictor_only_rotations_world=predictor_result["rotations"],
        predictor_only_root_yaw=predictor_result["root_yaw"],
        core_only_joints_world=core_result["positions"],
        core_only_rotations_world=core_result["rotations"],
        core_only_root_yaw=core_result["root_yaw"],
        all_six_joints_world=all_six_result["positions"],
        all_six_rotations_world=all_six_result["rotations"],
        all_six_root_yaw=all_six_result["root_yaw"],
        tracker_pos_world=tracker_positions,
    )
    report = {
        "source_path": str(source_path),
        "frame_start": start,
        "frame_end_exclusive": last,
        "frames": last - start,
        "fps": REALTIME_POSE_FPS,
        "sampling_steps": int(diffusion.num_timesteps),
        "sampling_noise_seed": int(args.seed),
        "dit_model_path": str(Path(args.dit_model_path).resolve()),
        "dit_weight_source": dit_weight_source,
        "metrics": metrics,
        "npz_path": str(npz_path),
        "video_path": str(video_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[visualize] render: {video_path.name}", flush=True)
    render_realtime_pose_four_way_comparison(
        output_path=video_path,
        reference_joints=reference_positions,
        predictor_joints=predictor_result["positions"],
        core_only_joints=core_result["positions"],
        all_six_joints=all_six_result["positions"],
        tracker_pos_world=tracker_positions,
        fps=int(round(float(args.visualize_fps))),
        stride=int(args.stride),
        camera_mode=str(args.camera_mode),
        frame_offset=start,
    )
    print(f"[visualize] wrote: {video_path}", flush=True)
    return {
        "video_path": video_path,
        "npz_path": npz_path,
        "report_path": report_path,
    }


def main(argv: list[str] | None = None) -> list[dict[str, Path]]:
    parser = build_arg_parser()
    args = parse_and_load_from_model(parser, argv)
    if not str(args.output_dir).strip():
        parser.error("必须指定 --output_dir。")
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
    output_dir = Path(args.output_dir).resolve()
    results = []
    for source_value in args.source_npz:
        source_path = Path(source_value).resolve()
        print(f"[visualize] start: {source_path}", flush=True)
        results.append(
            visualize_source(
                source_path=source_path,
                output_dir=output_dir,
                predictor=predictor,
                dit=dit,
                diffusion=diffusion,
                dit_weight_source=dit_weight_source,
                device=device,
                normalizer=normalizer,
                args=args,
            )
        )
    return results


if __name__ == "__main__":
    main()
