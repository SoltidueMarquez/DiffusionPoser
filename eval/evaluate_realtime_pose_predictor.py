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
from data_loaders.realtime_pose_geometry import (
    assemble_current_tracker_features_np,
    build_pose_target_np,
    build_tracker_measurements_np,
)
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from data_loaders.realtime_pose_predictor_features import (
    build_predictor_step_features_np,
    pose_head_to_world_rotations_np,
)
from data_loaders.sensor_masking import (
    CORE_THREE_AVAILABLE,
    HEAD_TRACKER_INDEX,
    PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH,
    REALTIME_POSE_EVAL_METRICS_START_FRAME,
    REALTIME_POSE_FPS,
)
from eval.realtime_pose_metrics import (
    aggregate_rpm_p2_mc_metrics,
    compute_rpm_p2_mc_metrics,
)
from sample.realtime_pose_runtime import decode_and_resolve_pose
from utils.fixseed import fixseed
from utils.model_util import load_realtime_pose_predictor
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import str2bool


PREDICTOR_EVAL_FIRST_GENERATED_FRAME = PREDICTOR_CORE_TRACKER_CONTEXT_LENGTH


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立评估 Predictor 的闭环长序列与 0～10 帧预测 horizon。"
    )
    base = parser.add_argument_group("base")
    base.add_argument("--cuda", default=True, type=str2bool)
    base.add_argument("--device", default=0, type=int)
    base.add_argument("--seed", default=10, type=int)
    group = parser.add_argument_group("predictor evaluation")
    group.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR)
    group.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR)
    group.add_argument("--split", default="test")
    group.add_argument("--normalizer_dir", required=True)
    group.add_argument("--predictor_model_path", required=True)
    group.add_argument("--normalize_input", default=True, type=str2bool)
    group.add_argument("--limit", default=0, type=int)
    group.add_argument(
        "--max_frames",
        default=0,
        type=int,
        help="P2 预热完成后最多计分的帧数；0 表示直到序列结束。",
    )
    group.add_argument("--output_json", required=True)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    fixseed(args.seed)
    device = torch.device(
        f"cuda:{args.device}" if args.cuda and torch.cuda.is_available() else "cpu"
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
    report = evaluate_predictor_entries(
        entries=entries,
        predictor=predictor,
        device=device,
        normalizer=normalizer,
        max_frames=args.max_frames,
    )
    payload = {
        "predictor_model_path": str(Path(args.predictor_model_path).resolve()),
        "split": str(args.split),
        "source_fps": float(REALTIME_POSE_FPS),
        "initial_context_frames": PREDICTOR_EVAL_FIRST_GENERATED_FRAME,
        "metrics_start_frame": REALTIME_POSE_EVAL_METRICS_START_FRAME,
        "history_policy": "closed-loop starts at frame 11; official metrics start at frame 30",
        "tracker_policy": "core trackers through current frame; no future tracker",
        **report,
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[predictor-eval] wrote {output}")
    return payload


def evaluate_predictor_entries(
    *,
    entries: list[dict],
    predictor: torch.nn.Module,
    device: torch.device,
    normalizer: RealtimePoseNormalizer,
    max_frames: int = 0,
) -> dict:
    """以 GT 烧入 10 帧后闭环滚动 Predictor，返回独立测试指标。"""

    if not entries:
        raise RuntimeError("Predictor 评估集为空。")
    totals = {
        "horizon_rotation_deg": np.zeros(11, dtype=np.float64),
        "horizon_count": np.zeros(11, dtype=np.int64),
        "first_30_rotation_deg": 0.0,
        "first_30_count": 0,
        "after_30_rotation_deg": 0.0,
        "after_30_count": 0,
    }
    generated_frame_count = 0
    frame_count = 0
    sequence_count = 0
    rpm_sequence_metrics: list[dict[str, float | None]] = []
    for entry in entries:
        source = load_realtime_source(resolve_source_entry_path(entry))
        world_rotations = compute_source_joint_rotations_world(source)
        last = evaluation_last_frame_exclusive(len(world_rotations), max_frames)
        if last <= REALTIME_POSE_EVAL_METRICS_START_FRAME:
            continue
        sequence_count += 1
        scored_predicted_rotations: list[np.ndarray] = []
        scored_target_rotations: list[np.ndarray] = []
        scored_predicted_positions: list[np.ndarray] = []
        scored_target_positions: list[np.ndarray] = []
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
            gt_horizon_length = min(11, len(world_rotations) - current)
            gt_horizon = build_pose_target_np(
                world_rotations[current : current + gt_horizon_length],
                features.current_head_yaw_world,
            )
            current_error = pose_rotation_error_deg(horizon[0], gt_horizon[0])

            current_tracker_raw = _build_current_tracker_raw(
                source, current, features.current_head_yaw_world
            )
            resolved = decode_and_resolve_pose(
                horizon[0],
                current_tracker_raw,
                features.current_head_yaw_world,
                source["tracker_pos_world"][current, HEAD_TRACKER_INDEX],
                float(source["root_pos_world"][current, 1]),
                source["joint_offsets_parent"],
                source["joint_rest_local_rotations_6d"],
                previous_root_yaw,
            )
            previous_root_yaw = resolved.root_yaw_world
            if current - PREDICTOR_EVAL_FIRST_GENERATED_FRAME < 30:
                totals["first_30_rotation_deg"] += current_error
                totals["first_30_count"] += 1
            else:
                totals["after_30_rotation_deg"] += current_error
                totals["after_30_count"] += 1
            generated_frame_count += 1

            # P2 先闭环运行满第一秒，再把当前帧及其 horizon 纳入正式指标。
            if current < REALTIME_POSE_EVAL_METRICS_START_FRAME:
                continue
            scored_predicted_rotations.append(resolved.joint_rotations_world)
            scored_target_rotations.append(world_rotations[current])
            scored_predicted_positions.append(resolved.joints_world)
            scored_target_positions.append(source["joints_world"][current])
            for offset in range(gt_horizon_length):
                totals["horizon_rotation_deg"][offset] += pose_rotation_error_deg(
                    horizon[offset], gt_horizon[offset]
                )
                totals["horizon_count"][offset] += 1
            frame_count += 1
        if scored_predicted_rotations:
            rpm_sequence_metrics.append(
                compute_rpm_p2_mc_metrics(
                    predicted_global_rotations=np.stack(
                        scored_predicted_rotations, axis=0
                    ),
                    target_global_rotations=np.stack(scored_target_rotations, axis=0),
                    predicted_joint_positions=np.stack(
                        scored_predicted_positions, axis=0
                    ),
                    target_joint_positions=np.stack(scored_target_positions, axis=0),
                    fps=REALTIME_POSE_FPS,
                )
            )
    if frame_count <= 0:
        raise RuntimeError("Predictor 评估没有可计分帧。")
    horizon = np.divide(
        totals["horizon_rotation_deg"],
        totals["horizon_count"],
        out=np.zeros(11, dtype=np.float64),
        where=totals["horizon_count"] > 0,
    )
    return {
        "evaluated_sequences": sequence_count,
        "generated_frames": generated_frame_count,
        "evaluated_frames": frame_count,
        "rpm_p2_mc": aggregate_rpm_p2_mc_metrics(rpm_sequence_metrics),
        "horizon_rotation_deg": horizon.tolist(),
        "horizon_counts": totals["horizon_count"].tolist(),
        "rolling": {
            "first_30_generated_rotation_deg": totals["first_30_rotation_deg"]
            / max(totals["first_30_count"], 1),
            "first_30_generated_count": totals["first_30_count"],
            "after_30_generated_rotation_deg": totals["after_30_rotation_deg"]
            / max(totals["after_30_count"], 1),
            "after_30_generated_count": totals["after_30_count"],
        },
    }


def evaluation_last_frame_exclusive(frame_count: int, max_frames: int) -> int:
    """返回评估循环上界；`max_frames` 只裁剪 P2 正式计分区间。"""

    last = int(frame_count)
    if int(max_frames) > 0:
        last = min(
            last,
            REALTIME_POSE_EVAL_METRICS_START_FRAME + int(max_frames),
        )
    return last


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
        measurements, np.asarray(CORE_THREE_AVAILABLE, dtype=bool)
    )


def pose_rotation_error_deg(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = rotation_6d_to_matrix_np(np.asarray(prediction).reshape(24, 6))
    gt = rotation_6d_to_matrix_np(np.asarray(target).reshape(24, 6))
    relative = np.swapaxes(pred, -1, -2) @ gt
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(cosine)).mean())


if __name__ == "__main__":
    main()
