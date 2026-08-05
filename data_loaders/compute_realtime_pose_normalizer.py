from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_task_store import load_shard_stats, read_store_metadata
from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM, TRACKER_CONTINUOUS_DIM, TRACKER_COUNT
from utils.normalizer import RealtimePoseNormalizer, build_pose_scale
from utils.run_dirs import resolve_latest_or_self, timestamped_child_dir, write_latest_pointer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 shard 合并 realtime pose normalizer 统计。")
    parser.add_argument(
        "--task_dir",
        default="dataset/AMASS_realtime_pose_144d_pelvis_residual_root_y0_stationary5_60hz_tasks",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset/meta_AMASS_realtime_pose_144d_pelvis_residual_root_y0_stationary5_60hz",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--run_name", default="auto")
    return parser


def compute_realtime_pose_normalizer(args: argparse.Namespace) -> dict[str, object]:
    task_dir = resolve_latest_or_self(args.task_dir, kind="tasks")
    split_dir = task_dir / str(args.split)
    metadata = read_store_metadata(split_dir)
    shards = sorted(metadata["shards"], key=lambda value: int(value["index"]))
    if not shards:
        raise RuntimeError(f"{split_dir} 没有 shard。")

    pose_sum = np.zeros(REALTIME_POSE_TARGET_DIM, dtype=np.float64)
    pose_sumsq = np.zeros_like(pose_sum)
    pose_count = 0
    tracker_sum = np.zeros((TRACKER_COUNT, TRACKER_CONTINUOUS_DIM), dtype=np.float64)
    tracker_sumsq = np.zeros_like(tracker_sum)
    tracker_count = np.zeros((TRACKER_COUNT, 1), dtype=np.float64)
    head_path_xz_sum = np.zeros(2, dtype=np.float64)
    head_path_xz_sumsq = np.zeros(2, dtype=np.float64)
    head_path_xz_count = 0
    head_height_sum = np.float64(0.0)
    head_height_sumsq = np.float64(0.0)
    head_height_count = 0
    for shard in shards:
        stats = load_shard_stats(split_dir, shard)
        pose_sum += stats["pose_sum"]
        pose_sumsq += stats["pose_sumsq"]
        pose_count += int(stats["pose_count"])
        tracker_sum += stats["tracker_sum"]
        tracker_sumsq += stats["tracker_sumsq"]
        tracker_count += stats["tracker_count"]
        head_path_xz_sum += stats["head_path_xz_sum"]
        head_path_xz_sumsq += stats["head_path_xz_sumsq"]
        head_path_xz_count += int(stats["head_path_xz_count"])
        head_height_sum += np.float64(stats["head_height_sum"])
        head_height_sumsq += np.float64(stats["head_height_sumsq"])
        head_height_count += int(stats["head_height_count"])

    pose_mean, pose_std = finalize_mean_std(pose_sum, pose_sumsq, pose_count, float(args.eps))
    pose_scale = build_pose_scale(pose_std, float(args.eps))
    tracker_mean, tracker_std = finalize_mean_std(
        tracker_sum, tracker_sumsq, tracker_count, float(args.eps)
    )
    head_path_xz_mean, head_path_xz_std = finalize_mean_std(
        head_path_xz_sum, head_path_xz_sumsq, head_path_xz_count, float(args.eps)
    )
    head_height_mean, head_height_std = finalize_mean_std(
        head_height_sum, head_height_sumsq, head_height_count, float(args.eps)
    )
    output_root = Path(args.output_dir).resolve()
    label = "rtp_144d_normalizer" if str(args.run_name).lower() in {"", "auto"} else str(args.run_name)
    output_dir = timestamped_child_dir(output_root, label)
    normalizer = RealtimePoseNormalizer(output_dir, eps=float(args.eps), disable=True)
    normalizer_metadata = {
        "generation_plan_hash": str(metadata["generation_plan_hash"]),
        "task_dir": str(task_dir),
        "split": str(args.split),
        "sample_count": int(metadata["sample_count"]),
        "pose_observation_count": int(pose_count),
        "tracker_observation_counts": tracker_count[:, 0].astype(int).tolist(),
        "std_definition": "population",
        "pose_scale_definition": "stabilized_pose_std_plus_eps",
    }
    normalizer.save(
        pose_mean,
        pose_scale,
        tracker_mean,
        tracker_std,
        head_path_xz_mean,
        head_path_xz_std,
        head_height_mean,
        head_height_std,
        metadata=normalizer_metadata,
    )
    result = {"output_dir": str(output_dir), **normalizer_metadata}
    write_latest_pointer(output_root, "normalizer", output_dir, result)
    return result


def finalize_mean_std(
    running_sum: np.ndarray,
    running_sumsq: np.ndarray,
    running_count: int | np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = np.asarray(running_count, dtype=np.float64)
    safe_count = np.maximum(count, 1.0)
    mean = running_sum / safe_count
    variance = np.maximum(running_sumsq / safe_count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(variance), float(eps))
    if count.shape:
        empty = count <= 0
        mean = np.where(empty, 0.0, mean)
        std = np.where(empty, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = build_argument_parser().parse_args(argv)
    result = compute_realtime_pose_normalizer(args)
    print(f"[normalizer] 完成：{result['output_dir']}")
    return result


if __name__ == "__main__":
    main()
