from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_task_store import load_shard_stats
from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    PREDICTOR_SPARSE_DIM,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
)
from utils.normalizer import RealtimePoseNormalizer, build_pose_scale


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="合并 Predictor + 单帧 DiT task shard 的 normalizer 统计。"
    )
    parser.add_argument("--task_dir", default="dataset/realtime_pose_tasks")
    parser.add_argument("--output_dir", default="dataset/realtime_pose_normalizer")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def compute_realtime_pose_normalizer(
    args: argparse.Namespace,
) -> dict[str, object]:
    task_dir = Path(args.task_dir).resolve()
    shards_root = task_dir / str(args.split) / "shards"
    shards = sorted(path for path in shards_root.glob("shard_*") if path.is_dir())
    if not shards:
        raise RuntimeError(f"{shards_root} 没有 shard_*/ 目录。")

    pose_sum = np.zeros(REALTIME_POSE_TARGET_DIM, dtype=np.float64)
    pose_sumsq = np.zeros_like(pose_sum)
    pose_count = 0
    tracker_sum = np.zeros(
        (TRACKER_COUNT, TRACKER_CONTINUOUS_DIM), dtype=np.float64
    )
    tracker_sumsq = np.zeros_like(tracker_sum)
    tracker_count = np.zeros((TRACKER_COUNT, 1), dtype=np.float64)
    predictor_sparse_sum = np.zeros(PREDICTOR_SPARSE_DIM, dtype=np.float64)
    predictor_sparse_sumsq = np.zeros_like(predictor_sparse_sum)
    predictor_sparse_count = 0
    for shard_dir in shards:
        stats = load_shard_stats(shard_dir)
        pose_sum += stats["pose_sum"]
        pose_sumsq += stats["pose_sumsq"]
        pose_count += int(stats["pose_count"])
        tracker_sum += stats["tracker_sum"]
        tracker_sumsq += stats["tracker_sumsq"]
        tracker_count += stats["tracker_count"]
        predictor_sparse_sum += stats["predictor_sparse_sum"]
        predictor_sparse_sumsq += stats["predictor_sparse_sumsq"]
        predictor_sparse_count += int(stats["predictor_sparse_count"])

    pose_mean, pose_std = finalize_mean_std(
        pose_sum, pose_sumsq, pose_count, float(args.eps)
    )
    pose_scale = build_pose_scale(pose_std, float(args.eps))
    tracker_mean, tracker_std = finalize_mean_std(
        tracker_sum, tracker_sumsq, tracker_count, float(args.eps)
    )
    predictor_sparse_mean, predictor_sparse_std = finalize_mean_std(
        predictor_sparse_sum,
        predictor_sparse_sumsq,
        predictor_sparse_count,
        float(args.eps),
    )

    output_dir = Path(args.output_dir).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    if output_dir.parent == output_dir or output_dir == repository_root:
        raise ValueError(
            f"拒绝将 normalizer output_dir 指向磁盘根目录或仓库根目录：{output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"normalizer 输出目录已存在: {output_dir}；"
            "请指定新目录或使用 --overwrite。"
        )
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        normalizer = RealtimePoseNormalizer(
            temporary_dir, eps=float(args.eps), disable=True
        )
        normalizer.save(
            pose_mean=pose_mean,
            pose_scale=pose_scale,
            tracker_mean=tracker_mean,
            tracker_std=tracker_std,
            predictor_sparse_mean=predictor_sparse_mean,
            predictor_sparse_std=predictor_sparse_std,
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary_dir.replace(output_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return {
        "output_dir": str(output_dir),
        "split": str(args.split),
        "pose_observation_count": int(pose_count),
        "tracker_observation_counts": tracker_count[:, 0].astype(int).tolist(),
        "predictor_sparse_observation_count": int(predictor_sparse_count),
    }


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
