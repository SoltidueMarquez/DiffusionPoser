from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data_loaders.realtime_pose_dataset import (
    find_manifest_path,
    load_materialized_task_npz,
    read_task_manifest,
)
from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
    TRACKER_MEASURED_VALID_OFFSET,
)
from utils.normalizer import RealtimePoseNormalizer
from utils.run_dirs import resolve_latest_or_self, timestamped_child_dir, write_latest_pointer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="统计 140D pose 与 [6,9] Tracker normalizer。")
    parser.add_argument(
        "--task_dir",
        default="dataset/AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz_tasks",
    )
    parser.add_argument(
        "--output_dir",
        default="dataset/meta_AMASS_realtime_pose_body_fbx_local_root_y0_stationary5_60hz",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--run_name", default="auto")
    # 保留 CLI 参数名，实际输出始终使用新时间戳目录，不覆盖旧统计。
    parser.add_argument("--overwrite", action="store_true")
    return parser


def compute_realtime_pose_normalizer(args: argparse.Namespace) -> dict[str, object]:
    task_dir = resolve_latest_or_self(args.task_dir, kind="tasks")
    manifest_path = find_manifest_path(task_dir, args.split)
    entries = read_task_manifest(manifest_path)
    if not entries:
        raise RuntimeError(f"{manifest_path} 没有 task。")

    pose_sum = np.zeros(REALTIME_POSE_TARGET_DIM, dtype=np.float64)
    pose_sumsq = np.zeros(REALTIME_POSE_TARGET_DIM, dtype=np.float64)
    pose_count = 0
    tracker_sum = np.zeros((TRACKER_COUNT, TRACKER_CONTINUOUS_DIM), dtype=np.float64)
    tracker_sumsq = np.zeros_like(tracker_sum)
    tracker_count = np.zeros((TRACKER_COUNT, 1), dtype=np.float64)

    for entry in tqdm(entries, desc="统计 140D normalizer", unit="task"):
        task = load_materialized_task_npz(manifest_path.parent, entry["task_path"])
        pose = np.concatenate(
            [task["pose_history"], task["current_target"][None]],
            axis=0,
        ).astype(np.float64)
        pose_sum += pose.sum(axis=0)
        pose_sumsq += np.square(pose).sum(axis=0)
        pose_count += pose.shape[0]

        tracker = task["tracker_window"].astype(np.float64)
        valid = tracker[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5
        continuous = tracker[..., :TRACKER_CONTINUOUS_DIM]
        tracker_sum += (continuous * valid[..., None]).sum(axis=0)
        tracker_sumsq += (np.square(continuous) * valid[..., None]).sum(axis=0)
        tracker_count += valid.sum(axis=0)[:, None]

    pose_mean, pose_std = finalize_mean_std(pose_sum, pose_sumsq, pose_count, float(args.eps))
    tracker_mean, tracker_std = finalize_mean_std(
        tracker_sum,
        tracker_sumsq,
        tracker_count,
        float(args.eps),
    )

    output_root = Path(args.output_dir).resolve()
    label = "rtp_140d_normalizer" if str(args.run_name).lower() in {"", "auto"} else str(args.run_name)
    output_dir = timestamped_child_dir(output_root, label)
    normalizer = RealtimePoseNormalizer(output_dir, eps=float(args.eps), disable=True)
    normalizer.save(
        pose_mean,
        pose_std,
        tracker_mean,
        tracker_std,
        metadata={
            "task_dir": str(task_dir),
            "split": str(args.split),
            "matched_tasks": len(entries),
            "pose_observation_count": int(pose_count),
            "tracker_valid_observation_counts": tracker_count[:, 0].astype(int).tolist(),
            "std_definition": "population",
        },
    )
    meta = {
        "output_dir": str(output_dir),
        "task_dir": str(task_dir),
        "matched_tasks": len(entries),
        "pose_observation_count": int(pose_count),
        "tracker_valid_observation_counts": tracker_count[:, 0].astype(int).tolist(),
    }
    write_latest_pointer(output_root, "normalizer", output_dir, meta)
    return meta


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
