from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data_loaders.generate_realtime_pose_tasks import (
    load_realtime_source,
    read_source_entries,
    read_split_keys,
    filter_entries_by_split,
)
from data_loaders.realtime_pose_dataset import encode_realtime_pose_features
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAMES,
    TRACKER_COUNT,
    get_schema_spec,
)
from utils.normalizer import RealtimePoseNormalizer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute realtime_pose mean/std normalizer from converted source data.")
    group = parser.add_argument_group("paths")
    group.add_argument("--source_dir", default="dataset/AMASS_realtime_pose_60hz", type=str)
    group.add_argument("--output_dir", default="dataset/meta_AMASS_realtime_pose_60hz", type=str)
    group.add_argument("--split_dir", default="data_loaders/splits", type=str)

    group = parser.add_argument_group("statistics")
    group.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES, type=str)
    group.add_argument("--split", default="train", type=str)
    group.add_argument("--eps", default=1e-8, type=float)
    group.add_argument("--overwrite", action="store_true")
    return parser


def compute_realtime_pose_normalizer(args: argparse.Namespace) -> dict[str, int | float | str]:
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_dir = Path(args.split_dir).resolve() if args.split_dir else None
    schema = get_schema_spec(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME))
    if not source_dir.exists():
        raise FileNotFoundError(f"{schema.name} 源数据目录不存在：{source_dir}")
    ensure_output_dir(output_dir=output_dir, overwrite=bool(args.overwrite))

    source_entries = read_source_entries(source_dir)
    split_keys = read_split_keys(split_dir=split_dir, split=args.split)
    split_entries = filter_entries_by_split(source_entries, split_keys)
    if not split_entries:
        raise RuntimeError(f"split={args.split} 没有匹配到 {schema.name} 源文件。")

    running_sum: np.ndarray | None = None
    running_sumsq: np.ndarray | None = None
    running_count = 0

    for entry in tqdm(split_entries, desc=f"统计 split={args.split} realtime normalizer", unit="file"):
        source = load_realtime_source(Path(entry["source_path"]), schema_name=schema.name)
        sensor_valid = np.ones((source["body_pose_parent_6d"].shape[0], TRACKER_COUNT), dtype=bool)
        features = encode_realtime_pose_features({**source, "sensor_valid": sensor_valid}, schema_name=schema.name)
        seq_sum = features.sum(axis=0, dtype=np.float64)
        seq_sumsq = np.square(features).sum(axis=0, dtype=np.float64)
        running_sum = seq_sum if running_sum is None else running_sum + seq_sum
        running_sumsq = seq_sumsq if running_sumsq is None else running_sumsq + seq_sumsq
        running_count += int(features.shape[0])

    if running_sum is None or running_sumsq is None or running_count <= 0:
        raise RuntimeError("没有成功统计到有效帧，无法生成 realtime_pose_v1 normalizer。")

    mean, std = finalize_mean_std(running_sum, running_sumsq, running_count, eps=float(args.eps))
    normalizer = RealtimePoseNormalizer(base_dir=output_dir, eps=float(args.eps), disable=True, schema_name=schema.name)
    normalizer.save(mean=mean, std=std)

    meta = {
        "schema_name": schema.name,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "split_dir": str(split_dir) if split_dir is not None else "",
        "split": args.split,
        "matched_source_files": len(split_entries),
        "total_frames": running_count,
        "feature_dim": schema.feature_dim,
        "eps": float(args.eps),
        "std_definition": "population",
    }
    save_meta(output_dir=output_dir, meta=meta)
    return meta


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    mean_path = output_dir / "mean.pt"
    std_path = output_dir / "std.pt"
    meta_path = output_dir / "normalizer_meta.json"
    existing = [path for path in (mean_path, std_path, meta_path) if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"normalizer 输出已存在：{existing_text}。如需重算，请添加 --overwrite。")
    output_dir.mkdir(parents=True, exist_ok=True)


def finalize_mean_std(
    running_sum: np.ndarray,
    running_sumsq: np.ndarray,
    running_count: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean = running_sum / float(running_count)
    second_moment = running_sumsq / float(running_count)
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.clip(std, a_min=eps, a_max=None)
    return mean.astype(np.float32), std.astype(np.float32)


def save_meta(output_dir: Path, meta: dict[str, int | float | str]) -> None:
    meta_path = output_dir / "normalizer_meta.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False, sort_keys=True)


def main(argv: list[str] | None = None) -> dict[str, int | float | str]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    meta = compute_realtime_pose_normalizer(args)
    print("[compute_realtime_pose_normalizer] 统计完成。")
    print(f"- 匹配源文件数：{meta['matched_source_files']}")
    print(f"- 累计有效帧数：{meta['total_frames']}")
    print(f"- 输出目录：{meta['output_dir']}")
    return meta


if __name__ == "__main__":
    main()
