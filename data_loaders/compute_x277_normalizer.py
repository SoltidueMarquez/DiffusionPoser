from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data_loaders.generate_x277_missing_tasks import (
    filter_entries_by_split,
    read_source_entries,
    read_split_keys,
)
from data_loaders.sensor_masking import X277_FEATURE_DIM
from utils.normalizer import X277Normalizer


# region 参数解析
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute X277 mean/std normalizer from converted AMASS data.")

    group = parser.add_argument_group("paths")
    group.add_argument("--source_dir", default="dataset/AMASS_x277_60hz", type=str, help="X277 源数据目录。")
    group.add_argument(
        "--output_dir",
        default="dataset/meta_AMASS_x277_60hz",
        type=str,
        help="保存 mean.pt、std.pt 和 normalizer_meta.json 的目录。",
    )
    group.add_argument(
        "--split_dir",
        default="data_loaders/splits",
        type=str,
        help="StableMotion 风格 split 目录；默认读取其中的 train.txt。",
    )

    group = parser.add_argument_group("statistics")
    group.add_argument("--split", default="train", type=str, help="用于统计 normalizer 的 split 名称。")
    group.add_argument("--eps", default=1e-8, type=float, help="std 最小值，避免除零。")
    group.add_argument("--overwrite", action="store_true", help="允许覆盖已有 mean.pt/std.pt。")
    return parser


# endregion


# region 流式统计
def compute_x277_normalizer(args: argparse.Namespace) -> dict[str, int | float | str]:
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_dir = Path(args.split_dir).resolve() if args.split_dir else None

    if not source_dir.exists():
        raise FileNotFoundError(f"X277 源数据目录不存在：{source_dir}")
    ensure_output_dir(output_dir=output_dir, overwrite=args.overwrite)

    source_entries = read_source_entries(source_dir=source_dir)
    split_keys = read_split_keys(split_dir=split_dir, split=args.split)
    split_entries = filter_entries_by_split(entries=source_entries, split_keys=split_keys)
    if not split_entries:
        raise RuntimeError(f"split={args.split} 没有匹配到任何 X277 源文件，请检查 split 文件和 manifest。")

    running_sum: np.ndarray | None = None
    running_sumsq: np.ndarray | None = None
    running_count = 0

    for entry in tqdm(split_entries, desc=f"统计 split={args.split} X277 normalizer", unit="file"):
        x277 = load_x277_array(Path(entry["source_path"]))
        seq_sum = x277.sum(axis=0, dtype=np.float64)
        seq_sumsq = np.square(x277).sum(axis=0, dtype=np.float64)

        if running_sum is None or running_sumsq is None:
            running_sum = seq_sum
            running_sumsq = seq_sumsq
        else:
            running_sum += seq_sum
            running_sumsq += seq_sumsq
        running_count += int(x277.shape[0])

    if running_sum is None or running_sumsq is None or running_count <= 0:
        raise RuntimeError("没有成功统计到任何有效帧，无法生成 X277 normalizer。")

    mean, std = finalize_mean_std(
        running_sum=running_sum,
        running_sumsq=running_sumsq,
        running_count=running_count,
        eps=float(args.eps),
    )

    normalizer = X277Normalizer(base_dir=output_dir, eps=float(args.eps), disable=True)
    normalizer.save(mean=mean, std=std)

    meta = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "split_dir": str(split_dir) if split_dir is not None else "",
        "split": args.split,
        "matched_source_files": len(split_entries),
        "total_frames": running_count,
        "feature_dim": X277_FEATURE_DIM,
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


def load_x277_array(source_path: Path) -> np.ndarray:
    with np.load(source_path, allow_pickle=False) as data:
        if "x" not in data:
            raise KeyError(f"{source_path} 缺少字段 `x`。")
        x277 = np.asarray(data["x"], dtype=np.float64)

    if x277.ndim != 2 or x277.shape[1] != X277_FEATURE_DIM:
        raise ValueError(f"{source_path} 的 x 应为 [T, {X277_FEATURE_DIM}]，实际为 {x277.shape}")
    if x277.shape[0] <= 0:
        raise ValueError(f"{source_path} 没有有效帧。")
    if not np.isfinite(x277).all():
        raise ValueError(f"{source_path} 包含 NaN 或 Inf，无法参与 normalizer 统计。")
    return x277


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


# endregion


def main(argv: list[str] | None = None) -> dict[str, int | float | str]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    meta = compute_x277_normalizer(args)
    print("[compute_x277_normalizer] 统计完成。")
    print(f"- 匹配源文件数：{meta['matched_source_files']}")
    print(f"- 累计有效帧数：{meta['total_frames']}")
    print(f"- 输出目录：{meta['output_dir']}")
    return meta


if __name__ == "__main__":
    main()
