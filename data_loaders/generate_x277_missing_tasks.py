from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data_loaders.sensor_masking import (
    MODEL_INPUT_DIM,
    X277_FEATURE_DIM,
    create_sensor_missing_task,
)


# region 参数解析
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed X277 sensor-missing tasks for DiffusionPoser.")

    group = parser.add_argument_group("paths")
    group.add_argument("--source_dir", default="dataset/AMASS_x277_60hz", type=str, help="X277 源数据目录。")
    group.add_argument(
        "--output_dir",
        default="dataset/AMASS_x277_60hz_missing_tasks",
        type=str,
        help="生成的缺失任务数据目录。",
    )
    group.add_argument(
        "--split_dir",
        default="",
        type=str,
        help="可选 StableMotion 风格 split 目录；若提供，将读取其中的 train.txt/test.txt 等文件过滤样本。",
    )

    group = parser.add_argument_group("task")
    group.add_argument("--splits", nargs="+", default=["train"], type=str, help="要生成的 split 名称列表。")
    group.add_argument("--seq_len", default=100, type=int, help="每条训练任务的固定帧长。")
    group.add_argument("--samples_per_file", default=4, type=int, help="每个源动作文件生成多少个固定窗口任务。")
    group.add_argument("--num_intervals", default=1, type=int, help="每条任务内连续缺失区间数量。")
    group.add_argument("--min_missing_length", default=10, type=int, help="每个缺失区间的最短帧数。")
    group.add_argument(
        "--max_missing_length",
        default=0,
        type=int,
        help="每个缺失区间的最长帧数；0 表示不超过 seq_len/有效长度。",
    )
    group.add_argument("--min_missing_sensors", default=1, type=int, help="每个区间最少缺失传感器数量。")
    group.add_argument("--max_missing_sensors", default=4, type=int, help="每个区间最多缺失传感器数量。")
    group.add_argument("--min_observed_sensors", default=2, type=int, help="每个区间至少保留的未缺失传感器数量。")
    group.add_argument("--limit", default=0, type=int, help="每个 split 最多使用多少个源样本；0 表示不限制。")

    group = parser.add_argument_group("runtime")
    group.add_argument("--seed", default=10, type=int, help="固定随机种子，保证离线任务可复现。")
    group.add_argument(
        "--compress_tasks",
        action="store_true",
        help="使用 np.savez_compressed 保存任务文件；文件更小但生成速度明显更慢。",
    )
    group.add_argument("--manifest_flush_interval", default=100, type=int, help="每写入多少条 manifest 后 flush 一次。")
    group.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出目录。")
    return parser


# endregion


# region 源数据与 split
def read_source_entries(source_dir: Path) -> list[dict]:
    """优先读取转换 manifest；不存在时退回到递归扫描 `.npz`。"""

    manifest_path = source_dir / "manifest.jsonl"
    if manifest_path.exists():
        return read_manifest_entries(source_dir=source_dir, manifest_path=manifest_path)
    return glob_source_entries(source_dir=source_dir)


def read_manifest_entries(source_dir: Path, manifest_path: Path) -> list[dict]:
    entries = []
    total_lines = count_text_lines(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as file:
        iterator = tqdm(
            enumerate(file, start=1),
            total=total_lines,
            desc="读取 X277 manifest",
            unit="条",
        )
        for line_number, line in iterator:
            if not line.strip():
                continue
            raw_entry = json.loads(line)
            if raw_entry.get("status", "converted") != "converted":
                continue

            source_path = resolve_manifest_output_path(source_dir=source_dir, entry=raw_entry)
            if not source_path.exists():
                raise FileNotFoundError(f"manifest 第 {line_number} 行指向的 X277 文件不存在：{source_path}")

            stablemotion_split_key = raw_entry.get("stablemotion_split_key")
            if not stablemotion_split_key:
                stablemotion_split_key = source_path_to_split_key(source_dir=source_dir, source_path=source_path)

            entries.append(
                {
                    "source_path": str(source_path),
                    "source_relative_path": source_path.relative_to(source_dir).as_posix(),
                    "stablemotion_split_key": normalize_slashes(stablemotion_split_key),
                    "frames": int(raw_entry.get("frames", 0)),
                    "feature_dim": int(raw_entry.get("feature_dim", X277_FEATURE_DIM)),
                    "is_mirrored": bool(raw_entry.get("is_mirrored", "M/" in stablemotion_split_key)),
                }
            )
    return entries


def resolve_manifest_output_path(source_dir: Path, entry: dict) -> Path:
    output_path = entry.get("output_path")
    if output_path:
        path = Path(output_path)
        if path.exists():
            return path

    relative_path = entry.get("source_relative_path") or entry.get("original_source_relative_path")
    if relative_path:
        return source_dir / normalize_slashes(relative_path)
    raise KeyError("manifest entry 缺少 output_path/source_relative_path，无法定位 X277 文件。")


def glob_source_entries(source_dir: Path) -> list[dict]:
    entries = []
    source_paths = sorted(source_dir.rglob("*.npz"))
    for source_path in tqdm(source_paths, desc="扫描 X277 npz", unit="个"):
        if "missing_tasks" in source_path.parts:
            continue
        entries.append(
            {
                "source_path": str(source_path),
                "source_relative_path": source_path.relative_to(source_dir).as_posix(),
                "stablemotion_split_key": source_path_to_split_key(source_dir=source_dir, source_path=source_path),
                "frames": 0,
                "feature_dim": X277_FEATURE_DIM,
                "is_mirrored": source_path.relative_to(source_dir).as_posix().startswith("M/"),
            }
        )
    return entries


def source_path_to_split_key(source_dir: Path, source_path: Path) -> str:
    return source_path.relative_to(source_dir).with_suffix(".npy").as_posix()


def read_split_keys(split_dir: Path | None, split: str) -> set[str] | None:
    if split_dir is None:
        return None
    split_path = split_dir / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"指定了 split_dir，但找不到 split 文件：{split_path}")

    keys = set()
    total_lines = count_text_lines(split_path)
    with split_path.open("r", encoding="utf-8") as file:
        for line in tqdm(file, total=total_lines, desc=f"读取 split={split}", unit="条"):
            key = normalize_split_key(line)
            if key:
                keys.add(key)
    return keys


def filter_entries_by_split(entries: list[dict], split_keys: set[str] | None) -> list[dict]:
    if split_keys is None:
        return entries
    filtered = []
    for entry in tqdm(entries, desc="匹配 split 与 X277 样本", unit="条"):
        if normalize_split_key(entry["stablemotion_split_key"]) in split_keys:
            filtered.append(entry)
    return filtered


def normalize_split_key(raw_key: str) -> str:
    """将 split 行和 manifest key 统一到无扩展名 POSIX 路径，便于兼容 `.npy`/`.npz` 和窗口元信息。"""

    key = raw_key.strip()
    if not key:
        return ""
    key = key.split(",", 1)[0].strip()
    key = normalize_slashes(key)
    if key.endswith(".npy") or key.endswith(".npz"):
        key = key[:-4]
    return key


def normalize_slashes(path: str) -> str:
    return path.replace("\\", "/")


def count_text_lines(path: Path) -> int:
    """给 tqdm 提供总量；只读文本行数，不解析内容。"""

    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


# endregion


# region 任务生成
def generate_missing_tasks(args: argparse.Namespace) -> dict[str, int]:
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    split_dir = Path(args.split_dir).resolve() if args.split_dir else None

    if not source_dir.exists():
        raise FileNotFoundError(f"X277 源数据目录不存在：{source_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在：{output_dir}。如需重建，请添加 --overwrite。")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_entries = read_source_entries(source_dir=source_dir)
    if not source_entries:
        raise RuntimeError(f"未在 {source_dir} 中找到可用 X277 `.npz` 文件。")
    print(f"[generate_x277_missing_tasks] 可用 X277 源样本数：{len(source_entries)}")

    counts = {}
    for split_index, split in enumerate(args.splits):
        split_keys = read_split_keys(split_dir=split_dir, split=split)
        split_entries = filter_entries_by_split(entries=source_entries, split_keys=split_keys)
        if args.limit > 0:
            split_entries = split_entries[: args.limit]
        if not split_entries:
            raise RuntimeError(f"split={split} 没有匹配到任何 X277 样本，请检查 split 文件或源 manifest。")
        total_tasks = len(split_entries) * args.samples_per_file
        print(
            f"[generate_x277_missing_tasks] split={split} 匹配源样本={len(split_entries)}，"
            f"预计生成任务={total_tasks}"
        )

        rng = np.random.default_rng(args.seed + split_index)
        counts[split] = generate_split_tasks(
            entries=split_entries,
            output_dir=output_dir,
            split=split,
            rng=rng,
            args=args,
        )
    return counts


def generate_split_tasks(
    entries: list[dict],
    output_dir: Path,
    split: str,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> int:
    split_dir = output_dir / split
    task_dir = split_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "manifest.jsonl"

    max_missing_length = args.max_missing_length if args.max_missing_length > 0 else None
    written = 0
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        progress = tqdm(
            total=len(entries) * args.samples_per_file,
            desc=f"生成 split={split} 缺失任务",
            unit="条",
        )
        for entry in entries:
            source_path = Path(entry["source_path"])
            source_x277 = load_x277_array(source_path=source_path, fallback_frames=entry.get("frames", 0))
            source_frames, feature_dim = source_x277.shape
            if feature_dim != X277_FEATURE_DIM:
                raise ValueError(f"{source_path} 的 x 特征维应为 277，实际为 {feature_dim}")

            for sample_index in range(args.samples_per_file):
                start_frame, valid_length = sample_window(
                    rng=rng,
                    source_frames=source_frames,
                    seq_len=args.seq_len,
                )
                sensor_missing_labels, inpaint_mask, intervals = create_sensor_missing_task(
                    seq_len=args.seq_len,
                    valid_length=valid_length,
                    rng=rng,
                    num_intervals=args.num_intervals,
                    min_missing_length=args.min_missing_length,
                    max_missing_length=max_missing_length,
                    min_missing_sensors=args.min_missing_sensors,
                    max_missing_sensors=args.max_missing_sensors,
                    min_observed_sensors=args.min_observed_sensors,
                )

                task_id = make_task_id(
                    split=split,
                    stablemotion_split_key=entry["stablemotion_split_key"],
                    sample_index=sample_index,
                )
                task_rel_path = Path("tasks") / f"{task_id}.npz"
                task_path = split_dir / task_rel_path
                x277_clip = create_x277_clip(
                    source_x277=source_x277,
                    start_frame=start_frame,
                    valid_length=valid_length,
                    seq_len=args.seq_len,
                )
                save_task_npz(
                    task_path=task_path,
                    compress=args.compress_tasks,
                    x277=x277_clip,
                    sensor_missing_labels=sensor_missing_labels,
                    inpaint_mask=inpaint_mask,
                    start_frame=np.int64(start_frame),
                    valid_length=np.int64(valid_length),
                    source_frames=np.int64(source_frames),
                    seq_len=np.int64(args.seq_len),
                )

                manifest_entry = {
                    "task_id": task_id,
                    "task_path": task_rel_path.as_posix(),
                    "split": split,
                    "source_path": str(source_path),
                    "source_relative_path": normalize_slashes(entry["source_relative_path"]),
                    "stablemotion_split_key": normalize_slashes(entry["stablemotion_split_key"]),
                    "start_frame": start_frame,
                    "valid_length": valid_length,
                    "source_frames": source_frames,
                    "seq_len": args.seq_len,
                    "feature_dim": MODEL_INPUT_DIM,
                    "task_format": "materialized_x277_v1",
                    "missing_intervals": [interval.to_dict() for interval in intervals],
                    "is_mirrored": bool(entry.get("is_mirrored", False)),
                }
                manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
                if args.manifest_flush_interval > 0 and written % args.manifest_flush_interval == 0:
                    manifest_file.flush()
                progress.update(1)
        progress.close()
    return written


def save_task_npz(task_path: Path, compress: bool, **arrays) -> None:
    """
    保存单条缺失任务。

    默认使用 `np.savez`，因为训练任务文件数量很多，压缩每个小文件会带来明显 CPU 开销。
    如果更在意磁盘占用，可以通过 `--compress_tasks` 切回压缩保存。
    """

    temp_path = task_path.with_name(task_path.name + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    with temp_path.open("wb") as file:
        if compress:
            np.savez_compressed(file, **arrays)
        else:
            np.savez(file, **arrays)
    temp_path.replace(task_path)


def load_x277_array(source_path: Path, fallback_frames: int = 0) -> np.ndarray:
    with np.load(source_path, allow_pickle=False) as data:
        if "x" not in data:
            raise KeyError(f"{source_path} 缺少字段 `x`。")
        x277 = data["x"].astype(np.float32, copy=True)
    if x277.ndim != 2:
        raise ValueError(f"{source_path} 的 x 应为 [T, 277]，实际为 {x277.shape}")
    frames, feature_dim = int(x277.shape[0]), int(x277.shape[1])
    if fallback_frames and fallback_frames != frames:
        raise ValueError(f"{source_path} manifest frames={fallback_frames}，实际 x 帧数={frames}")
    if feature_dim != X277_FEATURE_DIM:
        raise ValueError(f"{source_path} 的 x 特征维应为 {X277_FEATURE_DIM}，实际为 {feature_dim}")
    return x277


def inspect_x277_file(source_path: Path, fallback_frames: int = 0) -> tuple[int, int]:
    x277 = load_x277_array(source_path=source_path, fallback_frames=fallback_frames)
    return int(x277.shape[0]), int(x277.shape[1])


def create_x277_clip(source_x277: np.ndarray, start_frame: int, valid_length: int, seq_len: int) -> np.ndarray:
    """把源 X277 序列物化成固定长度训练窗口，padding 帧保持 0。"""

    end_frame = start_frame + valid_length
    if start_frame < 0 or valid_length <= 0 or end_frame > source_x277.shape[0]:
        raise ValueError(
            f"X277 clip 越界：start={start_frame}, valid_length={valid_length}, source_frames={source_x277.shape[0]}"
        )
    clip = np.zeros((seq_len, X277_FEATURE_DIM), dtype=np.float32)
    clip[:valid_length] = source_x277[start_frame:end_frame]
    return clip


def sample_window(rng: np.random.Generator, source_frames: int, seq_len: int) -> tuple[int, int]:
    if source_frames <= 0:
        raise ValueError("源序列帧数必须大于 0。")
    if seq_len <= 0:
        raise ValueError("seq_len 必须大于 0。")
    if source_frames <= seq_len:
        return 0, source_frames
    start_frame = int(rng.integers(0, source_frames - seq_len + 1))
    return start_frame, seq_len


def make_task_id(split: str, stablemotion_split_key: str, sample_index: int) -> str:
    key = normalize_slashes(stablemotion_split_key)
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(key).with_suffix("").as_posix()).strip("_")
    digest = hashlib.sha1(f"{split}:{key}:{sample_index}".encode("utf-8")).hexdigest()[:8]
    return f"{stem}_s{sample_index:04d}_{digest}"


# endregion


def main(argv: list[str] | None = None) -> dict[str, int]:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    counts = generate_missing_tasks(args)
    for split, count in counts.items():
        print(f"[generate_x277_missing_tasks] split={split} tasks={count}")
    return counts


if __name__ == "__main__":
    main()
