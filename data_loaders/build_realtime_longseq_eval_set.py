from __future__ import annotations

import argparse
import hashlib
import re
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any

from data_loaders.generate_realtime_pose_tasks import (
    filter_entries_by_split,
    load_realtime_source,
    normalize_split_key,
    read_source_entries,
    read_split_keys,
)
from data_loaders.sensor_masking import BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY, REALTIME_POSE_FPS


DEFAULT_SOURCE_DIR = (
    "dataset/AMASS_realtime_pose_body_fbx_local_pelvis_residual_root_y0_stationary5_60hz"
)
DEFAULT_SPLIT_DIR = "data_loaders/splits/RPM-P2"
MAX_LONGSEQ_FILE_STEM_CHARS = 32


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按 source 目录和 split 列出可用于长序列评估的动作。"
    )
    parser.add_argument("--source_dir", default=DEFAULT_SOURCE_DIR, type=str)
    parser.add_argument("--split_dir", default=DEFAULT_SPLIT_DIR, type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--min_frames", default=0, type=int)
    parser.add_argument("--include_mirror", default=False, action=BooleanOptionalAction)
    parser.add_argument("--limit", default=0, type=int)
    return parser


def read_longseq_source_entries(
    source_dir: str | Path,
    split_dir: str | Path,
    split: str = "test",
    min_frames: int = 0,
    include_mirror: bool = False,
) -> list[dict[str, Any]]:
    """从 source 目录读取长序列清单，不生成复制集或 manifest。"""

    source_root = Path(source_dir).resolve()
    split_root = Path(split_dir).resolve()
    split_keys = read_split_keys(split_root, str(split))
    all_entries = read_source_entries(source_root)
    selected_entries = filter_entries_by_split(all_entries, split_keys)

    available_keys = {
        normalize_split_key(entry["stablemotion_split_key"]) for entry in all_entries
    }
    missing = sorted((split_keys or set()).difference(available_keys))
    if missing:
        print(
            f"[longseq] split={split} 缺少 source={len(missing)}，将跳过；示例={missing[:3]}"
        )

    results: list[dict[str, Any]] = []
    used_sequence_ids: set[str] = set()
    for entry in selected_entries:
        relative = str(entry["source_relative_path"])
        is_mirrored = bool(entry["is_mirrored"])
        if is_mirrored and not bool(include_mirror):
            continue
        source_path = Path(entry["source_path"])
        source = load_realtime_source(source_path)
        frame_count = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
        if frame_count < int(min_frames):
            continue
        sequence_id = unique_sequence_id(make_sequence_id(relative), used_sequence_ids)
        results.append(
            {
                "sequence_id": sequence_id,
                "source_path": str(source_path),
                "source_relative_path": relative,
                "num_frames": frame_count,
                "fps": float(REALTIME_POSE_FPS),
                "is_mirrored": is_mirrored,
            }
        )
    results.sort(key=lambda item: (-int(item["num_frames"]), str(item["source_relative_path"])))
    return results


def resolve_source_entry_path(entry: dict[str, Any]) -> Path:
    path = Path(str(entry["source_path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"长序列 source 不存在: {path}")
    return path


def build_sequence_output_dir_name(entry: dict[str, Any]) -> str:
    return shorten_path_token(str(entry["sequence_id"]))


def build_replay_filename(entry: dict[str, Any]) -> str:
    return f"{shorten_path_token(str(entry['sequence_id']))}_replay.json"


def normalize_slashes(path: str) -> str:
    return str(path).replace("\\", "/")


def make_sequence_id(source_relative_path: str) -> str:
    stem = str(Path(normalize_slashes(source_relative_path)).with_suffix(""))
    return sanitize_path_token(stem.replace("/", "_"))


def sanitize_path_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return token or "sequence"


def shorten_path_token(value: str, max_chars: int = MAX_LONGSEQ_FILE_STEM_CHARS) -> str:
    token = sanitize_path_token(value)
    if len(token) <= max_chars:
        return token
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:10]
    keep = max(16, int(max_chars) - len(digest) - 1)
    head = keep // 2
    tail = keep - head
    return f"{token[:head]}_{token[-tail:]}_{digest}"


def unique_sequence_id(base_id: str, used: set[str]) -> str:
    candidate = base_id
    suffix = 2
    while candidate in used:
        candidate = f"{base_id}_{suffix:02d}"
        suffix += 1
    used.add(candidate)
    return candidate


def main(argv: list[str] | None = None) -> list[dict[str, Any]]:
    args = build_arg_parser().parse_args(argv)
    entries = read_longseq_source_entries(
        source_dir=args.source_dir,
        split_dir=args.split_dir,
        split=args.split,
        min_frames=int(args.min_frames),
        include_mirror=bool(args.include_mirror),
    )
    if int(args.limit) > 0:
        entries = entries[: int(args.limit)]
    print(f"[longseq] selected={len(entries)}")
    for entry in entries:
        print(
            f"{entry['sequence_id']}\t{entry['num_frames']}\t{entry['source_relative_path']}"
        )
    return entries


if __name__ == "__main__":
    main()
