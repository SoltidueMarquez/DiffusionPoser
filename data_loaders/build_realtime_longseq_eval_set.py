from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from argparse import BooleanOptionalAction
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.realtime_pose_task_store import read_store_metadata
from data_loaders.sensor_masking import BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY
from utils.run_dirs import read_latest_pointer, write_latest_pointer


DEFAULT_TASK_ROOT = "../artifactStore/DiffusionPoser/active/generated/tasks/realpose144_60hz"
DEFAULT_LONGSEQ_EVAL_ROOT = "../artifactStore/DiffusionPoser/active/generated/longseq_eval/realpose144_60hz"
DEFAULT_LONGSEQ_RUN_NAME = "v1_test_stress_long_seed10"
LONGSEQ_LATEST_KIND = "longseq_eval"
PRESET_STRESS_LONG = "stress_long"
SUPPORTED_PRESETS = (PRESET_STRESS_LONG,)
MAX_LONGSEQ_FILE_STEM_CHARS = 32


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a fixed long-sequence eval set from realtime_pose test tasks.")
    paths = parser.add_argument_group("paths")
    paths.add_argument("--task_dir", default=DEFAULT_TASK_ROOT, type=str)
    paths.add_argument("--task_run", default="latest", type=str, help="'latest', a run name under task_dir, or a direct path.")
    paths.add_argument("--output_root", default=DEFAULT_LONGSEQ_EVAL_ROOT, type=str)
    paths.add_argument("--run_name", default=DEFAULT_LONGSEQ_RUN_NAME, type=str)

    selection = parser.add_argument_group("selection")
    selection.add_argument("--preset", default=PRESET_STRESS_LONG, choices=SUPPORTED_PRESETS, type=str)
    selection.add_argument("--split", default="test", type=str)
    selection.add_argument("--min_frames", default=2000, type=int)
    selection.add_argument("--include_mirror", default=False, action=BooleanOptionalAction)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--overwrite", default=False, action=BooleanOptionalAction)
    return parser


def resolve_task_run_dir(task_dir: str | Path, task_run: str = "latest") -> Path:
    root = Path(task_dir).resolve()
    value = str(task_run or "latest").strip()
    if value.lower() == "latest":
        latest = read_latest_pointer(root, kind="tasks")
        if latest is not None:
            return latest.resolve()
        if (root / "test" / "sources.jsonl").exists() or (root / "train" / "sources.jsonl").exists():
            return root
        raise FileNotFoundError(f"Cannot resolve latest task run under {root}")

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (root / candidate).resolve()


def resolve_longseq_eval_dir(eval_root: str | Path, eval_set: str = "latest") -> Path:
    root = Path(eval_root).resolve()
    value = str(eval_set or "latest").strip()
    if value.lower() == "latest":
        latest = read_latest_pointer(root, kind=LONGSEQ_LATEST_KIND)
        if latest is None:
            raise FileNotFoundError(f"Cannot resolve latest longseq eval set under {root}")
        return latest.resolve()

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    return (root / candidate).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_longseq_manifest(eval_set_dir: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(eval_set_dir).resolve() / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Longseq eval manifest not found: {manifest_path}")
    return read_jsonl(manifest_path)


def resolve_manifest_source_path(eval_set_dir: str | Path, entry: dict[str, Any]) -> Path:
    raw_path = Path(str(entry["source_path"])).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (Path(eval_set_dir).resolve() / raw_path).resolve()


def build_replay_filename(entry: dict[str, Any]) -> str:
    return f"{shorten_path_token(str(entry['sequence_id']))}_replay.json"


def build_sequence_output_dir_name(entry: dict[str, Any]) -> str:
    return shorten_path_token(str(entry["sequence_id"]))


def build_realtime_longseq_eval_set(args: argparse.Namespace) -> Path:
    task_run_dir = resolve_task_run_dir(task_dir=args.task_dir, task_run=args.task_run)
    split_dir = task_run_dir / str(args.split)
    source_manifest_path = split_dir / "sources.jsonl"
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Task source manifest not found: {source_manifest_path}")

    store_metadata = read_store_metadata(split_dir)
    if str(store_metadata.get("split", "")) != str(args.split):
        raise ValueError(
            f"task_store split={store_metadata.get('split')} 与请求 split={args.split} 不一致。"
        )
    entries = read_jsonl(source_manifest_path)
    selected = select_longseq_entries(
        entries=entries,
        min_frames=int(args.min_frames),
        include_mirror=bool(args.include_mirror),
    )
    if not selected:
        raise RuntimeError(
            f"No long sequence matched preset={args.preset}, split={args.split}, "
            f"min_frames={args.min_frames}, include_mirror={args.include_mirror}"
        )

    output_root = Path(args.output_root).resolve()
    output_dir = (output_root / str(args.run_name)).resolve()
    reset_output_dir(output_root=output_root, output_dir=output_dir, overwrite=bool(args.overwrite))

    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = copy_selected_sources(
        selected=selected,
        output_dir=output_dir,
        sequence_dir=sequence_dir,
        task_run_dir=task_run_dir,
        source_manifest_path=source_manifest_path,
        split=str(args.split),
        preset=str(args.preset),
    )

    config = build_config(
        args=args,
        task_run_dir=task_run_dir,
        source_manifest_path=source_manifest_path,
        store_metadata=store_metadata,
    )
    summary = build_summary(
        manifest_entries=manifest_entries,
        config=config,
        output_dir=output_dir,
    )
    write_eval_set_files(output_dir=output_dir, manifest_entries=manifest_entries, config=config, summary=summary)
    write_latest_pointer(
        root_dir=output_root,
        kind=LONGSEQ_LATEST_KIND,
        output_dir=output_dir,
        metadata={
            "eval_set_dir": str(output_dir),
            "preset": str(args.preset),
            "split": str(args.split),
            "sequence_count": len(manifest_entries),
        },
    )
    return output_dir


def select_longseq_entries(
    entries: list[dict[str, Any]],
    min_frames: int,
    include_mirror: bool,
) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        source_relative_path = normalize_slashes(str(entry.get("source_relative_path", "")))
        if not source_relative_path:
            continue
        if source_relative_path not in deduped:
            deduped[source_relative_path] = entry

    selected = []
    for source_relative_path, entry in deduped.items():
        if "source_frames" not in entry:
            raise KeyError(f"task manifest 缺少 source_frames: {entry.get('task_id', source_relative_path)}")
        frames = int(entry["source_frames"])
        if frames <= 0:
            raise ValueError(f"task manifest source_frames 必须大于 0: {entry.get('task_id', source_relative_path)}")
        if frames < int(min_frames):
            continue
        is_mirrored = bool(entry.get("is_mirrored", False)) or source_relative_path.startswith("M/")
        if is_mirrored and not include_mirror:
            continue
        selected.append(entry)
    selected.sort(key=lambda item: (-int(item["source_frames"]), normalize_slashes(item["source_relative_path"])))
    return selected


def copy_selected_sources(
    selected: list[dict[str, Any]],
    output_dir: Path,
    sequence_dir: Path,
    task_run_dir: Path,
    source_manifest_path: Path,
    split: str,
    preset: str,
) -> list[dict[str, Any]]:
    manifest_entries = []
    used_sequence_ids: set[str] = set()
    for index, task_entry in enumerate(selected):
        original_source_path = resolve_task_source_path(
            task_entry=task_entry,
            source_manifest_path=source_manifest_path,
        )
        source = load_realtime_source(original_source_path)
        frame_count = int(source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY].shape[0])
        declared_frames = int(task_entry["source_frames"])
        if frame_count != declared_frames:
            raise ValueError(f"{original_source_path} frame count {frame_count} != manifest source_frames {declared_frames}")

        source_relative_path = normalize_slashes(str(task_entry["source_relative_path"]))
        sequence_id = unique_sequence_id(make_sequence_id(source_relative_path), used_sequence_ids)
        copied_name = f"{shorten_path_token(sequence_id)}__{frame_count}f.npz"
        copied_path = sequence_dir / copied_name
        shutil.copy2(original_source_path, copied_path)

        manifest_entries.append(
            {
                "sequence_id": sequence_id,
                "source_path": normalize_slashes(copied_path.relative_to(output_dir).as_posix()),
                "original_source_path": str(original_source_path),
                "source_relative_path": source_relative_path,
                "stablemotion_split_key": normalize_slashes(
                    str(task_entry.get("stablemotion_split_key") or task_entry.get("source_id", ""))
                ),
                "split": str(split),
                "num_frames": frame_count,
                "fps": float(task_entry["target_fps"]),
                "preset": preset,
                "is_mirrored": bool(task_entry.get("is_mirrored", False)) or source_relative_path.startswith("M/"),
                "rank": index,
                "task_run_dir": str(task_run_dir),
                "source_manifest_path": str(source_manifest_path),
            }
        )
    return manifest_entries


def resolve_task_source_path(task_entry: dict[str, Any], source_manifest_path: Path) -> Path:
    value = str(task_entry.get("source_path") or "")
    if not value:
        raise KeyError(f"Task entry missing source_path: {task_entry.get('task_id', '<unknown>')}")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (source_manifest_path.parent / path).resolve()


def reset_output_dir(output_root: Path, output_dir: Path, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not overwrite:
        raise FileExistsError(f"Output dir already exists: {output_dir}; pass --overwrite to rebuild it.")
    root = output_root.resolve()
    target = output_dir.resolve()
    if target == root:
        raise ValueError(f"Refusing to overwrite output root directly: {target}")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to overwrite path outside output_root: {target}") from exc
    shutil.rmtree(target)
    target.mkdir(parents=True)


def build_config(
    args: argparse.Namespace,
    task_run_dir: Path,
    source_manifest_path: Path,
    store_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": LONGSEQ_LATEST_KIND,
        "preset": str(args.preset),
        "split": str(args.split),
        "min_frames": int(args.min_frames),
        "include_mirror": bool(args.include_mirror),
        "task_run_dir": str(task_run_dir),
        "source_manifest_path": str(source_manifest_path),
        "generation_plan_hash": str(store_metadata["generation_plan_hash"]),
        "storage_mode": "copied_npz",
    }


def build_summary(
    manifest_entries: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    frame_counts = [int(entry["num_frames"]) for entry in manifest_entries]
    source_paths = [output_dir / str(entry["source_path"]) for entry in manifest_entries]
    total_bytes = sum(path.stat().st_size for path in source_paths if path.exists())
    datasets = Counter(first_dataset_token(str(entry["source_relative_path"])) for entry in manifest_entries)
    return {
        "kind": LONGSEQ_LATEST_KIND,
        "eval_set_dir": str(output_dir),
        "preset": config["preset"],
        "sequence_count": len(manifest_entries),
        "total_frames": int(sum(frame_counts)),
        "min_frames": int(min(frame_counts)),
        "max_frames": int(max(frame_counts)),
        "mean_frames": float(np.mean(frame_counts)),
        "total_bytes": int(total_bytes),
        "datasets": dict(sorted(datasets.items())),
        "config": config,
        "sequences": [
            {
                "sequence_id": entry["sequence_id"],
                "source_relative_path": entry["source_relative_path"],
                "num_frames": entry["num_frames"],
                "is_mirrored": entry["is_mirrored"],
            }
            for entry in manifest_entries
        ],
    }


def write_eval_set_files(
    output_dir: Path,
    manifest_entries: list[dict[str, Any]],
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for entry in manifest_entries:
            file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "summary.json", summary)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def normalize_slashes(path: str) -> str:
    return str(path).replace("\\", "/")


def first_dataset_token(source_relative_path: str) -> str:
    path = normalize_slashes(source_relative_path)
    if path.startswith("M/"):
        parts = path.split("/")
        return parts[1] if len(parts) > 1 else "M"
    return path.split("/", 1)[0]


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


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    output_dir = build_realtime_longseq_eval_set(args)
    print(f"[build_realtime_longseq_eval_set] output={output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
