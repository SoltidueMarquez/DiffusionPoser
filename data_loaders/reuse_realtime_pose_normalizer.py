from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.generate_realtime_pose_tasks import TASK_OUTPUT_MARKER
from data_loaders.realtime_pose_task_store import PLAN_HASH_FILE, read_store_metadata, write_json
from data_loaders.sensor_masking import (
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_MAX_ROLLOUT_STEPS,
    REALTIME_POSE_TARGET_DIM,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_PATTERN_CATEGORIES,
)
from utils.run_dirs import resolve_latest_or_self, timestamped_child_dir, write_latest_pointer


NORMALIZER_STAT_FILES = (
    "pose_mean.pt",
    "pose_std.pt",
    "tracker_mean.pt",
    "tracker_std.pt",
    "head_height_mean.pt",
    "head_height_std.pt",
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把正式 K4 normalizer 统计逐字节绑定到配对的 K15 task。"
    )
    parser.add_argument("--source_normalizer_dir", required=True)
    parser.add_argument("--reference_task_dir", required=True)
    parser.add_argument("--target_task_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--run_name", default="taid_rollout15_seed10")
    return parser


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_task_source_dir(task_dir: Path) -> Path:
    marker_path = task_dir / TASK_OUTPUT_MARKER
    if not marker_path.exists():
        raise FileNotFoundError(f"task 缺少 source marker：{marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    source_dir = marker.get("source_dir") if isinstance(marker, dict) else None
    if not source_dir:
        raise ValueError(f"{marker_path} 缺少 source_dir。")
    resolved = Path(str(source_dir)).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"task marker 指向的 source 不存在：{resolved}")
    return resolved


def _validate_task_array_contract(task_dir: Path, split: str, metadata: dict[str, Any]) -> None:
    shards = sorted(metadata["shards"], key=lambda value: int(value["index"]))
    if not shards:
        raise ValueError(f"{task_dir / split} 没有 shard。")
    shard_dir = task_dir / split / str(shards[0]["path"])
    required_shapes = {
        "current_target": (int(metadata["max_rollout_steps"]), REALTIME_POSE_TARGET_DIM),
        "tracker_history_continuous": (
            int(metadata["max_rollout_steps"]),
            REALTIME_POSE_HISTORY_LENGTH,
            TRACKER_COUNT,
            TRACKER_CONTINUOUS_DIM,
        ),
        "configured": (
            len(TRACKER_PATTERN_CATEGORIES),
            REALTIME_POSE_HISTORY_LENGTH + int(metadata["max_rollout_steps"]),
            TRACKER_COUNT,
        ),
    }
    for field, row_shape in required_shapes.items():
        path = shard_dir / f"{field}.npy"
        if not path.exists():
            raise FileNotFoundError(f"task shard 缺少 {path}。")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape[1:]) != tuple(row_shape):
            raise ValueError(
                f"{path} 行形状应为 {row_shape}，实际为 {tuple(array.shape[1:])}。"
            )


def _validate_task_plan_hash(task_dir: Path, metadata: dict[str, Any]) -> None:
    path = task_dir / PLAN_HASH_FILE
    if not path.exists():
        raise FileNotFoundError(f"task 缺少 generation plan hash：{path}")
    disk_hash = path.read_text(encoding="ascii").strip()
    metadata_hash = str(metadata["generation_plan_hash"])
    if disk_hash != metadata_hash:
        raise ValueError(
            f"task generation plan hash 与 store metadata 不一致：{disk_hash} != {metadata_hash}"
        )


def _validate_paired_tasks(
    reference_task_dir: Path,
    target_task_dir: Path,
    split: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    reference_metadata = read_store_metadata(reference_task_dir / split)
    target_metadata = read_store_metadata(target_task_dir / split)
    _validate_task_plan_hash(reference_task_dir, reference_metadata)
    _validate_task_plan_hash(target_task_dir, target_metadata)
    if int(reference_metadata["max_rollout_steps"]) != 4:
        raise ValueError("reference task 必须是正式 K4 task。")
    if int(target_metadata["max_rollout_steps"]) != REALTIME_POSE_MAX_ROLLOUT_STEPS:
        raise ValueError(
            f"target task 必须物化 K{REALTIME_POSE_MAX_ROLLOUT_STEPS}。"
        )
    for name, expected in (
        ("tracker_feature_dim", TRACKER_FEATURE_DIM),
        ("config_names", TRACKER_PATTERN_CATEGORIES),
    ):
        reference_value = reference_metadata[name]
        target_value = target_metadata[name]
        if name == "config_names":
            reference_value = tuple(reference_value)
            target_value = tuple(target_value)
        if reference_value != expected or target_value != expected:
            raise ValueError(f"reference/target task 的 {name} 不满足当前契约。")

    reference_source = _read_task_source_dir(reference_task_dir)
    target_source = _read_task_source_dir(target_task_dir)
    if reference_source != target_source:
        raise ValueError(
            "reference/target task 不是来自同一个 source："
            f"{reference_source} != {target_source}"
        )
    _validate_task_array_contract(reference_task_dir, split, reference_metadata)
    _validate_task_array_contract(target_task_dir, split, target_metadata)
    return reference_metadata, target_metadata, reference_source


def reuse_realtime_pose_normalizer(args: argparse.Namespace) -> dict[str, Any]:
    source_normalizer_dir = resolve_latest_or_self(
        args.source_normalizer_dir, kind="normalizer"
    )
    reference_task_dir = resolve_latest_or_self(args.reference_task_dir, kind="tasks")
    target_task_dir = resolve_latest_or_self(args.target_task_dir, kind="tasks")
    split = str(args.split)
    reference_metadata, target_metadata, source_dir = _validate_paired_tasks(
        reference_task_dir,
        target_task_dir,
        split,
    )

    source_meta_path = source_normalizer_dir / "normalizer_meta.json"
    if not source_meta_path.exists():
        raise FileNotFoundError(f"正式 normalizer 缺少元数据：{source_meta_path}")
    source_meta_bytes = source_meta_path.read_bytes()
    source_metadata = json.loads(source_meta_bytes.decode("utf-8"))
    if not isinstance(source_metadata, dict):
        raise ValueError(f"{source_meta_path} 必须是 JSON object。")
    reference_hash = str(reference_metadata["generation_plan_hash"])
    if str(source_metadata.get("generation_plan_hash", "")) != reference_hash:
        raise ValueError("正式 normalizer 的 plan hash 与 reference K4 task 不一致。")

    source_hashes: dict[str, str] = {}
    for filename in NORMALIZER_STAT_FILES:
        path = source_normalizer_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"正式 normalizer 缺少统计文件：{path}")
        source_hashes[filename] = sha256_file(path)

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = timestamped_child_dir(output_root, str(args.run_name))
    with tempfile.TemporaryDirectory(prefix=".normalizer_reuse_", dir=output_root) as raw_temp:
        temporary_dir = Path(raw_temp)
        for filename in NORMALIZER_STAT_FILES:
            shutil.copyfile(source_normalizer_dir / filename, temporary_dir / filename)
        copied_hashes = {
            filename: sha256_file(temporary_dir / filename)
            for filename in NORMALIZER_STAT_FILES
        }
        if copied_hashes != source_hashes:
            raise RuntimeError("复用后的 normalizer 统计文件不是逐字节一致副本。")

        metadata = {
            **source_metadata,
            "generation_plan_hash": str(target_metadata["generation_plan_hash"]),
            "task_dir": str(target_task_dir),
            "split": split,
            "source_dir": str(source_dir),
            "statistics_reused_without_recomputation": True,
            "statistics_source_normalizer_dir": str(source_normalizer_dir),
            "statistics_reference_task_dir": str(reference_task_dir),
            "statistics_reference_generation_plan_hash": reference_hash,
            "statistics_source_metadata_sha256": hashlib.sha256(source_meta_bytes).hexdigest(),
            "statistics_file_sha256": source_hashes,
        }
        write_json(temporary_dir / "normalizer_meta.json", metadata)
        temporary_dir.replace(output_dir)

    result = {
        "output_dir": str(output_dir),
        "generation_plan_hash": str(target_metadata["generation_plan_hash"]),
        "reference_generation_plan_hash": reference_hash,
        "statistics_file_sha256": source_hashes,
    }
    write_latest_pointer(output_root, "normalizer", output_dir, result)
    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_argument_parser().parse_args(argv)
    result = reuse_realtime_pose_normalizer(args)
    print(f"[reuse-normalizer] 完成：{result['output_dir']}")
    return result


if __name__ == "__main__":
    main()
