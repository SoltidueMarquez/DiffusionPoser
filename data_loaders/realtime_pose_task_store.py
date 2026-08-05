from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.lib.format import open_memmap

from data_loaders.sensor_masking import TRACKER_FEATURE_DIM, TRACKER_PATTERN_CATEGORIES


STORE_METADATA_FILE = "task_store.json"
PLAN_FILE = "generation_plan.jsonl"
PLAN_HASH_FILE = "generation_plan.sha256"
SHARD_STATS_FILE = "stats.npz"


@dataclass(frozen=True)
class ShardInfo:
    index: int
    row_count: int
    path: str


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_generation_plan(output_dir: Path, entries: Iterable[dict[str, Any]]) -> str:
    """先落盘不含绝对路径的确定性计划，再返回文件内容的 SHA-256。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / PLAN_FILE
    digest = hashlib.sha256()
    with plan_path.open("w", encoding="utf-8", newline="\n") as file:
        for entry in entries:
            line = canonical_json(entry) + "\n"
            file.write(line)
            digest.update(line.encode("utf-8"))
    plan_hash = digest.hexdigest()
    (output_dir / PLAN_HASH_FILE).write_text(plan_hash + "\n", encoding="ascii")
    return plan_hash


def read_generation_plan(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_store_metadata(split_dir: str | Path) -> dict[str, Any]:
    path = Path(split_dir) / STORE_METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"找不到 task store 元数据：{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON object。")
    required = {
        "generation_plan_hash",
        "split",
        "sample_count",
        "source_count",
        "two_point_phase_counts",
        "config_names",
        "tracker_feature_dim",
        "schema_fields",
        "shards",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"{path} 缺少新 task store 字段 {missing}；旧 task 不可复用。")
    if int(value["tracker_feature_dim"]) != TRACKER_FEATURE_DIM:
        raise ValueError(f"{path} Tracker 维度不是当前要求的 {TRACKER_FEATURE_DIM}。")
    if int(value["sample_count"]) <= 0 or int(value["source_count"]) <= 0:
        raise ValueError(f"{path} sample_count/source_count 必须大于 0。")
    if tuple(value["config_names"]) != TRACKER_PATTERN_CATEGORIES:
        raise ValueError(f"{path} 场景列表与当前五类训练契约不一致。")
    phase_counts = value["two_point_phase_counts"]
    if not isinstance(phase_counts, dict) or set(phase_counts) != {"dropout", "reconnect"}:
        raise ValueError(f"{path} 两点掉线 phase 统计结构无效。")
    phase_counts = {phase: int(count) for phase, count in phase_counts.items()}
    if min(phase_counts.values()) < 0 or sum(phase_counts.values()) != int(value["sample_count"]):
        raise ValueError(f"{path} 两点掉线 phase 统计与 sample_count 不一致。")
    allowed_difference = max(1, int(np.ceil(int(value["sample_count"]) * 0.2)))
    if abs(phase_counts["dropout"] - phase_counts["reconnect"]) > allowed_difference:
        raise ValueError(f"{path} 两点掉线 phase 统计不满足近似 1:1 契约。")
    required_fields = {
        "pose_window_clean",
        "tracker_window_continuous",
        "head_path_window",
        "configured",
        "measured_valid",
        "current_head_yaw_world",
    }
    if not required_fields.issubset(set(value["schema_fields"])):
        raise ValueError(f"{path} 不满足当前 task schema；旧 task 不可复用。")
    return value


class ShardWriter:
    """用临时 `.npy` memmap 写完整 shard，关闭后再逐文件原子重命名。"""

    def __init__(self, shard_dir: Path, row_count: int, fields: dict[str, tuple[tuple[int, ...], np.dtype]]):
        self.shard_dir = shard_dir
        self.temporary_dir = shard_dir.with_name(f".{shard_dir.name}.tmp")
        self.row_count = int(row_count)
        self.temporary_dir.mkdir(parents=True, exist_ok=False)
        self.arrays: dict[str, np.memmap] = {}
        for name, (row_shape, dtype) in fields.items():
            temporary_path = self.temporary_dir / f"{name}.npy"
            self.arrays[name] = open_memmap(
                temporary_path,
                mode="w+",
                dtype=np.dtype(dtype),
                shape=(self.row_count, *row_shape),
            )

    def write_row(self, row_index: int, values: dict[str, np.ndarray | int | float]) -> None:
        for name, array in self.arrays.items():
            array[row_index] = values[name]

    def finish(self) -> None:
        for array in self.arrays.values():
            array.flush()
        del array
        self.arrays.clear()
        self.temporary_dir.replace(self.shard_dir)


class ShardReader:
    """单 worker 最多持有两个已打开 shard；淘汰只释放 mmap 引用。"""

    def __init__(self, split_dir: Path, shards: list[dict[str, Any]], max_open_shards: int = 2):
        self.split_dir = split_dir
        self.shards = shards
        self.max_open_shards = int(max_open_shards)
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def get(self, shard_index: int) -> dict[str, np.ndarray]:
        index = int(shard_index)
        arrays = self._cache.pop(index, None)
        if arrays is None:
            shard_dir = self.split_dir / self.shards[index]["path"]
            arrays = {
                path.stem: np.load(path, mmap_mode="r", allow_pickle=False)
                for path in sorted(shard_dir.glob("*.npy"))
            }
            if not arrays:
                raise RuntimeError(f"shard 没有数组：{shard_dir}")
        self._cache[index] = arrays
        while len(self._cache) > self.max_open_shards:
            _, evicted = self._cache.popitem(last=False)
            self._close_arrays(evicted)
        return arrays

    def close(self) -> None:
        for arrays in self._cache.values():
            self._close_arrays(arrays)
        self._cache.clear()

    @staticmethod
    def _close_arrays(arrays: dict[str, np.ndarray]) -> None:
        for array in arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


def load_shard_stats(split_dir: Path, shard: dict[str, Any]) -> dict[str, np.ndarray]:
    path = split_dir / shard["path"] / SHARD_STATS_FILE
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}
