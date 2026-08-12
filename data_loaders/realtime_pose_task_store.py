from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.lib.format import open_memmap


SHARD_STATS_FILE = "stats.npz"


def discover_shards(
    split_dir: str | Path,
    required_fields: Iterable[str],
) -> list[dict[str, Any]]:
    """从目录发现 shard，并用数组首维建立最小读取索引。

    shard 目录及字段文件本身就是 task store 契约。
    初始化时只读取 `.npy` 头部，不扫描样本内容。
    """

    root = Path(split_dir).resolve()
    shards_root = root / "shards"
    if not shards_root.is_dir():
        raise FileNotFoundError(f"找不到 task shard 目录: {shards_root}")

    field_names = tuple(sorted(set(str(name) for name in required_fields)))
    shard_dirs = sorted(path for path in shards_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise RuntimeError(f"{shards_root} 中没有 shard_*/ 目录。")

    shards: list[dict[str, Any]] = []
    for index, shard_dir in enumerate(shard_dirs):
        row_count: int | None = None
        for field_name in field_names:
            field_path = shard_dir / f"{field_name}.npy"
            if not field_path.is_file():
                raise FileNotFoundError(f"task shard 缺少字段: {field_path}")
            array = np.load(field_path, mmap_mode="r", allow_pickle=False)
            try:
                if array.ndim < 1:
                    raise ValueError(f"task 字段必须带样本维: {field_path} shape={array.shape}")
                current_count = int(array.shape[0])
            finally:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            if row_count is None:
                row_count = current_count
            elif current_count != row_count:
                raise ValueError(
                    f"{shard_dir} 字段首维不一致: expected={row_count}, "
                    f"{field_name}={current_count}"
                )
        if row_count is None or row_count <= 0:
            raise ValueError(f"task shard 不能为空: {shard_dir}")
        shards.append(
            {
                "index": index,
                "row_count": row_count,
                "path": shard_dir.relative_to(root).as_posix(),
            }
        )
    return shards


class ShardWriter:
    """用临时目录写完一个 mmap shard，再原子重命名到正式目录。"""

    def __init__(
        self,
        shard_dir: Path,
        row_count: int,
        fields: dict[str, tuple[tuple[int, ...], np.dtype]],
    ):
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
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.arrays.clear()
        self.temporary_dir.replace(self.shard_dir)


class ShardReader:
    """按需 mmap shard；每个 Dataset worker 只保留少量已打开目录。"""

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
                raise RuntimeError(f"shard 没有数组: {shard_dir}")
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


def load_shard_stats(shard_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(shard_dir) / SHARD_STATS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"task shard 缺少 normalizer 统计: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}
