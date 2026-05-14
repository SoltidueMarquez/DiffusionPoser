from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loaders.sensor_masking import MODEL_INPUT_DIM, SENSOR_LABEL_DIM, X277_FEATURE_DIM


class X277MissingTaskDataset(Dataset):
    """
    读取离线生成的 X277 传感器缺失任务。

    每个样本在磁盘上拆成两部分：
    - 源 `.npz` 保存真实 X277 序列 `x: [T_source, 277]`；
    - task `.npz` 保存固定窗口起点、有效长度、6 维缺失标签和 `[T, 283]` inpaint mask。

    Dataset 返回训练 loop 直接需要的 `[C, T]` 张量，DataLoader 默认 collate 后得到 `[B, C, T]`。
    """

    def __init__(self, data_dir: str | Path, split: str = "train", seq_len: int = 100):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.manifest_path = find_manifest_path(data_dir=self.data_dir, split=split)
        self.manifest_dir = self.manifest_path.parent
        self.entries = read_task_manifest(self.manifest_path)

        if not self.entries:
            raise RuntimeError(f"{self.manifest_path} 中没有可用任务。")
        for entry in self.entries:
            entry_seq_len = int(entry.get("seq_len", seq_len))
            if entry_seq_len != seq_len:
                raise ValueError(
                    f"任务 {entry.get('task_id')} 的 seq_len={entry_seq_len}，"
                    f"但当前 DataLoader 请求 seq_len={seq_len}。"
                )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        task = load_task_npz(manifest_dir=self.manifest_dir, task_path=entry["task_path"])
        x277 = load_x277_sequence(entry=entry)

        start_frame = int(task["start_frame"])
        valid_length = int(task["valid_length"])
        if valid_length <= 0 or valid_length > self.seq_len:
            raise ValueError(f"valid_length 应位于 [1, {self.seq_len}]，实际为 {valid_length}")
        if start_frame < 0 or start_frame + valid_length > x277.shape[0]:
            raise ValueError(
                f"任务窗口越界：start={start_frame}, valid_length={valid_length}, source_frames={x277.shape[0]}"
            )

        clip = np.zeros((self.seq_len, X277_FEATURE_DIM), dtype=np.float32)
        clip[:valid_length] = x277[start_frame : start_frame + valid_length]

        valid_frame_mask = np.zeros(self.seq_len, dtype=bool)
        valid_frame_mask[:valid_length] = True

        sensor_missing_labels = validate_task_array(
            array=task["sensor_missing_labels"],
            shape=(self.seq_len, SENSOR_LABEL_DIM),
            name="sensor_missing_labels",
        ).astype(bool)
        inpaint_mask = validate_task_array(
            array=task["inpaint_mask"],
            shape=(self.seq_len, MODEL_INPUT_DIM),
            name="inpaint_mask",
        ).astype(bool)

        # padding 和标签维度都只作为条件上下文，不能进入扩散 loss。
        sensor_missing_labels[~valid_frame_mask] = False
        inpaint_mask[~valid_frame_mask] = False
        inpaint_mask[:, X277_FEATURE_DIM:MODEL_INPUT_DIM] = False

        x = np.concatenate([clip, sensor_missing_labels.astype(np.float32)], axis=1)

        return {
            "x": torch.from_numpy(x.T).float(),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask).bool(),
            "attention_mask": torch.from_numpy(valid_frame_mask).bool(),
            "sensor_missing_labels": torch.from_numpy(sensor_missing_labels.T).bool(),
            "inpaint_mask": torch.from_numpy(inpaint_mask.T).bool(),
            "length": valid_length,
            "keyid": entry.get("task_id", ""),
            "source_path": entry["source_path"],
        }


# region manifest 与文件读取
def find_manifest_path(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / split / "manifest.jsonl",
        data_dir / "manifest.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"找不到离线任务 manifest。已尝试：{', '.join(str(path) for path in candidates)}"
    )


def read_task_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_task_npz(manifest_dir: Path, task_path: str) -> dict[str, np.ndarray]:
    path = manifest_dir / task_path
    if not path.exists():
        raise FileNotFoundError(f"缺失任务文件不存在：{path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def load_x277_sequence(entry: dict) -> np.ndarray:
    source_path = Path(entry["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"源 X277 文件不存在：{source_path}")
    with np.load(source_path, allow_pickle=False) as data:
        if "x" not in data:
            raise KeyError(f"{source_path} 缺少字段 `x`。")
        x = data["x"].astype(np.float32, copy=True)

    if x.ndim != 2 or x.shape[1] != X277_FEATURE_DIM:
        raise ValueError(f"{source_path} 的 x 应为 [T, 277]，实际为 {x.shape}")
    return x


def validate_task_array(array: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} 应为 {shape}，实际为 {tuple(array.shape)}")
    return array


# endregion
