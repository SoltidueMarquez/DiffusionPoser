from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loaders.sensor_masking import MODEL_INPUT_DIM, SENSOR_LABEL_DIM, X277_FEATURE_DIM
from utils.normalizer import X277Normalizer


class X277MissingTaskDataset(Dataset):
    """
    读取离线生成的 X277 传感器缺失任务。

    每个样本由两部分组成：
    - 源 `.npz`：保存真实 X277 序列 `x: [T_source, 277]`；
    - task `.npz`：保存固定窗口、有效长度、`sensor_missing_labels: [T, 6]`
      和 `inpaint_mask: [T, 283]`。

    Dataset 返回 `[C, T]` 张量，DataLoader 默认 collate 后得到 `[B, C, T]`。
    当 `normalize_input=True` 时，只标准化有效帧的前 277 维 X277 特征，padding 仍保持 0。
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = 100,
        normalizer_dir: str | Path | None = None,
        normalize_input: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.normalize_input = bool(normalize_input)
        self.normalizer = create_normalizer(normalizer_dir=normalizer_dir, normalize_input=self.normalize_input)

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
        if self.normalizer is not None:
            # 只处理真实有效帧，padding 继续是模型可识别的 0。
            clip[:valid_length] = self.normalizer.normalize(clip[:valid_length])

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

        label_channels = encode_sensor_labels(
            sensor_missing_labels=sensor_missing_labels,
            valid_frame_mask=valid_frame_mask,
            normalize_input=self.normalize_input,
        )
        x = np.concatenate([clip, label_channels], axis=1)

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


# region normalizer 与标签编码
def create_normalizer(normalizer_dir: str | Path | None, normalize_input: bool) -> X277Normalizer | None:
    if not normalize_input:
        return None
    if normalizer_dir is None or str(normalizer_dir).strip() == "":
        raise ValueError("开启 normalize_input 时必须提供 normalizer_dir。")
    return X277Normalizer(base_dir=normalizer_dir)


def encode_sensor_labels(
    sensor_missing_labels: np.ndarray,
    valid_frame_mask: np.ndarray,
    normalize_input: bool,
) -> np.ndarray:
    """
    将 `[T, 6]` bool 缺失标签编码为模型输入通道。

    未归一化模式保留旧行为：False/True -> 0/1。
    归一化模式使用 StableMotion 风格的条件尺度：False/True -> -1/+1，padding 仍为 0。
    """

    label_channels = sensor_missing_labels.astype(np.float32)
    if normalize_input:
        label_channels = label_channels * 2.0 - 1.0
        label_channels[~valid_frame_mask] = 0.0
    return label_channels


# endregion


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
    return load_x277_sequence_by_path(str(entry["source_path"]))


@lru_cache(maxsize=512)
def load_x277_sequence_by_path(source_path_text: str) -> np.ndarray:
    # 同一个源动作会生成多个缺失任务，缓存源序列能减少重复 npz 解压和磁盘读取。
    source_path = Path(source_path_text)
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
