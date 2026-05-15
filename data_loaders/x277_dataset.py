from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loaders.sensor_masking import MODEL_INPUT_DIM, SENSOR_LABEL_DIM, X277_FEATURE_DIM
from utils.normalizer import X277Normalizer


class X277MissingTaskDataset(Dataset):
    """
    读取离线生成的 materialized X277 传感器缺失任务。

    每个 task `.npz` 必须包含：
    - `x277: [T, 277]`：已经从原始动作文件裁剪/padding 好的固定窗口。
    - `sensor_missing_labels: [T, 6]`：6 个传感器在每帧是否缺失。
    - `inpaint_mask: [T, 283]`：扩散训练中哪些位置需要补全并参与 loss。

    Dataset 返回 `[C, T]` 张量，DataLoader 默认 collate 后得到 `[B, C, T]`。
    训练阶段不再读取原始 `AMASS_x277_60hz`，避免每条样本同时解压 task 和源 npz。
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = 100,
        normalizer_dir: str | Path | None = None,
        normalize_input: bool = True,
        preload_data: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = int(seq_len)
        self.normalize_input = bool(normalize_input)
        self.preload_data = bool(preload_data)
        self.normalizer = create_normalizer(normalizer_dir=normalizer_dir, normalize_input=self.normalize_input)

        self.manifest_path = find_manifest_path(data_dir=self.data_dir, split=split)
        self.manifest_dir = self.manifest_path.parent
        self.entries = read_task_manifest(self.manifest_path)

        if not self.entries:
            raise RuntimeError(f"{self.manifest_path} 中没有可用任务。")
        for entry in self.entries:
            entry_seq_len = int(entry.get("seq_len", self.seq_len))
            if entry_seq_len != self.seq_len:
                raise ValueError(
                    f"任务 {entry.get('task_id')} 的 seq_len={entry_seq_len}，"
                    f"但当前 DataLoader 请求 seq_len={self.seq_len}。"
                )

        self.task_cache = None
        if self.preload_data:
            # 只预加载 materialized task 本身；normalizer 仍在 __getitem__ 中动态执行，
            # 这样切换 mean/std 或关闭标准化时不需要重新生成任务数据。
            self.task_cache = [
                load_materialized_task_npz(manifest_dir=self.manifest_dir, task_path=entry["task_path"])
                for entry in self.entries
            ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        task = self.load_task(index=index, entry=entry)

        valid_length = validate_scalar_int(task=task, name="valid_length")
        if valid_length <= 0 or valid_length > self.seq_len:
            raise ValueError(f"valid_length 应位于 [1, {self.seq_len}]，实际为 {valid_length}")

        clip = validate_task_array(
            array=task["x277"],
            shape=(self.seq_len, X277_FEATURE_DIM),
            name="x277",
        ).astype(np.float32, copy=True)
        # 物化任务理论上已经把 padding 写成 0；这里再清一次，避免异常文件把 padding 噪声带入模型。
        clip[valid_length:] = 0.0

        if self.normalizer is not None:
            # 只标准化真实有效帧。padding 保持 0，便于模型和 mask 一起识别无效区域。
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

        # padding 和 6 维缺失标签只作为条件上下文，不参与扩散补全 loss。
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
            "source_path": entry.get("source_path", ""),
        }

    def load_task(self, index: int, entry: dict) -> dict[str, np.ndarray]:
        if self.task_cache is not None:
            return self.task_cache[index]
        return load_materialized_task_npz(manifest_dir=self.manifest_dir, task_path=entry["task_path"])


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
    归一化模式使用 StableMotion 风格条件尺度：False/True -> -1/+1，padding 仍为 0。
    """

    label_channels = sensor_missing_labels.astype(np.float32)
    if normalize_input:
        label_channels = label_channels * 2.0 - 1.0
        label_channels[~valid_frame_mask] = 0.0
    return label_channels


# endregion


# region manifest 与任务读取
def find_manifest_path(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / split / "manifest.jsonl",
        data_dir / "manifest.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"找不到离线任务 manifest。已尝试：{', '.join(str(path) for path in candidates)}")


def read_task_manifest(manifest_path: Path) -> list[dict]:
    entries = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_materialized_task_npz(manifest_dir: Path, task_path: str) -> dict[str, np.ndarray]:
    path = manifest_dir / task_path
    if not path.exists():
        raise FileNotFoundError(f"缺失任务文件不存在：{path}")

    with np.load(path, allow_pickle=False) as data:
        task = {key: data[key].copy() for key in data.files}

    required_keys = {
        "x277",
        "sensor_missing_labels",
        "inpaint_mask",
        "start_frame",
        "valid_length",
        "source_frames",
        "seq_len",
    }
    missing_keys = sorted(required_keys.difference(task))
    if missing_keys:
        raise KeyError(
            f"{path} 不是新的 materialized X277 task，缺少字段：{missing_keys}。"
            "请使用 `python -m data_loaders.generate_x277_missing_tasks --overwrite ...` 重新生成任务数据。"
        )
    return task


def validate_scalar_int(task: dict[str, np.ndarray], name: str) -> int:
    if name not in task:
        raise KeyError(f"task 缺少字段 `{name}`。")
    value = np.asarray(task[name])
    if value.shape != ():
        raise ValueError(f"{name} 应为标量，实际 shape={value.shape}")
    return int(value.item())


def validate_task_array(array: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} 应为 {shape}，实际为 {tuple(array.shape)}")
    return array


# endregion
