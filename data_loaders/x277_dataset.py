from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data_loaders.manifest_utils import filter_entries_by_folder_path
from data_loaders.sensor_masking import (
    LAST_FRAME_RECONSTRUCTION_SEQ_LEN,
    MODEL_INPUT_DIM,
    SENSOR_LABEL_DIM,
    TASK_FORMAT_CURRENT277_BODY_RECONSTRUCTION_V2,
    X277_FEATURE_DIM,
    enforce_last_frame_reconstruction_task,
    sensor_feature_slices,
    validate_last_frame_seq_len,
    validate_last_frame_window,
)
from utils.normalizer import X277Normalizer


class X277MissingTaskDataset(Dataset):
    """
    读取离线生成的 X277 缺失任务。

    每个 task `.npz` 至少包含：
    - `x277: [T, 277]`
    - `sensor_missing_labels: [T, 6]`
    - `inpaint_mask: [T, 283]`

    Dataset 返回的是 `[C, T]` 布局，便于后面的 DataLoader 自动拼成 `[B, C, T]`。
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        seq_len: int = LAST_FRAME_RECONSTRUCTION_SEQ_LEN,
        normalizer_dir: str | Path | None = None,
        normalize_input: bool = True,
        preload_data: bool = False,
        folder_path: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = int(seq_len)
        validate_last_frame_seq_len(self.seq_len)
        self.normalize_input = bool(normalize_input)
        self.preload_data = bool(preload_data)
        self.normalizer = create_normalizer(normalizer_dir=normalizer_dir, normalize_input=self.normalize_input)

        # 先找到当前 split 对应的 manifest，再从 manifest 里读取任务条目。
        self.manifest_path = find_manifest_path(data_dir=self.data_dir, split=split)
        self.manifest_dir = self.manifest_path.parent
        self.entries = read_task_manifest(self.manifest_path)

        # folder_path 只是“额外筛选条件”，不会替代 split。
        if folder_path:
            self.entries = filter_entries_by_folder_path(self.entries, folder_path=folder_path)

        if not self.entries:
            raise RuntimeError(f"{self.manifest_path} 中没有可用任务。")

        # 数据入口只接受原生 11 帧 materialized task，不在读取阶段裁剪旧长窗口。
        for entry in self.entries:
            task_format = str(entry.get("task_format", "")).strip()
            if task_format and task_format != TASK_FORMAT_CURRENT277_BODY_RECONSTRUCTION_V2:
                raise ValueError(
                    f"任务 {entry.get('task_id')} 的 task_format={task_format}，"
                    f"当前只接受 {TASK_FORMAT_CURRENT277_BODY_RECONSTRUCTION_V2}。请重新生成 materialized task。"
                )
            if "seq_len" not in entry:
                raise KeyError(f"任务 {entry.get('task_id')} 缺少 manifest 字段 `seq_len`。")
            entry_seq_len = int(entry["seq_len"])
            if entry_seq_len != self.seq_len:
                raise ValueError(
                    f"任务 {entry.get('task_id')} 的 seq_len={entry_seq_len}，"
                    f"当前数据集只接受原生 {self.seq_len} 帧在线补全窗口。"
                )

        self.task_cache = None
        if self.preload_data:
            # 预加载只缓存原始 npz 内容，不提前做 normalizer 变换。
            # 这样同一份缓存既能复用，也不会把训练/测试阶段的归一化策略锁死。
            self.task_cache = [
                load_materialized_task_npz(manifest_dir=self.manifest_dir, task_path=entry["task_path"])
                for entry in self.entries
            ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        task = self.load_task(index=index, entry=entry)

        clip, sensor_missing_labels, inpaint_mask, valid_length = load_last_frame_task_arrays(
            task=task,
            seq_len=self.seq_len,
        )

        if self.normalizer is not None:
            # 11 帧全部是有效窗口；归一化只作用在 X277 物理特征上。
            clip = self.normalizer.normalize(clip)

        valid_frame_mask = np.ones(self.seq_len, dtype=bool)

        target_start, target_length = enforce_last_frame_reconstruction_task(
            sensor_missing_labels=sensor_missing_labels,
            inpaint_mask=inpaint_mask,
            valid_length=valid_length,
        )

        # 6 维缺失标签只作为条件通道输入，不作为重建目标。
        label_channels = encode_sensor_labels(
            sensor_missing_labels=sensor_missing_labels,
            valid_frame_mask=valid_frame_mask,
            normalize_input=self.normalize_input,
        )
        x = np.concatenate([clip, label_channels], axis=1)
        conditioned_clip = build_tracker_zero_conditioned_clip(
            clip=clip,
            sensor_missing_labels=sensor_missing_labels,
        )
        conditioned_x = np.concatenate([conditioned_clip, label_channels], axis=1)

        return {
            "x": torch.from_numpy(x.T).float(),
            "conditioned_x": torch.from_numpy(conditioned_x.T).float(),
            "valid_frame_mask": torch.from_numpy(valid_frame_mask).bool(),
            "attention_mask": torch.from_numpy(valid_frame_mask).bool(),
            "sensor_missing_labels": torch.from_numpy(sensor_missing_labels.T).bool(),
            "inpaint_mask": torch.from_numpy(inpaint_mask.T).bool(),
            "length": valid_length,
            "keyid": entry.get("task_id", ""),
            "source_path": entry.get("source_path", ""),
            "task_mode": entry.get("task_mode", ""),
            "schema_name": entry.get("schema_name", ""),
            "target_start": target_start,
            "target_length": target_length,
        }

    def load_task(self, index: int, entry: dict) -> dict[str, np.ndarray]:
        # 如果开启了 preload，就直接复用缓存；否则按需读盘。
        if self.task_cache is not None:
            return self.task_cache[index]
        return load_materialized_task_npz(manifest_dir=self.manifest_dir, task_path=entry["task_path"])


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
    把 `[T, 6]` 的传感器缺失标签编码成模型输入通道。

    - 未归一化模式：直接保留 0/1
    - 归一化模式：转成 -1/+1，并把 padding 帧重新置回 0
    """

    label_channels = sensor_missing_labels.astype(np.float32)
    if normalize_input:
        label_channels = label_channels * 2.0 - 1.0
        label_channels[~valid_frame_mask] = 0.0
    return label_channels


def build_tracker_zero_conditioned_clip(
    clip: np.ndarray,
    sensor_missing_labels: np.ndarray,
) -> np.ndarray:
    """
    构造模型实际看到的 X277 条件特征。

    缺失 tracker 的真实 pos/rot 仍保留在 `x` 里作为 GT 和导出参考；但训练/采样时这些通道
    不能泄漏给模型，所以在 `conditioned_x` 中按缺失标签置为 0。这里在归一化之后执行，
    保证“零条件”就是模型输入空间里的 0。
    """

    conditioned = np.asarray(clip, dtype=np.float32).copy()
    labels = np.asarray(sensor_missing_labels, dtype=bool)
    if conditioned.ndim != 2 or conditioned.shape[1] < X277_FEATURE_DIM:
        raise ValueError(f"clip 应为 [T, 277]，实际为 {conditioned.shape}")
    if labels.shape != (conditioned.shape[0], SENSOR_LABEL_DIM):
        raise ValueError(f"sensor_missing_labels 应为 [T, 6]，实际为 {labels.shape}")

    for sensor_index in range(SENSOR_LABEL_DIM):
        missing_frames = labels[:, sensor_index]
        if not missing_frames.any():
            continue
        pos_slice, rot_slice = sensor_feature_slices(sensor_index)
        conditioned[missing_frames, pos_slice] = 0.0
        conditioned[missing_frames, rot_slice] = 0.0
    return conditioned


def find_manifest_path(data_dir: Path, split: str) -> Path:
    # 优先找 `data_dir/split/manifest.jsonl`，找不到再回退到 `data_dir/manifest.jsonl`。
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


def load_last_frame_task_arrays(
    task: dict[str, np.ndarray],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """读取并校验原生 10 帧历史 + 第 11 帧补全任务，不做旧窗口裁剪。"""

    validate_last_frame_seq_len(seq_len)
    task_seq_len = validate_scalar_int(task=task, name="seq_len")
    validate_last_frame_seq_len(task_seq_len)
    valid_length = validate_scalar_int(task=task, name="valid_length")
    validate_last_frame_window(valid_length)

    clip = validate_task_array(task["x277"], shape=(seq_len, X277_FEATURE_DIM), name="x277").astype(
        np.float32,
        copy=True,
    )
    sensor_missing_labels = validate_task_array(
        task["sensor_missing_labels"], shape=(seq_len, SENSOR_LABEL_DIM), name="sensor_missing_labels"
    ).astype(bool)
    inpaint_mask = validate_task_array(
        task["inpaint_mask"], shape=(seq_len, MODEL_INPUT_DIM), name="inpaint_mask"
    ).astype(bool)

    return clip, sensor_missing_labels, inpaint_mask, valid_length


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
