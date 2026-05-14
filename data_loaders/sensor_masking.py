from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


X277_FEATURE_DIM = 277
SENSOR_LABEL_DIM = 6
MODEL_INPUT_DIM = X277_FEATURE_DIM + SENSOR_LABEL_DIM

TRACKER_POS_START = 216
TRACKER_POS_DIM = 3
TRACKER_ROT_START = 234
TRACKER_ROT_DIM = 6

SENSOR_NAMES = (
    "head",
    "left_wrist",
    "right_wrist",
    "waist",
    "left_foot",
    "right_foot",
)


@dataclass(frozen=True)
class MissingInterval:
    """记录一个连续缺失任务，便于写入 manifest 做复现实验。"""

    start: int
    length: int
    sensor_indices: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "length": self.length,
            "sensor_indices": list(self.sensor_indices),
            "sensor_names": [SENSOR_NAMES[index] for index in self.sensor_indices],
        }


# region 维度映射
def validate_sensor_index(sensor_index: int) -> None:
    if sensor_index < 0 or sensor_index >= SENSOR_LABEL_DIM:
        raise ValueError(f"sensor_index 必须位于 [0, {SENSOR_LABEL_DIM})，实际为 {sensor_index}")


def sensor_feature_slices(sensor_index: int) -> tuple[slice, slice]:
    """返回单个传感器在 X277 tracker position/rotation 中的两段特征切片。"""

    validate_sensor_index(sensor_index)
    pos_start = TRACKER_POS_START + TRACKER_POS_DIM * sensor_index
    rot_start = TRACKER_ROT_START + TRACKER_ROT_DIM * sensor_index
    return (
        slice(pos_start, pos_start + TRACKER_POS_DIM),
        slice(rot_start, rot_start + TRACKER_ROT_DIM),
    )


def tracker_feature_mask(sensor_indices: Iterable[int], feature_dim: int = MODEL_INPUT_DIM) -> np.ndarray:
    """
    将传感器编号转换成一维特征 mask。

    返回值形状为 `[feature_dim]`，其中 True 只出现在 X277 的 tracker position/rotation 维度。
    追加的 6 维缺失标签永远不作为重建目标，因此即使 `feature_dim=283` 也保持 False。
    """

    if feature_dim < MODEL_INPUT_DIM:
        raise ValueError(f"feature_dim 至少应为 {MODEL_INPUT_DIM}，实际为 {feature_dim}")

    mask = np.zeros(feature_dim, dtype=bool)
    for sensor_index in sensor_indices:
        pos_slice, rot_slice = sensor_feature_slices(int(sensor_index))
        mask[pos_slice] = True
        mask[rot_slice] = True
    return mask


# endregion


# region 缺失任务采样
def sample_missing_sensors(
    rng: np.random.Generator,
    min_missing_sensors: int = 1,
    max_missing_sensors: int = 4,
    min_observed_sensors: int = 2,
) -> tuple[int, ...]:
    """采样本段缺失的传感器集合，同时保证至少 `min_observed_sensors` 个传感器保留。"""

    max_allowed = SENSOR_LABEL_DIM - min_observed_sensors
    max_count = min(max_missing_sensors, max_allowed)
    if min_missing_sensors < 1 or min_missing_sensors > max_count:
        raise ValueError(
            "缺失传感器数量范围无效："
            f"min={min_missing_sensors}, max={max_missing_sensors}, min_observed={min_observed_sensors}"
        )

    missing_count = int(rng.integers(min_missing_sensors, max_count + 1))
    sensor_indices = rng.choice(SENSOR_LABEL_DIM, size=missing_count, replace=False)
    return tuple(sorted(int(index) for index in sensor_indices))


def sample_frame_interval(
    rng: np.random.Generator,
    valid_length: int,
    min_missing_length: int = 10,
    max_missing_length: int | None = None,
) -> tuple[int, int]:
    """在有效帧范围内采样连续缺失区间，返回 `[start, length]`。"""

    if valid_length <= 0:
        raise ValueError("valid_length 必须大于 0，才能生成缺失区间。")

    upper = valid_length if max_missing_length is None else min(max_missing_length, valid_length)
    lower = min(max(1, min_missing_length), upper)
    length = int(rng.integers(lower, upper + 1))
    start = int(rng.integers(0, valid_length - length + 1))
    return start, length


def create_sensor_missing_task(
    seq_len: int,
    valid_length: int,
    rng: np.random.Generator,
    num_intervals: int = 1,
    min_missing_length: int = 10,
    max_missing_length: int | None = None,
    min_missing_sensors: int = 1,
    max_missing_sensors: int = 4,
    min_observed_sensors: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[MissingInterval]]:
    """
    生成单条固定长度训练样本的缺失标签和 inpainting mask。

    `sensor_missing_labels` 形状为 `[T, 6]`，只表达某个传感器在某帧是否缺失。
    `inpaint_mask` 形状为 `[T, 283]`，True 表示该位置参与扩散加噪和 masked loss。
    padding 帧不生成缺失，标签通道 `[277:283]` 也始终不参与 loss。
    """

    if seq_len <= 0:
        raise ValueError("seq_len 必须大于 0。")
    if valid_length <= 0 or valid_length > seq_len:
        raise ValueError(f"valid_length 必须位于 [1, seq_len]，实际为 {valid_length} / {seq_len}")
    if num_intervals <= 0:
        raise ValueError("num_intervals 必须大于 0。")

    sensor_missing_labels = np.zeros((seq_len, SENSOR_LABEL_DIM), dtype=bool)
    inpaint_mask = np.zeros((seq_len, MODEL_INPUT_DIM), dtype=bool)
    intervals: list[MissingInterval] = []
    max_missing_per_frame = SENSOR_LABEL_DIM - min_observed_sensors

    for _ in range(num_intervals):
        for _attempt in range(100):
            start, length = sample_frame_interval(
                rng=rng,
                valid_length=valid_length,
                min_missing_length=min_missing_length,
                max_missing_length=max_missing_length,
            )
            sensor_indices = sample_missing_sensors(
                rng=rng,
                min_missing_sensors=min_missing_sensors,
                max_missing_sensors=max_missing_sensors,
                min_observed_sensors=min_observed_sensors,
            )

            candidate_labels = sensor_missing_labels.copy()
            candidate_inpaint_mask = inpaint_mask.copy()
            apply_sensor_missing_interval(
                sensor_missing_labels=candidate_labels,
                inpaint_mask=candidate_inpaint_mask,
                start=start,
                length=length,
                sensor_indices=sensor_indices,
            )
            if candidate_labels.sum(axis=1).max() <= max_missing_per_frame:
                sensor_missing_labels = candidate_labels
                inpaint_mask = candidate_inpaint_mask
                intervals.append(MissingInterval(start=start, length=length, sensor_indices=sensor_indices))
                break
        else:
            raise RuntimeError("无法在 100 次尝试内采样到满足可见传感器约束的缺失区间。")

    # 这两行是训练语义的保险丝：标签只是条件，不应被扩散模型预测；padding 也不应有监督。
    inpaint_mask[:, X277_FEATURE_DIM:MODEL_INPUT_DIM] = False
    inpaint_mask[valid_length:, :] = False
    sensor_missing_labels[valid_length:, :] = False
    return sensor_missing_labels, inpaint_mask, intervals


def apply_sensor_missing_interval(
    sensor_missing_labels: np.ndarray,
    inpaint_mask: np.ndarray,
    start: int,
    length: int,
    sensor_indices: Iterable[int],
) -> None:
    """把一个连续缺失区间写入 `[T, 6]` 标签和 `[T, 283]` inpaint mask。"""

    if sensor_missing_labels.ndim != 2 or sensor_missing_labels.shape[1] != SENSOR_LABEL_DIM:
        raise ValueError("sensor_missing_labels 必须为 [T, 6]。")
    if inpaint_mask.ndim != 2 or inpaint_mask.shape[1] < MODEL_INPUT_DIM:
        raise ValueError("inpaint_mask 必须为 [T, 283] 或更高特征维。")

    end = start + length
    if start < 0 or length <= 0 or end > sensor_missing_labels.shape[0]:
        raise ValueError(f"缺失区间越界：start={start}, length={length}, T={sensor_missing_labels.shape[0]}")

    for sensor_index in sensor_indices:
        validate_sensor_index(int(sensor_index))
        sensor_missing_labels[start:end, int(sensor_index)] = True
        pos_slice, rot_slice = sensor_feature_slices(int(sensor_index))
        inpaint_mask[start:end, pos_slice] = True
        inpaint_mask[start:end, rot_slice] = True

    inpaint_mask[:, X277_FEATURE_DIM:MODEL_INPUT_DIM] = False


# endregion
