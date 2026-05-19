from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


X277_FEATURE_DIM = 277
SENSOR_LABEL_DIM = 6
MODEL_INPUT_DIM = X277_FEATURE_DIM + SENSOR_LABEL_DIM

CURRENT277_SCHEMA_NAME = "current277_v1"
TASK_MODE_FULL_RECONSTRUCTION_CURRENT = "full_reconstruction_current"
TASK_MODES = (TASK_MODE_FULL_RECONSTRUCTION_CURRENT,)

BODY_ROT_START = 0
BODY_ROT_DIM = 144
BODY_VEL_START = 144
BODY_VEL_DIM = 72
TRACKER_POS_START = 216
TRACKER_POS_DIM = 3
TRACKER_ROT_START = 234
TRACKER_ROT_DIM = 6
ROOT_DELTA_START = 270
ROOT_DELTA_DIM = 2
ROOT_YAW_START = 272
ROOT_YAW_DIM = 1
CONTACT_START = 273
CONTACT_DIM = 4

FULL_RECONSTRUCTION_TARGET_SLICES = (
    slice(BODY_ROT_START, BODY_ROT_START + BODY_ROT_DIM),
    slice(BODY_VEL_START, BODY_VEL_START + BODY_VEL_DIM),
    slice(ROOT_DELTA_START, ROOT_DELTA_START + ROOT_DELTA_DIM),
    slice(ROOT_YAW_START, ROOT_YAW_START + ROOT_YAW_DIM),
    slice(CONTACT_START, CONTACT_START + CONTACT_DIM),
)

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


def sample_frame_interval_in_range(
    rng: np.random.Generator,
    start_frame: int,
    end_frame: int,
    min_missing_length: int = 10,
    max_missing_length: int | None = None,
) -> tuple[int, int]:
    """Sample a contiguous interval inside ``[start_frame, end_frame)``."""

    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError(f"invalid frame range: start={start_frame}, end={end_frame}")
    span = end_frame - start_frame
    upper = span if max_missing_length is None else min(max_missing_length, span)
    lower = min(max(1, min_missing_length), upper)
    length = int(rng.integers(lower, upper + 1))
    start = int(rng.integers(start_frame, end_frame - length + 1))
    return start, length


def mark_current_reconstruction_targets(inpaint_mask: np.ndarray, start: int, length: int) -> None:
    """Mark body/root/contact channels as reconstruction targets for a current277 window."""

    if inpaint_mask.ndim != 2 or inpaint_mask.shape[1] < MODEL_INPUT_DIM:
        raise ValueError("inpaint_mask must be [T, 283] or wider.")
    end = start + length
    if start < 0 or length <= 0 or end > inpaint_mask.shape[0]:
        raise ValueError(f"target range out of bounds: start={start}, length={length}, T={inpaint_mask.shape[0]}")
    for target_slice in FULL_RECONSTRUCTION_TARGET_SLICES:
        inpaint_mask[start:end, target_slice] = True
    inpaint_mask[:, X277_FEATURE_DIM:MODEL_INPUT_DIM] = False


def create_full_reconstruction_task(
    seq_len: int,
    valid_length: int,
    rng: np.random.Generator,
    num_intervals: int = 1,
    min_missing_length: int = 10,
    max_missing_length: int | None = None,
    min_missing_sensors: int = 1,
    max_missing_sensors: int = 4,
    min_observed_sensors: int = 2,
    min_history_frames: int = 10,
    min_target_frames: int = 1,
    max_target_frames: int | None = None,
    all_sensor_dropout_prob: float = 0.10,
    target_start: int | None = None,
    target_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[MissingInterval], int, int]:
    """
    Create a current277 full reconstruction task.

    The target suffix predicts body rotation, body velocity, root delta, yaw delta,
    contact labels, and tracker pos/rot only when that tracker is offline. The
    prefix before ``target_start`` remains condition context.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if valid_length <= 0 or valid_length > seq_len:
        raise ValueError(f"valid_length must be in [1, seq_len], got {valid_length} / {seq_len}")
    if num_intervals <= 0:
        raise ValueError("num_intervals must be positive.")
    if not 0.0 <= all_sensor_dropout_prob <= 1.0:
        raise ValueError("all_sensor_dropout_prob must be in [0, 1].")

    if target_start is None:
        target_start_upper = max(valid_length - max(1, min_target_frames), 0)
        target_start_lower = min(max(0, min_history_frames), target_start_upper)
        target_start = int(rng.integers(target_start_lower, target_start_upper + 1))
    else:
        target_start = int(target_start)
    if target_start < 0 or target_start >= valid_length:
        raise ValueError(f"target_start out of valid range: {target_start}, valid_length={valid_length}")

    max_available_target = valid_length - target_start
    if target_length is None:
        target_upper = max_available_target if max_target_frames is None else min(max_target_frames, max_available_target)
        target_lower = min(max(1, min_target_frames), target_upper)
        target_length = int(rng.integers(target_lower, target_upper + 1))
    else:
        target_length = int(target_length)
    if target_length <= 0 or target_start + target_length > valid_length:
        raise ValueError(
            f"target range out of valid frames: start={target_start}, length={target_length}, valid={valid_length}"
        )

    sensor_missing_labels = np.zeros((seq_len, SENSOR_LABEL_DIM), dtype=bool)
    inpaint_mask = np.zeros((seq_len, MODEL_INPUT_DIM), dtype=bool)
    intervals: list[MissingInterval] = []

    mark_current_reconstruction_targets(
        inpaint_mask=inpaint_mask,
        start=target_start,
        length=target_length,
    )

    target_end = target_start + target_length
    for _ in range(num_intervals):
        start, length = sample_frame_interval_in_range(
            rng=rng,
            start_frame=target_start,
            end_frame=target_end,
            min_missing_length=min_missing_length,
            max_missing_length=max_missing_length,
        )
        if rng.random() < all_sensor_dropout_prob:
            sensor_indices = tuple(range(SENSOR_LABEL_DIM))
        else:
            sensor_indices = sample_missing_sensors(
                rng=rng,
                min_missing_sensors=min_missing_sensors,
                max_missing_sensors=max_missing_sensors,
                min_observed_sensors=min_observed_sensors,
            )

        apply_sensor_missing_interval(
            sensor_missing_labels=sensor_missing_labels,
            inpaint_mask=inpaint_mask,
            start=start,
            length=length,
            sensor_indices=sensor_indices,
        )
        intervals.append(MissingInterval(start=start, length=length, sensor_indices=tuple(sensor_indices)))

    inpaint_mask[:, X277_FEATURE_DIM:MODEL_INPUT_DIM] = False
    inpaint_mask[valid_length:, :] = False
    sensor_missing_labels[valid_length:, :] = False
    return sensor_missing_labels, inpaint_mask, intervals, target_start, target_length


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
