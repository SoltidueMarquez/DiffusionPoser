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
HISTORY_CONTEXT_FRAMES = 10
LAST_FRAME_RECONSTRUCTION_SEQ_LEN = HISTORY_CONTEXT_FRAMES + 1

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


def validate_last_frame_seq_len(seq_len: int) -> None:
    """
    校验 current277 在线补全任务的物化窗口长度。

    当前需求不保留长窗口兼容：任务文件、训练、采样和导出都必须直接使用
    11 帧窗口，其中前 10 帧是历史条件，第 11 帧是唯一补全目标。
    """

    if int(seq_len) != LAST_FRAME_RECONSTRUCTION_SEQ_LEN:
        raise ValueError(
            "full_reconstruction_current 固定为 10 帧历史 + 第 11 帧补全，"
            f"seq_len 应为 {LAST_FRAME_RECONSTRUCTION_SEQ_LEN}，实际为 {seq_len}"
        )


def validate_last_frame_window(valid_length: int) -> None:
    """
    校验 current277 在线补全窗口长度。

    本任务固定读取 10 帧历史上下文，并补全第 11 帧；因此有效窗口长度必须
    正好是 11。这样训练、采样和评估里的 target 帧位置都保持一致。
    """

    if int(valid_length) != LAST_FRAME_RECONSTRUCTION_SEQ_LEN:
        raise ValueError(
            "full_reconstruction_current 固定为 10 帧历史 + 第 11 帧补全，"
            f"valid_length 应为 {LAST_FRAME_RECONSTRUCTION_SEQ_LEN}，实际为 {valid_length}"
        )


def current_reconstruction_target_frame(valid_length: int) -> int:
    """返回当前窗口唯一需要补全的帧索引，也就是第 11 帧。"""

    validate_last_frame_window(valid_length)
    return HISTORY_CONTEXT_FRAMES


def enforce_last_frame_reconstruction_task(
    sensor_missing_labels: np.ndarray,
    inpaint_mask: np.ndarray,
    valid_length: int,
) -> tuple[int, int]:
    """
    严格校验任务已经是 DiffusionPoser 风格的“只补当前帧”窗口。

    输入数组形状分别为 `[T, 6]` 和 `[T, 283]`。函数不再重写旧的长区间
    target；如果 mask 不是“前 10 帧全条件、第 11 帧补全”的原生任务，
    直接报错，要求重新生成 materialized task。
    """

    if sensor_missing_labels.ndim != 2 or sensor_missing_labels.shape[1] != SENSOR_LABEL_DIM:
        raise ValueError("sensor_missing_labels must be [T, 6].")
    if inpaint_mask.ndim != 2 or inpaint_mask.shape[1] < MODEL_INPUT_DIM:
        raise ValueError("inpaint_mask must be [T, 283] or wider.")
    if sensor_missing_labels.shape[0] != inpaint_mask.shape[0]:
        raise ValueError(
            f"sensor_missing_labels and inpaint_mask frame counts differ: "
            f"{sensor_missing_labels.shape[0]} vs {inpaint_mask.shape[0]}"
        )
    validate_last_frame_window(valid_length)
    if valid_length > inpaint_mask.shape[0]:
        raise ValueError(f"valid_length={valid_length} exceeds task length={inpaint_mask.shape[0]}")

    target_start = current_reconstruction_target_frame(valid_length)
    target_length = 1
    if sensor_missing_labels[:target_start].any() or sensor_missing_labels[target_start + 1 : valid_length].any():
        raise ValueError("sensor_missing_labels 只能在第 11 帧标记缺失传感器。")

    expected_mask = np.zeros_like(inpaint_mask, dtype=bool)
    mark_current_reconstruction_targets(
        inpaint_mask=expected_mask,
        start=target_start,
        length=target_length,
    )
    missing_sensors = np.flatnonzero(sensor_missing_labels[target_start])
    if missing_sensors.size:
        apply_sensor_missing_interval(
            sensor_missing_labels=np.zeros_like(sensor_missing_labels, dtype=bool),
            inpaint_mask=expected_mask,
            start=target_start,
            length=target_length,
            sensor_indices=missing_sensors,
        )

    if not np.array_equal(inpaint_mask.astype(bool), expected_mask):
        raise ValueError("inpaint_mask 必须只在第 11 帧标记 body/root/contact 和当前缺失 tracker。")
    return target_start, target_length


def create_full_reconstruction_task(
    seq_len: int,
    valid_length: int,
    rng: np.random.Generator,
    num_intervals: int = 1,
    min_missing_sensors: int = 1,
    max_missing_sensors: int = 4,
    min_observed_sensors: int = 2,
    all_sensor_dropout_prob: float = 0.10,
    target_start: int | None = None,
    target_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[MissingInterval], int, int]:
    """
    Create a current277 full reconstruction task.

    每个固定窗口只预测最后一个有效帧。历史帧全部作为条件，最后一帧预测
    body rotation、body velocity、root delta、yaw delta、contact，以及当前
    离线 tracker 的 position/rotation。这和 DiffusionPoser 的实时自回归
    inpainting 对齐：一个窗口对应“历史 + 当前观测”，输出当前帧。
    """

    validate_last_frame_seq_len(seq_len)
    validate_last_frame_window(valid_length)
    if num_intervals <= 0:
        raise ValueError("num_intervals must be positive.")
    if not 0.0 <= all_sensor_dropout_prob <= 1.0:
        raise ValueError("all_sensor_dropout_prob must be in [0, 1].")

    expected_target_start = current_reconstruction_target_frame(valid_length)
    if target_start is None:
        target_start = expected_target_start
    else:
        target_start = int(target_start)
        if target_start != expected_target_start:
            raise ValueError(
                "full_reconstruction_current 固定补第 11 帧："
                f"target_start 应为 {expected_target_start}，实际为 {target_start}"
            )
    if target_length is None:
        target_length = 1
    else:
        target_length = int(target_length)
        if target_length != 1:
            raise ValueError(
                "full_reconstruction_current 只补窗口最后一帧："
                f"target_length 应为 1，实际为 {target_length}"
            )

    sensor_missing_labels = np.zeros((seq_len, SENSOR_LABEL_DIM), dtype=bool)
    inpaint_mask = np.zeros((seq_len, MODEL_INPUT_DIM), dtype=bool)
    intervals: list[MissingInterval] = []

    mark_current_reconstruction_targets(
        inpaint_mask=inpaint_mask,
        start=target_start,
        length=target_length,
    )

    for _ in range(num_intervals):
        # 任务目标固定为当前窗口最后一帧；多次采样会把多个缺失传感器集合并到这一帧。
        start, length = target_start, target_length
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
