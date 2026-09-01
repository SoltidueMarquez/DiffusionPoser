from __future__ import annotations

import hashlib
import random

import numpy as np

from data_loaders.sensor_masking import (
    HAND_TRACKER_INDICES,
    PREDICTOR_FREE_RUNNING_MAX_STEPS,
    REALTIME_POSE_HISTORY_LENGTH,
    TRACKER_COUNT,
)


# RPM supplementary：每只手独立以 10% 概率丢弃一段输入，长度
# L ~ U(1, I+1+FR)。本项目与 RPM-P2 一样使用 I=10、FR=30。
RPM_HAND_DROPOUT_PROBABILITY = 0.1
RPM_HAND_DROPOUT_MIN_FRAMES = 1
RPM_HAND_DROPOUT_MAX_FRAMES = (
    REALTIME_POSE_HISTORY_LENGTH + 1 + PREDICTOR_FREE_RUNNING_MAX_STEPS
)
RPM_HAND_DROPOUT_TRAIN_SEED = 7


def rpm_hand_dropout_sample_key(task_seed: int) -> str:
    """把 Task Store 的稳定 seed 转成 Predictor/DiT 共用的样本键。"""

    return f"task_{int(task_seed):016x}"


def stable_rpm_hand_dropout_seed(
    base_seed: int,
    split: str,
    sample_key: str,
) -> int:
    """为每个训练样本派生稳定 seed，不依赖 worker 数量或读取顺序。"""

    payload = f"{int(base_seed)}\x1f{split}\x1f{sample_key}".encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False
    )


def build_rpm_training_hand_availability(
    *,
    frame_count: int,
    seed: int,
    probability: float = RPM_HAND_DROPOUT_PROBABILITY,
    min_frames: int = RPM_HAND_DROPOUT_MIN_FRAMES,
    max_frames: int = RPM_HAND_DROPOUT_MAX_FRAMES,
) -> np.ndarray:
    """复现 RPM 官方训练 masker，返回逐帧 ``[T,6]`` availability。

    官方实现按左手、右手顺序分别调用 Python ``random.random``，命中后再
    调用两次 ``random.randint`` 采样长度和起点。这里使用局部
    ``random.Random`` 保持完全相同的随机数算法与调用顺序，同时避免污染
    训练进程的全局随机状态。
    """

    count = int(frame_count)
    minimum = int(min_frames)
    maximum = int(max_frames)
    mask_probability = float(probability)
    if count <= 0:
        raise ValueError("frame_count 必须为正数。")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("min_frames/max_frames 必须满足 0 < min <= max。")
    if minimum > count:
        raise ValueError("min_frames 不能超过 frame_count。")
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("probability 必须位于 [0,1]。")

    available = np.ones((count, TRACKER_COUNT), dtype=bool)
    rng = random.Random(int(seed))
    if mask_probability <= 0.0:
        return available
    for tracker_index in HAND_TRACKER_INDICES:
        if rng.random() >= mask_probability:
            continue
        # 与 RPM mask_cond_segwise_by_idces 保持一致：randint 两端都包含。
        length = rng.randint(minimum, min(maximum, count))
        start = rng.randint(0, count - length)
        available[start : start + length, int(tracker_index)] = False
    return available


def build_rpm_predictor_training_availability(
    *,
    output_frame_count: int,
    seed: int,
) -> np.ndarray:
    """构造 Predictor ``-11～+40`` 序列使用的确定性手部 mask。

    第 0 帧是计算首个相对速度所需的额外 previous frame；RPM 的 41 帧
    tracking-input 区间从第 1 帧开始。末尾仅作 pose target 的帧保持在线。
    """

    output_count = int(output_frame_count)
    required = RPM_HAND_DROPOUT_MAX_FRAMES + 1
    if output_count < required:
        raise ValueError(
            f"Predictor availability 至少需要 {required} 帧，实际为 {output_count}。"
        )
    result = np.ones((output_count, TRACKER_COUNT), dtype=bool)
    result[1:required] = build_rpm_training_hand_availability(
        frame_count=RPM_HAND_DROPOUT_MAX_FRAMES,
        seed=seed,
    )
    return result


def build_rpm_dit_training_availability(*, seed: int) -> np.ndarray:
    """从官方 41 帧训练 mask 中截取居中的 ``[12,6]`` DiT 条件窗口。

    task store 的样本彼此独立，因此把当前帧放在官方 mask 区间中央，既保留
    原始 gap 随机过程，也能让 DiT 看到掉线前、掉线中和重连后的局部窗口。
    返回值含额外 previous frame，供 Predictor 速度通道正确关闭边界。
    """

    logical = build_rpm_training_hand_availability(
        frame_count=RPM_HAND_DROPOUT_MAX_FRAMES,
        seed=seed,
    )
    with_previous = np.ones(
        (RPM_HAND_DROPOUT_MAX_FRAMES + 1, TRACKER_COUNT), dtype=bool
    )
    with_previous[1:] = logical
    current = 1 + RPM_HAND_DROPOUT_MAX_FRAMES // 2
    return with_previous[current - 11 : current + 1].copy()
