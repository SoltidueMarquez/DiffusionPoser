from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    MISSING_AGE_CAP,
    NON_HEAD_TRACKER_INDICES,
    REALTIME_POSE_SEQ_LEN,
    SCENARIO_DROPOUT,
    SCENARIO_FIXED_SIX,
    SCENARIO_FIXED_THREE,
    SCENARIO_SIX_TO_THREE,
    SCENARIO_THREE_TO_SIX,
    TRACKER_COUNT,
    TRACKER_PATTERN_CATEGORIES,
    validate_tracker_states,
)


FIXED_THREE_CONFIG = np.asarray([True, True, True, False, False, False], dtype=bool)
FIXED_SIX_CONFIG = np.ones(TRACKER_COUNT, dtype=bool)


@dataclass(frozen=True)
class TrackerTimeline:
    """一个 source 的绝对帧 Tracker 状态，任何重叠窗口都只能从这里切片。"""

    configured: np.ndarray
    measured_valid: np.ndarray
    missing_age: np.ndarray
    missing_age_norm: np.ndarray

    def window(self, start_frame: int, seq_len: int = REALTIME_POSE_SEQ_LEN) -> "TrackerTimeline":
        start = int(start_frame)
        stop = start + int(seq_len)
        if start < 0 or stop > self.configured.shape[0]:
            raise IndexError(f"Tracker timeline 窗口越界: [{start},{stop}) / {self.configured.shape[0]}")
        return TrackerTimeline(
            configured=self.configured[start:stop].copy(),
            measured_valid=self.measured_valid[start:stop].copy(),
            missing_age=self.missing_age[start:stop].copy(),
            missing_age_norm=self.missing_age_norm[start:stop].copy(),
        )


def stable_source_seed(source_id: str, global_seed: int) -> int:
    payload = f"{int(global_seed)}:{source_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def build_tracker_timeline(
    source_id: str,
    frame_count: int,
    global_seed: int = 10,
    min_config_dwell: int = 180,
    max_config_dwell: int = 300,
    dropout_duration_min: int = 1,
    dropout_duration_max: int = MISSING_AGE_CAP,
) -> TrackerTimeline:
    """构造确定性的三/六点配置与掉线事件。

    配置区间故意长于两个 61 帧窗口，使固定、切换和掉线窗口可以从同一条
    绝对时间线中抽取，而不是对每个 task 单独随机。
    """

    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError(f"frame_count 必须大于 0，实际为 {frame_count}")
    if min_config_dwell < REALTIME_POSE_SEQ_LEN:
        raise ValueError("min_config_dwell 至少应覆盖一个完整 61 帧窗口。")
    if max_config_dwell < min_config_dwell:
        raise ValueError("max_config_dwell 不能小于 min_config_dwell。")

    rng = np.random.default_rng(stable_source_seed(source_id=source_id, global_seed=global_seed))
    configured = np.zeros((frame_count, TRACKER_COUNT), dtype=bool)
    blocks: list[tuple[int, int, bool]] = []
    cursor = 0
    use_six = bool(rng.integers(0, 2))
    while cursor < frame_count:
        dwell = int(rng.integers(min_config_dwell, max_config_dwell + 1))
        stop = min(frame_count, cursor + dwell)
        configured[cursor:stop] = FIXED_SIX_CONFIG if use_six else FIXED_THREE_CONFIG
        blocks.append((cursor, stop, use_six))
        cursor = stop
        use_six = not use_six

    measured_valid = configured.copy()
    # 每个足够长的稳定配置块最多放一个掉线事件，并给两侧保留固定窗口候选。
    margin = REALTIME_POSE_SEQ_LEN
    drop_count_phase = int(rng.integers(0, 2))
    for event_index, (block_start, block_stop, _) in enumerate(blocks):
        available = block_stop - block_start - 2 * margin
        if available < dropout_duration_min:
            continue
        max_duration = min(int(dropout_duration_max), available)
        duration = int(rng.integers(int(dropout_duration_min), max_duration + 1))
        latest_start = block_stop - margin - duration
        event_start = int(rng.integers(block_start + margin, latest_start + 1))
        event_stop = event_start + duration
        candidates = [
            tracker_index
            for tracker_index in NON_HEAD_TRACKER_INDICES
            if configured[event_start, tracker_index]
        ]
        if not candidates:
            continue
        # 在每条足够长的 source 时间线上交替生成单点/双点掉线，避免窗口级随机导致比例漂移。
        drop_count = min(1 + ((event_index + drop_count_phase) % 2), len(candidates))
        dropped = rng.choice(np.asarray(candidates), size=drop_count, replace=False)
        measured_valid[event_start:event_stop, dropped] = False

    validate_tracker_states(configured=configured, measured_valid=measured_valid)
    missing_age = compute_missing_age(configured=configured, measured_valid=measured_valid)
    return TrackerTimeline(
        configured=configured,
        measured_valid=measured_valid,
        missing_age=missing_age,
        missing_age_norm=missing_age.astype(np.float32) / float(MISSING_AGE_CAP),
    )


def compute_missing_age(
    configured: np.ndarray,
    measured_valid: np.ndarray,
    cap: int = MISSING_AGE_CAP,
) -> np.ndarray:
    configured, measured_valid = validate_tracker_states(configured, measured_valid)
    cap = int(cap)
    if cap <= 0:
        raise ValueError(f"missing age cap 必须大于 0，实际为 {cap}")
    age = np.zeros(configured.shape, dtype=np.int16)
    previous = np.zeros(TRACKER_COUNT, dtype=np.int16)
    for frame_index in range(configured.shape[0]):
        missing = configured[frame_index] & ~measured_valid[frame_index]
        current = np.zeros(TRACKER_COUNT, dtype=np.int16)
        current[missing] = np.minimum(previous[missing] + 1, cap)
        age[frame_index] = current
        previous = current
    age[:, HEAD_TRACKER_INDEX] = 0
    return age


def classify_tracker_window(
    configured: np.ndarray,
    measured_valid: np.ndarray,
) -> str | None:
    configured, measured_valid = validate_tracker_states(configured, measured_valid)
    if configured.shape[0] != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"场景分类固定读取 {REALTIME_POSE_SEQ_LEN} 帧，实际为 {configured.shape[0]}")

    if np.any(configured & ~measured_valid):
        return SCENARIO_DROPOUT

    transitions = np.flatnonzero(np.any(configured[1:] != configured[:-1], axis=1))
    if len(transitions) > 1:
        return None
    if len(transitions) == 1:
        before = int(configured[transitions[0]].sum())
        after = int(configured[transitions[0] + 1].sum())
        if before == 3 and after == 6:
            return SCENARIO_THREE_TO_SIX
        if before == 6 and after == 3:
            return SCENARIO_SIX_TO_THREE
        return None

    first = configured[0]
    if np.array_equal(first, FIXED_SIX_CONFIG):
        return SCENARIO_FIXED_SIX
    if np.array_equal(first, FIXED_THREE_CONFIG):
        return SCENARIO_FIXED_THREE
    return None


def candidate_starts_by_scenario(
    timeline: TrackerTimeline,
    seq_len: int = REALTIME_POSE_SEQ_LEN,
) -> dict[str, list[int]]:
    if int(seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"当前任务固定 seq_len={REALTIME_POSE_SEQ_LEN}")
    result = {category: [] for category in TRACKER_PATTERN_CATEGORIES}
    max_start = timeline.configured.shape[0] - seq_len
    for start in range(max_start + 1):
        stop = start + seq_len
        category = classify_tracker_window(
            configured=timeline.configured[start:stop],
            measured_valid=timeline.measured_valid[start:stop],
        )
        if category is not None:
            result[category].append(start)
    return result


def sample_balanced_starts(
    candidates: dict[str, list[int]],
    samples_per_category: int | dict[str, int],
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    if isinstance(samples_per_category, dict):
        sample_counts = {category: int(samples_per_category.get(category, 0)) for category in TRACKER_PATTERN_CATEGORIES}
    else:
        count = int(samples_per_category)
        sample_counts = {category: count for category in TRACKER_PATTERN_CATEGORIES}
    if any(count < 0 for count in sample_counts.values()) or not any(sample_counts.values()):
        raise ValueError("每类场景样本数必须非负且至少一类大于零。")
    selected: list[tuple[str, int]] = []
    for category in TRACKER_PATTERN_CATEGORIES:
        starts = candidates.get(category, [])
        count = sample_counts[category]
        if not starts or count <= 0:
            continue
        indices = rng.choice(
            np.asarray(starts, dtype=np.int64),
            size=count,
            replace=len(starts) < count,
        )
        selected.extend((category, int(start)) for start in indices.tolist())
    rng.shuffle(selected)
    return selected
