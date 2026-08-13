from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from data_loaders.realtime_pose_config import TrackerReliabilityConfig
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    NON_HEAD_TRACKER_INDICES,
    REALTIME_POSE_FUTURE_FRAME_COUNT,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    SCENARIO_FIXED_SIX,
    SCENARIO_FIXED_THREE,
    SCENARIO_SIX_TO_THREE,
    SCENARIO_THREE_TO_SIX,
    SCENARIO_TWO_POINT_DROPOUT_RECONNECT,
    TRACKER_COUNT,
    TRACKER_PATTERN_CATEGORIES,
    validate_tracker_states,
)
from data_loaders.tracker_reliability import compute_hard_rotation_state_np


FIXED_THREE_CONFIG = np.asarray([True, True, True, False, False, False], dtype=bool)
FIXED_SIX_CONFIG = np.ones(TRACKER_COUNT, dtype=bool)
DROPOUT_DURATION_MIN = 5
DROPOUT_DURATION_MAX = 30
RECOVERY_WINDOW_FRAMES = 15
SOURCE_EVENT_PERIOD = 120
TWO_POINT_TARGET_PHASES = ("dropout", "reconnect")


@dataclass(frozen=True)
class TrackerTimeline:
    """一个 source 的绝对帧状态；重叠窗口只能从同一时间线切片。"""

    configured: np.ndarray
    measured_valid: np.ndarray
    d_off: np.ndarray
    d_on: np.ndarray
    hard_rotation_state: np.ndarray

    def window(self, start_frame: int, seq_len: int = REALTIME_POSE_SEQ_LEN) -> "TrackerTimeline":
        start = int(start_frame)
        stop = start + int(seq_len)
        if start < 0 or stop > self.configured.shape[0]:
            raise IndexError(f"Tracker timeline 窗口越界: [{start},{stop}) / {self.configured.shape[0]}")
        return TrackerTimeline(**{
            name: np.asarray(getattr(self, name)[start:stop]).copy()
            for name in self.__dataclass_fields__
        })


@dataclass(frozen=True)
class TrackerScenarioBatch:
    """五类场景的 `[5,T,6]` 状态集合。"""

    configured: np.ndarray
    measured_valid: np.ndarray
    d_off: np.ndarray
    d_on: np.ndarray
    hard_rotation_state: np.ndarray


def stable_source_seed(source_id: str, global_seed: int) -> int:
    payload = f"{int(global_seed)}:{source_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def stable_context_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def build_task_config_plan(
    task_id: str,
    global_seed: int,
    max_rollout_steps: int,
    *,
    source_id: str | None = None,
    start_frame: int | None = None,
    source_frame_count: int | None = None,
) -> list[dict]:
    """为基础窗口生成五套场景；正式生成时事件来自 source 绝对时间线。"""

    rollout_steps = int(max_rollout_steps)
    if not 1 <= rollout_steps <= 4:
        raise ValueError("max_rollout_steps 必须在 [1,4]。")
    absolute_mode = source_id is not None and start_frame is not None and source_frame_count is not None
    absolute_target = int(start_frame or 0) + REALTIME_POSE_TARGET_START
    source_events = (
        build_source_scenario_events(
            str(source_id),
            int(source_frame_count),
            rollout_steps,
            global_seed,
        )
        if absolute_mode
        else []
    )
    selected_event = next(
        (
            event
            for event in source_events
            if 0 <= absolute_target - int(event["transition_frame"]) < RECOVERY_WINDOW_FRAMES
            and _event_accepts_target(event, absolute_target)
        ),
        None,
    )
    if absolute_mode and selected_event is None:
        raise ValueError(
            f"start_frame={start_frame} 不在 source={source_id} 的确定性动态场景候选内。"
        )

    plans: list[dict] = []
    for config_index, scenario in enumerate(TRACKER_PATTERN_CATEGORIES):
        rng = np.random.Generator(np.random.PCG64(stable_context_seed(global_seed, task_id, config_index)))
        plan: dict[str, object] = {"config_index": config_index, "scenario": scenario}
        if scenario in (SCENARIO_THREE_TO_SIX, SCENARIO_SIX_TO_THREE):
            if selected_event is not None:
                plan["transition_frame"] = int(selected_event["transition_frame"])
            else:
                post_transition_offset = int(rng.integers(0, RECOVERY_WINDOW_FRAMES))
                plan["transition_frame"] = REALTIME_POSE_TARGET_START - post_transition_offset
        elif scenario == SCENARIO_TWO_POINT_DROPOUT_RECONNECT:
            if selected_event is not None:
                duration = int(selected_event["dropout_duration"])
                dropped = np.asarray(selected_event["dropped_trackers"], dtype=np.int64)
                dropout_start = int(selected_event["dropout_start"])
                reconnect_frame = int(selected_event["reconnect_frame"])
                phase = str(selected_event["target_phase"])
                if reconnect_frame != dropout_start + duration:
                    raise RuntimeError("两点掉线事件的重连帧与掉线时长不一致。")
            else:
                duration = int(rng.integers(DROPOUT_DURATION_MIN, DROPOUT_DURATION_MAX + 1))
                dropped = rng.choice(np.asarray(NON_HEAD_TRACKER_INDICES), size=2, replace=False)
                phase = "dropout" if int(rng.integers(0, 2)) == 0 else "reconnect"
                if phase == "dropout":
                    elapsed = int(rng.integers(0, duration))
                    dropout_start = REALTIME_POSE_TARGET_START - elapsed
                else:
                    recovery_offset = int(rng.integers(0, RECOVERY_WINDOW_FRAMES))
                    reconnect_frame = REALTIME_POSE_TARGET_START - recovery_offset
                    dropout_start = reconnect_frame - duration
            plan.update(
                dropped_trackers=sorted(int(value) for value in dropped.tolist()),
                dropout_start=int(dropout_start),
                dropout_duration=duration,
                target_phase=phase,
            )
        plans.append(plan)
    return plans


def build_source_scenario_events(
    source_id: str,
    frame_count: int,
    max_rollout_steps: int,
    global_seed: int = 10,
) -> list[dict]:
    """构造所有动态场景共用的绝对事件锚点；同一 source 重复调用结果一致。"""

    last_target = (
        int(frame_count)
        - int(max_rollout_steps)
        - REALTIME_POSE_FUTURE_FRAME_COUNT
    )
    if last_target < REALTIME_POSE_TARGET_START:
        return []
    rng = np.random.Generator(np.random.PCG64(stable_source_seed(source_id, global_seed)))
    available_phase = min(SOURCE_EVENT_PERIOD - 1, last_target - REALTIME_POSE_TARGET_START)
    first_event = REALTIME_POSE_TARGET_START + int(rng.integers(0, available_phase + 1))
    # 同一事件的 15 个候选帧只承担一种阶段，避免从中点切分后
    # 恢复阶段最多只剩 7～8 帧。首个阶段由 source hash 决定，短 source 也不会固定偏向某一类。
    phase_offset = int(stable_context_seed(global_seed, source_id, "two_point_phase") % 2)
    events: list[dict] = []
    for event_index, transition_frame in enumerate(
        range(first_event, last_target + 1, SOURCE_EVENT_PERIOD)
    ):
        event_rng = np.random.Generator(
            np.random.PCG64(stable_context_seed(global_seed, source_id, event_index, "event"))
        )
        duration = int(event_rng.integers(DROPOUT_DURATION_MIN, DROPOUT_DURATION_MAX + 1))
        dropped = sorted(
            int(value)
            for value in event_rng.choice(
                np.asarray(NON_HEAD_TRACKER_INDICES), size=2, replace=False
            ).tolist()
        )
        event_window = min(RECOVERY_WINDOW_FRAMES, last_target - transition_frame + 1)
        target_phase = TWO_POINT_TARGET_PHASES[(event_index + phase_offset) % 2]
        # reconnect 事件从重连帧开始枚举 0～14；dropout 事件把重连点
        # 放在候选窗口之后，使被接受的目标帧仍严格处于掉线期。
        reconnect_frame = (
            int(transition_frame)
            if target_phase == "reconnect"
            else int(transition_frame + event_window)
        )
        dropout_start = reconnect_frame - duration
        events.append(
            {
                "transition_frame": int(transition_frame),
                "dropout_start": int(dropout_start),
                "dropout_duration": duration,
                "dropped_trackers": dropped,
                "reconnect_frame": reconnect_frame,
                "target_phase": target_phase,
            }
        )
    return events


def candidate_source_window_starts_by_phase(
    source_id: str,
    frame_count: int,
    max_rollout_steps: int,
    global_seed: int = 10,
) -> dict[str, list[int]]:
    """按两点掉线阶段返回满足动态场景契约的窗口起点。"""

    starts = {phase: [] for phase in TWO_POINT_TARGET_PHASES}
    for event in build_source_scenario_events(
        source_id, frame_count, max_rollout_steps, global_seed
    ):
        anchor = int(event["transition_frame"])
        phase = str(event["target_phase"])
        if phase not in starts:
            raise RuntimeError(f"未知的两点掉线目标阶段：{phase}")
        for offset in range(RECOVERY_WINDOW_FRAMES):
            target = anchor + offset
            start = target - REALTIME_POSE_TARGET_START
            if start < 0 or not _event_accepts_target(event, target):
                continue
            if (
                target
                + int(max_rollout_steps)
                - 1
                + REALTIME_POSE_FUTURE_FRAME_COUNT
                >= int(frame_count)
            ):
                continue
            starts[phase].append(start)
    return {phase: sorted(set(values)) for phase, values in starts.items()}


def candidate_source_window_starts(
    source_id: str,
    frame_count: int,
    max_rollout_steps: int,
    global_seed: int = 10,
) -> list[int]:
    """返回同时满足切换 0～14 帧和两点掉线/重连契约的窗口起点。"""

    starts_by_phase = candidate_source_window_starts_by_phase(
        source_id=source_id,
        frame_count=frame_count,
        max_rollout_steps=max_rollout_steps,
        global_seed=global_seed,
    )
    return sorted({start for values in starts_by_phase.values() for start in values})


def _event_accepts_target(event: dict, target_frame: int) -> bool:
    dropout_start = int(event["dropout_start"])
    reconnect_frame = dropout_start + int(event["dropout_duration"])
    target = int(target_frame)
    phase = str(event["target_phase"])
    if phase == "dropout":
        return dropout_start <= target < reconnect_frame
    if phase == "reconnect":
        return 0 <= target - reconnect_frame < RECOVERY_WINDOW_FRAMES
    raise ValueError(f"未知的两点掉线目标阶段：{phase}")


def materialize_task_configurations(
    config_plans: list[dict],
    frame_count: int = REALTIME_POSE_SEQ_LEN + REALTIME_POSE_FUTURE_FRAME_COUNT,
    duration_prefix: int = 60,
    reliability_config: TrackerReliabilityConfig | None = None,
    absolute_start_frame: int = 0,
) -> TrackerScenarioBatch:
    """物化五类场景，并用 60 帧前缀连续计算 duration。"""

    prefix = int(duration_prefix)
    absolute_frames = np.arange(
        int(absolute_start_frame) - prefix,
        int(absolute_start_frame) + int(frame_count),
        dtype=np.int64,
    )
    shape = (len(config_plans), int(frame_count), TRACKER_COUNT)
    values: dict[str, np.ndarray] = {
        "configured": np.empty(shape, dtype=bool),
        "measured_valid": np.empty(shape, dtype=bool),
        "d_off": np.empty(shape, dtype=np.uint8),
        "d_on": np.empty(shape, dtype=np.uint8),
        "hard_rotation_state": np.empty(shape, dtype=bool),
    }
    for plan in config_plans:
        index = int(plan["config_index"])
        configured, measured = _materialize_plan_states(plan, absolute_frames)
        d_off, d_on = compute_tracker_durations(configured, measured)
        hard = compute_hard_rotation_state_np(
            configured,
            measured,
            d_on,
            config=reliability_config,
        )
        for name, array in (
            ("configured", configured),
            ("measured_valid", measured),
            ("d_off", d_off),
            ("d_on", d_on),
            ("hard_rotation_state", hard),
        ):
            values[name][index] = array[prefix:]
    return TrackerScenarioBatch(**values)


def build_tracker_timeline(
    source_id: str,
    frame_count: int,
    global_seed: int = 10,
    min_config_dwell: int = 180,
    max_config_dwell: int = 300,
    dropout_duration_min: int = DROPOUT_DURATION_MIN,
    dropout_duration_max: int = DROPOUT_DURATION_MAX,
    reliability_config: TrackerReliabilityConfig | None = None,
) -> TrackerTimeline:
    """构造 source-absolute 三/六点配置和同步两点掉线事件。"""

    frame_count = int(frame_count)
    if frame_count <= 0:
        raise ValueError("frame_count 必须大于 0。")
    if min_config_dwell < REALTIME_POSE_SEQ_LEN or max_config_dwell < min_config_dwell:
        raise ValueError("配置驻留时长必须至少覆盖一个窗口，且 max>=min。")
    if not DROPOUT_DURATION_MIN <= dropout_duration_min <= dropout_duration_max <= DROPOUT_DURATION_MAX:
        raise ValueError("两点掉线时长必须限制在 [5,30]。")
    rng = np.random.default_rng(stable_source_seed(source_id, global_seed))
    configured = np.zeros((frame_count, TRACKER_COUNT), dtype=bool)
    blocks: list[tuple[int, int, bool]] = []
    cursor = 0
    use_six = bool(rng.integers(0, 2))
    while cursor < frame_count:
        stop = min(frame_count, cursor + int(rng.integers(min_config_dwell, max_config_dwell + 1)))
        configured[cursor:stop] = FIXED_SIX_CONFIG if use_six else FIXED_THREE_CONFIG
        blocks.append((cursor, stop, use_six))
        cursor = stop
        use_six = not use_six

    measured = configured.copy()
    margin = REALTIME_POSE_SEQ_LEN
    for block_start, block_stop, block_is_six in blocks:
        available = block_stop - block_start - 2 * margin
        if not block_is_six or available < dropout_duration_min:
            continue
        duration = int(rng.integers(dropout_duration_min, min(dropout_duration_max, available) + 1))
        latest_start = block_stop - margin - duration
        event_start = int(rng.integers(block_start + margin, latest_start + 1))
        dropped = rng.choice(np.asarray(NON_HEAD_TRACKER_INDICES), size=2, replace=False)
        measured[event_start : event_start + duration, dropped] = False

    validate_tracker_states(configured, measured)
    d_off, d_on = compute_tracker_durations(configured, measured)
    hard = compute_hard_rotation_state_np(
        configured,
        measured,
        d_on,
        config=reliability_config,
    )
    return TrackerTimeline(configured, measured, d_off, d_on, hard)


def build_isolated_condition_timeline(
    source_id: str,
    frame_count: int,
    condition: str,
    global_seed: int = 10,
    reliability_config: TrackerReliabilityConfig | None = None,
) -> TrackerTimeline:
    """为纯净对照实验构造只包含一类事件的绝对时间线。

    固定条件覆盖整条序列；切换条件只在序列中点发生一次目标方向的
    切换；两点掉线条件始终配置六点，仅注入确定性的掉线/重连事件。
    """

    frames = int(frame_count)
    scenario = str(condition)
    if frames <= 0:
        raise ValueError("frame_count 必须大于 0。")
    if scenario not in TRACKER_PATTERN_CATEGORIES:
        raise ValueError(f"未知独立评测条件：{scenario}")

    if scenario == SCENARIO_FIXED_THREE:
        configured = np.repeat(FIXED_THREE_CONFIG[None], frames, axis=0)
    elif scenario == SCENARIO_THREE_TO_SIX:
        configured = np.repeat(FIXED_THREE_CONFIG[None], frames, axis=0)
        configured[_isolated_transition_frame(frames) :] = FIXED_SIX_CONFIG
    elif scenario == SCENARIO_SIX_TO_THREE:
        configured = np.repeat(FIXED_SIX_CONFIG[None], frames, axis=0)
        configured[_isolated_transition_frame(frames) :] = FIXED_THREE_CONFIG
    else:
        configured = np.repeat(FIXED_SIX_CONFIG[None], frames, axis=0)

    measured = configured.copy()
    if scenario == SCENARIO_TWO_POINT_DROPOUT_RECONNECT:
        events = build_source_scenario_events(
            source_id=source_id,
            frame_count=frames,
            max_rollout_steps=1,
            global_seed=global_seed,
        )
        for event in events:
            raw_start = int(event["dropout_start"])
            start = max(0, raw_start)
            stop = min(frames, raw_start + int(event["dropout_duration"]))
            if stop <= start:
                continue
            dropped = np.asarray(event["dropped_trackers"], dtype=np.int64)
            measured[np.ix_(np.arange(start, stop), dropped)] = False

    validate_tracker_states(configured, measured)
    d_off, d_on = compute_tracker_durations(configured, measured)
    hard = compute_hard_rotation_state_np(
        configured,
        measured,
        d_on,
        config=reliability_config,
    )
    return TrackerTimeline(configured, measured, d_off, d_on, hard)


def isolated_condition_eval_mask(timeline: TrackerTimeline, condition: str) -> np.ndarray:
    """只让目标条件帧进入指标，但 rollout 仍保留完整历史与视频。"""

    scenario = str(condition)
    labels = np.asarray(
        [classify_tracker_frame(timeline, index) for index in range(len(timeline.configured))]
    )
    mask = labels == scenario
    if not np.any(mask):
        raise RuntimeError(f"独立条件 {scenario} 没有产生可评估帧。")
    return mask


def _isolated_transition_frame(frame_count: int) -> int:
    # 中点切换让前置条件充分填满 60 帧历史，后置条件也有足够观察区间。
    lower = min(REALTIME_POSE_TARGET_START, max(0, int(frame_count) - 1))
    upper = max(lower, int(frame_count) - RECOVERY_WINDOW_FRAMES)
    return min(max(int(frame_count) // 2, lower), upper)


def compute_tracker_durations(
    configured: np.ndarray,
    measured_valid: np.ndarray,
    cap: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    configured, measured = validate_tracker_states(configured, measured_valid)
    cap = int(cap)
    if cap <= 0:
        raise ValueError("duration cap 必须大于 0。")
    d_off = np.zeros(configured.shape, dtype=np.uint8)
    d_on = np.zeros(configured.shape, dtype=np.uint8)
    previous_off = np.zeros(TRACKER_COUNT, dtype=np.int64)
    previous_on = np.zeros(TRACKER_COUNT, dtype=np.int64)
    for frame_index in range(configured.shape[0]):
        valid = configured[frame_index] & measured[frame_index]
        missing = configured[frame_index] & ~measured[frame_index]
        current_off = np.zeros(TRACKER_COUNT, dtype=np.int64)
        current_on = np.zeros(TRACKER_COUNT, dtype=np.int64)
        current_off[missing] = np.minimum(previous_off[missing] + 1, cap)
        current_on[valid] = np.minimum(previous_on[valid] + 1, cap)
        d_off[frame_index] = current_off.astype(np.uint8)
        d_on[frame_index] = current_on.astype(np.uint8)
        previous_off, previous_on = current_off, current_on
    return d_off, d_on


def classify_tracker_window(configured: np.ndarray, measured_valid: np.ndarray) -> str | None:
    configured, measured = validate_tracker_states(configured, measured_valid)
    if configured.shape[0] != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"场景分类固定读取 {REALTIME_POSE_SEQ_LEN} 帧。")
    missing = configured & ~measured
    if np.any(missing):
        active_counts = missing.sum(axis=1)
        if np.any((active_counts != 0) & (active_counts != 2)):
            return None
        current_missing = active_counts[-1] == 2
        reconnect_rows = np.flatnonzero((active_counts[:-1] == 2) & (active_counts[1:] == 0)) + 1
        recently_reconnected = bool(len(reconnect_rows) and 0 <= REALTIME_POSE_TARGET_START - reconnect_rows[-1] < 15)
        return SCENARIO_TWO_POINT_DROPOUT_RECONNECT if current_missing or recently_reconnected else None

    transitions = np.flatnonzero(np.any(configured[1:] != configured[:-1], axis=1))
    if len(transitions) > 1:
        return None
    if len(transitions) == 1:
        post_frame = int(transitions[0] + 1)
        if not 0 <= REALTIME_POSE_TARGET_START - post_frame < RECOVERY_WINDOW_FRAMES:
            return None
        before = int(configured[post_frame - 1].sum())
        after = int(configured[post_frame].sum())
        if before == 3 and after == 6:
            return SCENARIO_THREE_TO_SIX
        if before == 6 and after == 3:
            return SCENARIO_SIX_TO_THREE
        return None
    if np.array_equal(configured[0], FIXED_SIX_CONFIG):
        return SCENARIO_FIXED_SIX
    if np.array_equal(configured[0], FIXED_THREE_CONFIG):
        return SCENARIO_FIXED_THREE
    return None


def classify_tracker_frame(
    timeline: TrackerTimeline,
    frame_index: int,
    recovery_window_frames: int = RECOVERY_WINDOW_FRAMES,
) -> str:
    """按当前状态和最近事件分类长序列帧，配置切换与掉线重连使用不同状态边沿。"""

    index = int(frame_index)
    window = int(recovery_window_frames)
    frame_count = int(timeline.configured.shape[0])
    if not 0 <= index < frame_count:
        raise IndexError(f"frame_index 必须在 [0,{frame_count})，实际为 {index}")
    if window <= 0:
        raise ValueError("recovery_window_frames 必须大于 0。")

    current_configured = timeline.configured[index]
    current_measured = timeline.measured_valid[index]
    current_missing = current_configured & ~current_measured
    if int(current_missing.sum()) == 2:
        return SCENARIO_TWO_POINT_DROPOUT_RECONNECT

    start = max(1, index - window + 1)
    configured_before = timeline.configured[start - 1 : index]
    configured_after = timeline.configured[start : index + 1]
    measured_before = timeline.measured_valid[start - 1 : index]
    measured_after = timeline.measured_valid[start : index + 1]

    # 配置变化属于 three_to_six / six_to_three。必须先识别它，不能把新配置
    # Tracker 的 measured=False→True 当成掉线后的重连。
    transition_rows = np.flatnonzero(np.any(configured_after != configured_before, axis=1))
    if len(transition_rows):
        transition_frame = start + int(transition_rows[-1])
        before_count = int(timeline.configured[transition_frame - 1].sum())
        after_count = int(timeline.configured[transition_frame].sum())
        if before_count == 3 and after_count == 6:
            return SCENARIO_THREE_TO_SIX
        if before_count == 6 and after_count == 3:
            return SCENARIO_SIX_TO_THREE

    # 真正的重连要求 Tracker 在前后两帧都保持 configured，并且上一帧处于
    # configured-but-missing 状态；这会排除配置新增产生的有效性上升沿。
    previous_missing = configured_before & ~measured_before
    reconnect = previous_missing & configured_after & measured_after
    if np.any(reconnect):
        return SCENARIO_TWO_POINT_DROPOUT_RECONNECT

    if np.array_equal(current_configured, FIXED_SIX_CONFIG):
        return SCENARIO_FIXED_SIX
    if np.array_equal(current_configured, FIXED_THREE_CONFIG):
        return SCENARIO_FIXED_THREE
    raise ValueError(f"当前帧不是受支持的三点或六点配置：{current_configured.tolist()}")


def candidate_starts_by_scenario(
    timeline: TrackerTimeline,
    seq_len: int = REALTIME_POSE_SEQ_LEN,
) -> dict[str, list[int]]:
    if int(seq_len) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"当前任务固定 seq_len={REALTIME_POSE_SEQ_LEN}")
    result = {category: [] for category in TRACKER_PATTERN_CATEGORIES}
    for start in range(timeline.configured.shape[0] - int(seq_len) + 1):
        category = classify_tracker_window(
            timeline.configured[start : start + seq_len],
            timeline.measured_valid[start : start + seq_len],
        )
        if category is not None:
            result[category].append(start)
    return result


def sample_balanced_starts(
    candidates: dict[str, list[int]],
    samples_per_category: int | dict[str, int],
    rng: np.random.Generator,
) -> list[tuple[str, int]]:
    counts = (
        {category: int(samples_per_category.get(category, 0)) for category in TRACKER_PATTERN_CATEGORIES}
        if isinstance(samples_per_category, dict)
        else {category: int(samples_per_category) for category in TRACKER_PATTERN_CATEGORIES}
    )
    if any(value < 0 for value in counts.values()) or not any(counts.values()):
        raise ValueError("每类场景样本数必须非负且至少一类大于零。")
    selected: list[tuple[str, int]] = []
    for category in TRACKER_PATTERN_CATEGORIES:
        starts = candidates.get(category, [])
        count = counts[category]
        if count <= 0 or not starts:
            continue
        indices = rng.choice(np.asarray(starts), size=count, replace=len(starts) < count)
        selected.extend((category, int(value)) for value in indices.tolist())
    rng.shuffle(selected)
    return selected


def _materialize_plan_states(plan: dict, absolute_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scenario = str(plan["scenario"])
    if scenario in (SCENARIO_FIXED_SIX, SCENARIO_TWO_POINT_DROPOUT_RECONNECT):
        configured = np.repeat(FIXED_SIX_CONFIG[None], len(absolute_frames), axis=0)
    elif scenario == SCENARIO_FIXED_THREE:
        configured = np.repeat(FIXED_THREE_CONFIG[None], len(absolute_frames), axis=0)
    elif scenario in (SCENARIO_THREE_TO_SIX, SCENARIO_SIX_TO_THREE):
        before = FIXED_THREE_CONFIG if scenario == SCENARIO_THREE_TO_SIX else FIXED_SIX_CONFIG
        after = FIXED_SIX_CONFIG if scenario == SCENARIO_THREE_TO_SIX else FIXED_THREE_CONFIG
        configured = np.repeat(before[None], len(absolute_frames), axis=0)
        configured[absolute_frames >= int(plan["transition_frame"])] = after
    else:
        raise ValueError(f"未知 Tracker 场景：{scenario}")
    measured = configured.copy()
    if scenario == SCENARIO_TWO_POINT_DROPOUT_RECONNECT:
        start = int(plan["dropout_start"])
        stop = start + int(plan["dropout_duration"])
        rows = np.flatnonzero((absolute_frames >= start) & (absolute_frames < stop))
        dropped = np.asarray(plan["dropped_trackers"], dtype=np.int64)
        if dropped.shape != (2,) or HEAD_TRACKER_INDEX in dropped or len(np.unique(dropped)) != 2:
            raise ValueError("两点掉线必须选择两个不同的非 Head Tracker。")
        measured[np.ix_(rows, dropped)] = False
    validate_tracker_states(configured, measured)
    return configured, measured
