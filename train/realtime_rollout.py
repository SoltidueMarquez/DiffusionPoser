"""RealtimePose 单进程自回归课程与学习率调度。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


REALTIME_ROLLOUT_DEFAULTS: dict[str, float | int] = {
    "short_rollout_prob": 0.50,
    "long_rollout_prob": 0.25,
    "short_rollout_loss_weight": 0.50,
    "long_rollout_loss_weight": 0.50,
    "rollout_h1_start_step": 30_000,
    "rollout_h2_start_step": 60_000,
    "rollout_h4_start_step": 70_000,
    "rollout_h8_start_step": 90_000,
    "rollout_prob_ramp_steps": 10_000,
    "rollout_max_horizon_prob": 0.50,
    "long_rollout_transition_prob": 0.50,
    "long_rollout_smooth_l1_beta": 1.0,
    "rollout_ddim_steps": 10,
}


REALTIME_LR_DEFAULTS: dict[str, float | int] = {
    "lr_warmup_start": 1e-6,
    "lr_warmup_steps": 2_000,
    "lr_min": 1e-5,
}


TRAINING_SCHEDULE_SIGNATURE_FIELDS = (
    "num_steps",
    "rollout_steps",
    *REALTIME_ROLLOUT_DEFAULTS.keys(),
    "lr",
    *REALTIME_LR_DEFAULTS.keys(),
)


@dataclass(frozen=True)
class RolloutCurriculumState:
    """某个 global step 唯一对应的 rollout 训练状态。"""

    phase: str
    active_rollout_steps: int
    max_horizon: int
    short_prob: float
    long_prob: float
    max_horizon_prob: float

    def to_log_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


def _linear_ramp(global_step: int, start_step: int, ramp_steps: int) -> float:
    if int(global_step) < int(start_step):
        return 0.0
    if int(ramp_steps) <= 0:
        return 1.0
    return min(max((int(global_step) - int(start_step)) / float(ramp_steps), 0.0), 1.0)


def validate_rollout_schedule(
    *,
    rollout_steps: int,
    h1_start_step: int,
    h2_start_step: int,
    h4_start_step: int,
    h8_start_step: int,
    ramp_steps: int,
) -> None:
    if not 1 <= int(rollout_steps) <= 9:
        raise ValueError("rollout_steps 必须在 [1,9]，其中 9 表示 base+H1～H8。")
    starts = [int(h1_start_step), int(h2_start_step), int(h4_start_step), int(h8_start_step)]
    if any(value < 0 for value in starts):
        raise ValueError("rollout horizon 起始 step 不能为负数。")
    if starts != sorted(starts):
        raise ValueError("rollout horizon 起始 step 必须满足 H1 <= H2 <= H4 <= H8。")
    if int(ramp_steps) < 0:
        raise ValueError("rollout_prob_ramp_steps 不能为负数。")


def rollout_curriculum_state(
    *,
    global_step: int,
    rollout_steps: int,
    short_rollout_prob: float,
    long_rollout_prob: float,
    rollout_h1_start_step: int,
    rollout_h2_start_step: int,
    rollout_h4_start_step: int,
    rollout_h8_start_step: int,
    rollout_prob_ramp_steps: int,
    rollout_max_horizon_prob: float,
) -> RolloutCurriculumState:
    """按 global step 返回唯一课程状态；恢复训练不会重新从 base 开始。"""

    validate_rollout_schedule(
        rollout_steps=rollout_steps,
        h1_start_step=rollout_h1_start_step,
        h2_start_step=rollout_h2_start_step,
        h4_start_step=rollout_h4_start_step,
        h8_start_step=rollout_h8_start_step,
        ramp_steps=rollout_prob_ramp_steps,
    )
    for name, value in (
        ("short_rollout_prob", short_rollout_prob),
        ("long_rollout_prob", long_rollout_prob),
        ("rollout_max_horizon_prob", rollout_max_horizon_prob),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} 必须在 [0,1]，实际为 {value}")

    step = max(0, int(global_step))
    available_max = int(rollout_steps) - 1
    if step < int(rollout_h1_start_step) or available_max <= 0:
        return RolloutCurriculumState("base", 1, 0, 0.0, 0.0, 0.0)

    short_prob = float(short_rollout_prob) * _linear_ramp(
        step, int(rollout_h1_start_step), int(rollout_prob_ramp_steps)
    )
    if step < int(rollout_h2_start_step) or available_max < 2:
        return RolloutCurriculumState("h1", min(int(rollout_steps), 2), 1, short_prob, 0.0, 1.0)

    long_prob = float(long_rollout_prob) * _linear_ramp(
        step, int(rollout_h2_start_step), int(rollout_prob_ramp_steps)
    )
    if step < int(rollout_h4_start_step) or available_max < 4:
        max_horizon = min(2, available_max)
        return RolloutCurriculumState("h2", max_horizon + 1, max_horizon, short_prob, long_prob, 1.0)
    if step < int(rollout_h8_start_step) or available_max < 8:
        max_horizon = min(4, available_max)
        return RolloutCurriculumState(
            "h4",
            max_horizon + 1,
            max_horizon,
            short_prob,
            long_prob,
            float(rollout_max_horizon_prob),
        )

    max_horizon = min(8, available_max)
    max_prob = float(rollout_max_horizon_prob) * _linear_ramp(
        step, int(rollout_h8_start_step), int(rollout_prob_ramp_steps)
    )
    return RolloutCurriculumState(
        "h8",
        max_horizon + 1,
        max_horizon,
        short_prob,
        long_prob,
        max_prob,
    )


def rollout_curriculum_state_from_args(args, global_step: int) -> RolloutCurriculumState:
    return rollout_curriculum_state(
        global_step=global_step,
        rollout_steps=int(getattr(args, "rollout_steps", 1)),
        short_rollout_prob=float(
            getattr(args, "short_rollout_prob", REALTIME_ROLLOUT_DEFAULTS["short_rollout_prob"])
        ),
        long_rollout_prob=float(
            getattr(args, "long_rollout_prob", REALTIME_ROLLOUT_DEFAULTS["long_rollout_prob"])
        ),
        rollout_h1_start_step=int(
            getattr(args, "rollout_h1_start_step", REALTIME_ROLLOUT_DEFAULTS["rollout_h1_start_step"])
        ),
        rollout_h2_start_step=int(
            getattr(args, "rollout_h2_start_step", REALTIME_ROLLOUT_DEFAULTS["rollout_h2_start_step"])
        ),
        rollout_h4_start_step=int(
            getattr(args, "rollout_h4_start_step", REALTIME_ROLLOUT_DEFAULTS["rollout_h4_start_step"])
        ),
        rollout_h8_start_step=int(
            getattr(args, "rollout_h8_start_step", REALTIME_ROLLOUT_DEFAULTS["rollout_h8_start_step"])
        ),
        rollout_prob_ramp_steps=int(
            getattr(args, "rollout_prob_ramp_steps", REALTIME_ROLLOUT_DEFAULTS["rollout_prob_ramp_steps"])
        ),
        rollout_max_horizon_prob=float(
            getattr(args, "rollout_max_horizon_prob", REALTIME_ROLLOUT_DEFAULTS["rollout_max_horizon_prob"])
        ),
    )


def scheduled_learning_rate(
    *,
    global_step: int,
    num_steps: int,
    lr: float,
    lr_warmup_start: float,
    lr_warmup_steps: int,
    lr_min: float,
) -> float:
    """先线性 warmup，再从峰值连续 cosine decay 到 lr_min。"""

    peak = float(lr)
    start = float(lr_warmup_start)
    minimum = float(lr_min)
    warmup_steps = int(lr_warmup_steps)
    total_steps = int(num_steps)
    step = min(max(int(global_step), 0), max(total_steps, 0))
    if peak <= 0.0 or start <= 0.0 or minimum <= 0.0:
        raise ValueError("lr、lr_warmup_start 和 lr_min 都必须大于 0。")
    if warmup_steps < 0 or total_steps <= 0:
        raise ValueError("学习率调度要求 lr_warmup_steps >= 0 且 num_steps > 0。")
    effective_warmup_steps = min(warmup_steps, total_steps)
    if effective_warmup_steps > 0 and step < effective_warmup_steps:
        return start + (peak - start) * (step / float(effective_warmup_steps))
    decay_steps = total_steps - effective_warmup_steps
    if decay_steps <= 0:
        return peak
    progress = min(max((step - effective_warmup_steps) / float(decay_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (peak - minimum) * cosine


def training_schedule_payload(args) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for name in TRAINING_SCHEDULE_SIGNATURE_FIELDS:
        value = getattr(args, name)
        if isinstance(value, bool):
            payload[name] = int(value)
        elif isinstance(value, int):
            payload[name] = int(value)
        else:
            payload[name] = float(value)
    return payload


def training_schedule_signature(args) -> str:
    encoded = json.dumps(
        training_schedule_payload(args),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sampling_epoch_for_global_step(
    *,
    global_step: int,
    batches_per_epoch: int,
    phase_start_steps: tuple[int, ...],
    resume_mid_epoch: bool,
) -> int:
    """从课程区间和 global step 纯函数派生在线采样 epoch。"""

    batches = int(batches_per_epoch)
    if batches <= 0:
        raise ValueError("batches_per_epoch 必须为正数。")
    step = max(0, int(global_step))
    starts = sorted({0, *(max(0, int(value)) for value in phase_start_steps)})
    current_index = max(index for index, value in enumerate(starts) if value <= step)
    completed_epochs = 0
    for index in range(current_index):
        interval = starts[index + 1] - starts[index]
        completed_epochs += (interval + batches - 1) // batches
    offset = step - starts[current_index]
    if resume_mid_epoch and offset > 0:
        return completed_epochs + (offset + batches - 1) // batches
    return completed_epochs + offset // batches
