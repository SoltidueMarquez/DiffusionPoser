"""RealtimePose 双层 rollout 训练的单一默认配置。"""

from __future__ import annotations


REALTIME_ROLLOUT_V3_DEFAULTS: dict[str, float | int] = {
    "short_rollout_prob": 0.50,
    "long_rollout_prob": 0.25,
    "short_rollout_loss_weight": 0.50,
    "long_rollout_loss_weight": 0.50,
    "long_rollout_phase1_steps": 500,
    "long_rollout_phase2_steps": 1500,
    "long_rollout_phase1_max_horizon": 2,
    "long_rollout_phase2_max_horizon": 4,
    "long_rollout_transition_prob": 0.50,
    "long_rollout_smooth_l1_beta": 1.0,
    "rollout_ddim_steps": 10,
}


def long_rollout_max_horizon(
    *,
    global_step: int,
    rollout_steps: int,
    phase1_steps: int,
    phase2_steps: int,
    phase1_max_horizon: int,
    phase2_max_horizon: int,
) -> int:
    """按当前训练 step 返回允许的 H 上限；long rollout 的最小 H 固定为 2。"""

    available_max = int(rollout_steps) - 1
    if available_max < 2:
        return 0
    if int(global_step) < int(phase1_steps):
        configured_max = int(phase1_max_horizon)
    elif int(global_step) < int(phase2_steps):
        configured_max = int(phase2_max_horizon)
    else:
        configured_max = available_max
    return max(2, min(available_max, configured_max))
