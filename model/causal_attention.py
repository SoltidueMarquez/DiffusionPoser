from __future__ import annotations

import torch


def build_frame_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """构造帧级 causal mask，True 表示该 query 不能读取对应 key。"""
    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


def build_target_dit_causal_mask(
    seq_len: int,
    tracker_count: int,
    target_frame: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    构造 target_dit 的 token 级可见性。

    token 顺序固定为 [target, sensor0..sensorN, frame0..frameT]。
    frame token 只读过去/当前帧；sensor 和 target token 可读当前目标帧之前的历史帧。
    """
    seq_len = int(seq_len)
    tracker_count = int(tracker_count)
    target_frame = int(target_frame)
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if tracker_count <= 0:
        raise ValueError(f"tracker_count must be positive, got {tracker_count}")
    if target_frame < 0 or target_frame >= seq_len:
        raise ValueError(f"target_frame must be in [0, {seq_len}), got {target_frame}")

    target_index = 0
    sensor_start = 1
    frame_start = sensor_start + tracker_count
    total_tokens = frame_start + seq_len

    mask = torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device)
    visible_frame_end = frame_start + target_frame + 1

    # target token 保留自身扩散状态，同时读取当前 tracker 条件和历史/当前帧。
    mask[target_index, target_index] = False
    mask[target_index, sensor_start:frame_start] = False
    mask[target_index, frame_start:visible_frame_end] = False

    # sensor token 作为当前观测条件，不反向读取 target token 或目标帧之后的信息。
    mask[sensor_start:frame_start, sensor_start:frame_start] = False
    mask[sensor_start:frame_start, frame_start:visible_frame_end] = False

    # frame token 不能读取 target/sensor token，只能沿时间读取过去和当前帧。
    mask[frame_start:, frame_start:] = build_frame_causal_mask(seq_len, device=device)
    return mask
