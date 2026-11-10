"""掉线局部动态与 GT 接触诊断；所有输入均为共享骨架的世界位置。"""

from __future__ import annotations

import numpy as np

from data_loaders.realtime_pose_kinematics import JOINT_INDEX


WRIST_JOINTS = (JOINT_INDEX["left_wrist"], JOINT_INDEX["right_wrist"])
FOOT_JOINTS = (JOINT_INDEX["left_foot"], JOINT_INDEX["right_foot"])


def validate_positions(values: np.ndarray) -> np.ndarray:
    """位置 `[T,J,3]`，至少包含 SMPL22，单位米、y 向上。"""
    positions = np.asarray(values, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1] < 22 or positions.shape[2] != 3:
        raise ValueError("位置必须为 [T,J>=22,3]。")
    if not np.isfinite(positions).all():
        raise ValueError("位置包含非有限数值。")
    return positions


def frame_velocities(positions: np.ndarray, fps: float) -> np.ndarray:
    """后向差分的第 t 项对齐源帧 t；首帧无速度，记为 NaN。"""
    positions = validate_positions(positions)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps 必须为正数。")
    velocities = np.full_like(positions, np.nan)
    velocities[1:] = np.diff(positions, axis=0) * fps
    return velocities


def require_aligned_frames(actual: np.ndarray, expected: np.ndarray) -> None:
    """拒绝静默截短或错一帧，避免将时间偏差当成模型延迟。"""
    if not np.array_equal(actual, expected):
        raise ValueError("方法之间的 source_frames 不一致。")


def decompose_auj(predicted_curve, target_curve) -> dict[str, float]:
    predicted = np.asarray(predicted_curve, dtype=np.float64)
    target = np.asarray(target_curve, dtype=np.float64)
    if predicted.ndim != 1 or predicted.shape != target.shape or not len(predicted):
        raise ValueError("jerk 曲线必须是等长非空一维数组。")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("jerk 曲线包含非有限数值。")
    difference = predicted - target
    above = float(np.maximum(difference, 0).sum())
    below = float(np.maximum(-difference, 0).sum())
    return {"auj": above + below, "above_gt": above, "below_gt": below,
            "peak_jerk": float(predicted.max())}


def transition_events(gaps, frame_count: int, fps: float = 30.0, score_start: int = 30):
    """gap 使用 source 帧，复用官方 transition 的有效事件边界。"""
    window = int(0.5 * fps)
    events = []
    for hand, intervals in enumerate(gaps):
        for start, end in intervals:
            start, end = int(start), int(end)
            if start < 0 or end > frame_count or start >= end:
                raise ValueError("gap 超出 source 时间轴或区间为空。")
            common = {"hand": hand, "gap_start": start, "gap_end": end}
            if start - score_start - 3 > 0 and end - start >= window and start + window <= frame_count:
                events.append({**common, "phase": "tracking_to_synthesis", "start": start, "stop": start + window})
            if end - score_start - 3 > 0 and end + window < frame_count:
                events.append({**common, "phase": "synthesis_to_tracking", "start": end, "stop": end + window})
    return events


def gt_contact_mask(target_positions, fps: float = 30.0):
    """GT 接触代理 `[T,2]`；阈值对所有方法共用，不从预测估计地面。"""
    target = validate_positions(target_positions)
    feet = target[:, FOOT_JOINTS]
    floor_y = float(np.percentile(feet[..., 1], 5))
    speed = np.linalg.norm(frame_velocities(target, fps)[:, FOOT_JOINTS], axis=-1)
    contact = (feet[..., 1] <= floor_y + 0.05) & (speed <= 0.1)
    return contact, floor_y


def wrist_event_metrics(predicted_positions, target_positions, event, fps=30.0):
    predicted = validate_positions(predicted_positions)
    target = validate_positions(target_positions)
    if predicted.shape != target.shape:
        raise ValueError("预测与 GT 的形状必须一致。")
    start, stop = event["start"], event["stop"]
    if not 1 <= start < stop <= len(target):
        raise ValueError("腕部窗口缺少前一帧速度上下文。")
    joint = WRIST_JOINTS[event["hand"]]
    # 仅计算对应窗口，避免每个 event 重复计算整条序列。
    pred = predicted[start:stop, joint]
    gt = target[start:stop, joint]
    pred_v = (pred - predicted[start - 1:stop - 1, joint]) * fps
    gt_v = (gt - target[start - 1:stop - 1, joint]) * fps
    return {
        "wrist_mpjpe_cm": float(np.linalg.norm(pred - gt, axis=-1).mean() * 100),
        "wrist_mpjve_cm_s": float(np.linalg.norm(pred_v - gt_v, axis=-1).mean() * 100),
        "pred_speed_m_s": float(np.linalg.norm(pred_v, axis=-1).mean()),
        "gt_speed_m_s": float(np.linalg.norm(gt_v, axis=-1).mean()),
    }


def aggregate_wrist_events(values):
    if not values:
        return {"event_count": 0, "wrist_mpjpe_cm": None, "wrist_mpjve_cm_s": None,
                "speed_ratio": None, "pred_speed_m_s": None, "gt_speed_m_s": None}
    result = {key: float(np.mean([item[key] for item in values])) for key in values[0]}
    result["event_count"] = len(values)
    result["speed_ratio"] = result["pred_speed_m_s"] / result["gt_speed_m_s"] if result["gt_speed_m_s"] > 1e-8 else None
    return result


def contact_statistics(predicted_positions, target_positions, contact_mask, selected_frames, fps=30.0):
    """阶段内取帧并集，避免两只手的重叠掉线重复计数脚部接触。"""
    pred = validate_positions(predicted_positions)
    gt = validate_positions(target_positions)
    contact = np.asarray(contact_mask, dtype=bool)
    selected = np.asarray(selected_frames, dtype=bool)
    if pred.shape != gt.shape or contact.shape != (len(gt), 2) or selected.shape != (len(gt),):
        raise ValueError("接触掩码与轨迹的形状不匹配。")
    mask = contact & selected[:, None]
    pred_feet = pred[:, FOOT_JOINTS]
    height = np.maximum(pred_feet[..., 1] - gt[:, FOOT_JOINTS, 1], 0)
    velocity = frame_velocities(pred, fps)[:, FOOT_JOINTS]
    horizontal_speed = np.linalg.norm(velocity[..., [0, 2]], axis=-1)
    return {
        "contact_samples": int(mask.sum()), "eligible_samples": int(selected.sum() * 2),
        "positive_height_sum_cm": float(height[mask].sum() * 100),
        "horizontal_speed_sum_cm_s": float(horizontal_speed[mask].sum() * 100),
    }


def aggregate_contact_statistics(values):
    totals = {key: sum(item[key] for item in values) for key in (
        "contact_samples", "eligible_samples", "positive_height_sum_cm", "horizontal_speed_sum_cm_s",
    )}
    count, denominator = totals["contact_samples"], totals["eligible_samples"]
    return {
        "contact_samples": count, "eligible_samples": denominator,
        "contact_coverage": count / denominator if denominator else None,
        "foot_positive_height_cm": totals["positive_height_sum_cm"] / count if count else None,
        "foot_horizontal_speed_cm_s": totals["horizontal_speed_sum_cm_s"] / count if count else None,
    }


def select_video_events(events):
    """按预先指定的 GT 动态排序选例，并明确加入一个 Ours 最差事件。"""
    eligible = [e for e in events if e["phase"] == "tracking_to_synthesis"
                and e["gap_start"] - 15 >= e["generated_start_frame"]
                and e["gap_end"] + 15 <= e["frame_count"]]
    # 双手若同时掉线且区间相同，视频实际上是同一片段，不能占两个选例名额。
    clip_key = lambda e: (e["sequence"], e["gap_start"], e["gap_end"])
    if len({clip_key(e) for e in eligible}) < 4:
        raise ValueError("不足四个具备完整前后上下文的掉线事件。")
    tie = lambda e: (e["sequence"], e["gap_start"], e["hand"])
    ordered = sorted(eligible, key=lambda e: (e["gt_speed_m_s"], *tie(e)))
    selected, used = [], set()
    for label, fraction in (("GT speed median", 0.5), ("GT speed p90", 0.9), ("GT speed maximum", 1.0)):
        index = int(round(fraction * (len(ordered) - 1)))
        candidates = ordered[index:] + ordered[:index]
        chosen = next(e for e in candidates if clip_key(e) not in used)
        used.add(clip_key(chosen))
        selected.append({**chosen, "selection": label})
    worst = sorted(eligible, key=lambda e: (-e["ours_wrist_mpjpe_cm"], *tie(e)))
    chosen = next(e for e in worst if clip_key(e) not in used)
    selected.append({**chosen, "selection": "Worst remaining Ours wrist error"})
    return selected
