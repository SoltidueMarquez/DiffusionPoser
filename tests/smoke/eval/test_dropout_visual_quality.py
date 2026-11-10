import numpy as np
import pytest

from eval.dropout_visual_quality import (
    aggregate_contact_statistics, aggregate_wrist_events, contact_statistics,
    decompose_auj, gt_contact_mask, require_aligned_frames, select_video_events,
    transition_events, wrist_event_metrics,
)


def test_auj_decomposition_distinguishes_missing_and_excess_dynamics():
    result = decompose_auj([1, 5, 2], [3, 2, 2])
    assert result == {"auj": 5, "above_gt": 3, "below_gt": 2, "peak_jerk": 5}
    assert decompose_auj([1, 2], [1, 2])["auj"] == 0
    assert decompose_auj([0, 0], [2, 3])["below_gt"] == 5


def test_wrong_source_frame_alignment_is_rejected():
    require_aligned_frames(np.arange(1, 10), np.arange(1, 10))
    with pytest.raises(ValueError, match="source_frames"):
        require_aligned_frames(np.arange(0, 9), np.arange(1, 10))


def test_transition_events_use_absolute_source_frames_and_half_second():
    events = transition_events([[(40, 60)], [(70, 90)]], 120)
    assert [(e["start"], e["stop"], e["hand"]) for e in events] == [
        (40, 55, 0), (60, 75, 0), (70, 85, 1), (90, 105, 1),
    ]
    # 正式计分区间前三帧不足以构成 jerk，保持官方排除边界。
    assert not [e for e in transition_events([[(32, 50)], []], 70) if e["phase"] == "tracking_to_synthesis"]


def test_wrist_metrics_detect_attenuation_and_ignore_other_hand():
    target = np.zeros((20, 24, 3))
    target[:, 20, 0] = np.arange(20) / 30
    pred = target.copy()
    pred[:, 20, 0] *= 0.5
    pred[:, 21] = 100  # 右腕大误差不得污染左腕事件。
    event = {"start": 5, "stop": 20, "hand": 0}
    metrics = wrist_event_metrics(pred, target, event)
    assert metrics["wrist_mpjve_cm_s"] == pytest.approx(50)
    assert aggregate_wrist_events([metrics])["speed_ratio"] == pytest.approx(0.5)


def test_speed_ratio_uses_ratio_of_means_including_stationary_events():
    a = {"pred_speed_m_s": 0.2, "gt_speed_m_s": 0.0}
    b = {"pred_speed_m_s": 0.8, "gt_speed_m_s": 2.0}
    assert aggregate_wrist_events([a, b])["speed_ratio"] == pytest.approx(0.5)
    assert aggregate_wrist_events([{"pred_speed_m_s": 0.0, "gt_speed_m_s": 0.0}])["speed_ratio"] is None


def test_contact_is_from_gt_and_pools_only_selected_frames():
    target = np.zeros((40, 24, 3))
    target[:, 11, 1] = 0.3  # 右脚高于地面，不计为接触。
    contact, floor = gt_contact_mask(target)
    assert floor == 0
    assert contact[1:, 0].all()
    assert not contact[:, 1].any()
    pred = target.copy()
    pred[:, 10, 1] += 0.1
    pred[:, 10, 0] = np.arange(40) / 30 * 0.2
    selected = np.zeros(40, bool)
    selected[10:25] = True
    selected[20:30] = True  # 重叠区间只计算一次。
    result = aggregate_contact_statistics([contact_statistics(pred, target, contact, selected)])
    assert result["contact_samples"] == 20
    assert result["contact_coverage"] == 0.5
    assert result["foot_positive_height_cm"] == pytest.approx(10)
    assert result["foot_horizontal_speed_cm_s"] == pytest.approx(20)
    empty = aggregate_contact_statistics([contact_statistics(pred, target, np.zeros_like(contact), selected)])
    assert empty["foot_positive_height_cm"] is None


def test_video_selection_is_deterministic_and_distinct():
    events = [dict(phase="tracking_to_synthesis", gap_start=40, gap_end=60,
                   generated_start_frame=11, frame_count=100, sequence=str(i), hand=0,
                   gt_speed_m_s=float(i), ours_wrist_mpjpe_cm=float(10 - i)) for i in range(10)]
    selected = select_video_events(events)
    assert [e["sequence"] for e in selected] == ["4", "8", "9", "0"]
    assert select_video_events(events[::-1]) == selected
