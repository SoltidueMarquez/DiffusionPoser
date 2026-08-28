from __future__ import annotations

import numpy as np
import pytest

from data_loaders.sensor_masking import (
    CORE_THREE_AVAILABLE,
    CORE_TRACKER_INDICES,
    OPTIONAL_TRACKER_INDICES,
)
from sample import evaluate_tracker_pair_reconnection as pair_reconnection


def test_pair_reconnection_protocols_cover_all_optional_pairs() -> None:
    protocols = pair_reconnection.build_tracker_pair_reconnection_protocols(
        stage_frames=150
    )

    assert [protocol.name for protocol in protocols] == [
        "reconnect_hip_and_left_foot",
        "reconnect_hip_and_right_foot",
        "reconnect_left_foot_and_right_foot",
    ]
    for protocol in protocols:
        assert protocol.warmup_tracker_available == CORE_THREE_AVAILABLE
        assert [sum(stage.tracker_available) for stage in protocol.stages] == [3, 5]
        assert all(
            all(stage.tracker_available[index] for index in CORE_TRACKER_INDICES)
            for stage in protocol.stages
        )
        before = np.asarray(protocol.stages[0].tracker_available, dtype=bool)
        after = np.asarray(protocol.stages[1].tracker_available, dtype=bool)
        changed = np.flatnonzero(~before & after)
        assert changed.shape == (2,)
        assert set(changed.tolist()).issubset(OPTIONAL_TRACKER_INDICES)


def test_pair_reconnection_schedule_switches_exactly_two_trackers() -> None:
    tracker_indices = tuple(
        pair_reconnection.RECONNECT_TRACKER_NAME_TO_INDEX[name]
        for name in ("hip", "right_foot")
    )
    schedule = pair_reconnection.build_tracker_pair_reconnection_schedule(
        scored_frame_count=300,
        reconnect_tracker_indices=tracker_indices,
        stage_frames=150,
    )

    assert schedule.stage_indices.tolist() == [0] * 150 + [1] * 150
    assert schedule.tracker_available[:150].sum(axis=1).tolist() == [3] * 150
    assert schedule.tracker_available[150:].sum(axis=1).tolist() == [5] * 150
    assert not schedule.tracker_available[:150, list(tracker_indices)].any()
    assert schedule.tracker_available[150:, list(tracker_indices)].all()


def test_pair_reconnection_rejects_invalid_pair_and_selection() -> None:
    hip_index = pair_reconnection.RECONNECT_TRACKER_NAME_TO_INDEX["hip"]
    with pytest.raises(ValueError, match="两个不同"):
        pair_reconnection.build_tracker_pair_reconnection_schedule(
            scored_frame_count=300,
            reconnect_tracker_indices=(hip_index, hip_index),
            stage_frames=150,
        )
    with pytest.raises(ValueError, match="不能与具体组合"):
        pair_reconnection.resolve_reconnect_pair_names(
            ["all", "hip_and_left_foot"]
        )
    with pytest.raises(ValueError, match="不能包含重复"):
        pair_reconnection.resolve_reconnect_pair_names(
            ["hip_and_left_foot", "hip_and_left_foot"]
        )
