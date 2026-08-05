from __future__ import annotations

from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_stationary_source_field_is_available_for_contact_supervision():
    source = build_toy_realtime_source(frame_count=70)
    assert "stationary_prob_5" in source
