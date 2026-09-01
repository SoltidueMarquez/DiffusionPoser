from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sample.postprocess_unity_tracker_recording import (
    TrackerGap,
    clean_tracker_recording,
    parse_tracker_gap,
    repair_tracker_gaps,
)


def test_parse_tracker_gap_uses_half_open_frame_range() -> None:
    assert parse_tracker_gap("10:15:4") == TrackerGap(10, 15, 4)
    with pytest.raises(ValueError, match="三个字段必须都是整数"):
        parse_tracker_gap("1.5:10:4")
    with pytest.raises(ValueError, match="tracker_index"):
        parse_tracker_gap("10:15:6")


def test_repair_tracker_gap_only_changes_selected_tracker_and_frames() -> None:
    frame_count = 20
    times = np.arange(frame_count, dtype=np.float64) / 72.0
    positions = np.zeros((frame_count, 6, 3), dtype=np.float64)
    positions[:, 4, 0] = np.linspace(0.0, 1.0, frame_count)
    positions[8:12, 4, 1] = np.asarray([0.5, -0.4, 0.7, -0.3])
    identity = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rotations = np.broadcast_to(identity, (frame_count, 6, 4)).copy()
    rotations[8:12, 4] = Rotation.from_rotvec([0.0, 1.2, 0.0]).as_quat()

    repaired_positions, repaired_rotations, changed = repair_tracker_gaps(
        times=times,
        positions=positions,
        rotations_xyzw=rotations,
        gaps=(TrackerGap(8, 12, 4),),
    )

    assert changed[8:12, 4].all()
    assert not changed[:8].any()
    assert not changed[12:].any()
    np.testing.assert_array_equal(repaired_positions[:, 5], positions[:, 5])
    np.testing.assert_array_equal(repaired_rotations[:, 5], rotations[:, 5])
    np.testing.assert_array_equal(repaired_positions[:8, 4], positions[:8, 4])
    np.testing.assert_array_equal(repaired_positions[12:, 4], positions[12:, 4])
    assert np.abs(repaired_positions[8:12, 4, 1]).max() == pytest.approx(0.0)
    repaired_angles = np.linalg.norm(
        Rotation.from_quat(repaired_rotations[8:12, 4]).as_rotvec(), axis=-1
    )
    assert repaired_angles.max() == pytest.approx(0.0)
    np.testing.assert_allclose(
        np.linalg.norm(repaired_rotations, axis=-1),
        1.0,
        atol=1e-7,
    )


def test_clean_tracker_recording_preserves_unselected_json_values(tmp_path) -> None:
    frame_count = 8
    frames = []
    for frame_index in range(frame_count):
        rotations = [[0.0, 0.0, 0.0, 2.0] for _ in range(6)]
        rotations[4] = [0.0, float(frame_index), 0.0, 2.0]
        frames.append(
            {
                "time": frame_index / 72.0,
                "positions": [
                    [float(frame_index), float(tracker_index), 0.0]
                    for tracker_index in range(6)
                ],
                "rotations": rotations,
                "available": [True, True, True, False, True, True],
            }
        )
    input_path = tmp_path / "recording.json"
    output_path = tmp_path / "recording_cleaned.json"
    input_path.write_text(
        json.dumps({"floorY": 0.0, "frames": frames}),
        encoding="utf-8",
    )

    clean_tracker_recording(
        input_path=input_path,
        output_path=output_path,
        gaps=(TrackerGap(3, 5, 4),),
        overwrite=False,
    )

    source = json.loads(input_path.read_text(encoding="utf-8"))
    cleaned = json.loads(output_path.read_text(encoding="utf-8"))
    for frame_index in range(frame_count):
        for tracker_index in range(6):
            if 3 <= frame_index < 5 and tracker_index == 4:
                continue
            assert (
                cleaned["frames"][frame_index]["positions"][tracker_index]
                == source["frames"][frame_index]["positions"][tracker_index]
            )
            assert (
                cleaned["frames"][frame_index]["rotations"][tracker_index]
                == source["frames"][frame_index]["rotations"][tracker_index]
            )
