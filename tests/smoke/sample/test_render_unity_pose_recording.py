from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sample.infer_unity_recording import (
    ResampledTrackerRecording,
    UnityTrackerRecording,
    apply_tracker_availability_overrides,
)
from sample.render_unity_pose_recording import (
    build_front_camera_direction,
    estimate_initial_face_direction,
    load_unity_pose_recording,
    select_tracker_frame_indices,
)


def test_load_unity_pose_recording_preserves_transform_contract(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pose.json"
    frames = []
    for frame_index in range(3):
        frames.append(
            {
                "time": frame_index / 30.0,
                "rootPosition": [0.0, 0.0, 0.0],
                "rootRotation": [0.0, 0.0, 0.0, 1.0],
                "pelvisLocalPosition": [0.0, 0.9, 0.0],
                "localRotations": [[0.0, 0.0, 0.0, 1.0]] * 24,
            }
        )
    path.write_text(json.dumps({"fps": 30, "frames": frames}), encoding="utf-8")

    pose = load_unity_pose_recording(path)

    assert pose.frame_count == 3
    assert pose.fps == 30
    assert pose.local_rotations_xyzw.shape == (3, 24, 4)


def test_tracker_alignment_replays_inference_warmup_policy() -> None:
    frame_count = 50
    times = np.arange(frame_count, dtype=np.float64) / 30.0
    available = np.ones((frame_count, 6), dtype=bool)
    recording = ResampledTrackerRecording(
        times=times,
        positions=np.zeros((frame_count, 6, 3), dtype=np.float32),
        rotations_xyzw=np.broadcast_to(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (frame_count, 6, 4),
        ).copy(),
        rotations_6d=np.zeros((frame_count, 6, 6), dtype=np.float32),
        rotations_world=np.broadcast_to(
            np.eye(3, dtype=np.float32), (frame_count, 6, 3, 3)
        ).copy(),
        available=available,
        floor_y=0.0,
    )
    expected_indices = np.arange(14, frame_count, dtype=np.int64)
    pose_times = times[expected_indices] - times[expected_indices[0]]

    selected = select_tracker_frame_indices(
        recording=recording,
        pose_times=pose_times,
        warmup_frames=3,
    )

    np.testing.assert_array_equal(selected, expected_indices)


def test_ignore_feet_masks_inference_availability_without_mutating_source() -> None:
    source_available = np.ones((4, 6), dtype=bool)
    recording = UnityTrackerRecording(
        times=np.arange(4, dtype=np.float64) / 30.0,
        positions=np.zeros((4, 6, 3), dtype=np.float32),
        rotations_xyzw=np.broadcast_to(
            np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (4, 6, 4),
        ).copy(),
        available=source_available,
        floor_y=0.0,
    )

    masked = apply_tracker_availability_overrides(
        recording,
        ignore_hip=False,
        ignore_feet=True,
    )

    assert recording.available.all()
    assert masked.available[:, :4].all()
    assert not masked.available[:, 4:].any()


def test_front_camera_uses_rendered_mesh_face_direction() -> None:
    face_directions = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.995],
            [-0.1, 0.0, 0.995],
        ],
        dtype=np.float32,
    )

    face = estimate_initial_face_direction(face_directions, reference_frames=3)
    camera = build_front_camera_direction(
        face_direction=face,
        yaw_offset_deg=0.0,
        elevation_deg=0.0,
    )
    rear = build_front_camera_direction(
        face_direction=face,
        yaw_offset_deg=180.0,
        elevation_deg=0.0,
    )

    np.testing.assert_allclose(camera, [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(rear, [0.0, 0.0, -1.0], atol=1e-6)
