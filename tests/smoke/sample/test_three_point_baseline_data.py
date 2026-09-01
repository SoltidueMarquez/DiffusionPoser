from pathlib import Path

import numpy as np
import pytest

from sample.run_official_agrol_three_point import choose_window_start
from sample.render_three_point_method_comparison import (
    PANEL_HEIGHT,
    PANEL_WIDTH,
    METHOD_ORDER,
    build_method_metrics,
    replay_playback_label,
    select_agrol_motion,
    select_rpm_motion,
    stitch_independent_panel_views,
)
from sample.realtime_pose_smpl_rendering import SmplMeshSequence
from sample.three_point_baseline_data import (
    StreamingMoments,
    baseline_motion_6d_to_rotation_matrices,
    build_sparse_features_from_trackers,
    matrix_to_first_two_columns_6d,
    resample_tracker_signals,
)


def test_streaming_moments_match_numpy_unbiased_std():
    values = np.arange(60, dtype=np.float32).reshape(20, 3)
    moments = StreamingMoments.empty(3)
    moments = moments.update(values[:7])
    moments = moments.update(values[7:])
    mean, std = moments.finalize()

    np.testing.assert_allclose(mean, values.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(std, values.std(axis=0, ddof=1), atol=1e-6)


def test_replay_playback_label_matches_slowdown_factor():
    assert replay_playback_label(2) == "0.5× replay"
    assert replay_playback_label(4) == "0.25× replay"
    with pytest.raises(ValueError, match="必须 >= 1"):
        replay_playback_label(0)


def test_sparse_features_follow_official_field_order():
    positions = np.zeros((2, 3, 3), dtype=np.float32)
    positions[1, :, 0] = np.asarray([1.0, 2.0, 3.0])
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None], 6, axis=0)
    rotations = rotations.reshape(2, 3, 3, 3)

    sparse = build_sparse_features_from_trackers(positions, rotations)

    assert sparse.shape == (1, 54)
    identity_6d = matrix_to_first_two_columns_6d(np.eye(3)[None])[0]
    np.testing.assert_allclose(sparse[0, :18], np.tile(identity_6d, 3))
    np.testing.assert_allclose(sparse[0, 18:36], np.tile(identity_6d, 3))
    np.testing.assert_allclose(sparse[0, 36:45], positions[1].reshape(-1))
    np.testing.assert_allclose(sparse[0, 45:54], positions[1].reshape(-1))


def test_tracker_resampling_keeps_30hz_samples_on_even_60hz_frames():
    positions = np.zeros((4, 3, 3), dtype=np.float32)
    positions[:, :, 0] = np.arange(4, dtype=np.float32)[:, None]
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None], 12, axis=0)
    rotations = rotations.reshape(4, 3, 3, 3)

    positions_60, rotations_60 = resample_tracker_signals(
        positions, rotations, source_fps=30.0, target_fps=60.0
    )

    assert positions_60.shape[0] == 7
    np.testing.assert_allclose(positions_60[::2], positions, atol=1e-6)
    np.testing.assert_allclose(rotations_60[::2], rotations, atol=1e-6)


def test_agrol_window_contains_requested_source_frames():
    start = choose_window_start(
        feature_count=700,
        source_frame_start=60,
        source_frame_end_exclusive=120,
    )
    target_start = 2 * 60 - 1
    target_end = 2 * 120 - 2

    assert start <= target_start
    assert start + 196 >= target_end


def test_baseline_6d_projection_returns_orthonormal_rotations():
    motion = np.tile(
        np.asarray([2.0, 0.0, 0.0, 0.1, 3.0, 0.0], dtype=np.float32),
        (2, 22),
    )

    matrices = baseline_motion_6d_to_rotation_matrices(motion)

    identity = np.swapaxes(matrices, -1, -2) @ matrices
    expected_identity = np.broadcast_to(np.eye(3), identity.shape)
    np.testing.assert_allclose(identity, expected_identity, atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(matrices), 1.0, atol=1e-6)


def test_baseline_outputs_align_to_source_frame_numbers(tmp_path: Path):
    rpm_path = tmp_path / "rpm.npz"
    rpm_motion = np.arange(20 * 132, dtype=np.float32).reshape(20, 132)
    np.savez(rpm_path, rpm_local_rotations_6d=rpm_motion, source_frame_offset=1)
    selected_rpm = select_rpm_motion(rpm_path, 4, 7)
    np.testing.assert_array_equal(selected_rpm, rpm_motion[3:6])

    agrol_path = tmp_path / "agrol.npz"
    agrol_motion = np.arange(196 * 132, dtype=np.float32).reshape(196, 132)
    np.savez(
        agrol_path,
        agrol_local_rotations_6d_60hz=agrol_motion,
        agrol_feature_window_start=5,
    )
    selected_agrol = select_agrol_motion(agrol_path, 4, 7)
    np.testing.assert_array_equal(selected_agrol, agrol_motion[[2, 4, 6]])


def test_independent_scene_panels_stitch_without_changing_pixels():
    panels = [
        np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), value, dtype=np.uint8)
        for value in (10, 20, 30, 40)
    ]

    stitched = stitch_independent_panel_views(panels)

    assert stitched.shape == (PANEL_HEIGHT, PANEL_WIDTH * 4, 3)
    for index, expected in enumerate((10, 20, 30, 40)):
        start = index * PANEL_WIDTH
        assert np.all(stitched[:, start : start + PANEL_WIDTH] == expected)


def test_independent_scene_panels_support_direct_4k_rendering():
    panel_width = PANEL_WIDTH * 2
    panel_height = PANEL_HEIGHT * 2
    panels = [
        np.full((panel_height, panel_width, 3), value, dtype=np.uint8)
        for value in (10, 20, 30, 40)
    ]

    stitched = stitch_independent_panel_views(
        panels,
        panel_width=panel_width,
        panel_height=panel_height,
    )

    assert stitched.shape == (2160, 3840, 3)


def test_pose_only_metrics_remove_each_methods_root_translation():
    joints = np.zeros((1, 22, 3), dtype=np.float32)
    joints[0, :, 1] = np.linspace(0.0, 1.0, 22)
    vertices = joints[:, :8].copy()
    translations = {
        "GT": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "RPM": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "Ours": np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
        "AGRoL": np.asarray([0.0, 0.0, -3.0], dtype=np.float32),
    }
    sequences = {
        name: SmplMeshSequence(
            vertices_world=vertices + translations[name],
            joints_world=joints + translations[name],
        )
        for name in METHOD_ORDER
    }
    identity = np.broadcast_to(
        np.eye(3, dtype=np.float32), (1, 22, 3, 3)
    ).copy()
    rotations = {name: identity.copy() for name in METHOD_ORDER}

    metrics = build_method_metrics(sequences, rotations, fps=30.0)

    for method in METHOD_ORDER[1:]:
        assert metrics[method]["mpjpe_cm"] > 0.0
        assert np.isclose(metrics[method]["root_aligned_mpjpe_cm"], 0.0, atol=1e-5)
        assert np.isclose(metrics[method]["root_aligned_pve_cm"], 0.0, atol=1e-5)
        assert np.isclose(metrics[method]["local_geodesic_deg"], 0.0, atol=1e-8)
