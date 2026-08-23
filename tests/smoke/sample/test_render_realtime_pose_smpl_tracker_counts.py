import numpy as np

from sample.realtime_pose_smpl_rendering import SmplMeshSequence
from sample.render_realtime_pose_smpl_presentation import build_presentation_layout
from sample.render_realtime_pose_smpl_tracker_counts import (
    METHOD_ORDER,
    TRACKER_AVAILABLE_BY_METHOD,
    build_active_tracker_points,
    validate_tracker_count_configuration,
)


def build_toy_sequences(frame_count: int = 4):
    vertices_template = np.asarray(
        [
            [-0.25, 0.0, -0.15],
            [0.25, 0.0, -0.15],
            [-0.25, 0.0, 0.15],
            [0.25, 0.0, 0.15],
            [-0.25, 1.7, -0.15],
            [0.25, 1.7, -0.15],
            [-0.25, 1.7, 0.15],
            [0.25, 1.7, 0.15],
        ],
        dtype=np.float32,
    )
    pelvis = np.zeros((frame_count, 3), dtype=np.float32)
    pelvis[:, 0] = np.linspace(0.0, 0.6, frame_count, dtype=np.float32)
    sequences = {}
    for method_index, method_name in enumerate(METHOD_ORDER):
        vertices = np.repeat(vertices_template[None], frame_count, axis=0)
        vertices += pelvis[:, None]
        vertices[..., 2] += method_index * 0.01
        joints = np.repeat(pelvis[:, None], 22, axis=1)
        joints[:, 15, 1] = 1.55
        sequences[method_name] = SmplMeshSequence(vertices, joints)
    trackers = np.repeat(pelvis[:, None], 6, axis=1)
    trackers[:, 0, 1] = 1.55
    trackers[:, 1, 0] -= 0.45
    trackers[:, 1, 1] = 1.05
    trackers[:, 2, 0] += 0.45
    trackers[:, 2, 1] = 1.05
    trackers[:, 3, 1] = 0.85
    trackers[:, 4, 0] -= 0.12
    trackers[:, 4, 1] = 0.05
    trackers[:, 5, 0] += 0.12
    trackers[:, 5, 1] = 0.05
    return sequences, trackers


def test_tracker_count_masks_are_exact_and_nested():
    validate_tracker_count_configuration()
    masks = np.asarray(TRACKER_AVAILABLE_BY_METHOD, dtype=bool)
    assert [int(mask.sum()) for mask in masks] == [3, 4, 5, 6]
    np.testing.assert_array_equal(masks[0], [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(masks[1], [1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(masks[2], [1, 1, 1, 1, 0, 1])
    np.testing.assert_array_equal(masks[3], [1, 1, 1, 1, 1, 1])


def test_tracker_count_layout_and_active_points_preserve_inputs():
    sequences, trackers = build_toy_sequences()
    original_trackers = trackers.copy()
    original_vertices = {
        method_name: sequence.vertices_world.copy()
        for method_name, sequence in sequences.items()
    }
    layout = build_presentation_layout(
        sequences=sequences,
        tracker_pos_world=trackers,
        method_order=METHOD_ORDER,
        tracker_available_by_method=TRACKER_AVAILABLE_BY_METHOD,
        follow_method_name=METHOD_ORDER[0],
    )
    active_points = build_active_tracker_points(
        trackers[0],
        layout.method_offsets,
    )
    assert [len(points) for points in active_points] == [3, 4, 5, 6]
    for method_index, points in enumerate(active_points):
        np.testing.assert_allclose(
            points - layout.method_offsets[method_index],
            trackers[0, TRACKER_AVAILABLE_BY_METHOD[method_index]],
            atol=1e-7,
        )
    np.testing.assert_array_equal(trackers, original_trackers)
    for method_name in METHOD_ORDER:
        np.testing.assert_array_equal(
            sequences[method_name].vertices_world,
            original_vertices[method_name],
        )
    np.testing.assert_allclose(
        layout.method_offsets[0],
        -layout.method_offsets[3],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        layout.method_offsets[1],
        -layout.method_offsets[2],
        atol=1e-7,
    )
    assert np.allclose(layout.camera_poses[:, 1, 3], layout.camera_poses[0, 1, 3])
