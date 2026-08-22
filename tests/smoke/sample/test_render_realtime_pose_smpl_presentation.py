import numpy as np

from sample.render_realtime_pose_smpl_comparison import METHOD_ORDER, SmplMeshSequence
from sample.render_realtime_pose_smpl_presentation import (
    INTRO_FRAME_COUNT,
    METHOD_GAP_METERS,
    build_intro_tracker_points,
    build_presentation_frame_schedule,
    build_presentation_layout,
)


def build_synthetic_sequences():
    """构造带水平位移和起伏的三路小网格，避免测试依赖 SMPL 模型文件。"""

    frame_count = 4
    pelvis = np.asarray(
        [
            [0.0, 0.95, 0.0],
            [0.3, 1.08, -0.2],
            [0.7, 1.20, -0.5],
            [1.0, 1.02, -0.8],
        ],
        dtype=np.float32,
    )
    body_points = np.asarray(
        [
            [-0.32, -0.95, -0.18],
            [0.32, -0.95, -0.18],
            [-0.42, 0.10, 0.0],
            [0.42, 0.10, 0.0],
            [-0.20, 0.82, 0.08],
            [0.20, 0.82, 0.08],
        ],
        dtype=np.float32,
    )
    sequences = {}
    for method_index, method_name in enumerate(METHOD_ORDER):
        method_delta = np.asarray(
            [0.03 * method_index, 0.0, -0.02 * method_index],
            dtype=np.float32,
        )
        vertices = pelvis[:, None, :] + body_points[None] + method_delta
        joints = np.repeat(pelvis[:, None, :], 24, axis=1) + method_delta
        joints[:, 9] += np.asarray([0.0, 0.45, 0.0], dtype=np.float32)
        joints[:, 10] += np.asarray([-0.15, -0.92, 0.10], dtype=np.float32)
        joints[:, 11] += np.asarray([0.15, -0.92, 0.10], dtype=np.float32)
        sequences[method_name] = SmplMeshSequence(
            vertices_world=vertices.astype(np.float32),
            joints_world=joints.astype(np.float32),
        )
    trackers = np.zeros((frame_count, 6, 3), dtype=np.float32)
    trackers[:, 0] = pelvis + np.asarray([0.0, 0.75, 0.0], dtype=np.float32)
    trackers[:, 1] = pelvis + np.asarray([-0.55, 0.28, 0.0], dtype=np.float32)
    trackers[:, 2] = pelvis + np.asarray([0.55, 0.28, 0.0], dtype=np.float32)
    return sequences, trackers


def project_to_ndc(points_world: np.ndarray, camera_pose: np.ndarray, yfov: float):
    camera_coordinates = (
        np.asarray(points_world, dtype=np.float64) - camera_pose[:3, 3]
    ) @ camera_pose[:3, :3]
    depth = -camera_coordinates[:, 2]
    tan_half_y = np.tan(float(yfov) * 0.5)
    tan_half_x = tan_half_y * (1920.0 / 1080.0)
    ndc = np.stack(
        [
            camera_coordinates[:, 0] / (depth * tan_half_x),
            camera_coordinates[:, 1] / (depth * tan_half_y),
        ],
        axis=-1,
    )
    return ndc, depth


def test_shared_stage_offsets_camera_follow_and_frustum():
    """三路应对称分开，共用固定尺度相机，并在整段内完整进入视锥。"""

    sequences, trackers = build_synthetic_sequences()
    original_vertices = {
        name: value.vertices_world.copy() for name, value in sequences.items()
    }
    layout = build_presentation_layout(
        sequences=sequences,
        tracker_pos_world=trackers,
    )

    np.testing.assert_allclose(layout.method_offsets[1], 0.0, atol=1e-7)
    np.testing.assert_allclose(
        layout.method_offsets[0],
        -layout.method_offsets[2],
        atol=1e-7,
    )
    camera_right = layout.base_camera.pose[:3, 0]
    for offset in layout.method_offsets:
        perpendicular = offset - camera_right * float(offset @ camera_right)
        np.testing.assert_allclose(perpendicular, 0.0, atol=1e-6)

    # 整段任一姿态的屏幕横向包围盒都不能侵入相邻方法的展示区域。
    projected_intervals = []
    for method_index, method_name in enumerate(METHOD_ORDER):
        centered = (
            sequences[method_name].vertices_world
            - layout.follow_offsets[:, None, :]
            + layout.method_offsets[method_index]
        )
        projected = centered @ camera_right
        projected_intervals.append((float(np.min(projected)), float(np.max(projected))))
    assert projected_intervals[1][0] - projected_intervals[0][1] >= METHOD_GAP_METERS - 1e-6
    assert projected_intervals[2][0] - projected_intervals[1][1] >= METHOD_GAP_METERS - 1e-6

    # 开场的两份 tracker 去除各自展示偏移后必须逐点完全相同。
    intro_trackers = build_intro_tracker_points(
        trackers[0, :3],
        layout.method_offsets,
    )
    np.testing.assert_allclose(
        intro_trackers[0] - layout.method_offsets[1],
        intro_trackers[1] - layout.method_offsets[2],
        atol=1e-7,
    )

    # 展示偏移不能污染调用方持有的原始模型结果。
    for method_name in METHOD_ORDER:
        np.testing.assert_array_equal(
            sequences[method_name].vertices_world,
            original_vertices[method_name],
        )

    # 相机只跟随 XZ；旋转和 Y 高度在跳跃帧中也必须完全固定。
    np.testing.assert_allclose(
        layout.camera_poses[:, :3, :3],
        np.repeat(
            layout.camera_poses[0:1, :3, :3],
            layout.camera_poses.shape[0],
            axis=0,
        ),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        layout.camera_poses[:, 1, 3],
        layout.camera_poses[0, 1, 3],
        atol=1e-8,
    )
    np.testing.assert_allclose(layout.follow_offsets[:, 1], 0.0, atol=1e-8)

    # 每一帧三路网格以及开场的两份相同三点都必须处于同一个透视视锥内。
    for frame_index, camera_pose in enumerate(layout.camera_poses):
        points = []
        for method_index, method_name in enumerate(METHOD_ORDER):
            points.append(
                sequences[method_name].vertices_world[frame_index]
                + layout.method_offsets[method_index]
            )
        points.extend(
            [
                trackers[frame_index, :3] + layout.method_offsets[1],
                trackers[frame_index, :3] + layout.method_offsets[2],
            ]
        )
        ndc, depth = project_to_ndc(
            np.concatenate(points, axis=0),
            camera_pose,
            layout.base_camera.yfov,
        )
        assert np.all(depth > 0.0)
        assert np.all(np.abs(ndc) <= 1.0 + 1e-6)


def test_presentation_schedule_has_exact_intro_normal_and_replay_mapping():
    """60 帧动作必须得到 21 帧开场、60 帧原速和 120 帧重复慢放。"""

    schedule = build_presentation_frame_schedule(60)
    assert len(schedule) == INTRO_FRAME_COUNT + 60 + 120 == 201
    assert all(
        frame.source_frame_index == 0
        and frame.show_trackers
        and frame.playback_label == "Input setup"
        for frame in schedule[:INTRO_FRAME_COUNT]
    )
    normal = schedule[INTRO_FRAME_COUNT : INTRO_FRAME_COUNT + 60]
    assert [frame.source_frame_index for frame in normal] == list(range(60))
    assert all(not frame.show_trackers and frame.playback_label == "1.0×" for frame in normal)
    replay = schedule[INTRO_FRAME_COUNT + 60 :]
    assert [frame.source_frame_index for frame in replay] == [
        frame_index
        for frame_index in range(60)
        for _ in range(2)
    ]
    assert all(
        not frame.show_trackers and frame.playback_label == "0.5× replay"
        for frame in replay
    )
