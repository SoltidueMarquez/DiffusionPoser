import math

import numpy as np
from scipy.spatial.transform import Rotation

from data_converter.amass_smpl_utils import (
    local_to_global_rotations,
    transform_rotations_to_unity,
)
from data_loaders.body_fbx_kinematics import (
    build_synthetic_body_fbx_rest,
    fk_body_fbx_local_delta,
    source_global_rotations_to_body_fbx_local_delta_6d,
)
from data_loaders.realtime_pose_kinematics import SMPL_PARENTS
from sample.render_realtime_pose_smpl_comparison import (
    PERSPECTIVE_CAMERA_SIDE_YAW,
    body_fbx_world_to_smpl_local_rotations,
    build_horizontal_clip_direction,
    build_horizontal_pelvis_follow_offsets,
    build_horizontal_travel_directions,
    create_arrow_mesh,
    fit_travel_oblique_perspective_camera,
    transform_faces_to_unity_winding,
)


def test_body_fbx_world_to_smpl_local_rotations_round_trip():
    """验证 renderer 精确逆转 converter 的 heading、rest 与坐标基处理。"""

    frame_count = 4
    rotvec = np.linspace(
        -0.22,
        0.27,
        frame_count * 24 * 3,
        dtype=np.float64,
    ).reshape(frame_count, 24, 3)
    local_amass = Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(
        frame_count, 24, 3, 3
    )
    global_amass = local_to_global_rotations(local_amass, SMPL_PARENTS)
    global_unity = transform_rotations_to_unity(global_amass)
    root_heading = np.asarray([-0.35, -0.1, 0.18, 0.42], dtype=np.float32)
    body_pose_6d = source_global_rotations_to_body_fbx_local_delta_6d(
        global_unity,
        root_heading=root_heading,
    )
    rest = build_synthetic_body_fbx_rest()
    _, body_global = fk_body_fbx_local_delta(
        body_pose_local_delta_6d=body_pose_6d,
        actor_root_pos_world=np.zeros((frame_count, 3), dtype=np.float32),
        root_heading=root_heading,
        rest=rest,
    )

    recovered_local_amass = body_fbx_world_to_smpl_local_rotations(
        body_global,
        root_heading,
        rest.rest_local_rotations,
        rest.parents,
    )

    np.testing.assert_allclose(
        recovered_local_amass[:, :22],
        local_amass[:, :22],
        atol=1e-5,
        rtol=1e-5,
    )


def test_unity_reflection_reverses_face_winding_without_mutating_source():
    """AMASS→Unity 是反射变换，SMPL 面绕序必须同步反转。"""

    source = np.asarray([[0, 1, 2], [4, 5, 6]], dtype=np.int64)
    original = source.copy()

    transformed = transform_faces_to_unity_winding(source)

    np.testing.assert_array_equal(transformed, [[0, 2, 1], [4, 6, 5]])
    np.testing.assert_array_equal(source, original)


def test_direction_arrows_use_smoothed_horizontal_travel():
    """移动箭头只解释水平轨迹，不应被 AMASS 高度变化带偏。"""

    # AMASS [x,y,z] 转 Unity 后为 [x,z,y]；因此 AMASS -Y 表示 Unity -Z 移动。
    translations = np.zeros((7, 3), dtype=np.float32)
    translations[:, 1] = np.linspace(0.0, -1.2, 7)
    translations[:, 2] = np.asarray([0.0, 0.1, 0.25, 0.4, 0.25, 0.1, 0.0])
    travel = build_horizontal_travel_directions(translations, half_window=2)
    np.testing.assert_allclose(
        travel,
        np.repeat(np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32), 7, axis=0),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        build_horizontal_clip_direction(translations),
        np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_perspective_camera_is_oblique_side_view_and_fits_viewport():
    """透视相机应略偏离纯侧视，并把所有点放进实际宽高比的视锥。"""

    points = np.asarray(
        [
            [-0.4, 0.0, 0.0],
            [0.4, 1.8, -2.0],
            [-0.2, 0.9, -1.0],
        ],
        dtype=np.float32,
    )
    travel = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    camera = fit_travel_oblique_perspective_camera(
        points,
        travel_direction=travel,
        viewport_width=640,
        viewport_height=880,
    )
    horizontal_view = camera.pose[:3, 2].copy()
    horizontal_view[1] = 0.0
    horizontal_view /= np.linalg.norm(horizontal_view)
    np.testing.assert_allclose(
        abs(float(horizontal_view @ travel)),
        math.sin(PERSPECTIVE_CAMERA_SIDE_YAW),
        atol=1e-7,
    )
    np.testing.assert_allclose(camera.aspect_ratio, 640.0 / 880.0)

    camera_coordinates = (
        points - camera.pose[:3, 3]
    ) @ camera.pose[:3, :3]
    depth = -camera_coordinates[:, 2]
    tan_half_y = math.tan(camera.yfov * 0.5)
    tan_half_x = tan_half_y * camera.aspect_ratio
    assert np.all(depth > 0.0)
    assert np.all(np.abs(camera_coordinates[:, 0] / depth) <= tan_half_x)
    assert np.all(np.abs(camera_coordinates[:, 1] / depth) <= tan_half_y)


def test_pelvis_follow_offsets_only_track_horizontal_motion():
    """腿部近景应跟随 pelvis 的 XZ 位移，但必须保留真实起跳高度。"""

    joints = np.zeros((3, 22, 3), dtype=np.float32)
    joints[:, 0] = np.asarray(
        [
            [0.2, 0.9, -0.4],
            [0.5, 1.3, -0.1],
            [0.9, 1.0, 0.3],
        ],
        dtype=np.float32,
    )
    offsets = build_horizontal_pelvis_follow_offsets(joints)
    np.testing.assert_allclose(
        offsets,
        np.asarray(
            [
                [0.2, 0.0, -0.4],
                [0.5, 0.0, -0.1],
                [0.9, 0.0, 0.3],
            ],
            dtype=np.float32,
        ),
    )


def test_arrow_mesh_tip_follows_requested_direction():
    """箭头网格的起点和尖端必须与输入方向一致。"""

    origin = np.asarray([0.3, 1.2, -0.4])
    direction = np.asarray([1.0, -1.0, 0.0])
    arrow = create_arrow_mesh(origin, direction, length=0.38)
    unit_direction = direction / np.linalg.norm(direction)
    along_arrow = (np.asarray(arrow.vertices) - origin) @ unit_direction
    np.testing.assert_allclose(float(along_arrow.min()), 0.0, atol=1e-7)
    np.testing.assert_allclose(float(along_arrow.max()), 0.38, atol=1e-7)
