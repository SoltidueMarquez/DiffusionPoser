from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from data_converter.amass_smpl_utils import (
    AMASS_TO_UNITY,
    SOURCE_BODY_JOINT_COUNT,
    local_to_global_rotations,
)
from data_loaders.realtime_pose_kinematics import (
    JOINT_INDEX,
    SMPL_PARENTS,
    TRACKER_JOINT_INDICES,
    global_to_parent_local_rotations,
)
from eval.realtime_pose_metrics import compute_rpm_p2_mc_metrics
from sample.realtime_pose_smpl_rendering import (
    SmplMeshSequence,
    body_fbx_world_to_smpl_local_rotations,
    build_horizontal_pelvis_follow_offsets,
    create_front_marker_mesh,
    create_smplh_model,
    create_sphere_cloud,
    create_static_scene,
    decode_png,
    encode_png,
    load_comparison_clip,
    load_font,
    normalize_vector,
    rotation_matrices_to_axis_angle,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)
from sample.render_realtime_pose_smpl_presentation import (
    build_follow_camera_poses,
    build_visible_tracker_glyph_points,
    create_material,
    draw_centered_text,
    fit_fixed_presentation_camera,
    validate_mesh_sequences,
)
from sample.three_point_baseline_data import (
    baseline_motion_6d_to_rotation_matrices,
)
from utils.video_io import Mp4FrameWriter


OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
PANEL_WIDTH = OUTPUT_WIDTH // 4
PANEL_HEIGHT = OUTPUT_HEIGHT
INTRO_FRAME_COUNT = 18
METHOD_ORDER = ("GT", "RPM", "Ours", "AGRoL")
METHOD_COLORS = {
    "GT": (0.45, 0.66, 0.86, 1.0),
    "RPM": (0.82, 0.50, 0.52, 1.0),
    "Ours": (0.28, 0.72, 0.69, 1.0),
    "AGRoL": (0.63, 0.52, 0.82, 1.0),
}
TRACKER_COLOR = (1.0, 0.55, 0.05, 1.0)
CHEST_MARKER_COLOR = (1.0, 0.88, 0.18, 1.0)
LEFT_FOOT_COLOR = (0.05, 0.82, 0.92, 1.0)
RIGHT_FOOT_COLOR = (0.86, 0.24, 0.78, 1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="统一渲染 GT、RPM、Ours 与 AGRoL 的同输入三点 SMPL-H Demo。"
    )
    parser.add_argument("--comparison_npz", required=True, type=Path)
    parser.add_argument("--comparison_report_json", required=True, type=Path)
    parser.add_argument("--amass_npz", required=True, type=Path)
    parser.add_argument("--smpl_model_dir", required=True, type=Path)
    parser.add_argument("--rpm_npz", required=True, type=Path)
    parser.add_argument("--agrol_npz", required=True, type=Path)
    parser.add_argument("--source_frame_start", required=True, type=int)
    parser.add_argument("--source_frame_end_exclusive", required=True, type=int)
    parser.add_argument("--output_mp4", required=True, type=Path)
    parser.add_argument("--output_report_json", required=True, type=Path)
    parser.add_argument("--output_preview_png", default=None, type=Path)
    parser.add_argument(
        "--resolution_scale",
        default=1,
        type=int,
        help="渲染分辨率倍率；2 表示直接以 3840×2160 渲染，而不是插值放大。",
    )
    parser.add_argument(
        "--replay_slowdown_factor",
        default=4,
        type=int,
        help="回放段相对原速的慢放倍数；4 表示以 0.25× 速度播放。",
    )
    return parser


def replay_playback_label(replay_slowdown_factor: int) -> str:
    """返回与回放慢放倍数一致的画面标签。"""
    factor = int(replay_slowdown_factor)
    if factor < 1:
        raise ValueError(f"replay_slowdown_factor 必须 >= 1，实际为 {factor}")
    return f"{1.0 / factor:g}× replay"


def select_rpm_motion(
    rpm_npz: Path,
    source_frame_start: int,
    source_frame_end_exclusive: int,
) -> np.ndarray:
    with np.load(rpm_npz, allow_pickle=False) as payload:
        motion = np.asarray(payload["rpm_local_rotations_6d"], dtype=np.float32)
        offset = int(np.asarray(payload["source_frame_offset"]).item())
    start = int(source_frame_start) - offset
    end = int(source_frame_end_exclusive) - offset
    if start < 0 or end > motion.shape[0]:
        raise ValueError(f"RPM 请求切片 [{start},{end}) 超过输出长度 {motion.shape[0]}")
    return motion[start:end]


def select_agrol_motion(
    agrol_npz: Path,
    source_frame_start: int,
    source_frame_end_exclusive: int,
) -> np.ndarray:
    with np.load(agrol_npz, allow_pickle=False) as payload:
        motion = np.asarray(
            payload["agrol_local_rotations_6d_60hz"], dtype=np.float32
        )
        window_start = int(
            np.asarray(payload["agrol_feature_window_start"]).item()
        )
    source_frames = np.arange(
        int(source_frame_start), int(source_frame_end_exclusive), dtype=np.int64
    )
    # 60 Hz feature index 0 对应 1/60 s；30 Hz source frame n 对应索引 2n-1。
    indices = 2 * source_frames - 1 - window_start
    if np.any(indices < 0) or np.any(indices >= motion.shape[0]):
        raise ValueError(
            f"AGRoL 对齐索引 [{indices.min()},{indices.max()}] 超过窗口 {motion.shape[0]}"
        )
    return motion[indices]


def local_amass_to_global_unity(local_rotations: np.ndarray) -> np.ndarray:
    local = np.asarray(local_rotations, dtype=np.float64)
    parents = np.asarray(SMPL_PARENTS[:SOURCE_BODY_JOINT_COUNT], dtype=np.int64)
    global_amass = local_to_global_rotations(local, parents)
    global_unity = (
        AMASS_TO_UNITY[None, None]
        @ global_amass
        @ AMASS_TO_UNITY.T[None, None]
    )
    return global_unity.astype(np.float32)


def build_comparison_sequences(
    *,
    clip,
    rpm_motion_6d: np.ndarray,
    agrol_motion_6d: np.ndarray,
    smpl_model_dir: Path,
) -> tuple[dict[str, SmplMeshSequence], dict[str, np.ndarray], np.ndarray]:
    gt_local = Rotation.from_rotvec(
        np.asarray(clip.gt_pose_axis_angle, dtype=np.float64).reshape(-1, 3)
    ).as_matrix().reshape(clip.frame_count, SOURCE_BODY_JOINT_COUNT, 3, 3)
    rpm_local = baseline_motion_6d_to_rotation_matrices(rpm_motion_6d)
    agrol_local = baseline_motion_6d_to_rotation_matrices(agrol_motion_6d)
    ours_local = body_fbx_world_to_smpl_local_rotations(
        clip.diffusion_rotations_world,
        clip.diffusion_root_yaw,
        clip.body_fbx_rest.rest_local_rotations,
        clip.body_fbx_rest.parents,
    )[:, :SOURCE_BODY_JOINT_COUNT]
    rpm_pose = rotation_matrices_to_axis_angle(rpm_local)
    agrol_pose = rotation_matrices_to_axis_angle(agrol_local)

    model = create_smplh_model(
        model_dir=smpl_model_dir,
        gender=clip.gender,
        batch_size=clip.frame_count,
    )
    sequences = {
        "GT": run_smplh_forward(
            model=model,
            pose_axis_angle=clip.gt_pose_axis_angle,
            betas=clip.betas,
            translation_amass=clip.gt_translation_amass,
        ),
        "RPM": run_smplh_forward(
            model=model,
            pose_axis_angle=rpm_pose,
            betas=clip.betas,
            translation_amass=clip.gt_translation_amass,
        ),
        "Ours": run_smplh_forward(
            model=model,
            pose_axis_angle=rotation_matrices_to_axis_angle(
                # Ours 保存的是 body.fbx 世界旋转；沿用已验收的逆变换函数恢复 SMPL local。
                ours_local
            ),
            betas=clip.betas,
            translation_amass=clip.gt_translation_amass,
        ),
        "AGRoL": run_smplh_forward(
            model=model,
            pose_axis_angle=agrol_pose,
            betas=clip.betas,
            translation_amass=clip.gt_translation_amass,
        ),
    }
    rotations_world = {
        # 四种方法全部转换成同一 SMPL global/Unity 世界基底，供朝向标记和
        # rotation 指标使用；不能混入 body.fbx world rotation。
        "GT": local_amass_to_global_unity(gt_local),
        "RPM": local_amass_to_global_unity(rpm_local),
        "Ours": local_amass_to_global_unity(ours_local),
        "AGRoL": local_amass_to_global_unity(agrol_local),
    }
    faces = transform_faces_to_unity_winding(model.faces)
    return sequences, rotations_world, faces


def build_method_metrics(
    sequences: dict[str, SmplMeshSequence],
    rotations_world: dict[str, np.ndarray],
    fps: float,
) -> dict[str, dict[str, float | None]]:
    def pad_to_smpl24(value: np.ndarray) -> np.ndarray:
        rotations = np.asarray(value, dtype=np.float32)
        if rotations.shape[1] == 24:
            return rotations
        if rotations.shape[1] != SOURCE_BODY_JOINT_COUNT:
            raise ValueError(f"指标旋转应为 SMPL22/24，实际为 {rotations.shape}")
        # 指标只读取前 22 个身体关节；补齐手部根节点以满足现有 SMPL24 接口。
        identity = np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (rotations.shape[0], 24 - SOURCE_BODY_JOINT_COUNT, 3, 3),
        )
        return np.concatenate([rotations, identity], axis=1)

    result: dict[str, dict[str, float | None]] = {}
    target_global = pad_to_smpl24(rotations_world["GT"])
    target_local = global_to_parent_local_rotations(target_global)[
        :, :SOURCE_BODY_JOINT_COUNT
    ]
    for method in METHOD_ORDER[1:]:
        predicted_global = pad_to_smpl24(rotations_world[method])
        metrics = compute_rpm_p2_mc_metrics(
            predicted_global_rotations=predicted_global,
            target_global_rotations=target_global,
            predicted_joint_positions=sequences[method].joints_world,
            target_joint_positions=sequences["GT"].joints_world,
            fps=float(fps),
        )
        # 四种方法共享同一 SMPL-H topology/body shape，因此逐顶点欧氏距离可直接
        # 反映画面表面轮廓差异，比只看 22 个关节更接近定性图的视觉判断。
        vertex_error = np.linalg.norm(
            sequences[method].vertices_world - sequences["GT"].vertices_world,
            axis=-1,
        )
        metrics["pve_cm"] = float(vertex_error.mean() * 100.0)

        # 定性姿态对比需要排除 root 平移的影响。分别减去各自 pelvis 后，
        # MPJPE/PVE 只反映人体构型；局部旋转测地误差则避免 axis-angle 分量口径
        # 可能掩盖的三维旋转差异。
        predicted_root = sequences[method].joints_world[:, :1]
        target_root = sequences["GT"].joints_world[:, :1]
        predicted_joints = (
            sequences[method].joints_world[:, :SOURCE_BODY_JOINT_COUNT]
            - predicted_root
        )
        target_joints = (
            sequences["GT"].joints_world[:, :SOURCE_BODY_JOINT_COUNT]
            - target_root
        )
        metrics["root_aligned_mpjpe_cm"] = float(
            np.linalg.norm(target_joints - predicted_joints, axis=-1).mean()
            * 100.0
        )
        predicted_vertices = sequences[method].vertices_world - predicted_root
        target_vertices = sequences["GT"].vertices_world - target_root
        metrics["root_aligned_pve_cm"] = float(
            np.linalg.norm(target_vertices - predicted_vertices, axis=-1).mean()
            * 100.0
        )

        predicted_local = global_to_parent_local_rotations(predicted_global)[
            :, :SOURCE_BODY_JOINT_COUNT
        ]
        relative_local = (
            np.swapaxes(target_local, -1, -2) @ predicted_local
        )
        metrics["local_geodesic_deg"] = float(
            np.degrees(
                Rotation.from_matrix(relative_local.reshape(-1, 3, 3)).magnitude()
            ).mean()
        )
        result[method] = metrics
    return result


def render_single_method_view(
    *,
    renderer,
    scene,
    method_name: str,
    frame_index: int,
    sequences: dict[str, SmplMeshSequence],
    rotations_world: dict[str, np.ndarray],
    faces: np.ndarray,
    tracker_pos_world: np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    """在独立场景中渲染一种方法；四个调用共享完全相同的相机位姿。"""

    import pyrender
    import trimesh

    dynamic_nodes = []

    def add_mesh(mesh, material, smooth: bool = True) -> None:
        dynamic_nodes.append(
            scene.add(
                pyrender.Mesh.from_trimesh(mesh, material=material, smooth=smooth)
            )
        )

    try:
        vertices = sequences[method_name].vertices_world[frame_index]
        joints = sequences[method_name].joints_world[frame_index]
        add_mesh(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            create_material(pyrender, METHOD_COLORS[method_name], 0.92),
        )
        torso_front = -rotations_world[method_name][
            frame_index, JOINT_INDEX["spine3"], :, 2
        ]
        torso_front = normalize_vector(torso_front)
        add_mesh(
            create_front_marker_mesh(
                joints[JOINT_INDEX["spine3"]] + torso_front * 0.12,
                torso_front,
            ),
            create_material(pyrender, CHEST_MARKER_COLOR, 0.45),
        )
        for joint_index, color in (
            (JOINT_INDEX["left_foot"], LEFT_FOOT_COLOR),
            (JOINT_INDEX["right_foot"], RIGHT_FOOT_COLOR),
        ):
            point = joints[joint_index] + np.asarray([0.0, 0.018, 0.0])
            add_mesh(
                create_sphere_cloud(point[None], radius=0.027),
                create_material(pyrender, color, 0.52),
            )

        if method_name != "GT":
            tracker_points = build_visible_tracker_glyph_points(
                np.asarray(tracker_pos_world[frame_index, :3], dtype=np.float64),
                np.asarray(camera_pose, dtype=np.float64)[:3, 3],
            )
            add_mesh(
                create_sphere_cloud(tracker_points, radius=0.036),
                create_material(pyrender, TRACKER_COLOR, 0.48),
            )
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
        return np.asarray(color[..., :3], dtype=np.uint8)
    finally:
        for node in dynamic_nodes:
            scene.remove_node(node)


def stitch_independent_panel_views(
    panel_views: list[np.ndarray],
    panel_width: int = PANEL_WIDTH,
    panel_height: int = PANEL_HEIGHT,
) -> np.ndarray:
    """把四个独立场景的 `[H,W,3]` RGB 结果按 GT/RPM/Ours/AGRoL 拼接。"""

    if len(panel_views) != len(METHOD_ORDER):
        raise ValueError(
            f"独立场景数量应为 {len(METHOD_ORDER)}，实际为 {len(panel_views)}"
        )
    expected_shape = (int(panel_height), int(panel_width), 3)
    panels = [np.asarray(value, dtype=np.uint8) for value in panel_views]
    if any(value.shape != expected_shape for value in panels):
        raise ValueError(
            f"每个独立场景应为 {expected_shape}，"
            f"实际为 {[value.shape for value in panels]}"
        )
    return np.concatenate(panels, axis=1)


def rgba(color: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(int(round(channel * 255.0)) for channel in color)


def compose_frame(
    *,
    viewport_rgb: np.ndarray,
    playback_label: str,
    source_frame: int,
    show_input_note: bool,
    resolution_scale: int = 1,
) -> Image.Image:
    scale = int(resolution_scale)
    if scale < 1:
        raise ValueError(f"resolution_scale 必须 >= 1，实际为 {scale}")

    def px(value: int) -> int:
        return int(round(value * scale))

    output_width = OUTPUT_WIDTH * scale
    output_height = OUTPUT_HEIGHT * scale
    image = Image.fromarray(viewport_rgb).convert("RGBA")
    if image.size != (output_width, output_height):
        raise ValueError(
            f"viewport 尺寸应为 {(output_width, output_height)}，实际为 {image.size}"
        )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    segment_width = output_width // len(METHOD_ORDER)
    labels = {
        "GT": "GT",
        "RPM": "RPM · 3 trackers",
        "Ours": "Ours · 3 trackers",
        "AGRoL": "AGRoL · 3 trackers",
    }
    for index, method in enumerate(METHOD_ORDER):
        left = index * segment_width
        right = output_width if index == len(METHOD_ORDER) - 1 else (index + 1) * segment_width
        draw.rectangle((left, 0, right, px(78)), fill=rgba(METHOD_COLORS[method]))
        draw_centered_text(
            draw,
            (left, px(5), right, px(60)),
            labels[method],
            load_font(px(22)),
            (255, 255, 255, 255),
        )
        if method != "GT":
            draw_centered_text(
                draw,
                (left, px(47), right, px(76)),
                "Head + left/right wrist",
                load_font(px(12)),
                (255, 255, 255, 235),
            )
    for boundary_index in range(1, len(METHOD_ORDER)):
        x = boundary_index * segment_width
        draw.line(
            (x, px(78), x, output_height - px(70)),
            fill=(190, 196, 205, 220),
            width=px(2),
        )

    playback_text = f"{playback_label}  |  frame {source_frame}"
    playback_font = load_font(px(13))
    playback_bbox = draw.textbbox((0, 0), playback_text, font=playback_font)
    # 0.25× 标签比原来的 0.5× 多一个字符，按真实文本宽度扩展徽标，
    # 避免四倍慢放时帧号被挤出背景框。
    playback_width = max(px(187), playback_bbox[2] - playback_bbox[0] + px(28))
    playback_right = px(24) + playback_width
    draw.rounded_rectangle(
        (px(24), px(94), playback_right, px(134)),
        radius=px(13),
        fill=(31, 41, 55, 220),
    )
    draw_centered_text(
        draw,
        (px(24), px(94), playback_right, px(134)),
        playback_text,
        playback_font,
        (255, 255, 255, 255),
    )
    if show_input_note:
        draw.rounded_rectangle(
            (output_width // 2 - px(330), px(94), output_width // 2 + px(330), px(142)),
            radius=px(15),
            fill=(255, 255, 255, 238),
            outline=(235, 142, 24, 245),
            width=px(2),
        )
        draw.ellipse(
            (
                output_width // 2 - px(294),
                px(110),
                output_width // 2 - px(278),
                px(126),
            ),
            fill=rgba(TRACKER_COLOR),
        )
        draw.text(
            (output_width // 2 - px(260), px(105)),
            "Identical 3-point observations for all methods",
            font=load_font(px(17)),
            fill=(31, 41, 55, 255),
        )

    footer_top = output_height - px(70)
    draw.rectangle((0, footer_top, output_width, output_height), fill=(255, 255, 255, 235))
    draw.line(
        (0, footer_top, output_width, footer_top),
        fill=(202, 207, 214, 240),
        width=px(2),
    )
    draw_centered_text(
        draw,
        (px(18), footer_top + px(3), output_width - px(18), output_height - px(3)),
        "Four independent scenes  |  Identical camera, body shape, GT root translation and timeline  |  AGRoL: 60→30 Hz",
        load_font(px(17)),
        (31, 41, 55, 255),
    )
    return Image.alpha_composite(image, overlay).convert("RGB")


def render_video(
    *,
    output_mp4: Path,
    output_preview_png: Path | None,
    clip,
    sequences: dict[str, SmplMeshSequence],
    rotations_world: dict[str, np.ndarray],
    faces: np.ndarray,
    resolution_scale: int = 1,
    replay_slowdown_factor: int = 4,
) -> Path:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender

    scale = int(resolution_scale)
    if scale < 1:
        raise ValueError(f"resolution_scale 必须 >= 1，实际为 {scale}")
    slowdown_factor = int(replay_slowdown_factor)
    slow_playback_label = replay_playback_label(slowdown_factor)
    panel_width = PANEL_WIDTH * scale
    panel_height = PANEL_HEIGHT * scale
    frame_count, _ = validate_mesh_sequences(sequences, METHOD_ORDER)
    tracker_masks = np.zeros((len(METHOD_ORDER), 6), dtype=bool)
    tracker_masks[1:, :3] = True
    layout_trackers = np.asarray(clip.tracker_pos_world, dtype=np.float32).copy()
    layout_trackers[:, :3] = sequences["GT"].joints_world[
        :, TRACKER_JOINT_INDICES[:3]
    ]
    # 四种方法不再横向平移到同一舞台。相机在 GT pelvis 的 XZ 轨迹上跟随，
    # 并用四种方法的重叠包围盒一次性拟合；随后把同一 CameraSpec/pose 复制到
    # 四个独立场景，保证每栏的外参、内参、灯光和地面完全一致。
    follow_offsets = build_horizontal_pelvis_follow_offsets(
        sequences["GT"].joints_world
    )
    shared_camera = fit_fixed_presentation_camera(
        sequences=sequences,
        tracker_pos_world=layout_trackers,
        follow_offsets=follow_offsets,
        method_offsets=np.zeros((len(METHOD_ORDER), 3), dtype=np.float32),
        viewport_width=panel_width,
        viewport_height=panel_height,
        method_order=METHOD_ORDER,
        tracker_available_by_method=tracker_masks,
        padding=1.12,
    )
    camera_poses = build_follow_camera_poses(shared_camera.pose, follow_offsets)
    all_vertices = np.concatenate(
        [sequences[name].vertices_world.reshape(-1, 3) for name in METHOD_ORDER],
        axis=0,
    )
    floor_y = float(np.min(all_vertices[:, 1]))
    horizontal_min = np.min(all_vertices[:, [0, 2]], axis=0)
    horizontal_max = np.max(all_vertices[:, [0, 2]], axis=0)
    horizontal_center = (horizontal_min + horizontal_max) * 0.5
    grid_center = np.asarray(
        [horizontal_center[0], floor_y, horizontal_center[1]], dtype=np.float64
    )
    grid_size = max(6.0, float(np.max(horizontal_max - horizontal_min)) + 3.0)
    scenes = []
    camera_nodes = []
    for _ in METHOD_ORDER:
        scene, camera_node = create_static_scene(
            shared_camera,
            floor_y=floor_y,
            grid_size=grid_size,
            grid_center=grid_center,
        )
        scenes.append(scene)
        camera_nodes.append(camera_node)
    # 高分辨率模式直接增加 OpenGL 采样像素，人物网格边缘和手脚细节不会依赖
    # 对 1080p 图片做后处理插值。
    renderer = pyrender.OffscreenRenderer(panel_width, panel_height)
    output = output_mp4.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    slow_frames: list[bytes] = []
    preview_index = frame_count // 2
    # 画面中的共享橙色标记采用 GT SMPL 的 Head/双腕世界位置；RPM/AGRoL 的
    # 官方输入正由这些关节生成，Ours 使用其 body.fbx 等价表示。
    shared_tracker_pos_world = sequences["GT"].joints_world[
        :, TRACKER_JOINT_INDICES[:3]
    ]
    try:
        for frame_index in range(frame_count):
            panel_views = []
            for method_index, method_name in enumerate(METHOD_ORDER):
                scenes[method_index].set_pose(
                    camera_nodes[method_index], pose=camera_poses[frame_index]
                )
                panel_views.append(
                    render_single_method_view(
                        renderer=renderer,
                        scene=scenes[method_index],
                        method_name=method_name,
                        frame_index=frame_index,
                        sequences=sequences,
                        rotations_world=rotations_world,
                        faces=faces,
                        tracker_pos_world=shared_tracker_pos_world,
                        camera_pose=camera_poses[frame_index],
                    )
                )
            viewport = stitch_independent_panel_views(
                panel_views,
                panel_width=panel_width,
                panel_height=panel_height,
            )
            normal = compose_frame(
                viewport_rgb=viewport,
                playback_label="1.0×",
                source_frame=clip.source_frame_start + frame_index,
                show_input_note=False,
                resolution_scale=scale,
            )
            slow = compose_frame(
                viewport_rgb=viewport,
                playback_label=slow_playback_label,
                source_frame=clip.source_frame_start + frame_index,
                show_input_note=False,
                resolution_scale=scale,
            )
            if writer is None:
                intro = compose_frame(
                    viewport_rgb=viewport,
                    playback_label="Input setup",
                    source_frame=clip.source_frame_start,
                    show_input_note=True,
                    resolution_scale=scale,
                )
                intro_rgb = np.asarray(intro, dtype=np.uint8)
                writer = Mp4FrameWriter(output_path=output, frame_rgb=intro_rgb, fps=clip.fps)
                for _ in range(INTRO_FRAME_COUNT):
                    writer.append(intro_rgb)
            writer.append(np.asarray(normal, dtype=np.uint8))
            slow_frames.append(encode_png(slow))
            if output_preview_png is not None and frame_index == preview_index:
                preview = output_preview_png.expanduser().resolve()
                preview.parent.mkdir(parents=True, exist_ok=True)
                normal.save(preview)
            print(
                f"[three-point-demo] rendered {frame_index + 1}/{frame_count} "
                f"(source frame {clip.source_frame_start + frame_index})",
                flush=True,
            )
        for encoded in slow_frames:
            rgb = decode_png(encoded)
            # 每个源帧重复 slowdown_factor 次，保持 30 FPS 编码的同时得到
            # 精确的 1 / slowdown_factor 回放速度。
            for _ in range(slowdown_factor):
                writer.append(rgb)
    finally:
        if writer is not None:
            writer.close()
        with suppress(Exception):
            renderer.delete()
    return output


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    clip = load_comparison_clip(
        comparison_npz=args.comparison_npz,
        report_json=args.comparison_report_json,
        amass_npz=args.amass_npz,
        source_frame_start=args.source_frame_start,
        source_frame_end_exclusive=args.source_frame_end_exclusive,
        diffusion_variant="core_only",
    )
    rpm_motion = select_rpm_motion(
        args.rpm_npz,
        args.source_frame_start,
        args.source_frame_end_exclusive,
    )
    agrol_motion = select_agrol_motion(
        args.agrol_npz,
        args.source_frame_start,
        args.source_frame_end_exclusive,
    )
    sequences, rotations_world, faces = build_comparison_sequences(
        clip=clip,
        rpm_motion_6d=rpm_motion,
        agrol_motion_6d=agrol_motion,
        smpl_model_dir=args.smpl_model_dir.expanduser().resolve(),
    )
    metrics = build_method_metrics(sequences, rotations_world, fps=clip.fps)
    output = render_video(
        output_mp4=args.output_mp4,
        output_preview_png=args.output_preview_png,
        clip=clip,
        sequences=sequences,
        rotations_world=rotations_world,
        faces=faces,
        resolution_scale=args.resolution_scale,
        replay_slowdown_factor=args.replay_slowdown_factor,
    )
    report = {
        "source_frame_start": int(args.source_frame_start),
        "source_frame_end_exclusive": int(args.source_frame_end_exclusive),
        "fps": int(clip.fps),
        "input_trackers": ["Head", "Left Wrist", "Right Wrist"],
        "shared_gt_root_translation": True,
        "baseline_tracker_coordinates": "official AMASS/SMPL world Head + wrists",
        "ours_tracker_coordinates": "equivalent body.fbx world Head + wrists",
        "render_layout": (
            "four independent scenes with identical camera intrinsics/extrinsics, "
            "lighting, floor and unshifted actor world coordinates"
        ),
        "output_resolution": [
            int(OUTPUT_WIDTH * args.resolution_scale),
            int(OUTPUT_HEIGHT * args.resolution_scale),
        ],
        "replay_slowdown_factor": int(args.replay_slowdown_factor),
        "agrol_alignment": (
            "build official AMASS/SMPL trackers at 60 Hz, infer one "
            "196-frame window, sample displayed poses back to 30 Hz"
        ),
        "metrics_on_rendered_clip": metrics,
        "rpm_npz": str(args.rpm_npz.expanduser().resolve()),
        "agrol_npz": str(args.agrol_npz.expanduser().resolve()),
        "output_mp4": str(output),
    }
    report_path = args.output_report_json.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[three-point-demo] wrote: {output}", flush=True)
    print(f"[three-point-demo] report: {report_path}", flush=True)
    return output


if __name__ == "__main__":
    main()
