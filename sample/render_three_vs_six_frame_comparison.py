from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_loaders.realtime_pose_kinematics import TRACKER_JOINT_INDICES
from sample.realtime_pose_smpl_rendering import (
    SmplMeshSequence,
    build_mesh_sequences,
    build_surface_aligned_glyph_points,
    create_sphere_cloud,
    create_static_scene,
    load_comparison_clip,
    load_font,
)
from sample.render_realtime_pose_smpl_presentation import (
    create_material,
    draw_centered_text,
    fit_fixed_presentation_camera,
    presentation_view_direction_unity,
)


PANEL_WIDTH = 1920
OUTPUT_HEIGHT = 2160
TRACKER_GLYPH_RADIUS = 0.032
TRACKER_GLYPH_OUTWARD_RATIO = 0.18
BODY_COLOR = (0x75 / 255.0, 0xCD / 255.0, 0xCF / 255.0, 1.0)
TRACKER_COLOR = (0.96, 0.0, 0.76, 1.0)
METHOD_NAMES = ("3 trackers", "6 trackers")
TRACKER_AVAILABLE_BY_METHOD = np.asarray(
    [
        [True, True, True, False, False, False],
        [True, True, True, True, True, True],
    ],
    dtype=bool,
)


# region CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="独立渲染同一帧三点/六点结果，并生成左右拼接的 4K 对比图。"
    )
    parser.add_argument("--comparison_npz", required=True, type=Path)
    parser.add_argument("--report_json", required=True, type=Path)
    parser.add_argument("--amass_npz", required=True, type=Path)
    parser.add_argument("--smpl_model_dir", required=True, type=Path)
    parser.add_argument("--source_frame", required=True, type=int)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


# endregion


# region 数据与相机


def load_rendered_method(
    *,
    comparison_npz: Path,
    report_json: Path,
    amass_npz: Path,
    smpl_model_dir: Path,
    source_frame: int,
    diffusion_variant: str,
) -> tuple[object, SmplMeshSequence, np.ndarray]:
    """加载单帧预测并转换成 SMPL-H；只返回最终 Diffusion 结果。"""

    clip = load_comparison_clip(
        comparison_npz=comparison_npz,
        report_json=report_json,
        amass_npz=amass_npz,
        source_frame_start=int(source_frame),
        source_frame_end_exclusive=int(source_frame) + 1,
        diffusion_variant=diffusion_variant,
    )
    sequences, faces = build_mesh_sequences(clip, smpl_model_dir)
    return clip, sequences["+ Diffusion"], np.asarray(faces, dtype=np.int64)


def build_shared_camera(
    *,
    three_sequence: SmplMeshSequence,
    six_sequence: SmplMeshSequence,
    tracker_pos_world: np.ndarray,
):
    """两个人体重叠拟合相机，保证独立场景使用完全相同的观察参数。"""

    sequences = {
        METHOD_NAMES[0]: three_sequence,
        METHOD_NAMES[1]: six_sequence,
    }
    return fit_fixed_presentation_camera(
        sequences=sequences,
        tracker_pos_world=np.asarray(tracker_pos_world, dtype=np.float32),
        follow_offsets=np.zeros((1, 3), dtype=np.float32),
        method_offsets=np.zeros((2, 3), dtype=np.float32),
        viewport_width=PANEL_WIDTH,
        viewport_height=OUTPUT_HEIGHT,
        padding=1.22,
        method_order=METHOD_NAMES,
        tracker_available_by_method=TRACKER_AVAILABLE_BY_METHOD,
        camera_view_direction=presentation_view_direction_unity(),
    )


# endregion


# region 渲染与排版


def render_method_panel(
    *,
    sequence: SmplMeshSequence,
    faces: np.ndarray,
    camera_spec,
    active_tracker_mask: np.ndarray,
    floor_y: float,
    grid_center: np.ndarray,
) -> Image.Image:
    """在独立 scene 中渲染一个人体，tracker 图标贴到该人体可见表面。"""

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    scene, _ = create_static_scene(
        camera_spec=camera_spec,
        floor_y=float(floor_y),
        grid_size=6.0,
        grid_center=np.asarray(grid_center, dtype=np.float64),
    )
    vertices = np.asarray(sequence.vertices_world[0], dtype=np.float64)
    body = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    scene.add(
        pyrender.Mesh.from_trimesh(
            body,
            material=create_material(pyrender, BODY_COLOR, 0.92),
            smooth=True,
        )
    )

    # 紫球仅表达当前方法启用了哪些 tracker 位置。以渲染骨架关节作为视觉
    # 锚点，避免 Body-FBX 与 SMPL-H 骨长差异被误看成模型预测误差。
    tracker_mask = np.asarray(active_tracker_mask, dtype=bool)
    tracker_indices = TRACKER_JOINT_INDICES[tracker_mask]
    anchor_points = np.asarray(
        sequence.joints_world[0, tracker_indices],
        dtype=np.float64,
    )
    glyph_points = build_surface_aligned_glyph_points(
        anchor_points=anchor_points,
        camera_position=np.asarray(camera_spec.pose[:3, 3], dtype=np.float64),
        body_vertices=vertices,
        body_faces=faces,
        glyph_radius=TRACKER_GLYPH_RADIUS,
        outward_offset_ratio=TRACKER_GLYPH_OUTWARD_RATIO,
    )
    tracker_cloud = create_sphere_cloud(
        glyph_points,
        radius=TRACKER_GLYPH_RADIUS,
    )
    scene.add(
        pyrender.Mesh.from_trimesh(
            tracker_cloud,
            material=create_material(pyrender, TRACKER_COLOR, 0.42),
            smooth=True,
        )
    )

    renderer = pyrender.OffscreenRenderer(
        viewport_width=PANEL_WIDTH,
        viewport_height=OUTPUT_HEIGHT,
    )
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
    finally:
        renderer.delete()
    return Image.fromarray(np.asarray(color[..., :3], dtype=np.uint8), mode="RGB")


def decorate_panel(
    image: Image.Image,
    *,
    title: str,
    subtitle: str,
    source_frame: int,
) -> Image.Image:
    """添加标题和同序列/同帧说明，不遮挡人体主体。"""

    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_box = (320, 26, PANEL_WIDTH - 320, 194)
    draw.rounded_rectangle(
        title_box,
        radius=34,
        fill=(252, 253, 254, 246),
        outline=(193, 201, 211, 255),
        width=4,
    )
    draw_centered_text(
        draw,
        (title_box[0], title_box[1] + 14, title_box[2], title_box[1] + 88),
        title,
        load_font(62),
        (28, 37, 50, 255),
    )
    draw_centered_text(
        draw,
        (title_box[0], title_box[1] + 86, title_box[2], title_box[3] - 10),
        subtitle,
        load_font(30),
        (70, 80, 97, 255),
    )

    footer_text = f"Same sequence | source frame {source_frame} | 3/4 view"
    footer_font = load_font(28)
    footer_bounds = draw.textbbox((0, 0), footer_text, font=footer_font)
    footer_width = footer_bounds[2] - footer_bounds[0] + 48
    footer_box = (
        PANEL_WIDTH - footer_width - 38,
        OUTPUT_HEIGHT - 112,
        PANEL_WIDTH - 38,
        OUTPUT_HEIGHT - 42,
    )
    draw.rounded_rectangle(footer_box, radius=15, fill=(31, 41, 55, 230))
    draw_centered_text(
        draw,
        footer_box,
        footer_text,
        footer_font,
        (247, 249, 252, 255),
    )
    return Image.alpha_composite(result, overlay).convert("RGB")


def compose_panels(three_panel: Image.Image, six_panel: Image.Image) -> Image.Image:
    """左右拼成精确 3840x2160，并增加中性分隔线。"""

    combined = Image.new("RGB", (PANEL_WIDTH * 2, OUTPUT_HEIGHT))
    combined.paste(three_panel, (0, 0))
    combined.paste(six_panel, (PANEL_WIDTH, 0))
    draw = ImageDraw.Draw(combined)
    draw.rectangle(
        (PANEL_WIDTH - 3, 0, PANEL_WIDTH + 3, OUTPUT_HEIGHT),
        fill=(190, 198, 208),
    )
    return combined


# endregion


def tracker_alignment_cm(clip, sequence: SmplMeshSequence) -> list[float]:
    """记录原始 tracker 与渲染关节的差异，便于审计可视化转换。"""

    distances = np.linalg.norm(
        np.asarray(clip.tracker_pos_world[0], dtype=np.float64)
        - np.asarray(
            sequence.joints_world[0, TRACKER_JOINT_INDICES],
            dtype=np.float64,
        ),
        axis=1,
    )
    return [float(value * 100.0) for value in distances]


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    three_clip, three_sequence, three_faces = load_rendered_method(
        comparison_npz=args.comparison_npz,
        report_json=args.report_json,
        amass_npz=args.amass_npz,
        smpl_model_dir=args.smpl_model_dir,
        source_frame=args.source_frame,
        diffusion_variant="core_only",
    )
    six_clip, six_sequence, six_faces = load_rendered_method(
        comparison_npz=args.comparison_npz,
        report_json=args.report_json,
        amass_npz=args.amass_npz,
        smpl_model_dir=args.smpl_model_dir,
        source_frame=args.source_frame,
        diffusion_variant="all_six",
    )
    np.testing.assert_array_equal(three_faces, six_faces)
    camera_spec = build_shared_camera(
        three_sequence=three_sequence,
        six_sequence=six_sequence,
        tracker_pos_world=three_clip.tracker_pos_world,
    )
    floor_y = min(
        float(np.min(three_sequence.vertices_world[..., 1])),
        float(np.min(six_sequence.vertices_world[..., 1])),
    )
    combined_vertices = np.concatenate(
        [three_sequence.vertices_world[0], six_sequence.vertices_world[0]],
        axis=0,
    )
    horizontal_min = np.min(combined_vertices[:, [0, 2]], axis=0)
    horizontal_max = np.max(combined_vertices[:, [0, 2]], axis=0)
    horizontal_center = (horizontal_min + horizontal_max) * 0.5
    grid_center = np.asarray(
        [horizontal_center[0], floor_y, horizontal_center[1]],
        dtype=np.float64,
    )

    three_panel = render_method_panel(
        sequence=three_sequence,
        faces=three_faces,
        camera_spec=camera_spec,
        active_tracker_mask=TRACKER_AVAILABLE_BY_METHOD[0],
        floor_y=floor_y,
        grid_center=grid_center,
    )
    six_panel = render_method_panel(
        sequence=six_sequence,
        faces=six_faces,
        camera_spec=camera_spec,
        active_tracker_mask=TRACKER_AVAILABLE_BY_METHOD[1],
        floor_y=floor_y,
        grid_center=grid_center,
    )
    three_panel = decorate_panel(
        three_panel,
        title="3 Trackers",
        subtitle="Head + left/right wrist",
        source_frame=args.source_frame,
    )
    six_panel = decorate_panel(
        six_panel,
        title="6 Trackers",
        subtitle="Head + wrists + hip + left/right foot",
        source_frame=args.source_frame,
    )
    combined = compose_panels(three_panel, six_panel)

    three_path = output_dir / (
        f"three_trackers_source_frame_{args.source_frame}_three_quarter_1920x2160.png"
    )
    six_path = output_dir / (
        f"six_trackers_source_frame_{args.source_frame}_three_quarter_1920x2160.png"
    )
    combined_path = output_dir / (
        f"three_vs_six_source_frame_{args.source_frame}_three_quarter_4k.png"
    )
    three_panel.save(three_path)
    six_panel.save(six_path)
    combined.save(combined_path)

    metadata = {
        "comparison_npz": str(args.comparison_npz.expanduser().resolve()),
        "report_json": str(args.report_json.expanduser().resolve()),
        "amass_npz": str(args.amass_npz.expanduser().resolve()),
        "source_frame": int(args.source_frame),
        "same_camera": True,
        "same_body_shape": True,
        "independent_scenes": True,
        "combined_resolution": [PANEL_WIDTH * 2, OUTPUT_HEIGHT],
        "tracker_source_data_modified": False,
        "tracker_glyph_anchor": "corresponding rendered SMPL-H joint",
        "tracker_glyph_placement": "first visible body-surface ray intersection",
        "fixed_camera_offset_m": 0.0,
        "tracker_glyph_radius_m": TRACKER_GLYPH_RADIUS,
        "three_source_tracker_to_rendered_joint_cm": tracker_alignment_cm(
            three_clip, three_sequence
        ),
        "six_source_tracker_to_rendered_joint_cm": tracker_alignment_cm(
            six_clip, six_sequence
        ),
        "outputs": {
            "three_trackers": str(three_path),
            "six_trackers": str(six_path),
            "combined_4k": str(combined_path),
        },
    }
    (output_dir / "render_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(combined_path)


if __name__ == "__main__":
    main()
