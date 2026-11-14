"""复用 SMPL-H 渲染，比较掉线实验中各方法的姿态。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_converter.amass_smpl_utils import load_motion_source
from data_loaders.body_fbx_kinematics import load_body_fbx_rest
from sample.analyze_dropout_visual_quality import KEYS
from sample.realtime_pose_smpl_rendering import (
    SmplMeshSequence, build_horizontal_pelvis_follow_offsets, create_static_scene, load_font,
    body_fbx_world_to_smpl_local_rotations, create_smplh_model, rotation_matrices_to_axis_angle,
    run_smplh_forward, transform_faces_to_unity_winding,
)
from sample.render_realtime_pose_smpl_presentation import (
    build_follow_camera_poses, create_material, fit_fixed_presentation_camera,
)
from utils.video_io import Mp4FrameWriter


ORDER = ("GT", "Ours", "RPM-Reactive", "RPM-Smooth")
RGBA = {"GT": (0.40, 0.44, 0.50, 1), "Ours": (0.05, 0.61, 0.61, 1),
        "RPM-Reactive": (0.81, 0.43, 0.20, 1), "RPM-Smooth": (0.49, 0.38, 0.69, 1)}
PANEL_WIDTH, PANEL_HEIGHT = 480, 720


def build_case_meshes(payload, start, end, smpl_model_dir):
    """沿用项目的逆重定向和 SMPL-H FK，所有方法共用 GT 身形和头部锚点。"""
    raw_path = Path(str(payload["amass_path"]))
    motion = load_motion_source(raw_path, raw_path.parent, target_fps=30)
    rest = load_body_fbx_rest(str(payload["rest_path"]))
    method_keys = {method: KEYS[method] for method in ORDER}
    count = end - start
    model = create_smplh_model(smpl_model_dir, motion.gender, count)
    sequences = {"GT": run_smplh_forward(
        model=model, pose_axis_angle=motion.poses[start:end, :66].reshape(count, 22, 3),
        betas=motion.betas, translation_amass=motion.trans[start:end],
    )}
    for method, key in method_keys.items():
        if method == "GT":
            continue
        local = body_fbx_world_to_smpl_local_rotations(
            payload[f"{key}_rotations"][start:end], payload[f"{key}_root_yaw"][start:end],
            rest.rest_local_rotations, rest.parents,
        )[:, :22]
        mesh = run_smplh_forward(
            model=model, pose_axis_angle=rotation_matrices_to_axis_angle(local),
            betas=motion.betas, translation_amass=np.zeros((count, 3), dtype=np.float32),
        )
        # 仅通过在线头部定位 avatar，保留预测头部残差；不把 GT 根平移直接赋给预测。
        head_residual = payload[f"{key}_positions"][start:end, 15] - payload["gt_positions"][start:end, 15]
        translation = sequences["GT"].joints_world[:, 15] + head_residual - mesh.joints_world[:, 15]
        sequences[method] = SmplMeshSequence(
            vertices_world=mesh.vertices_world + translation[:, None],
            joints_world=mesh.joints_world + translation[:, None],
        )
    return sequences, transform_faces_to_unity_winding(model.faces)


def render_mesh_view(renderer, scene, vertices, faces, color):
    import pyrender
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=create_material(pyrender, color, 0.92), smooth=True))
    try:
        pixels, _ = renderer.render(scene)
        return pixels[..., :3]
    finally:
        scene.remove_node(node)


def render_case(event, root, renderer, writer, case_index, smpl_model_dir):
    output_dir = root
    output_dir.mkdir(parents=True, exist_ok=True)
    order = ORDER
    panel_width = PANEL_WIDTH
    with np.load(root / "aligned" / event["file"], allow_pickle=False) as archive:
        payload = dict(archive)
        start, end = event["gap_start"] - 15, event["gap_end"] + 15
        sequences, faces = build_case_meshes(payload, start, end, smpl_model_dir)
        available = payload["tracker_available"][start:end]
        # 地面仅由 GT 定义，不能根据各方法的最低脚位置分别移动地面。
        floor_y = float(np.percentile(sequences["GT"].vertices_world[..., 1].min(axis=1), 5))
    trajectories = {m: sequences[m].joints_world for m in order}
    follow = build_horizontal_pelvis_follow_offsets(trajectories["GT"])
    trackers = np.zeros((end - start, 6, 3), dtype=np.float32)
    trackers[:, :3] = trajectories["GT"][:, [15, 20, 21]]
    camera = fit_fixed_presentation_camera(
        sequences=sequences, tracker_pos_world=trackers, follow_offsets=follow,
        method_offsets=np.zeros((len(order), 3)), viewport_width=panel_width, viewport_height=PANEL_HEIGHT,
        method_order=order, tracker_available_by_method=np.tile([1, 1, 1, 0, 0, 0], (len(order), 1)).astype(bool), padding=1.2,
    )
    poses = build_follow_camera_poses(camera.pose, follow)
    center = trajectories["GT"][:, 0].mean(axis=0)
    center[1] = floor_y
    scenes, cameras = [], []
    for method in order:
        scene, camera_node = create_static_scene(camera, floor_y=floor_y, grid_size=8.0, grid_center=center)
        scenes.append(scene)
        cameras.append(camera_node)
    title_font, body_font = load_font(27), load_font(21)
    preview_at = min(event["gap_start"] + 8, event["gap_end"] - 1) - start
    case_writer = None
    try:
        for local_index, source_frame in enumerate(range(start, end)):
            panels = []
            for method_index, method in enumerate(order):
                scenes[method_index].set_pose(cameras[method_index], pose=poses[local_index])
                panels.append(render_mesh_view(renderer, scenes[method_index], sequences[method].vertices_world[local_index], faces, RGBA[method]))
            image = Image.new("RGB", (1920, 900), "#f5f6f8")
            image.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, 100))
            draw = ImageDraw.Draw(image)
            draw.text((24, 12), f"Case {case_index + 1}/4 | {event['selection']} | 1x, 30 fps", fill="#172230", font=title_font)
            for index, method in enumerate(order):
                draw.text((index * panel_width + 24, 58), method, fill="#172230", font=title_font)
            hand_status = "  ".join(f"{label}: {'ON' if available[local_index, tracker] else 'LOST'}"
                                    for label, tracker in (("Left hand", 1), ("Right hand", 2)))
            draw.text((24, 830), f"t={(source_frame - event['gap_start']) / 30:+.2f}s | {hand_status} | source frame {source_frame}", fill="#172230", font=body_font)
            draw.text((24, 862), f"{event['sequence']} | Shared SMPL-H body; no display smoothing", fill="#485464", font=body_font)
            frame = np.asarray(image)
            if writer is None:
                writer = Mp4FrameWriter(output_dir / "dropout_comparison.mp4", frame, fps=30)
            if case_writer is None:
                case_writer = Mp4FrameWriter(output_dir / f"case_{case_index + 1:02d}.mp4", frame, fps=30)
            writer.append(frame)
            case_writer.append(frame)
            if local_index == preview_at:
                image.save(output_dir / f"case_{case_index + 1:02d}_preview.png")
    finally:
        if case_writer is not None:
            case_writer.close()
    return writer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--smpl_model_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import torch
    torch.set_num_threads(2)

    selected = json.loads((args.experiment_dir / "selected_events.json").read_text())
    output_dir = args.experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_width = PANEL_WIDTH
    renderer = pyrender.OffscreenRenderer(panel_width, PANEL_HEIGHT)
    writer = Mp4FrameWriter(output_dir / "dropout_comparison.mp4", np.zeros((900, 1920, 3), dtype=np.uint8), fps=30)
    try:
        for index, event in enumerate(selected):
            writer = render_case(event, args.experiment_dir, renderer, writer, index, args.smpl_model_dir)
            print(f"[dropout-render] {index + 1}/4 {event['selection']}", flush=True)
    finally:
        renderer.delete()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
