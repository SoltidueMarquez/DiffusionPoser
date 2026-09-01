from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from data_converter.amass_smpl_utils import AMASS_TO_UNITY
from data_loaders.body_fbx_kinematics import BodyFbxRest, load_body_fbx_rest
from data_loaders.realtime_pose_geometry import extract_forward_yaw_np
from data_loaders.sensor_masking import CORE_TRACKER_INDICES, REALTIME_POSE_FPS
from sample.infer_unity_recording import (
    DEFAULT_WARMUP_FRAMES,
    ResampledTrackerRecording,
    apply_tracker_availability_overrides,
    find_first_core_window,
    load_unity_tracker_recording,
    resample_tracker_recording,
)
from sample.realtime_pose_smpl_rendering import (
    SmplMeshSequence,
    body_fbx_world_to_smpl_local_rotations,
    create_smplh_model,
    create_sphere_cloud,
    create_static_scene,
    load_font,
    normalize_vector,
    require_directory,
    require_file,
    rotation_matrices_to_axis_angle,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)
from sample.render_realtime_pose_smpl_presentation import (
    build_presentation_layout,
    build_visible_tracker_glyph_points,
    create_material,
)
from utils.video_io import Mp4FrameWriter


OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
DEFAULT_CAMERA_REFERENCE_FRAMES = 10
DEFAULT_CAMERA_ELEVATION_DEG = 12.0
DEFAULT_CAMERA_FIT_PADDING = 1.22
BODY_COLOR = (0.22, 0.72, 0.69, 1.0)
TRACKER_COLOR = (1.0, 0.55, 0.05, 1.0)


# region 数据契约


@dataclass(frozen=True)
class UnityPoseRecording:
    """离线推理输出的 Unity Actor Transform 时间序列。"""

    fps: int
    times: np.ndarray  # [T]
    root_positions: np.ndarray  # [T,3]
    root_rotations_xyzw: np.ndarray  # [T,4]
    pelvis_local_positions: np.ndarray  # [T,3]
    local_rotations_xyzw: np.ndarray  # [T,24,4]

    @property
    def frame_count(self) -> int:
        return int(self.times.shape[0])


@dataclass(frozen=True)
class UnityRenderInputs:
    """完成时间轴对齐、可直接转网格和渲染的 Unity 数据。"""

    pose: UnityPoseRecording
    rest: BodyFbxRest
    tracker_positions: np.ndarray  # [T,6,3]
    tracker_available: np.ndarray  # [T,6]
    floor_y: float
    world_positions: np.ndarray  # [T,24,3]
    world_rotations: np.ndarray  # [T,24,3,3]
    root_yaw: np.ndarray  # [T]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "把 infer_unity_recording 输出的 Pose JSON 渲染为独立 SMPL-H 视频；"
            "相机根据开头姿态自动放到人物正面。"
        )
    )
    paths = parser.add_argument_group("Unity recording paths")
    paths.add_argument("--pose_json", required=True, type=Path)
    paths.add_argument("--tracker_json", required=True, type=Path)
    paths.add_argument("--body_fbx_rest_json", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_mp4", required=True, type=Path)
    timeline = parser.add_argument_group("timeline")
    timeline.add_argument(
        "--warmup_frames",
        default=DEFAULT_WARMUP_FRAMES,
        type=int,
        help="必须与生成 pose_json 时 infer_unity_recording 的 warmup_frames 一致。",
    )
    timeline.add_argument(
        "--ignore_hip",
        action="store_true",
        help="不显示 Hip，并按五点配置拟合相机；应与推理配置保持一致。",
    )
    timeline.add_argument(
        "--ignore_feet",
        action="store_true",
        help="不显示左右脚 Tracker；应与三点推理的 --ignore_feet 保持一致。",
    )
    camera = parser.add_argument_group("front camera")
    camera.add_argument(
        "--camera_reference_frames",
        default=DEFAULT_CAMERA_REFERENCE_FRAMES,
        type=int,
        help="用开头多少帧估计渲染网格的正面方向。",
    )
    camera.add_argument(
        "--camera_yaw_offset_deg",
        default=0.0,
        type=float,
        help="在自动正面方向上追加水平旋转；180 可查看背面。",
    )
    camera.add_argument(
        "--camera_elevation_deg",
        default=DEFAULT_CAMERA_ELEVATION_DEG,
        type=float,
    )
    camera.add_argument(
        "--camera_fit_padding",
        default=DEFAULT_CAMERA_FIT_PADDING,
        type=float,
    )
    camera.add_argument("--overwrite", action="store_true")
    return parser


def load_unity_pose_recording(path: str | Path) -> UnityPoseRecording:
    pose_path = require_file(Path(path), "pose_json")
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    frames = payload.get("frames", ())
    if not isinstance(frames, list) or not frames:
        raise ValueError("pose_json.frames 必须是非空数组。")
    fps = int(payload.get("fps", 0))
    if fps <= 0:
        raise ValueError("pose_json.fps 必须为正整数。")

    times = np.asarray([frame["time"] for frame in frames], dtype=np.float64)
    root_positions = np.asarray(
        [frame["rootPosition"] for frame in frames], dtype=np.float32
    )
    root_rotations = np.asarray(
        [frame["rootRotation"] for frame in frames], dtype=np.float64
    )
    pelvis_positions = np.asarray(
        [frame["pelvisLocalPosition"] for frame in frames], dtype=np.float32
    )
    local_rotations = np.asarray(
        [frame["localRotations"] for frame in frames], dtype=np.float64
    )
    frame_count = len(frames)
    expected_shapes = {
        "time": (frame_count,),
        "rootPosition": (frame_count, 3),
        "rootRotation": (frame_count, 4),
        "pelvisLocalPosition": (frame_count, 3),
        "localRotations": (frame_count, 24, 4),
    }
    values = {
        "time": times,
        "rootPosition": root_positions,
        "rootRotation": root_rotations,
        "pelvisLocalPosition": pelvis_positions,
        "localRotations": local_rotations,
    }
    for name, expected_shape in expected_shapes.items():
        value = values[name]
        if value.shape != expected_shape:
            raise ValueError(f"{name} 应为 {expected_shape}，实际为 {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} 含 NaN/Inf。")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("pose_json.time 必须严格递增。")

    for name, quaternions in (
        ("rootRotation", root_rotations),
        ("localRotations", local_rotations),
    ):
        norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        if np.any(norms <= 1e-8):
            raise ValueError(f"{name} 含零 quaternion。")
        quaternions /= norms
    return UnityPoseRecording(
        fps=fps,
        times=times,
        root_positions=root_positions,
        root_rotations_xyzw=root_rotations.astype(np.float32),
        pelvis_local_positions=pelvis_positions,
        local_rotations_xyzw=local_rotations.astype(np.float32),
    )


def select_tracker_frame_indices(
    *,
    recording: ResampledTrackerRecording,
    pose_times: np.ndarray,
    warmup_frames: int,
) -> np.ndarray:
    """严格复现推理入口的输出时序，避免用“截取最后 N 帧”猜测对齐。"""

    start = find_first_core_window(recording.available)
    output_indices: list[int] = []
    has_previous_pose = False
    for current in range(start + 11, len(recording.times)):
        core_available = bool(
            recording.available[current, list(CORE_TRACKER_INDICES)].all()
        )
        if core_available:
            has_previous_pose = True
            output_indices.append(current)
        elif has_previous_pose:
            # 推理入口在核心点短时掉线时保持上一帧，但仍写入当前时间戳。
            output_indices.append(current)

    warmup = max(0, int(warmup_frames))
    selected = np.asarray(output_indices[warmup:], dtype=np.int64)
    times = np.asarray(pose_times, dtype=np.float64)
    if selected.shape != times.shape:
        raise ValueError(
            "pose_json 帧数与 tracker_json/warmup_frames 不一致："
            f"pose={times.shape[0]}, tracker={selected.shape[0]}。"
        )
    expected_times = recording.times[selected] - recording.times[selected[0]]
    if not np.allclose(expected_times, times, rtol=0.0, atol=1e-6):
        raise ValueError("pose_json.time 与 tracker_json 重建时间轴不一致。")
    return selected


def reconstruct_body_fbx_world(
    pose: UnityPoseRecording,
    rest: BodyFbxRest,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 Actor root + 24 个父局部旋转恢复为 `[T,24]` 世界 FK。"""

    frame_count = pose.frame_count
    heading_rotations = Rotation.from_quat(
        pose.root_rotations_xyzw.astype(np.float64)
    ).as_matrix()
    local_rotations = Rotation.from_quat(
        pose.local_rotations_xyzw.reshape(-1, 4).astype(np.float64)
    ).as_matrix().reshape(frame_count, 24, 3, 3)
    world_positions = np.empty((frame_count, 24, 3), dtype=np.float64)
    world_rotations = np.empty((frame_count, 24, 3, 3), dtype=np.float64)
    for joint_index, parent_index in enumerate(rest.parents.tolist()):
        if parent_index < 0:
            world_rotations[:, joint_index] = (
                heading_rotations @ local_rotations[:, joint_index]
            )
            world_positions[:, joint_index] = pose.root_positions + np.einsum(
                "tij,tj->ti",
                heading_rotations,
                pose.pelvis_local_positions,
            )
        else:
            parent_rotations = world_rotations[:, parent_index]
            world_rotations[:, joint_index] = (
                parent_rotations @ local_rotations[:, joint_index]
            )
            world_positions[:, joint_index] = (
                world_positions[:, parent_index]
                + np.einsum(
                    "tij,j->ti",
                    parent_rotations,
                    rest.rest_local_positions[joint_index],
                )
            )
    root_yaw = extract_forward_yaw_np(heading_rotations)
    return (
        world_positions.astype(np.float32),
        world_rotations.astype(np.float32),
        root_yaw.astype(np.float32),
    )


def load_render_inputs(
    *,
    pose_json: Path,
    tracker_json: Path,
    body_fbx_rest_json: Path,
    warmup_frames: int,
    ignore_hip: bool,
    ignore_feet: bool,
) -> UnityRenderInputs:
    pose = load_unity_pose_recording(pose_json)
    if pose.fps != int(REALTIME_POSE_FPS):
        raise ValueError(
            f"Unity 渲染当前只支持 {REALTIME_POSE_FPS:g}Hz，实际为 {pose.fps}Hz。"
        )
    raw_recording = apply_tracker_availability_overrides(
        load_unity_tracker_recording(tracker_json),
        ignore_hip=bool(ignore_hip),
        ignore_feet=bool(ignore_feet),
    )
    recording = resample_tracker_recording(raw_recording, fps=float(pose.fps))
    selected = select_tracker_frame_indices(
        recording=recording,
        pose_times=pose.times,
        warmup_frames=warmup_frames,
    )
    rest = load_body_fbx_rest(body_fbx_rest_json)
    world_positions, world_rotations, root_yaw = reconstruct_body_fbx_world(
        pose, rest
    )
    return UnityRenderInputs(
        pose=pose,
        rest=rest,
        tracker_positions=np.asarray(recording.positions[selected], dtype=np.float32),
        tracker_available=np.asarray(recording.available[selected], dtype=bool),
        floor_y=float(recording.floor_y),
        world_positions=world_positions,
        world_rotations=world_rotations,
        root_yaw=root_yaw,
    )


# endregion


# region SMPL-H 与正面相机


def build_smpl_mesh_sequence(
    *,
    inputs: UnityRenderInputs,
    smpl_model_dir: Path,
) -> tuple[SmplMeshSequence, np.ndarray, np.ndarray]:
    """恢复渲染网格，并返回逐帧 SMPL 正面方向。"""

    smpl_local = body_fbx_world_to_smpl_local_rotations(
        inputs.world_rotations,
        inputs.root_yaw,
        inputs.rest.rest_local_rotations,
        inputs.rest.parents,
    )
    pose_axis_angle = rotation_matrices_to_axis_angle(smpl_local[:, :22])
    model = create_smplh_model(
        model_dir=require_directory(smpl_model_dir, "smpl_model_dir"),
        gender="neutral",
        batch_size=inputs.pose.frame_count,
    )
    zero_translation = np.zeros((inputs.pose.frame_count, 3), dtype=np.float32)
    sequence_zero = run_smplh_forward(
        model=model,
        pose_axis_angle=pose_axis_angle,
        betas=np.zeros((10,), dtype=np.float32),
        translation_amass=zero_translation,
    )
    # Actor root 不是 AMASS transl。以两边共同的 pelvis joint 做逐帧刚性平移，
    # 保证录制 Tracker、runtime FK 和展示网格处于同一 Unity 世界坐标。
    translation = inputs.world_positions[:, 0] - sequence_zero.joints_world[:, 0]
    sequence = SmplMeshSequence(
        vertices_world=(
            sequence_zero.vertices_world + translation[:, None]
        ).astype(np.float32),
        joints_world=(sequence_zero.joints_world + translation[:, None]).astype(
            np.float32
        ),
    )

    # 中性 SMPL-H 在 AMASS 坐标中朝 +Y；转换到 Unity 后对应 +Z。
    face_amass = np.einsum(
        "tij,j->ti",
        smpl_local[:, 0].astype(np.float64),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    )
    face_unity = face_amass @ AMASS_TO_UNITY.T
    face_unity[:, 1] = 0.0
    norms = np.linalg.norm(face_unity, axis=-1, keepdims=True)
    face_unity = face_unity / np.maximum(norms, 1e-8)
    return (
        sequence,
        transform_faces_to_unity_winding(model.faces),
        face_unity.astype(np.float32),
    )


def estimate_initial_face_direction(
    face_directions: np.ndarray,
    reference_frames: int,
) -> np.ndarray:
    """用开头短窗口的圆均值抵消单帧姿态噪声，返回水平正面方向。"""

    directions = np.asarray(face_directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3 or not len(directions):
        raise ValueError(f"face_directions 应为非空 [T,3]，实际为 {directions.shape}")
    count = min(len(directions), max(1, int(reference_frames)))
    selected = directions[:count].copy()
    selected[:, 1] = 0.0
    selected_norms = np.linalg.norm(selected, axis=-1)
    selected = selected[selected_norms > 1e-8]
    if not len(selected):
        raise ValueError("开头姿态无法确定水平正面方向。")
    selected /= np.linalg.norm(selected, axis=-1, keepdims=True)
    mean_direction = np.mean(selected, axis=0)
    mean_direction[1] = 0.0
    return normalize_vector(mean_direction)


def build_front_camera_direction(
    *,
    face_direction: np.ndarray,
    yaw_offset_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """把人物正面水平向量转换成“目标到相机”的三维观察方向。"""

    horizontal = normalize_vector(np.asarray(face_direction, dtype=np.float64))
    horizontal[1] = 0.0
    horizontal = normalize_vector(horizontal)
    yaw_rotation = Rotation.from_euler("y", float(yaw_offset_deg), degrees=True)
    horizontal = yaw_rotation.apply(horizontal)
    elevation = math.radians(float(elevation_deg))
    direction = np.asarray(
        [
            horizontal[0] * math.cos(elevation),
            math.sin(elevation),
            horizontal[2] * math.cos(elevation),
        ],
        dtype=np.float64,
    )
    return normalize_vector(direction)


# endregion


# region 视频渲染


def compose_unity_frame(
    *,
    viewport_rgb: np.ndarray,
    frame_index: int,
    frame_count: int,
    time_seconds: float,
    tracker_count: int,
    ignore_hip: bool,
    ignore_feet: bool,
) -> np.ndarray:
    image = Image.fromarray(np.asarray(viewport_rgb, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    tracker_text = f"{tracker_count} trackers"
    if ignore_hip and ignore_feet:
        tracker_text += " · Hip/feet ignored"
    elif ignore_hip:
        tracker_text += " · Hip ignored"
    elif ignore_feet:
        tracker_text += " · Feet ignored"
    draw.rounded_rectangle(
        (OUTPUT_WIDTH - 390, 20, OUTPUT_WIDTH - 24, 82),
        radius=14,
        fill=(255, 255, 255, 232),
        outline=(205, 212, 220, 245),
        width=2,
    )
    draw.text(
        (OUTPUT_WIDTH - 365, 36),
        tracker_text,
        font=load_font(20),
        fill=(202, 87, 32, 255),
    )
    footer_top = OUTPUT_HEIGHT - 48
    draw.rectangle(
        (0, footer_top, OUTPUT_WIDTH, OUTPUT_HEIGHT), fill=(255, 255, 255, 232)
    )
    progress = frame_index / max(1, frame_count - 1)
    draw.rectangle(
        (0, footer_top, int(round(OUTPUT_WIDTH * progress)), footer_top + 5),
        fill=(45, 184, 166, 255),
    )
    draw.text(
        (24, footer_top + 14),
        (
            f"{frame_index + 1}/{frame_count}   ·   "
            f"{time_seconds:.2f} s   ·   {REALTIME_POSE_FPS:g} Hz"
        ),
        font=load_font(18),
        fill=(55, 65, 81, 255),
    )
    return np.asarray(image, dtype=np.uint8)


def render_unity_pose_video(
    *,
    inputs: UnityRenderInputs,
    sequence: SmplMeshSequence,
    faces: np.ndarray,
    camera_view_direction: np.ndarray,
    camera_fit_padding: float,
    output_mp4: Path,
    ignore_hip: bool,
    ignore_feet: bool,
    overwrite: bool,
) -> Path:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import pyrender
        import trimesh
    except ImportError as exc:
        raise ImportError("缺少 pyrender/trimesh，无法执行 Unity 离屏渲染。") from exc

    output = Path(output_mp4).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("output_mp4 必须使用 .mp4 后缀。")
    if output.exists() and not bool(overwrite):
        raise FileExistsError(f"输出已存在；如需覆盖请传 --overwrite：{output}")
    if float(camera_fit_padding) <= 0.0:
        raise ValueError("camera_fit_padding 必须为正数。")

    sequences = {"Prediction": sequence}
    visible_tracker_mask = np.any(inputs.tracker_available, axis=0)[None]
    layout = build_presentation_layout(
        sequences=sequences,
        tracker_pos_world=inputs.tracker_positions,
        method_order=("Prediction",),
        tracker_available_by_method=visible_tracker_mask,
        follow_method_name="Prediction",
        camera_fit_padding=float(camera_fit_padding),
        camera_view_direction=np.asarray(camera_view_direction, dtype=np.float64),
    )
    scene, camera_node = create_static_scene(
        layout.base_camera,
        floor_y=float(inputs.floor_y),
        grid_size=layout.grid_size,
        grid_center=layout.grid_center,
    )
    renderer = pyrender.OffscreenRenderer(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    body_material = create_material(pyrender, BODY_COLOR, 0.92)
    tracker_material = create_material(pyrender, TRACKER_COLOR, 0.48)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: Mp4FrameWriter | None = None
    try:
        for frame_index in range(inputs.pose.frame_count):
            scene.set_pose(camera_node, pose=layout.camera_poses[frame_index])
            nodes = []
            try:
                body = trimesh.Trimesh(
                    vertices=np.asarray(
                        sequence.vertices_world[frame_index], dtype=np.float64
                    ),
                    faces=np.asarray(faces, dtype=np.int64),
                    process=False,
                )
                nodes.append(
                    scene.add(
                        pyrender.Mesh.from_trimesh(
                            body, material=body_material, smooth=True
                        )
                    )
                )
                tracker_mask = np.asarray(
                    inputs.tracker_available[frame_index], dtype=bool
                )
                tracker_points = build_visible_tracker_glyph_points(
                    np.asarray(
                        inputs.tracker_positions[frame_index], dtype=np.float64
                    ),
                    np.asarray(layout.camera_poses[frame_index])[:3, 3],
                )
                tracker_cloud = create_sphere_cloud(
                    tracker_points[tracker_mask], radius=0.032
                )
                if tracker_cloud is not None:
                    nodes.append(
                        scene.add(
                            pyrender.Mesh.from_trimesh(
                                tracker_cloud,
                                material=tracker_material,
                                smooth=True,
                            )
                        )
                    )
                viewport_rgb, _ = renderer.render(
                    scene, flags=pyrender.RenderFlags.NONE
                )
            finally:
                for node in nodes:
                    scene.remove_node(node)

            frame_rgb = compose_unity_frame(
                viewport_rgb=np.asarray(viewport_rgb[..., :3], dtype=np.uint8),
                frame_index=frame_index,
                frame_count=inputs.pose.frame_count,
                time_seconds=float(inputs.pose.times[frame_index]),
                tracker_count=int(inputs.tracker_available[frame_index].sum()),
                ignore_hip=bool(ignore_hip),
                ignore_feet=bool(ignore_feet),
            )
            if writer is None:
                writer = Mp4FrameWriter(output, frame_rgb, inputs.pose.fps)
            writer.append(frame_rgb)
            if frame_index % 30 == 0 or frame_index + 1 == inputs.pose.frame_count:
                print(
                    f"[unity-render] {frame_index + 1}/{inputs.pose.frame_count}",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()
        with suppress(Exception):
            renderer.delete()
    return output


# endregion


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    if int(args.warmup_frames) < 0:
        raise ValueError("warmup_frames 不能为负数。")
    if int(args.camera_reference_frames) <= 0:
        raise ValueError("camera_reference_frames 必须为正整数。")
    inputs = load_render_inputs(
        pose_json=args.pose_json,
        tracker_json=args.tracker_json,
        body_fbx_rest_json=args.body_fbx_rest_json,
        warmup_frames=int(args.warmup_frames),
        ignore_hip=bool(args.ignore_hip),
        ignore_feet=bool(args.ignore_feet),
    )
    sequence, faces, face_directions = build_smpl_mesh_sequence(
        inputs=inputs,
        smpl_model_dir=args.smpl_model_dir,
    )
    face_direction = estimate_initial_face_direction(
        face_directions,
        reference_frames=int(args.camera_reference_frames),
    )
    camera_view_direction = build_front_camera_direction(
        face_direction=face_direction,
        yaw_offset_deg=float(args.camera_yaw_offset_deg),
        elevation_deg=float(args.camera_elevation_deg),
    )
    output = render_unity_pose_video(
        inputs=inputs,
        sequence=sequence,
        faces=faces,
        camera_view_direction=camera_view_direction,
        camera_fit_padding=float(args.camera_fit_padding),
        output_mp4=args.output_mp4,
        ignore_hip=bool(args.ignore_hip),
        ignore_feet=bool(args.ignore_feet),
        overwrite=bool(args.overwrite),
    )
    sidecar = {
        "experiment": "unity_recording_pose_front_camera_render",
        "pose_json": str(require_file(args.pose_json, "pose_json")),
        "tracker_json": str(require_file(args.tracker_json, "tracker_json")),
        "frames": inputs.pose.frame_count,
        "fps": inputs.pose.fps,
        "ignore_hip": bool(args.ignore_hip),
        "ignore_feet": bool(args.ignore_feet),
        "warmup_frames": int(args.warmup_frames),
        "camera_reference_frames": int(args.camera_reference_frames),
        "camera_yaw_offset_deg": float(args.camera_yaw_offset_deg),
        "camera_elevation_deg": float(args.camera_elevation_deg),
        "camera_fit_padding": float(args.camera_fit_padding),
        "estimated_face_direction": face_direction.tolist(),
        "camera_view_direction": camera_view_direction.tolist(),
        "video_path": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[unity-render] wrote {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
