from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_converter.amass_smpl_utils import (
    SOURCE_BODY_JOINT_COUNT,
    build_smpl_local_rotations,
    load_motion_source,
    local_to_global_rotations,
    transform_rotations_to_unity,
)
from data_loaders.realtime_pose_kinematics import JOINT_INDEX, SMPL_PARENTS
from sample.realtime_pose_smpl_rendering import (
    camera_pose_look_at,
    create_smplh_model,
    require_directory,
    require_file,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)
from sample.render_rpm_past_motion import build_front_alignment_rotation


TARGET_FPS = 30.0
DEFAULT_CURRENT_FRAME = 180
DEFAULT_FRAME_INDICES = (160, 165, 170, 175, 179, 180)
TRACKER_FRAME_COUNT = len(DEFAULT_FRAME_INDICES)
HISTORY_FRAME_COUNT = TRACKER_FRAME_COUNT - 1
DEFAULT_OUTPUT = Path(
    "output/主方法图所需材料与参考/"
    "Predictor_Tracker选择_五帧六点示意.png"
)
OUTPUT_WIDTH = 1900
OUTPUT_HEIGHT = 620
STAGE_GAP_METERS = 0.18
CURRENT_FRAME_EXTRA_GAP_METERS = 0.14
CAMERA_PADDING = 1.16
WHITE = (255, 255, 255, 255)
BODY_COLOR = (0.73, 0.74, 0.76, 1.0)
BODY_OPACITY = 0.55
TRACKER_FILL = (37, 124, 192, 255)
TRACKER_BORDER = (19, 72, 117, 255)
CORE_HALO = (49, 178, 119, 62)
CROSS_COLOR = (224, 54, 65, 255)
TRACKER_SYMBOL_SCALE = 1.5

# Predictor 的 54D sparse feature 只读取 Head + 双腕。其余三个 Tracker
# 即使真实存在于观测中，也不进入 Predictor 的 core tracker context。
TRACKER_SPECS = (
    ("head", JOINT_INDEX["head"], True),
    ("left_wrist", JOINT_INDEX["left_wrist"], True),
    ("right_wrist", JOINT_INDEX["right_wrist"], True),
    ("hip", JOINT_INDEX["pelvis"], False),
    ("left_foot", JOINT_INDEX["left_foot"], False),
    ("right_foot", JOINT_INDEX["right_foot"], False),
)


@dataclass(frozen=True)
class TrackerMotion:
    """五帧历史加当前帧的男性 SMPL-H 运动与展示元数据。"""

    vertices: np.ndarray  # [6,V,3]
    joints: np.ndarray  # [6,22,3]
    faces: np.ndarray  # [F,3]
    frame_indices: np.ndarray  # [6]
    current_frame: int
    presentation_yaw_deg: float


@dataclass(frozen=True)
class TrackerLayout:
    """五帧历史加当前帧横排后的网格、关节和相机参数。"""

    vertices: np.ndarray  # [6,V,3]
    joints: np.ndarray  # [6,22,3]
    target: np.ndarray  # [3]
    camera_pose: np.ndarray  # [4,4]
    xmag: float
    ymag: float


# region CLI 与动作加载


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "绘制五帧历史加当前帧的六点 Tracker 人体序列，并给不进入 "
            "Predictor 的腰部与双脚 Tracker 叠加红叉。"
        )
    )
    parser.add_argument("--amass_npz", required=True, type=Path)
    parser.add_argument("--smpl_model_dir", required=True, type=Path)
    parser.add_argument("--output_png", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument(
        "--frame_indices",
        nargs=TRACKER_FRAME_COUNT,
        default=DEFAULT_FRAME_INDICES,
        type=int,
        metavar=("F0", "F1", "F2", "F3", "F4", "CURRENT"),
    )
    parser.add_argument(
        "--current_frame", default=DEFAULT_CURRENT_FRAME, type=int
    )
    return parser


def validate_frames(
    frame_indices: np.ndarray,
    *,
    current_frame: int,
    frame_count: int,
) -> np.ndarray:
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.shape != (TRACKER_FRAME_COUNT,):
        raise ValueError(
            "Tracker 序列必须包含五帧历史加一个当前帧："
            f"{frames.shape}。"
        )
    if not np.all(np.diff(frames) > 0):
        raise ValueError(f"frame_indices 必须严格递增：{frames.tolist()}。")
    if np.any(frames < 0) or np.any(frames >= int(frame_count)):
        raise ValueError(
            f"frame_indices 超出 [0,{frame_count})：{frames.tolist()}。"
        )
    if int(frames[-1]) != int(current_frame):
        raise ValueError(
            "最后一个 Tracker 帧必须等于 current_frame："
            f"{int(frames[-1])} != {int(current_frame)}。"
        )
    return frames


def load_tracker_motion(
    *,
    amass_npz: Path,
    smpl_model_dir: Path,
    frame_indices: np.ndarray,
    current_frame: int,
) -> TrackerMotion:
    """加载同一 Salsa 序列的五帧历史和当前帧，并转移到男性 SMPL-H。"""

    source_path = require_file(amass_npz, "amass_npz")
    source = load_motion_source(
        path=source_path,
        amass_dir=source_path.parent,
        target_fps=TARGET_FPS,
    )
    frames = validate_frames(
        frame_indices,
        current_frame=int(current_frame),
        frame_count=int(source.poses.shape[0]),
    )
    pose_axis_angle = np.asarray(
        source.poses[frames, : SOURCE_BODY_JOINT_COUNT * 3],
        dtype=np.float32,
    ).reshape(TRACKER_FRAME_COUNT, SOURCE_BODY_JOINT_COUNT, 3)
    model = create_smplh_model(
        model_dir=require_directory(smpl_model_dir, "smpl_model_dir"),
        gender="male",
        batch_size=TRACKER_FRAME_COUNT,
    )
    sequence = run_smplh_forward(
        model=model,
        pose_axis_angle=pose_axis_angle,
        betas=np.zeros((10,), dtype=np.float32),
        translation_amass=np.zeros(
            (TRACKER_FRAME_COUNT, 3), dtype=np.float32
        ),
    )

    # 与 Past Motion 图使用同一正面展示朝向，确保两块输入模块可以并排。
    selected_poses = np.asarray(source.poses[frames], dtype=np.float64)
    local_rotations = build_smpl_local_rotations(selected_poses)
    global_rotations = local_to_global_rotations(local_rotations, SMPL_PARENTS)
    global_rotations_unity = transform_rotations_to_unity(global_rotations)
    # 该共享 helper 固定接收五帧；取末五帧只用于计算最后一个 current
    # frame 的正面展示 yaw，不会丢弃最早历史帧的姿态。
    front_rotation, presentation_yaw_deg = build_front_alignment_rotation(
        global_rotations_unity[-HISTORY_FRAME_COUNT:]
    )
    return TrackerMotion(
        vertices=(
            np.asarray(sequence.vertices_world, dtype=np.float64)
            @ front_rotation.T
        ).astype(np.float32),
        joints=(
            np.asarray(sequence.joints_world, dtype=np.float64)
            @ front_rotation.T
        ).astype(np.float32),
        faces=transform_faces_to_unity_winding(model.faces),
        frame_indices=frames,
        current_frame=int(current_frame),
        presentation_yaw_deg=float(presentation_yaw_deg),
    )


# endregion


# region 横排布局与人体渲染


def build_layout(motion: TrackerMotion) -> TrackerLayout:
    vertices = np.asarray(motion.vertices, dtype=np.float64).copy()
    joints = np.asarray(motion.joints, dtype=np.float64).copy()
    cursor = 0.0
    for frame_slot in range(TRACKER_FRAME_COUNT):
        pelvis = joints[frame_slot, JOINT_INDEX["pelvis"]].copy()
        vertices[frame_slot, :, 0] -= pelvis[0]
        vertices[frame_slot, :, 2] -= pelvis[2]
        joints[frame_slot, :, 0] -= pelvis[0]
        joints[frame_slot, :, 2] -= pelvis[2]

        floor_y = float(np.min(vertices[frame_slot, :, 1]))
        vertices[frame_slot, :, 1] -= floor_y
        joints[frame_slot, :, 1] -= floor_y
        frame_min_x = float(np.min(vertices[frame_slot, :, 0]))
        frame_max_x = float(np.max(vertices[frame_slot, :, 0]))
        if frame_slot == HISTORY_FRAME_COUNT:
            # 右侧 current frame 与五帧历史拉开一点距离，不依赖文字也能
            # 表达它是额外加入的当前输入。
            cursor += CURRENT_FRAME_EXTRA_GAP_METERS
        stage_x = cursor - frame_min_x
        vertices[frame_slot, :, 0] += stage_x
        joints[frame_slot, :, 0] += stage_x
        cursor += (frame_max_x - frame_min_x) + STAGE_GAP_METERS

    all_vertices = vertices.reshape(-1, 3)
    center_x = 0.5 * (
        float(np.min(all_vertices[:, 0]))
        + float(np.max(all_vertices[:, 0]))
    )
    vertices[:, :, 0] -= center_x
    joints[:, :, 0] -= center_x
    all_vertices = vertices.reshape(-1, 3)
    mins = np.min(all_vertices, axis=0)
    maxs = np.max(all_vertices, axis=0)
    target = 0.5 * (mins + maxs)
    content_width = float(maxs[0] - mins[0])
    content_height = float(maxs[1] - mins[1])
    aspect = float(OUTPUT_WIDTH) / float(OUTPUT_HEIGHT)
    ymag = max(content_height * 0.5, content_width * 0.5 / aspect)
    ymag *= CAMERA_PADDING
    xmag = ymag * aspect
    eye = target + np.asarray([0.0, 0.0, 6.0], dtype=np.float64)
    return TrackerLayout(
        vertices=vertices.astype(np.float32),
        joints=joints.astype(np.float32),
        target=target,
        camera_pose=camera_pose_look_at(eye, target),
        xmag=float(xmag),
        ymag=float(ymag),
    )


def render_bodies(motion: TrackerMotion, layout: TrackerLayout) -> Image.Image:
    try:
        import pyrender
        import trimesh
    except ImportError as exc:
        raise ImportError("缺少 pyrender/trimesh，无法渲染 Tracker 序列。") from exc

    scene = pyrender.Scene(
        bg_color=np.ones(4),
        ambient_light=np.asarray([0.50, 0.50, 0.50]),
    )
    scene.add(
        pyrender.OrthographicCamera(
            xmag=layout.xmag,
            ymag=layout.ymag,
            znear=0.05,
            zfar=20.0,
        ),
        pose=layout.camera_pose,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.9),
        pose=camera_pose_look_at(
            layout.target + np.asarray([-2.6, 3.4, 4.5]), layout.target
        ),
    )
    scene.add(
        pyrender.DirectionalLight(
            color=np.asarray([0.88, 0.92, 1.0]), intensity=0.7
        ),
        pose=camera_pose_look_at(
            layout.target + np.asarray([3.0, 1.8, 3.8]), layout.target
        ),
    )
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=BODY_COLOR,
        metallicFactor=0.0,
        roughnessFactor=0.88,
    )
    for frame_slot, frame_vertices in enumerate(layout.vertices):
        # Current frame 只表达刚到达系统的 Tracker observation，不渲染
        # SMPL 人体；它的关节坐标仍保留，用于把六个设备放到正确位置。
        if frame_slot == HISTORY_FRAME_COUNT:
            continue
        mesh = trimesh.Trimesh(
            vertices=np.asarray(frame_vertices, dtype=np.float64),
            faces=np.asarray(motion.faces, dtype=np.int64),
            process=False,
        )
        scene.add(
            pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
        )

    renderer = pyrender.OffscreenRenderer(OUTPUT_WIDTH, OUTPUT_HEIGHT)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
    finally:
        with suppress(Exception):
            renderer.delete()
    return Image.fromarray(np.asarray(color[..., :3], dtype=np.uint8)).convert(
        "RGBA"
    )


def project_points(points: np.ndarray, layout: TrackerLayout) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    x = (
        (values[:, 0] - layout.target[0])
        / layout.xmag
        * OUTPUT_WIDTH
        * 0.5
        + OUTPUT_WIDTH * 0.5
    )
    y = (
        OUTPUT_HEIGHT * 0.5
        - (values[:, 1] - layout.target[1])
        / layout.ymag
        * OUTPUT_HEIGHT
        * 0.5
    )
    return np.stack([x, y], axis=-1).astype(np.float32)


# endregion


# region Tracker 符号


def draw_tracker_device(
    draw: ImageDraw.ImageDraw,
    center: np.ndarray,
    *,
    used_by_predictor: bool,
) -> None:
    point = np.asarray(center, dtype=np.float64)
    center_x = int(round(point[0]))
    center_y = int(round(point[1]))
    if used_by_predictor:
        draw.ellipse(
            (center_x - 24, center_y - 24, center_x + 24, center_y + 24),
            fill=CORE_HALO,
        )
    box = (center_x - 15, center_y - 10, center_x + 15, center_y + 10)
    draw.rounded_rectangle(
        box,
        radius=6,
        fill=TRACKER_FILL,
        outline=WHITE,
        width=4,
    )
    draw.rounded_rectangle(
        box,
        radius=6,
        outline=TRACKER_BORDER,
        width=3,
    )
    draw.ellipse(
        (center_x - 3, center_y - 3, center_x + 3, center_y + 3),
        fill=WHITE,
    )
    if used_by_predictor:
        return
    cross_radius = 15
    segments = (
        (
            center_x - cross_radius,
            center_y - cross_radius,
            center_x + cross_radius,
            center_y + cross_radius,
        ),
        (
            center_x - cross_radius,
            center_y + cross_radius,
            center_x + cross_radius,
            center_y - cross_radius,
        ),
    )
    for segment in segments:
        draw.line(segment, fill=WHITE, width=8)
        draw.line(segment, fill=CROSS_COLOR, width=5)


def overlay_tracker_sequence(
    image: Image.Image,
    motion: TrackerMotion,
    layout: TrackerLayout,
) -> Image.Image:
    # 在白底上先降低人体对比度，再叠加完全不透明的 Tracker。这样视觉上
    # 等价于 55% 人体透明度，同时仍保持最终 PNG 的纯白背景契约。
    white_background = Image.new("RGBA", image.size, WHITE)
    faded_body = Image.blend(
        white_background,
        image.convert("RGBA"),
        BODY_OPACITY,
    )
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    projected = project_points(layout.joints, layout).reshape(
        TRACKER_FRAME_COUNT, 22, 2
    )
    for frame_slot in range(TRACKER_FRAME_COUNT):
        pelvis_point = projected[frame_slot, JOINT_INDEX["pelvis"]]
        for _, joint_index, used_by_predictor in TRACKER_SPECS:
            tracker_point = projected[frame_slot, joint_index].copy()
            if joint_index == JOINT_INDEX["head"]:
                # SMPL 的 head joint 位于面部中心偏下；上移到额头更符合 HMD
                # 或头戴 Tracker 的实际佩戴位置。
                tracker_point[1] -= 18.0
            elif joint_index in (
                JOINT_INDEX["left_foot"],
                JOINT_INDEX["right_foot"],
            ):
                # 并腿帧中左右脚点非常接近，沿各自远离 pelvis 的方向稍微
                # 错开叉号，避免两个“未送入 Predictor”标记完全重叠。
                outward = np.sign(tracker_point[0] - pelvis_point[0])
                tracker_point[0] += float(outward if outward != 0 else 1.0) * 14.0
            draw_tracker_device(
                draw,
                tracker_point,
                used_by_predictor=used_by_predictor,
            )
    return Image.alpha_composite(faded_body, overlay).convert("RGB")


# endregion


# region 输出


def validate_image(image: Image.Image) -> None:
    if image.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        raise RuntimeError(
            f"输出尺寸错误：{image.size}，期望 {(OUTPUT_WIDTH, OUTPUT_HEIGHT)}。"
        )
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    corners = rgb[[0, 0, -1, -1], [0, -1, 0, -1]]
    if not np.all(corners == 255):
        raise RuntimeError("Tracker 序列四角不是纯白背景。")
    if int(np.any(rgb < 248, axis=-1).sum()) < 10000:
        raise RuntimeError("Tracker 序列前景像素过少，可能渲染失败。")


def write_report(
    *,
    output_png: Path,
    amass_npz: Path,
    motion: TrackerMotion,
) -> Path:
    report_path = output_png.with_suffix(".json")
    report = {
        "asset": "predictor_tracker_selection_sequence",
        "source_path": str(Path(amass_npz).expanduser().resolve()),
        "frame_indices": motion.frame_indices.astype(int).tolist(),
        "current_frame": int(motion.current_frame),
        "tracker_layout": "five_history_frames_plus_current_frame",
        "history_frames": motion.frame_indices[:HISTORY_FRAME_COUNT]
        .astype(int)
        .tolist(),
        "current_input_frame": int(motion.frame_indices[-1]),
        "current_frame_extra_gap_m": CURRENT_FRAME_EXTRA_GAP_METERS,
        "predictor_used": [
            name for name, _, used in TRACKER_SPECS if used
        ],
        "predictor_not_used": [
            name for name, _, used in TRACKER_SPECS if not used
        ],
        "cross_semantics": (
            "红叉只表示该 Tracker 不进入 Predictor core tracker context；"
            "不表示设备不存在，也不排除后续 IK/DiT 使用。"
        ),
        "body_model": "SMPL-H male, zeros(10) betas",
        "body_opacity_on_white": BODY_OPACITY,
        "current_frame_body_opacity": 0.0,
        "tracker_symbol_scale": TRACKER_SYMBOL_SCALE,
        "projection": "front orthographic",
        "presentation_yaw_deg": float(motion.presentation_yaw_deg),
        "resolution": [OUTPUT_WIDTH, OUTPUT_HEIGHT],
        "background_rgb": [255, 255, 255],
        "output_png": str(output_png),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report_path


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    motion = load_tracker_motion(
        amass_npz=args.amass_npz,
        smpl_model_dir=args.smpl_model_dir,
        frame_indices=np.asarray(args.frame_indices, dtype=np.int64),
        current_frame=int(args.current_frame),
    )
    layout = build_layout(motion)
    image = overlay_tracker_sequence(
        render_bodies(motion, layout), motion, layout
    )
    validate_image(image)

    output_png = Path(args.output_png).expanduser().resolve()
    if output_png.suffix.lower() != ".png":
        raise ValueError(f"output_png 必须使用 .png 后缀：{output_png}")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    report_path = write_report(
        output_png=output_png,
        amass_npz=args.amass_npz,
        motion=motion,
    )
    print(
        f"[predictor-tracker-selection] frames: "
        f"{motion.frame_indices.tolist()}",
        flush=True,
    )
    print(f"[predictor-tracker-selection] wrote: {output_png}", flush=True)
    print(f"[predictor-tracker-selection] wrote: {report_path}", flush=True)
    return output_png, report_path


# endregion


if __name__ == "__main__":
    main()
