from __future__ import annotations

import argparse
from contextlib import suppress
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from data_converter.amass_smpl_utils import (
    SOURCE_BODY_JOINT_COUNT,
    build_smpl_local_rotations,
    load_motion_source,
    local_to_global_rotations,
    transform_rotations_to_unity,
)
from data_loaders.realtime_pose_kinematics import JOINT_INDEX, SMPL_PARENTS
from sample.realtime_pose_smpl_rendering import (
    SmplMeshSequence,
    camera_pose_look_at,
    create_smplh_model,
    require_directory,
    require_file,
    run_smplh_forward,
    transform_faces_to_unity_winding,
)


TARGET_FPS = 30.0
DEFAULT_CURRENT_FRAME = 180
# 这五帧覆盖当前时刻前约 0.63 秒，比逐帧紧邻采样更容易读出 Salsa 动作变化。
DEFAULT_FRAME_INDICES = (160, 165, 170, 175, 179)
DEFAULT_OUTPUT = Path(
    "output/主方法图所需材料与参考/PastMotion_5帧_SMPL男性_正面.png"
)
DEFAULT_WIDTH = 2500
DEFAULT_HEIGHT = 900
STAGE_GAP_METERS = 0.16
CAMERA_PADDING = 1.08
BODY_COLOR = (0.50, 0.51, 0.52, 1.0)
WHITE_RGBA = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class PastMotionRender:
    """Past Motion 渲染所需的男性 SMPL-H 网格和可复现元数据。"""

    sequence: SmplMeshSequence
    faces: np.ndarray
    frame_indices: np.ndarray
    current_frame: int
    source_fps: float
    presentation_yaw_deg: float


@dataclass(frozen=True)
class OrthographicLayout:
    """五个人体横向排布后的网格与正交相机参数。"""

    vertices_world: np.ndarray  # [5,V,3]
    camera_pose: np.ndarray  # [4,4]
    xmag: float
    ymag: float


# region CLI


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "从 AMASS 动作生成 RPM 风格的五帧 Past Motion：男性 SMPL-H、"
            "正面正交投影和纯白背景。"
        )
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument("--amass_npz", required=True, type=Path)
    paths.add_argument("--smpl_model_dir", required=True, type=Path)
    paths.add_argument("--output_png", default=DEFAULT_OUTPUT, type=Path)
    motion = parser.add_argument_group("motion")
    motion.add_argument(
        "--frame_indices",
        nargs=5,
        default=DEFAULT_FRAME_INDICES,
        type=int,
        metavar=("F0", "F1", "F2", "F3", "F4"),
        help="严格递增的五个 30 Hz source frame，且必须早于 current_frame。",
    )
    motion.add_argument(
        "--current_frame",
        default=DEFAULT_CURRENT_FRAME,
        type=int,
        help="Past Motion 对应的当前时刻；仅用于校验和记录。",
    )
    image = parser.add_argument_group("image")
    image.add_argument("--width", default=DEFAULT_WIDTH, type=int)
    image.add_argument("--height", default=DEFAULT_HEIGHT, type=int)
    return parser


# endregion


# region 动作与男性 SMPL-H


def validate_frame_indices(
    frame_indices: np.ndarray,
    *,
    current_frame: int,
    frame_count: int,
) -> np.ndarray:
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.shape != (5,):
        raise ValueError(f"Past Motion 必须正好选择 5 帧，实际为 {frames.shape}。")
    if not np.all(np.diff(frames) > 0):
        raise ValueError(f"frame_indices 必须严格递增，实际为 {frames.tolist()}。")
    if np.any(frames < 0) or np.any(frames >= int(frame_count)):
        raise ValueError(
            f"frame_indices 超出 [0,{frame_count})：{frames.tolist()}。"
        )
    if not 0 <= int(current_frame) < int(frame_count):
        raise ValueError(
            f"current_frame={current_frame} 超出 [0,{frame_count})。"
        )
    if int(current_frame) <= int(frames[-1]):
        raise ValueError(
            "五帧必须全部属于过去："
            f"最后一帧 {int(frames[-1])} 不早于 current_frame={current_frame}。"
        )
    return frames


def build_front_alignment_rotation(
    joint_rotations_world: np.ndarray,
) -> tuple[np.ndarray, float]:
    """让最后一帧胸腹面正对位于 +Z 的相机，仅改变展示朝向。

    AMASS 原始 SMPL 的 Spine3 局部 ``+Z`` 是稳定的躯干正面参考。使用最后一个过去帧
    统一对齐整段序列，能保留五帧之间真实的躯干转动，而不是把每一帧分别
    掰成完全相同的朝向。
    """

    rotations = np.asarray(joint_rotations_world, dtype=np.float64)
    if rotations.shape != (5, len(SMPL_PARENTS), 3, 3):
        raise ValueError(
            "joint_rotations_world 应为 [5,24,3,3]，"
            f"实际为 {rotations.shape}。"
        )
    torso_front = rotations[-1, JOINT_INDEX["spine3"], :, 2]
    torso_front_xz = torso_front[[0, 2]]
    if float(np.linalg.norm(torso_front_xz)) <= 1e-8:
        raise ValueError("Spine3 正面方向的水平分量为零，无法建立正面视角。")
    current_angle = math.atan2(
        float(torso_front_xz[1]),
        float(torso_front_xz[0]),
    )
    desired_angle = math.pi * 0.5  # +Z，正对放置在 +Z 方向的相机。
    presentation_yaw = desired_angle - current_angle
    cos_yaw = math.cos(presentation_yaw)
    sin_yaw = math.sin(presentation_yaw)
    rotation = np.asarray(
        [
            [cos_yaw, 0.0, -sin_yaw],
            [0.0, 1.0, 0.0],
            [sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float64,
    )
    return rotation, math.degrees(presentation_yaw)


def load_male_past_motion(
    *,
    amass_npz: Path,
    smpl_model_dir: Path,
    frame_indices: np.ndarray,
    current_frame: int,
) -> PastMotionRender:
    """把指定 30 Hz AMASS 姿态转移到标准男性 SMPL-H 身形。"""

    amass_path = require_file(amass_npz, "amass_npz")
    model_dir = require_directory(smpl_model_dir, "smpl_model_dir")
    source = load_motion_source(
        path=amass_path,
        amass_dir=amass_path.parent,
        target_fps=TARGET_FPS,
    )
    frames = validate_frame_indices(
        frame_indices,
        current_frame=int(current_frame),
        frame_count=int(source.poses.shape[0]),
    )
    # 只使用 SMPL 身体的 22 个关节；手指保持男性模型的自然平手均值。
    pose_axis_angle = np.asarray(
        source.poses[frames, : SOURCE_BODY_JOINT_COUNT * 3],
        dtype=np.float32,
    ).reshape(5, SOURCE_BODY_JOINT_COUNT, 3)
    model = create_smplh_model(
        model_dir=model_dir,
        gender="male",
        batch_size=5,
    )
    sequence = run_smplh_forward(
        model=model,
        pose_axis_angle=pose_axis_angle,
        # 零 beta 表示标准男性模板，避免把原女性演员的身形参数带入结果。
        betas=np.zeros((10,), dtype=np.float32),
        # Past Motion 只表达姿态；五帧稍后按时间顺序独立横向排布。
        translation_amass=np.zeros((5, 3), dtype=np.float32),
    )
    selected_poses = np.asarray(source.poses[frames], dtype=np.float64)
    local_rotations = build_smpl_local_rotations(selected_poses)
    global_rotations = local_to_global_rotations(local_rotations, SMPL_PARENTS)
    global_rotations_unity = transform_rotations_to_unity(global_rotations)
    front_rotation, presentation_yaw_deg = build_front_alignment_rotation(
        global_rotations_unity
    )
    oriented_sequence = SmplMeshSequence(
        vertices_world=(
            np.asarray(sequence.vertices_world, dtype=np.float64)
            @ front_rotation.T
        ).astype(np.float32),
        joints_world=(
            np.asarray(sequence.joints_world, dtype=np.float64)
            @ front_rotation.T
        ).astype(np.float32),
    )
    return PastMotionRender(
        sequence=oriented_sequence,
        faces=transform_faces_to_unity_winding(model.faces),
        frame_indices=frames,
        current_frame=int(current_frame),
        source_fps=float(TARGET_FPS),
        presentation_yaw_deg=float(presentation_yaw_deg),
    )


# endregion


# region 横向排版与正交相机


def build_orthographic_layout(
    render: PastMotionRender,
    *,
    width: int,
    height: int,
) -> OrthographicLayout:
    """把任意数量的姿态从左到右打包，并拟合统一正交相机。"""

    if int(width) <= 0 or int(height) <= 0:
        raise ValueError(f"输出尺寸必须为正数，实际为 {width}x{height}。")
    vertices = np.asarray(render.sequence.vertices_world, dtype=np.float64).copy()
    joints = np.asarray(render.sequence.joints_world, dtype=np.float64)
    if vertices.ndim != 3 or vertices.shape[0] < 1 or vertices.shape[2] != 3:
        raise ValueError(
            f"vertices_world 应为 [T,V,3] 且 T>=1，实际为 {vertices.shape}。"
        )
    if joints.ndim != 3 or joints.shape[0] != vertices.shape[0]:
        raise ValueError(
            "joints_world 与 vertices_world 的时间长度必须一致："
            f"{joints.shape} vs {vertices.shape}。"
        )

    cursor = 0.0
    for frame_index in range(vertices.shape[0]):
        pelvis = joints[frame_index, JOINT_INDEX["pelvis"]]
        vertices[frame_index, :, 0] -= float(pelvis[0])
        vertices[frame_index, :, 2] -= float(pelvis[2])
        # 每个姿态独立落地，防止原动作的轻微根高度变化造成视觉漂浮。
        vertices[frame_index, :, 1] -= float(
            np.min(vertices[frame_index, :, 1])
        )
        frame_min_x = float(np.min(vertices[frame_index, :, 0]))
        frame_max_x = float(np.max(vertices[frame_index, :, 0]))
        vertices[frame_index, :, 0] += cursor - frame_min_x
        cursor += (frame_max_x - frame_min_x) + STAGE_GAP_METERS

    all_vertices = vertices.reshape(-1, 3)
    center_x = 0.5 * (
        float(np.min(all_vertices[:, 0])) + float(np.max(all_vertices[:, 0]))
    )
    vertices[:, :, 0] -= center_x
    all_vertices = vertices.reshape(-1, 3)
    mins = np.min(all_vertices, axis=0)
    maxs = np.max(all_vertices, axis=0)
    target = 0.5 * (mins + maxs)
    content_width = float(maxs[0] - mins[0])
    content_height = float(maxs[1] - mins[1])
    aspect_ratio = float(width) / float(height)
    ymag = max(
        content_height * 0.5,
        content_width * 0.5 / aspect_ratio,
    ) * CAMERA_PADDING
    xmag = ymag * aspect_ratio
    if not np.isfinite([xmag, ymag]).all() or xmag <= 0.0 or ymag <= 0.0:
        raise ValueError("正交相机拟合得到无效范围。")
    # 正交投影不依赖相机距离；6 米只用于给近远裁剪面留足安全范围。
    eye = np.asarray([target[0], target[1], target[2] + 6.0], dtype=np.float64)
    return OrthographicLayout(
        vertices_world=vertices.astype(np.float32),
        camera_pose=camera_pose_look_at(eye, target),
        xmag=float(xmag),
        ymag=float(ymag),
    )


# endregion


# region 离屏渲染与输出


def render_past_motion_image(
    *,
    render: PastMotionRender,
    layout: OrthographicLayout,
    width: int,
    height: int,
) -> Image.Image:
    """在纯白背景上渲染五个柔和灰色男性 SMPL-H。"""

    try:
        import pyrender
        import trimesh
    except ImportError as exc:
        raise ImportError("缺少 pyrender/trimesh，无法生成 Past Motion 图。") from exc

    scene = pyrender.Scene(
        bg_color=WHITE_RGBA,
        ambient_light=np.asarray([0.48, 0.48, 0.48], dtype=np.float64),
    )
    camera = pyrender.OrthographicCamera(
        xmag=float(layout.xmag),
        ymag=float(layout.ymag),
        znear=0.05,
        zfar=20.0,
    )
    scene.add(camera, pose=np.asarray(layout.camera_pose, dtype=np.float64))
    target = np.asarray(layout.camera_pose[:3, 3], dtype=np.float64).copy()
    target[2] -= 6.0
    # 主光和补光都接近相机方向，使正面轮廓清楚，同时保留胸腹与四肢体积。
    key_pose = camera_pose_look_at(
        target + np.asarray([-2.6, 3.4, 4.5], dtype=np.float64),
        target,
    )
    fill_pose = camera_pose_look_at(
        target + np.asarray([3.0, 1.8, 3.8], dtype=np.float64),
        target,
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.9),
        pose=key_pose,
    )
    scene.add(
        pyrender.DirectionalLight(
            color=np.asarray([0.88, 0.92, 1.0]),
            intensity=0.7,
        ),
        pose=fill_pose,
    )
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=BODY_COLOR,
        metallicFactor=0.0,
        roughnessFactor=0.88,
    )
    for frame_vertices in layout.vertices_world:
        body = trimesh.Trimesh(
            vertices=np.asarray(frame_vertices, dtype=np.float64),
            faces=np.asarray(render.faces, dtype=np.int64),
            process=False,
        )
        scene.add(
            pyrender.Mesh.from_trimesh(
                body,
                material=material,
                smooth=True,
            )
        )

    renderer = pyrender.OffscreenRenderer(int(width), int(height))
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
    finally:
        with suppress(Exception):
            renderer.delete()
    return Image.fromarray(np.asarray(color[..., :3], dtype=np.uint8), mode="RGB")


def validate_rendered_image(image: Image.Image, *, width: int, height: int) -> None:
    if image.size != (int(width), int(height)):
        raise RuntimeError(
            f"Past Motion 输出尺寸错误：{image.size}，期望 {(width, height)}。"
        )
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    corner_pixels = rgb[[0, 0, -1, -1], [0, -1, 0, -1]]
    if not np.all(corner_pixels == 255):
        raise RuntimeError("Past Motion 四角不是纯白，背景契约被破坏。")
    foreground = np.any(rgb < 248, axis=-1)
    if not np.any(foreground):
        raise RuntimeError("Past Motion 没有检测到人体前景。")
    foreground_y, foreground_x = np.nonzero(foreground)
    margin = 8
    if (
        int(np.min(foreground_x)) < margin
        or int(np.max(foreground_x)) >= int(width) - margin
        or int(np.min(foreground_y)) < margin
        or int(np.max(foreground_y)) >= int(height) - margin
    ):
        raise RuntimeError("Past Motion 人体贴近画布边缘，可能发生裁切。")


def write_sidecar(
    *,
    output_png: Path,
    amass_npz: Path,
    render: PastMotionRender,
    layout: OrthographicLayout,
    width: int,
    height: int,
) -> Path:
    report = {
        "experiment": "rpm_style_past_motion",
        "source_path": str(Path(amass_npz).expanduser().resolve()),
        "source_fps": float(render.source_fps),
        "current_frame": int(render.current_frame),
        "past_frame_indices": render.frame_indices.astype(int).tolist(),
        "body_model": "SMPL-H male",
        "betas": "zeros(10)",
        "hand_pose": "flat_hand_mean",
        "camera": {
            "projection": "orthographic",
            "view": "front",
            "presentation_yaw_deg": float(render.presentation_yaw_deg),
            "xmag": float(layout.xmag),
            "ymag": float(layout.ymag),
        },
        "background_rgb": [255, 255, 255],
        "resolution": [int(width), int(height)],
        "output_png": str(Path(output_png).expanduser().resolve()),
    }
    output_json = Path(output_png).with_suffix(".json").expanduser().resolve()
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_json


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    args = build_arg_parser().parse_args(argv)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    output_png = Path(args.output_png).expanduser().resolve()
    if output_png.suffix.lower() != ".png":
        raise ValueError(f"output_png 必须使用 .png 后缀，实际为 {output_png}。")
    render = load_male_past_motion(
        amass_npz=args.amass_npz,
        smpl_model_dir=args.smpl_model_dir,
        frame_indices=np.asarray(args.frame_indices, dtype=np.int64),
        current_frame=int(args.current_frame),
    )
    layout = build_orthographic_layout(
        render,
        width=int(args.width),
        height=int(args.height),
    )
    image = render_past_motion_image(
        render=render,
        layout=layout,
        width=int(args.width),
        height=int(args.height),
    )
    validate_rendered_image(image, width=int(args.width), height=int(args.height))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)
    output_json = write_sidecar(
        output_png=output_png,
        amass_npz=args.amass_npz,
        render=render,
        layout=layout,
        width=int(args.width),
        height=int(args.height),
    )
    print(f"[rpm-past-motion] frames: {render.frame_indices.tolist()}", flush=True)
    print(f"[rpm-past-motion] wrote: {output_png}", flush=True)
    print(f"[rpm-past-motion] report: {output_json}", flush=True)
    return output_png, output_json


# endregion


if __name__ == "__main__":
    main()
