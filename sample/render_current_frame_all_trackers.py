from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from sample.render_predictor_tracker_selection import (
    DEFAULT_CURRENT_FRAME,
    DEFAULT_FRAME_INDICES,
    TRACKER_SPECS,
    TrackerMotion,
    load_tracker_motion,
)


DEFAULT_OUTPUT = Path(
    "output/主方法图所需材料与参考/"
    "CurrentFrame_全部六点Tracker_绿圈_无人体.png"
)
OUTPUT_WIDTH = 700
OUTPUT_HEIGHT = 800
CAMERA_PADDING = 1.12
WHITE = (255, 255, 255, 255)
TRACKER_FILL = (37, 124, 192, 255)
TRACKER_BORDER = (19, 72, 117, 255)
HALO_FILL = (49, 178, 119, 44)
HALO_BORDER = (29, 168, 112, 255)


# region CLI 与当前帧布局


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "绘制当前帧 Head、双腕、Hip 与双脚六个 Tracker；"
            "所有 Tracker 使用绿色圆环，不绘制 SMPL 人体或红叉。"
        )
    )
    parser.add_argument("--amass_npz", required=True, type=Path)
    parser.add_argument("--smpl_model_dir", required=True, type=Path)
    parser.add_argument("--output_png", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument(
        "--current_frame", default=DEFAULT_CURRENT_FRAME, type=int
    )
    return parser


def build_loading_frames(current_frame: int) -> np.ndarray:
    """构造 helper 所需的五帧历史加当前帧索引。"""

    if int(current_frame) == DEFAULT_CURRENT_FRAME:
        return np.asarray(DEFAULT_FRAME_INDICES, dtype=np.int64)
    if int(current_frame) < 5:
        raise ValueError("current_frame 至少为 5，才能构造正面朝向参考。")
    return np.arange(int(current_frame) - 5, int(current_frame) + 1)


def project_current_tracker_points(
    motion: TrackerMotion,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """把当前帧六点投影到画布，并保留正面人体比例下的空间关系。"""

    # SMPL 网格只用于计算不可见的取景范围，最终图片完全不渲染人体。
    vertices = np.asarray(motion.vertices[-1], dtype=np.float64).copy()
    joints = np.asarray(motion.joints[-1], dtype=np.float64).copy()
    pelvis = joints[JOINT_INDEX["pelvis"]].copy()
    vertices[:, 0] -= pelvis[0]
    vertices[:, 2] -= pelvis[2]
    joints[:, 0] -= pelvis[0]
    joints[:, 2] -= pelvis[2]

    floor_y = float(np.min(vertices[:, 1]))
    vertices[:, 1] -= floor_y
    joints[:, 1] -= floor_y
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    target = 0.5 * (mins + maxs)
    content_width = float(maxs[0] - mins[0])
    content_height = float(maxs[1] - mins[1])
    aspect = float(OUTPUT_WIDTH) / float(OUTPUT_HEIGHT)
    ymag = max(content_height * 0.5, content_width * 0.5 / aspect)
    ymag *= CAMERA_PADDING
    xmag = ymag * aspect

    tracker_world: dict[str, np.ndarray] = {}
    tracker_pixels: dict[str, np.ndarray] = {}
    pelvis_point = joints[JOINT_INDEX["pelvis"]]
    for name, joint_index, _ in TRACKER_SPECS:
        point = joints[joint_index].copy()
        pixel = np.asarray(
            [
                (point[0] - target[0]) / xmag * OUTPUT_WIDTH * 0.5
                + OUTPUT_WIDTH * 0.5,
                OUTPUT_HEIGHT * 0.5
                - (point[1] - target[1])
                / ymag
                * OUTPUT_HEIGHT
                * 0.5,
            ],
            dtype=np.float64,
        )
        if name == "head":
            # 将 SMPL 头部关节点上移到额头位置，表达头戴式 Tracker。
            pixel[1] -= 24.0
        elif name in ("left_foot", "right_foot"):
            # 正面并腿时双脚很接近，轻微向外错开以保持两个设备可辨认。
            outward = np.sign(point[0] - pelvis_point[0])
            pixel[0] += float(outward if outward != 0 else 1.0) * 18.0
        tracker_world[name] = point.astype(np.float32)
        tracker_pixels[name] = pixel.astype(np.float32)
    return tracker_world, tracker_pixels


# endregion


# region Tracker 绘制


def draw_tracker_device(
    draw: ImageDraw.ImageDraw,
    center: np.ndarray,
) -> None:
    """绘制放大的蓝色 Tracker，并为所有设备添加同样的绿色圆环。"""

    center_x = int(round(float(center[0])))
    center_y = int(round(float(center[1])))
    halo_radius = 38
    draw.ellipse(
        (
            center_x - halo_radius,
            center_y - halo_radius,
            center_x + halo_radius,
            center_y + halo_radius,
        ),
        fill=HALO_FILL,
        outline=HALO_BORDER,
        width=5,
    )
    half_width = 23
    half_height = 15
    box = (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )
    draw.rounded_rectangle(
        box,
        radius=9,
        fill=TRACKER_FILL,
        outline=WHITE,
        width=5,
    )
    draw.rounded_rectangle(
        box,
        radius=9,
        outline=TRACKER_BORDER,
        width=3,
    )
    draw.ellipse(
        (center_x - 5, center_y - 5, center_x + 5, center_y + 5),
        fill=WHITE,
    )


def render_all_trackers(
    tracker_pixels: dict[str, np.ndarray],
) -> Image.Image:
    image = Image.new("RGBA", (OUTPUT_WIDTH, OUTPUT_HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    for name, _, _ in TRACKER_SPECS:
        draw_tracker_device(draw, tracker_pixels[name])
    return image.convert("RGB")


# endregion


# region 校验与输出


def validate_image(image: Image.Image) -> None:
    if image.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        raise RuntimeError(
            f"输出尺寸错误：{image.size}，期望 {(OUTPUT_WIDTH, OUTPUT_HEIGHT)}。"
        )
    rgb = np.asarray(image, dtype=np.uint8)
    corners = rgb[[0, 0, -1, -1], [0, -1, 0, -1]]
    if not np.all(corners == 255):
        raise RuntimeError("当前帧 Tracker 图四角不是纯白背景。")
    # 六个绿色圆环和蓝色设备应产生足够的非白像素。
    if int(np.any(rgb < 248, axis=-1).sum()) < 7000:
        raise RuntimeError("Tracker 前景像素过少，可能绘制失败。")


def write_report(
    *,
    output_png: Path,
    amass_npz: Path,
    motion: TrackerMotion,
    tracker_world: dict[str, np.ndarray],
    tracker_pixels: dict[str, np.ndarray],
) -> Path:
    report_path = output_png.with_suffix(".json")
    report = {
        "asset": "current_frame_all_six_trackers_without_body",
        "source_path": str(Path(amass_npz).expanduser().resolve()),
        "current_frame": int(motion.current_frame),
        "trackers": [name for name, _, _ in TRACKER_SPECS],
        "tracker_world_xyz": {
            name: point.astype(float).tolist()
            for name, point in tracker_world.items()
        },
        "tracker_pixel_xy": {
            name: point.astype(float).tolist()
            for name, point in tracker_pixels.items()
        },
        "all_tracker_halos": "green",
        "crosses": "none",
        "visible_body": "none",
        "body_model_for_positioning_only": (
            "SMPL-H male, zeros(10) betas; not rendered"
        ),
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
    current_frame = int(args.current_frame)
    motion = load_tracker_motion(
        amass_npz=args.amass_npz,
        smpl_model_dir=args.smpl_model_dir,
        frame_indices=build_loading_frames(current_frame),
        current_frame=current_frame,
    )
    tracker_world, tracker_pixels = project_current_tracker_points(motion)
    image = render_all_trackers(tracker_pixels)
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
        tracker_world=tracker_world,
        tracker_pixels=tracker_pixels,
    )
    print(
        f"[current-frame-all-trackers] current frame: {current_frame}",
        flush=True,
    )
    print(f"[current-frame-all-trackers] wrote: {output_png}", flush=True)
    print(f"[current-frame-all-trackers] wrote: {report_path}", flush=True)
    return output_png, report_path


# endregion


if __name__ == "__main__":
    main()
