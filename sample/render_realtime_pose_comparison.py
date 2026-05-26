from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from data_loaders.realtime_pose_kinematics import SMPL_PARENTS
from data_loaders.sensor_masking import TRACKER_COUNT


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render realtime_pose GT vs prediction comparison MP4.")
    parser.add_argument("--input_npz", required=True, type=str)
    parser.add_argument("--output_mp4", required=True, type=str)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--camera_mode", default="global", choices=["global", "follow"], type=str)
    parser.add_argument("--layout", default="split", choices=["split", "overlay"], type=str)
    parser.add_argument("--local_radius", default=1.25, type=float)
    return parser


def as_unbatched(array: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim >= 1 and value.shape[0] == 1:
        return value[0]
    if value.ndim >= 2:
        return value
    raise ValueError(f"{name} 维度不合法：{value.shape}")


def fixed_axis_limits(*arrays: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    points = []
    for array in arrays:
        value = np.asarray(array, dtype=np.float32)
        if value.size == 0:
            continue
        points.append(value.reshape(-1, 3))
    if not points:
        return (-1.0, 1.0), (0.0, 2.0), (-1.0, 1.0)

    stacked = np.concatenate(points, axis=0)
    mins = np.nanmin(stacked, axis=0)
    maxs = np.nanmax(stacked, axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    radius = max(radius, 0.75)
    return (
        (float(center[0] - radius), float(center[0] + radius)),
        (float(max(0.0, center[1] - radius)), float(center[1] + radius)),
        (float(center[2] - radius), float(center[2] + radius)),
    )


def set_axes_equal(ax, axis_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> None:
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[2])
    ax.set_zlim(*axis_limits[1])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.view_init(elev=14, azim=-72)


def follow_axis_limits(
    reference_frame: np.ndarray,
    predicted_frame: np.ndarray,
    tracker_frame: np.ndarray | None,
    local_radius: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """围绕当前帧人体取景，避免长序列全局范围把人物缩成很小。"""

    points = [np.asarray(reference_frame, dtype=np.float32).reshape(-1, 3)]
    points.append(np.asarray(predicted_frame, dtype=np.float32).reshape(-1, 3))
    if tracker_frame is not None:
        points.append(np.asarray(tracker_frame, dtype=np.float32).reshape(-1, 3))
    stacked = np.concatenate(points, axis=0)
    mins = np.nanmin(stacked, axis=0)
    maxs = np.nanmax(stacked, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(float(local_radius), float(np.max(maxs - mins) * 0.75))
    return (
        (float(center[0] - radius), float(center[0] + radius)),
        (float(max(0.0, center[1] - radius)), float(center[1] + radius)),
        (float(center[2] - radius), float(center[2] + radius)),
    )


def draw_skeleton(ax, joints: np.ndarray, color: str, label: str) -> None:
    joints = np.asarray(joints, dtype=np.float32)
    for joint_index, parent_index in enumerate(SMPL_PARENTS.tolist()):
        if parent_index < 0:
            continue
        xs = [joints[parent_index, 0], joints[joint_index, 0]]
        ys = [joints[parent_index, 2], joints[joint_index, 2]]
        zs = [joints[parent_index, 1], joints[joint_index, 1]]
        ax.plot(xs, ys, zs, color=color, linewidth=1.8)
    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], color=color, s=8, depthshade=False, label=label)


def draw_trackers(ax, tracker_pos: np.ndarray, sensor_valid: np.ndarray) -> None:
    trackers = np.asarray(tracker_pos, dtype=np.float32)
    valid = np.asarray(sensor_valid, dtype=bool)
    if trackers.shape != (TRACKER_COUNT, 3) or valid.shape != (TRACKER_COUNT,):
        return
    visible = trackers[valid]
    if visible.size == 0:
        return
    ax.scatter(visible[:, 0], visible[:, 2], visible[:, 1], color="#f59e0b", marker="o", s=24, depthshade=False)


def figure_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "tostring_rgb"):
        return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3].copy()


class Mp4FrameWriter:
    def __init__(self, output_path: Path, frame_rgb: np.ndarray, fps: int):
        ffmpeg_exe = find_ffmpeg_exe()
        width, height = frame_rgb.shape[1], frame_rgb.shape[0]
        self.output_path = Path(output_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="realtime_pose_mp4_"))
        self.temp_output_path = self.temp_dir / "render.mp4"
        self.process = subprocess.Popen(
            [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                str(int(fps)),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(self.temp_output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def append(self, frame_rgb: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin 已关闭。")
        self.process.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = b""
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
        return_code = self.process.wait()
        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg 写 MP4 失败，returncode={return_code}: {message}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            try:
                self.output_path.unlink()
            except PermissionError as exc:
                raise PermissionError(f"无法覆盖 MP4，目标文件可能正被播放器占用：{self.output_path}") from exc
        shutil.move(str(self.temp_output_path), str(self.output_path))
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def find_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except ModuleNotFoundError:
        pass

    roots = [os.environ.get("CONDA_PREFIX", ""), sys.prefix]
    patterns = [
        Path("Lib") / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg*.exe",
        Path("lib") / "python*" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg*",
    ]
    for root_value in roots:
        if not root_value:
            continue
        root = Path(root_value)
        for pattern in patterns:
            matches = sorted(root.glob(str(pattern)))
            if matches:
                return str(matches[0])
    raise ModuleNotFoundError("找不到 imageio-ffmpeg 或 ffmpeg 可执行文件，无法写 MP4。")


def render_realtime_pose_comparison(
    output_path: Path,
    reference_joints: np.ndarray,
    predicted_joints: np.ndarray,
    tracker_pos_world: np.ndarray | None = None,
    sensor_valid: np.ndarray | None = None,
    eval_frame_mask: np.ndarray | None = None,
    root_yaw_reference: np.ndarray | None = None,
    root_yaw_predicted: np.ndarray | None = None,
    fps: int = 30,
    stride: int = 1,
    camera_mode: str = "global",
    layout: str = "split",
    local_radius: float = 1.25,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = as_unbatched(np.asarray(reference_joints, dtype=np.float32), "reference_joints")
    predicted = as_unbatched(np.asarray(predicted_joints, dtype=np.float32), "predicted_joints")
    if reference.shape != predicted.shape or reference.ndim != 3 or reference.shape[1:] != (24, 3):
        raise ValueError(f"joints 应为相同的 [T,24,3]，实际 reference={reference.shape}, predicted={predicted.shape}")

    trackers = None if tracker_pos_world is None else as_unbatched(np.asarray(tracker_pos_world, dtype=np.float32), "tracker_pos_world")
    valid = None if sensor_valid is None else as_unbatched(np.asarray(sensor_valid, dtype=bool), "sensor_valid")
    eval_mask = (
        np.ones((reference.shape[0],), dtype=bool)
        if eval_frame_mask is None
        else as_unbatched(np.asarray(eval_frame_mask, dtype=bool), "eval_frame_mask")
    )
    yaw_ref = None if root_yaw_reference is None else as_unbatched(np.asarray(root_yaw_reference, dtype=np.float32), "root_yaw_reference")
    yaw_pred = None if root_yaw_predicted is None else as_unbatched(np.asarray(root_yaw_predicted, dtype=np.float32), "root_yaw_predicted")
    if camera_mode not in {"global", "follow"}:
        raise ValueError(f"camera_mode 必须是 global/follow，实际为 {camera_mode}")
    if layout not in {"split", "overlay"}:
        raise ValueError(f"layout 必须是 split/overlay，实际为 {layout}")

    axis_inputs = [reference, predicted]
    if trackers is not None:
        axis_inputs.append(trackers)
    global_axis_limits = fixed_axis_limits(*axis_inputs)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_indices = range(0, reference.shape[0], max(1, int(stride)))
    writer: Mp4FrameWriter | None = None
    try:
        for frame_index in frame_indices:
            joint_error = float(np.linalg.norm(predicted[frame_index] - reference[frame_index], axis=-1).mean())
            yaw_error = 0.0
            if yaw_ref is not None and yaw_pred is not None:
                yaw_error = float((yaw_pred[frame_index] - yaw_ref[frame_index] + np.pi) % (2.0 * np.pi) - np.pi)
                yaw_error = abs(yaw_error)

            tracker_frame = trackers[frame_index] if trackers is not None else None
            axis_limits = (
                global_axis_limits
                if camera_mode == "global"
                else follow_axis_limits(
                    reference_frame=reference[frame_index],
                    predicted_frame=predicted[frame_index],
                    tracker_frame=tracker_frame,
                    local_radius=float(local_radius),
                )
            )
            if layout == "overlay":
                fig = plt.figure(figsize=(7.2, 6.4), dpi=140)
                axes = [fig.add_subplot(1, 1, 1, projection="3d")]
                draw_skeleton(axes[0], reference[frame_index], color="#2563eb", label="GT")
                draw_skeleton(axes[0], predicted[frame_index], color="#dc2626", label="Pred")
                if trackers is not None and valid is not None:
                    draw_trackers(axes[0], trackers[frame_index], valid[frame_index])
                set_axes_equal(axes[0], axis_limits)
                axes[0].set_title("GT vs Pred")
                axes[0].legend(loc="upper right")
            else:
                fig = plt.figure(figsize=(9.6, 4.8), dpi=100)
                axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")]
                titles = ("GT", "Pred")
                for ax, joints, color, title in zip(axes, (reference[frame_index], predicted[frame_index]), ("#2563eb", "#dc2626"), titles):
                    draw_skeleton(ax, joints, color=color, label=title)
                    if trackers is not None and valid is not None:
                        draw_trackers(ax, trackers[frame_index], valid[frame_index])
                    set_axes_equal(ax, axis_limits)
                    ax.set_title(title)
            status = "eval" if bool(eval_mask[frame_index]) else "warm-up/skip"
            fig.suptitle(
                f"frame={frame_index} status={status} MPJPE={joint_error:.4f} yaw_err={yaw_error:.4f}",
                fontsize=11,
            )
            fig.tight_layout()
            frame_rgb = figure_to_rgb(fig)
            if writer is None:
                writer = Mp4FrameWriter(output_path=output_path, frame_rgb=frame_rgb, fps=int(fps))
            writer.append(frame_rgb)
            plt.close(fig)
    finally:
        if writer is not None:
            writer.close()
    return output_path


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_arg_parser().parse_args(argv)
    input_path = Path(args.input_npz).resolve()
    output_path = Path(args.output_mp4).resolve()
    with np.load(input_path, allow_pickle=True) as data:
        render_realtime_pose_comparison(
            output_path=output_path,
            reference_joints=data["reference_joints_world"],
            predicted_joints=data["predicted_joints_world"],
            tracker_pos_world=data["tracker_pos_world"] if "tracker_pos_world" in data.files else None,
            sensor_valid=data["sensor_valid"] if "sensor_valid" in data.files else None,
            eval_frame_mask=data["eval_frame_mask"] if "eval_frame_mask" in data.files else None,
            root_yaw_reference=data["root_yaw_reference"] if "root_yaw_reference" in data.files else None,
            root_yaw_predicted=data["root_yaw_predicted"] if "root_yaw_predicted" in data.files else None,
            fps=int(args.fps),
            stride=int(args.stride),
            camera_mode=str(args.camera_mode),
            layout=str(args.layout),
            local_radius=float(args.local_radius),
        )
    print(f"[render_realtime_pose_comparison] output={output_path}")
    return {"output_path": output_path}


if __name__ == "__main__":
    main()
