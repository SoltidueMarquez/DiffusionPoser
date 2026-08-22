from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


class Mp4FrameWriter:
    """把连续 RGB 帧通过 imageio-ffmpeg 编码成 H.264 MP4。"""

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
            raise RuntimeError(
                f"ffmpeg 写 MP4 失败，returncode={return_code}: {message}"
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            try:
                self.output_path.unlink()
            except PermissionError as exc:
                raise PermissionError(
                    f"无法覆盖 MP4，目标文件可能正被播放器占用：{self.output_path}"
                ) from exc
        shutil.move(str(self.temp_output_path), str(self.output_path))
        shutil.rmtree(self.temp_dir, ignore_errors=True)


def find_ffmpeg_exe() -> str:
    """优先使用 imageio-ffmpeg，并兼容 Conda 中的二进制目录。"""

    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except ModuleNotFoundError:
        pass

    roots = [os.environ.get("CONDA_PREFIX", ""), sys.prefix]
    patterns = [
        Path("Lib")
        / "site-packages"
        / "imageio_ffmpeg"
        / "binaries"
        / "ffmpeg*.exe",
        Path("lib")
        / "python*"
        / "site-packages"
        / "imageio_ffmpeg"
        / "binaries"
        / "ffmpeg*",
    ]
    for root_value in roots:
        if not root_value:
            continue
        root = Path(root_value)
        for pattern in patterns:
            matches = sorted(root.glob(str(pattern)))
            if matches:
                return str(matches[0])
    raise ModuleNotFoundError(
        "找不到 imageio-ffmpeg 或 ffmpeg 可执行文件，无法写 MP4。"
    )
