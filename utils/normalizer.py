from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
    TRACKER_FEATURE_DIM,
    TRACKER_MEASURED_VALID_OFFSET,
)
REALTIME_POSE_MIN_NORMALIZER_STD = 1e-4


def _stabilize_std(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clone()
    if torch.any(value <= 0):
        raise ValueError("normalizer scale/std 必须全部大于零。")
    value[value < REALTIME_POSE_MIN_NORMALIZER_STD] = 1.0
    return value


def _validate_eps(value: Any) -> float:
    eps = float(value)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("normalizer eps 必须是有限非负数。")
    return eps


def build_pose_scale(pose_std: Any, eps: float) -> torch.Tensor:
    """把统计标准差固化为所有 Pose 正反归一化共同使用的 `[144]` 尺度。"""

    return _stabilize_std(torch.as_tensor(pose_std, dtype=torch.float32).reshape(-1)) + _validate_eps(eps)


class RealtimePoseNormalizer:
    """144 维姿态与六类 Tracker 连续量的独立 normalizer。"""

    def __init__(
        self,
        base_dir: str | Path,
        eps: float = 1e-8,
        disable: bool = False,
    ):
        self.eps = _validate_eps(eps)
        self.disable = bool(disable)
        self.base_dir = Path(base_dir).resolve()
        self.pose_mean_path = self.base_dir / "pose_mean.pt"
        self.pose_scale_path = self.base_dir / "pose_scale.pt"
        self.tracker_mean_path = self.base_dir / "tracker_mean.pt"
        self.tracker_std_path = self.base_dir / "tracker_std.pt"
        self.head_path_xz_mean_path = self.base_dir / "head_path_xz_mean.pt"
        self.head_path_xz_std_path = self.base_dir / "head_path_xz_std.pt"
        self.head_height_mean_path = self.base_dir / "head_height_mean.pt"
        self.head_height_std_path = self.base_dir / "head_height_std.pt"
        self.pose_mean: torch.Tensor | None = None
        self.pose_scale: torch.Tensor | None = None
        self.tracker_mean: torch.Tensor | None = None
        self.tracker_std: torch.Tensor | None = None
        self.head_path_xz_mean: torch.Tensor | None = None
        self.head_path_xz_std: torch.Tensor | None = None
        self.head_height_mean: torch.Tensor | None = None
        self.head_height_std: torch.Tensor | None = None
        if not self.disable:
            self.load()

    def load(self) -> None:
        required = (
            self.pose_mean_path,
            self.pose_scale_path,
            self.tracker_mean_path,
            self.tracker_std_path,
            self.head_path_xz_mean_path,
            self.head_path_xz_std_path,
            self.head_height_mean_path,
            self.head_height_std_path,
        )
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.base_dir} 缺少 144 维 normalizer 文件 {missing}；旧契约的统计不能复用。"
            )
        self.pose_mean = torch.load(self.pose_mean_path, map_location="cpu", weights_only=True).float()
        self.pose_scale = torch.load(
            self.pose_scale_path, map_location="cpu", weights_only=True
        ).float()
        self.tracker_mean = torch.load(self.tracker_mean_path, map_location="cpu", weights_only=True).float()
        self.tracker_std = _stabilize_std(
            torch.load(self.tracker_std_path, map_location="cpu", weights_only=True).float()
        )
        self.head_path_xz_mean = torch.load(
            self.head_path_xz_mean_path, map_location="cpu", weights_only=True
        ).float().reshape(2)
        self.head_path_xz_std = _stabilize_std(
            torch.load(self.head_path_xz_std_path, map_location="cpu", weights_only=True).float().reshape(2)
        )
        self.head_height_mean = torch.load(
            self.head_height_mean_path, map_location="cpu", weights_only=True
        ).float().reshape(())
        self.head_height_std = _stabilize_std(
            torch.load(self.head_height_std_path, map_location="cpu", weights_only=True).float().reshape(1)
        ).reshape(())
        self._validate_stats()

    def save(
        self,
        pose_mean: Any,
        pose_scale: Any,
        tracker_mean: Any,
        tracker_std: Any,
        head_path_xz_mean: Any,
        head_path_xz_std: Any,
        head_height_mean: Any,
        head_height_std: Any,
    ) -> None:
        self.pose_mean = torch.as_tensor(pose_mean, dtype=torch.float32).reshape(-1)
        self.pose_scale = torch.as_tensor(pose_scale, dtype=torch.float32).reshape(-1)
        self.tracker_mean = torch.as_tensor(tracker_mean, dtype=torch.float32).reshape(
            TRACKER_COUNT, TRACKER_CONTINUOUS_DIM
        )
        self.tracker_std = _stabilize_std(
            torch.as_tensor(tracker_std, dtype=torch.float32).reshape(
                TRACKER_COUNT, TRACKER_CONTINUOUS_DIM
            )
        )
        self.head_path_xz_mean = torch.as_tensor(
            head_path_xz_mean, dtype=torch.float32
        ).reshape(2)
        self.head_path_xz_std = _stabilize_std(
            torch.as_tensor(head_path_xz_std, dtype=torch.float32).reshape(2)
        )
        self.head_height_mean = torch.as_tensor(head_height_mean, dtype=torch.float32).reshape(())
        self.head_height_std = _stabilize_std(
            torch.as_tensor(head_height_std, dtype=torch.float32).reshape(1)
        ).reshape(())
        self._validate_stats()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.pose_mean, self.pose_mean_path)
        torch.save(self.pose_scale, self.pose_scale_path)
        torch.save(self.tracker_mean, self.tracker_mean_path)
        torch.save(self.tracker_std, self.tracker_std_path)
        torch.save(self.head_path_xz_mean, self.head_path_xz_mean_path)
        torch.save(self.head_path_xz_std, self.head_path_xz_std_path)
        torch.save(self.head_height_mean, self.head_height_mean_path)
        torch.save(self.head_height_std, self.head_height_std_path)

    def normalize_pose(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.disable:
            return value
        mean, scale = self._pose_stats_for(value)
        return (value - mean) / scale

    def inverse_pose(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.disable:
            return value
        mean, scale = self._pose_stats_for(value)
        return value * scale + mean

    def normalize_tracker(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """归一化前 9 维，并在归一化后严格清零 invalid Tracker 连续量。"""

        if value.shape[-2:] != (TRACKER_COUNT, TRACKER_FEATURE_DIM):
            raise ValueError(
                f"tracker_window 尾部必须为 [{TRACKER_COUNT},{TRACKER_FEATURE_DIM}]，实际为 {tuple(value.shape)}"
            )
        if self.disable:
            return value.clone() if isinstance(value, torch.Tensor) else value.copy()
        mean, std = self._tracker_stats_for(value)
        result = value.clone() if isinstance(value, torch.Tensor) else value.copy()
        continuous = (result[..., :TRACKER_CONTINUOUS_DIM] - mean) / (std + self.eps)
        measured = result[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5
        if isinstance(result, torch.Tensor):
            continuous = torch.where(measured[..., None], continuous, torch.zeros_like(continuous))
        else:
            continuous = np.where(measured[..., None], continuous, np.zeros_like(continuous))
        result[..., :TRACKER_CONTINUOUS_DIM] = continuous
        return result

    def inverse_tracker_continuous(
        self,
        value: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        if value.shape[-2:] != (TRACKER_COUNT, TRACKER_CONTINUOUS_DIM):
            raise ValueError(
                f"Tracker 连续量尾部必须为 [{TRACKER_COUNT},{TRACKER_CONTINUOUS_DIM}]，实际为 {tuple(value.shape)}"
            )
        if self.disable:
            return value
        mean, std = self._tracker_stats_for(value)
        return value * (std + self.eps) + mean

    def normalize_head_height(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """只归一化 trajectory 的 Head height 标量，其他四维保持物理量。"""

        mean, std = self._head_height_stats_for(value)
        return (value - mean) / (std + self.eps)

    def normalize_head_path(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """归一化 Head 路径的 XZ 与高度，sin/cos 保持原始物理语义。"""

        if value.shape[-1] != 5:
            raise ValueError(f"head_path 最后一维必须为 5，实际为 {value.shape[-1]}")
        result = value.clone() if isinstance(value, torch.Tensor) else value.copy()
        xz_mean, xz_std = self._head_path_xz_stats_for(result)
        height_mean, height_std = self._head_height_stats_for(result)
        result[..., :2] = (result[..., :2] - xz_mean) / (xz_std + self.eps)
        result[..., 2] = (result[..., 2] - height_mean) / (height_std + self.eps)
        return result

    def inverse_head_height(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        mean, std = self._head_height_stats_for(value)
        return value * (std + self.eps) + mean

    def _pose_stats_for(
        self, value: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if value.shape[-1] != REALTIME_POSE_TARGET_DIM:
            raise ValueError(
                f"姿态最后一维必须为 {REALTIME_POSE_TARGET_DIM}，实际为 {value.shape[-1]}；旧 154/214 维数据不可用。"
            )
        if self.pose_mean is None or self.pose_scale is None:
            self.load()
        assert self.pose_mean is not None and self.pose_scale is not None
        if isinstance(value, np.ndarray):
            return (
                self.pose_mean.numpy().astype(value.dtype, copy=False),
                self.pose_scale.numpy().astype(value.dtype, copy=False),
            )
        return (
            self.pose_mean.to(device=value.device, dtype=value.dtype),
            self.pose_scale.to(device=value.device, dtype=value.dtype),
        )

    def _tracker_stats_for(
        self, value: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if self.tracker_mean is None or self.tracker_std is None:
            self.load()
        assert self.tracker_mean is not None and self.tracker_std is not None
        if isinstance(value, np.ndarray):
            return (
                self.tracker_mean.numpy().astype(value.dtype, copy=False),
                self.tracker_std.numpy().astype(value.dtype, copy=False),
            )
        return (
            self.tracker_mean.to(device=value.device, dtype=value.dtype),
            self.tracker_std.to(device=value.device, dtype=value.dtype),
        )

    def _validate_stats(self) -> None:
        assert self.pose_mean is not None and self.pose_scale is not None
        assert self.tracker_mean is not None and self.tracker_std is not None
        assert self.head_path_xz_mean is not None and self.head_path_xz_std is not None
        assert self.head_height_mean is not None and self.head_height_std is not None
        if tuple(self.pose_mean.shape) != (REALTIME_POSE_TARGET_DIM,) or tuple(self.pose_scale.shape) != (
            REALTIME_POSE_TARGET_DIM,
        ):
            raise ValueError("Pose normalizer 必须为 [144]。")
        expected_tracker = (TRACKER_COUNT, TRACKER_CONTINUOUS_DIM)
        if tuple(self.tracker_mean.shape) != expected_tracker or tuple(self.tracker_std.shape) != expected_tracker:
            raise ValueError("Tracker normalizer 必须为 [6,9]。")
        tensors = (
            self.pose_mean,
            self.pose_scale,
            self.tracker_mean,
            self.tracker_std,
            self.head_path_xz_mean,
            self.head_path_xz_std,
            self.head_height_mean,
            self.head_height_std,
        )
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("normalizer 统计包含 NaN 或 Inf。")
        if torch.any(self.pose_scale <= 0) or torch.any(self.tracker_std <= 0) or self.head_height_std <= 0:
            raise ValueError("normalizer std 必须全大于零。")

        if torch.any(self.head_path_xz_std <= 0):
            raise ValueError("Head 路径 XZ normalizer std 必须全部大于零。")

    def _head_path_xz_stats_for(
        self, value: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if self.head_path_xz_mean is None or self.head_path_xz_std is None:
            self.load()
        assert self.head_path_xz_mean is not None and self.head_path_xz_std is not None
        if isinstance(value, np.ndarray):
            return (
                self.head_path_xz_mean.numpy().astype(value.dtype, copy=False),
                self.head_path_xz_std.numpy().astype(value.dtype, copy=False),
            )
        return (
            self.head_path_xz_mean.to(device=value.device, dtype=value.dtype),
            self.head_path_xz_std.to(device=value.device, dtype=value.dtype),
        )

    def _head_height_stats_for(
        self, value: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if self.head_height_mean is None or self.head_height_std is None:
            self.load()
        assert self.head_height_mean is not None and self.head_height_std is not None
        if isinstance(value, np.ndarray):
            return (
                np.asarray(self.head_height_mean.item(), dtype=value.dtype),
                np.asarray(self.head_height_std.item(), dtype=value.dtype),
            )
        return (
            self.head_height_mean.to(device=value.device, dtype=value.dtype),
            self.head_height_std.to(device=value.device, dtype=value.dtype),
        )
