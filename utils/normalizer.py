from __future__ import annotations

import json
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
from utils.run_dirs import resolve_latest_or_self


REALTIME_POSE_MIN_NORMALIZER_STD = 1e-4


def _stabilize_std(value: torch.Tensor) -> torch.Tensor:
    value = value.float().clone()
    value[value < REALTIME_POSE_MIN_NORMALIZER_STD] = 1.0
    return value


class RealtimePoseNormalizer:
    """144 维姿态与六类 Tracker 连续量的独立 normalizer。"""

    def __init__(
        self,
        base_dir: str | Path,
        eps: float = 1e-8,
        disable: bool = False,
    ):
        self.eps = float(eps)
        self.disable = bool(disable)
        self.base_dir = Path(base_dir) if disable else resolve_latest_or_self(base_dir, kind="normalizer")
        self.pose_mean_path = self.base_dir / "pose_mean.pt"
        self.pose_std_path = self.base_dir / "pose_std.pt"
        self.tracker_mean_path = self.base_dir / "tracker_mean.pt"
        self.tracker_std_path = self.base_dir / "tracker_std.pt"
        self.meta_path = self.base_dir / "normalizer_meta.json"
        self.pose_mean: torch.Tensor | None = None
        self.pose_std: torch.Tensor | None = None
        self.tracker_mean: torch.Tensor | None = None
        self.tracker_std: torch.Tensor | None = None
        self.metadata: dict[str, Any] = {}
        # 训练循环仍通过 mean/std 读取姿态统计；这里明确只指 144 维姿态。
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if not self.disable:
            self.load()

    def load(self) -> None:
        required = (
            self.pose_mean_path,
            self.pose_std_path,
            self.tracker_mean_path,
            self.tracker_std_path,
        )
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{self.base_dir} 缺少 144 维 normalizer 文件 {missing}；旧契约的统计不能复用。"
            )
        self.pose_mean = torch.load(self.pose_mean_path, map_location="cpu", weights_only=True).float()
        self.pose_std = _stabilize_std(
            torch.load(self.pose_std_path, map_location="cpu", weights_only=True).float()
        )
        self.tracker_mean = torch.load(self.tracker_mean_path, map_location="cpu", weights_only=True).float()
        self.tracker_std = _stabilize_std(
            torch.load(self.tracker_std_path, map_location="cpu", weights_only=True).float()
        )
        self._validate_stats()
        if self.meta_path.exists():
            value = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                self.metadata = value
        self.mean = self.pose_mean
        self.std = self.pose_std

    def save(
        self,
        pose_mean: Any,
        pose_std: Any,
        tracker_mean: Any,
        tracker_std: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.pose_mean = torch.as_tensor(pose_mean, dtype=torch.float32).reshape(-1)
        self.pose_std = _stabilize_std(torch.as_tensor(pose_std, dtype=torch.float32).reshape(-1))
        self.tracker_mean = torch.as_tensor(tracker_mean, dtype=torch.float32).reshape(
            TRACKER_COUNT, TRACKER_CONTINUOUS_DIM
        )
        self.tracker_std = _stabilize_std(
            torch.as_tensor(tracker_std, dtype=torch.float32).reshape(
                TRACKER_COUNT, TRACKER_CONTINUOUS_DIM
            )
        )
        self._validate_stats()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.pose_mean, self.pose_mean_path)
        torch.save(self.pose_std, self.pose_std_path)
        torch.save(self.tracker_mean, self.tracker_mean_path)
        torch.save(self.tracker_std, self.tracker_std_path)
        self._write_meta(metadata or {})
        self.metadata = {"eps": self.eps, **(metadata or {})}
        self.mean = self.pose_mean
        self.std = self.pose_std

    def normalize_pose(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.disable:
            return value
        mean, std = self._pose_stats_for(value)
        return (value - mean) / (std + self.eps)

    def inverse_pose(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.disable:
            return value
        mean, std = self._pose_stats_for(value)
        return value * (std + self.eps) + mean

    # 兼容仓库内只针对姿态张量的简写；不再接受 214 维混合特征。
    normalize = normalize_pose
    inverse = inverse_pose

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

    def _pose_stats_for(
        self, value: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if value.shape[-1] != REALTIME_POSE_TARGET_DIM:
            raise ValueError(
                f"姿态最后一维必须为 {REALTIME_POSE_TARGET_DIM}，实际为 {value.shape[-1]}；旧 154/214 维数据不可用。"
            )
        if self.pose_mean is None or self.pose_std is None:
            self.load()
        assert self.pose_mean is not None and self.pose_std is not None
        if isinstance(value, np.ndarray):
            return (
                self.pose_mean.numpy().astype(value.dtype, copy=False),
                self.pose_std.numpy().astype(value.dtype, copy=False),
            )
        return (
            self.pose_mean.to(device=value.device, dtype=value.dtype),
            self.pose_std.to(device=value.device, dtype=value.dtype),
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
        assert self.pose_mean is not None and self.pose_std is not None
        assert self.tracker_mean is not None and self.tracker_std is not None
        if tuple(self.pose_mean.shape) != (REALTIME_POSE_TARGET_DIM,) or tuple(self.pose_std.shape) != (
            REALTIME_POSE_TARGET_DIM,
        ):
            raise ValueError("Pose normalizer 必须为 [144]。")
        expected_tracker = (TRACKER_COUNT, TRACKER_CONTINUOUS_DIM)
        if tuple(self.tracker_mean.shape) != expected_tracker or tuple(self.tracker_std.shape) != expected_tracker:
            raise ValueError("Tracker normalizer 必须为 [6,9]。")
        tensors = (self.pose_mean, self.pose_std, self.tracker_mean, self.tracker_std)
        if not all(torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("normalizer 统计包含 NaN 或 Inf。")
        if torch.any(self.pose_std <= 0) or torch.any(self.tracker_std <= 0):
            raise ValueError("normalizer std 必须全大于零。")

    def _write_meta(self, extra: dict[str, Any]) -> None:
        meta = {
            "eps": self.eps,
            **extra,
        }
        with self.meta_path.open("w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2, sort_keys=True)
