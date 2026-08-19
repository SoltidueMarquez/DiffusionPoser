from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    PREDICTOR_SPARSE_DIM,
    TRACKER_CONTINUOUS_DIM,
    TRACKER_COUNT,
)


REALTIME_POSE_MIN_NORMALIZER_STD = 1e-4


def _validate_eps(value: Any) -> float:
    eps = float(value)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("normalizer eps 必须是有限非负数。")
    return eps


def _stabilize_std(value: torch.Tensor) -> torch.Tensor:
    result = value.float().clone()
    if torch.any(result <= 0):
        raise ValueError("normalizer scale/std 必须全部大于零。")
    result[result < REALTIME_POSE_MIN_NORMALIZER_STD] = 1.0
    return result


def build_pose_scale(pose_std: Any, eps: float) -> torch.Tensor:
    """把统计标准差固化为所有 Pose 正反归一化共同使用的 `[144]` 尺度。"""

    return _stabilize_std(
        torch.as_tensor(pose_std, dtype=torch.float32).reshape(-1)
    ) + _validate_eps(eps)


class RealtimePoseNormalizer:
    """Predictor 与单帧 DiT 共用的 Pose、Tracker 和 54D sparse normalizer。"""

    def __init__(
        self,
        base_dir: str | Path,
        eps: float = 1e-8,
        disable: bool = False,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.eps = _validate_eps(eps)
        self.disable = bool(disable)
        self.pose_mean_path = self.base_dir / "pose_mean.pt"
        self.pose_scale_path = self.base_dir / "pose_scale.pt"
        self.tracker_mean_path = self.base_dir / "tracker_mean.pt"
        self.tracker_std_path = self.base_dir / "tracker_std.pt"
        self.predictor_sparse_mean_path = self.base_dir / "predictor_sparse_mean.pt"
        self.predictor_sparse_std_path = self.base_dir / "predictor_sparse_std.pt"

        self.pose_mean: torch.Tensor | None = None
        self.pose_scale: torch.Tensor | None = None
        self.tracker_mean: torch.Tensor | None = None
        self.tracker_std: torch.Tensor | None = None
        self.predictor_sparse_mean: torch.Tensor | None = None
        self.predictor_sparse_std: torch.Tensor | None = None
        if not self.disable:
            self.load()

    def load(self) -> None:
        paths = (
            self.pose_mean_path,
            self.pose_scale_path,
            self.tracker_mean_path,
            self.tracker_std_path,
            self.predictor_sparse_mean_path,
            self.predictor_sparse_std_path,
        )
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{self.base_dir} 缺少 normalizer 文件：{missing}")
        self.pose_mean = torch.load(
            self.pose_mean_path, map_location="cpu", weights_only=True
        ).float()
        self.pose_scale = _stabilize_std(
            torch.load(
                self.pose_scale_path, map_location="cpu", weights_only=True
            ).float()
        )
        self.tracker_mean = torch.load(
            self.tracker_mean_path, map_location="cpu", weights_only=True
        ).float()
        self.tracker_std = _stabilize_std(
            torch.load(
                self.tracker_std_path, map_location="cpu", weights_only=True
            ).float()
        )
        self.predictor_sparse_mean = torch.load(
            self.predictor_sparse_mean_path, map_location="cpu", weights_only=True
        ).float()
        self.predictor_sparse_std = _stabilize_std(
            torch.load(
                self.predictor_sparse_std_path, map_location="cpu", weights_only=True
            ).float()
        )
        self._validate_stats()

    def save(
        self,
        pose_mean: Any,
        pose_scale: Any,
        tracker_mean: Any,
        tracker_std: Any,
        predictor_sparse_mean: Any,
        predictor_sparse_std: Any,
    ) -> None:
        self.pose_mean = torch.as_tensor(pose_mean, dtype=torch.float32).reshape(
            REALTIME_POSE_TARGET_DIM
        )
        self.pose_scale = _stabilize_std(
            torch.as_tensor(pose_scale, dtype=torch.float32).reshape(
                REALTIME_POSE_TARGET_DIM
            )
        )
        self.tracker_mean = torch.as_tensor(
            tracker_mean, dtype=torch.float32
        ).reshape(TRACKER_COUNT, TRACKER_CONTINUOUS_DIM)
        self.tracker_std = _stabilize_std(
            torch.as_tensor(tracker_std, dtype=torch.float32).reshape(
                TRACKER_COUNT, TRACKER_CONTINUOUS_DIM
            )
        )
        self.predictor_sparse_mean = torch.as_tensor(
            predictor_sparse_mean, dtype=torch.float32
        ).reshape(PREDICTOR_SPARSE_DIM)
        self.predictor_sparse_std = _stabilize_std(
            torch.as_tensor(predictor_sparse_std, dtype=torch.float32).reshape(
                PREDICTOR_SPARSE_DIM
            )
        )
        self._validate_stats()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.pose_mean, self.pose_mean_path)
        torch.save(self.pose_scale, self.pose_scale_path)
        torch.save(self.tracker_mean, self.tracker_mean_path)
        torch.save(self.tracker_std, self.tracker_std_path)
        torch.save(self.predictor_sparse_mean, self.predictor_sparse_mean_path)
        torch.save(self.predictor_sparse_std, self.predictor_sparse_std_path)

    def normalize_pose(
        self, value: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if self.disable:
            return value
        mean, scale = self._stats_for(value, "pose")
        return (value - mean) / scale

    def inverse_pose(
        self, value: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if self.disable:
            return value
        mean, scale = self._stats_for(value, "pose")
        return value * scale + mean

    def normalize_tracker_continuous(
        self, value: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if value.shape[-2:] != (TRACKER_COUNT, TRACKER_CONTINUOUS_DIM):
            raise ValueError("Tracker 连续量尾维必须为 [6,9]。")
        if self.disable:
            return value
        mean, std = self._stats_for(value, "tracker")
        return (value - mean) / (std + self.eps)

    def inverse_tracker_continuous(
        self, value: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if value.shape[-2:] != (TRACKER_COUNT, TRACKER_CONTINUOUS_DIM):
            raise ValueError("Tracker 连续量尾维必须为 [6,9]。")
        if self.disable:
            return value
        mean, std = self._stats_for(value, "tracker")
        return value * (std + self.eps) + mean

    def normalize_predictor_sparse(
        self, value: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if value.shape[-1] != PREDICTOR_SPARSE_DIM:
            raise ValueError(f"Predictor sparse 最后一维必须为 {PREDICTOR_SPARSE_DIM}。")
        if self.disable:
            return value
        mean, std = self._stats_for(value, "predictor_sparse")
        return (value - mean) / (std + self.eps)

    def inverse_predictor_sparse(
        self, value: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if value.shape[-1] != PREDICTOR_SPARSE_DIM:
            raise ValueError(f"Predictor sparse 最后一维必须为 {PREDICTOR_SPARSE_DIM}。")
        if self.disable:
            return value
        mean, std = self._stats_for(value, "predictor_sparse")
        return value * (std + self.eps) + mean

    def _stats_for(
        self,
        value: np.ndarray | torch.Tensor,
        kind: str,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if self.pose_mean is None:
            self.load()
        mapping = {
            "pose": (self.pose_mean, self.pose_scale),
            "tracker": (self.tracker_mean, self.tracker_std),
            "predictor_sparse": (self.predictor_sparse_mean, self.predictor_sparse_std),
        }
        mean, scale = mapping[kind]
        assert mean is not None and scale is not None
        if isinstance(value, np.ndarray):
            return (
                mean.numpy().astype(value.dtype, copy=False),
                scale.numpy().astype(value.dtype, copy=False),
            )
        return (
            mean.to(device=value.device, dtype=value.dtype),
            scale.to(device=value.device, dtype=value.dtype),
        )

    def _validate_stats(self) -> None:
        values = (
            self.pose_mean,
            self.pose_scale,
            self.tracker_mean,
            self.tracker_std,
            self.predictor_sparse_mean,
            self.predictor_sparse_std,
        )
        if any(value is None for value in values):
            raise RuntimeError("normalizer 统计尚未完整设置。")
        assert self.pose_mean is not None and self.pose_scale is not None
        assert self.tracker_mean is not None and self.tracker_std is not None
        assert self.predictor_sparse_mean is not None and self.predictor_sparse_std is not None
        if tuple(self.pose_mean.shape) != (REALTIME_POSE_TARGET_DIM,):
            raise ValueError("pose_mean 必须为 [144]。")
        if tuple(self.pose_scale.shape) != (REALTIME_POSE_TARGET_DIM,):
            raise ValueError("pose_scale 必须为 [144]。")
        if tuple(self.tracker_mean.shape) != (
            TRACKER_COUNT,
            TRACKER_CONTINUOUS_DIM,
        ) or tuple(self.tracker_std.shape) != (
            TRACKER_COUNT,
            TRACKER_CONTINUOUS_DIM,
        ):
            raise ValueError("Tracker normalizer 必须为 [6,9]。")
        if tuple(self.predictor_sparse_mean.shape) != (PREDICTOR_SPARSE_DIM,) or tuple(
            self.predictor_sparse_std.shape
        ) != (PREDICTOR_SPARSE_DIM,):
            raise ValueError(
                f"Predictor sparse normalizer 必须为 [{PREDICTOR_SPARSE_DIM}]。"
            )
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("normalizer 统计包含 NaN 或 Inf。")
        if bool((self.pose_scale <= 0).any()) or bool(
            (self.tracker_std <= 0).any()
        ) or bool((self.predictor_sparse_std <= 0).any()):
            raise ValueError("normalizer scale/std 必须全部大于零。")
