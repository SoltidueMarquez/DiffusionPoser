from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_loaders.sensor_masking import (
    REALTIME_POSE_SCHEMA_NAME,
    get_schema_spec,
)


def enforce_realtime_pose_normalizer_contract(
    mean: torch.Tensor,
    std: torch.Tensor,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> tuple[torch.Tensor, torch.Tensor]:
    """固定 sensor_valid/contact 为条件标签，避免标准化破坏二值语义。"""

    schema = get_schema_spec(schema_name)
    mean = mean.float().flatten().clone()
    std = std.float().flatten().clone()
    if mean.numel() == schema.feature_dim and std.numel() == schema.feature_dim:
        sensor_slice = schema.sensor_valid_slice()
        mean[sensor_slice] = 0.0
        std[sensor_slice] = 1.0
        if schema.supports_contact:
            contact_slice = schema.foot_contact_slice()
            mean[contact_slice] = 0.0
            std[contact_slice] = 1.0
    return mean, std


class RealtimePoseNormalizer:
    """realtime_pose schema-specific mean/std 标准化工具。"""

    def __init__(
        self,
        base_dir: str | Path,
        eps: float = 1e-8,
        disable: bool = False,
        schema_name: str = REALTIME_POSE_SCHEMA_NAME,
    ):
        self.base_dir = Path(base_dir)
        self.mean_path = self.base_dir / "mean.pt"
        self.std_path = self.base_dir / "std.pt"
        self.eps = float(eps)
        self.disable = bool(disable)
        self.schema = get_schema_spec(schema_name)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        if not self.disable:
            self.load()

    def load(self) -> None:
        if not self.mean_path.exists() or not self.std_path.exists():
            raise FileNotFoundError(
                "找不到 realtime_pose_v1 normalizer 文件，请先运行 "
                "`python -m data_loaders.compute_realtime_pose_normalizer ...`。"
            )
        mean = torch.load(self.mean_path, map_location="cpu", weights_only=True).float().flatten()
        std = torch.load(self.std_path, map_location="cpu", weights_only=True).float().flatten()
        mean, std = enforce_realtime_pose_normalizer_contract(mean, std, schema_name=self.schema.name)
        self._validate_stats(mean=mean, std=std)
        self.mean = mean
        self.std = torch.clamp(std, min=self.eps)

    def save(self, mean: Any, std: Any) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std_tensor = torch.as_tensor(std, dtype=torch.float32).flatten()
        mean_tensor, std_tensor = enforce_realtime_pose_normalizer_contract(
            mean_tensor,
            std_tensor,
            schema_name=self.schema.name,
        )
        self._validate_stats(mean=mean_tensor, std=std_tensor)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mean_tensor, self.mean_path)
        torch.save(torch.clamp(std_tensor, min=self.eps), self.std_path)
        self.mean = mean_tensor
        self.std = torch.clamp(std_tensor, min=self.eps)

    def normalize(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.disable:
            return x
        mean, std = self._stats_for_value(x)
        return (x - mean) / (std + self.eps)

    def inverse(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.disable:
            return x
        mean, std = self._stats_for_value(x)
        return x * (std + self.eps) + mean

    def _validate_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        expected_dim = self.schema.feature_dim
        if mean.numel() != expected_dim or std.numel() != expected_dim:
            raise ValueError(
                f"realtime_pose normalizer 需要 {expected_dim} 维，"
                f"实际 mean={mean.numel()}, std={std.numel()}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("realtime_pose normalizer mean/std 包含 NaN 或 Inf。")
        if torch.any(std <= 0):
            raise ValueError("realtime_pose normalizer std 必须全部大于 0。")

    def _stats_for_value(self, x: np.ndarray | torch.Tensor) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if self.mean is None or self.std is None:
            self.load()
        if x.shape[-1] != self.schema.feature_dim:
            raise ValueError(f"输入最后一维应为 {self.schema.feature_dim}，实际为 {x.shape[-1]}")

        assert self.mean is not None and self.std is not None
        if isinstance(x, np.ndarray):
            mean = self.mean.numpy().astype(x.dtype, copy=False)
            std = self.std.numpy().astype(x.dtype, copy=False)
            return mean, std
        return self.mean.to(device=x.device, dtype=x.dtype), self.std.to(device=x.device, dtype=x.dtype)
