from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from data_loaders.sensor_masking import X277_FEATURE_DIM


class X277Normalizer:
    """
    X277 原始动作特征的归一化工具。

    本工具只负责 `x: [T, 277]` 的真实动作特征，不包含追加的 6 维传感器缺失标签。
    标签通道是离散条件，是否映射为 -1/+1 由 dataloader 在拼接 `[T, 283]` 时处理。
    """

    def __init__(self, base_dir: str | Path, eps: float = 1e-8, disable: bool = False):
        self.base_dir = Path(base_dir)
        self.mean_path = self.base_dir / "mean.pt"
        self.std_path = self.base_dir / "std.pt"
        self.eps = float(eps)
        self.disable = disable
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

        if not disable:
            self.load()

    def load(self) -> None:
        if not self.mean_path.exists() or not self.std_path.exists():
            raise FileNotFoundError(
                "找不到 X277 normalizer 文件，请先运行："
                "python -m data_loaders.compute_x277_normalizer "
                "--source_dir dataset/AMASS_x277_60hz "
                "--output_dir dataset/meta_AMASS_x277_60hz "
                "--split_dir data_loaders/splits --split train --overwrite"
            )

        mean = torch.load(self.mean_path, map_location="cpu", weights_only=True).float().flatten()
        std = torch.load(self.std_path, map_location="cpu", weights_only=True).float().flatten()
        self._validate_stats(mean=mean, std=std)
        self.mean = mean
        self.std = torch.clamp(std, min=self.eps)

    def save(self, mean: Any, std: Any) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std_tensor = torch.as_tensor(std, dtype=torch.float32).flatten()
        self._validate_stats(mean=mean_tensor, std=std_tensor)

        self.base_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mean_tensor, self.mean_path)
        torch.save(torch.clamp(std_tensor, min=self.eps), self.std_path)
        self.mean = mean_tensor
        self.std = torch.clamp(std_tensor, min=self.eps)

    def normalize(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        将 X277 特征标准化。

        输入和输出形状保持一致，最后一维必须是 277，例如 `[T, 277]` 或 `[B, T, 277]`。
        """

        if self.disable:
            return x
        mean, std = self._stats_for_value(x)
        return (x - mean) / (std + self.eps)

    def inverse(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        将标准化后的 X277 特征还原到原始尺度。

        输入和输出形状保持一致，最后一维必须是 277。
        """

        if self.disable:
            return x
        mean, std = self._stats_for_value(x)
        return x * (std + self.eps) + mean

    @staticmethod
    def _validate_stats(mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.numel() != X277_FEATURE_DIM or std.numel() != X277_FEATURE_DIM:
            raise ValueError(
                f"X277 normalizer 需要 {X277_FEATURE_DIM} 维 mean/std，"
                f"实际为 mean={mean.numel()}, std={std.numel()}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("X277 normalizer 的 mean/std 包含 NaN 或 Inf。")
        if torch.any(std <= 0):
            raise ValueError("X277 normalizer 的 std 必须全部大于 0。")

    def _stats_for_value(self, x: np.ndarray | torch.Tensor) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        if self.mean is None or self.std is None:
            self.load()

        if x.shape[-1] != X277_FEATURE_DIM:
            raise ValueError(f"输入最后一维应为 {X277_FEATURE_DIM}，实际为 {x.shape[-1]}")

        assert self.mean is not None and self.std is not None
        if isinstance(x, np.ndarray):
            mean = self.mean.numpy().astype(x.dtype, copy=False)
            std = self.std.numpy().astype(x.dtype, copy=False)
            return mean, std

        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return mean, std
