from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


TRAINING_PRECISIONS = ("fp32", "bf16")


class TrainingPrecision:
    """只对模型 forward 使用 BF16，返回 FP32 结果供几何与 diffusion loss 使用。"""

    def __init__(self, name: str, device: torch.device | str):
        self.name = str(name).lower()
        self.device = torch.device(device)
        if self.name not in TRAINING_PRECISIONS:
            raise ValueError(f"precision 必须是 {TRAINING_PRECISIONS}，实际为 {name}。")
        if self.name == "bf16" and (
            self.device.type != "cuda" or not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("BF16 训练需要支持 BF16 的 CUDA GPU。")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def forward(self, model: torch.nn.Module, *args, **kwargs) -> Any:
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.name == "bf16"
            else nullcontext()
        )
        with context:
            output = model(*args, **kwargs)
        return _floating_to_fp32(output)


def _floating_to_fp32(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _floating_to_fp32(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_floating_to_fp32(item) for item in value)
    return value
