from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint_model(model, model_path: str | Path, device, use_ema: bool = True):
    """加载当前模型 checkpoint；常规 step checkpoint 优先取同 step EMA。"""

    model_path = Path(model_path)
    if use_ema:
        ema_path = model_path.with_name(model_path.name.replace("model", "ema", 1))
        if ema_path.exists():
            model.load_state_dict(
                torch.load(ema_path, map_location="cpu", weights_only=True)
            )
            model.to(device).eval()
            return model, "ema"

    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.to(device)
    model.eval()
    return model, "model"
