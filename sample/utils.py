from __future__ import annotations

import json
from pathlib import Path

import torch


def choose_sampler(diffusion, use_ddim: bool):
    """根据配置选择 DDIM 或标准扩散采样器。"""

    return diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop


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


def load_json(path: Path) -> dict:
    """以固定 UTF-8 编码读取 JSON。"""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
