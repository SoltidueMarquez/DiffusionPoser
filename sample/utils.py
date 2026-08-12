from __future__ import annotations

import json
from pathlib import Path

import torch
from ema_pytorch import EMA


def choose_sampler(diffusion, use_ddim: bool):
    """根据配置选择 DDIM 或标准扩散采样器。"""

    return diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop


def load_checkpoint_model(model, model_path: str | Path, device, use_ema: bool = True):
    """加载 checkpoint；存在 EMA 权重时优先用于推理。"""

    model_path = Path(model_path)
    if use_ema:
        ema_path = model_path.with_name(model_path.name.replace("model", "ema", 1))
        if ema_path.exists():
            ema = EMA(model, include_online_model=False)
            ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
            inference_model = ema.ema_model
            inference_model.to(device)
            inference_model.eval()
            return inference_model, "ema"

    state_dict = torch.load(model_path, map_location="cpu")
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible_keys.missing_keys)
    unexpected_keys = list(incompatible_keys.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "checkpoint 与当前模型结构不匹配，已停止测试以避免生成不可信结果。"
            f" missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    model.to(device)
    model.eval()
    return model, "model"


def load_json(path: Path) -> dict:
    """以固定 UTF-8 编码读取 JSON。"""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
