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
            try:
                ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
            except (RuntimeError, KeyError) as exc:
                raise RuntimeError(
                    "checkpoint 模型结构不兼容，无法加载 EMA 权重；"
                    "新增轻量 Tracker 条件编码器后必须使用新训练权重。"
                ) from exc
            inference_model = ema.ema_model
            inference_model.to(device)
            inference_model.eval()
            return inference_model, "ema"

    state_dict = torch.load(model_path, map_location="cpu")
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(incompatible_keys.missing_keys)
    unexpected_keys = list(incompatible_keys.unexpected_keys)
    if missing_keys or unexpected_keys:
        if any(
            key.startswith("current_joint_condition_input.") for key in missing_keys
        ):
            raise RuntimeError(
                "旧 checkpoint 缺少轻量 Tracker 条件编码器；"
                "请使用新结构从头训练得到的权重。"
            )
        if (
            "joint_diffusion_horizon_length" in missing_keys
            or any("future_leg_head" in key for key in unexpected_keys)
        ):
            raise RuntimeError("单帧 checkpoint 与联合 11 帧模型不兼容。")
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
