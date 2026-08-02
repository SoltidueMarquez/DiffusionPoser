from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from ema_pytorch import EMA


def build_output_dir(args) -> Path:
    """根据 checkpoint、seed 和 folder_path 推导测试输出目录。"""

    output_dir = getattr(args, "output_dir", "")
    if output_dir:
        # 如果命令行已经显式指定输出目录，就尊重调用方，避免默认命名覆盖人工设置。
        return Path(output_dir)

    model_path = Path(args.model_path).resolve()
    model_dir = model_path.parent
    ckpt_name = model_path.stem
    if ckpt_name.startswith("model"):
        ckpt_name = ckpt_name[len("model") :]
    folder_token = sanitize_path_token(getattr(args, "folder_path", ""))

    # 这里的目录命名尽量把“是哪一个 checkpoint、哪个 seed、哪个子目录”说清楚，
    # 这样同一个模型跑多个测试子集时不容易混。
    parts = ["FixTest", model_dir.name, ckpt_name, f"seed{getattr(args, 'seed', 0)}"]
    if folder_token:
        parts.append(folder_token)
    return model_dir / "_".join(parts)


def sanitize_path_token(path_value: str) -> str:
    # 文件夹名可能包含斜杠、空格或中文符号；输出目录里只保留安全字符，避免路径失效。
    token = str(path_value).strip().replace("\\", "/")
    token = re.sub(r"[^A-Za-z0-9._/-]+", "_", token)
    token = token.strip("/").replace("/", "_")
    return token


def choose_sampler(diffusion, use_ddim: bool):
    """根据 `ts_respace` 选择采样器。"""

    return diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop


def load_checkpoint_model(model, model_path: str | Path, device, use_ema: bool = True):
    """
    加载模型权重。

    优先顺序是：
    1. 如果允许使用 EMA 且存在对应的 EMA 文件，就先加载 EMA；
    2. 否则退回到原始 model 权重。

    这样能尽量贴近训练时通常观察到的稳定效果。
    """

    model_path = Path(model_path)
    if use_ema:
        ema_path = model_path.with_name(model_path.name.replace("model", "ema", 1))
        if ema_path.exists():
            ema = EMA(model, include_online_model=False)
            ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
            # 采样链路需要调用 prepare_conditioning 等模型自定义方法，因此统一返回内部推理模型。
            inference_model = ema.ema_model
            inference_model.to(device)
            inference_model.eval()
            return inference_model, "ema"

    state_dict = torch.load(model_path, map_location="cpu")
    # 采样入口最怕“权重没完整加载但程序继续跑”：结果文件会生成，却没有可解释性。
    # 因此这里保留自定义错误信息，而不是静默接受 missing / unexpected keys。
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
    # 小工具函数：读 json 时统一固定编码，避免 Windows 下中文参数乱码。
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
