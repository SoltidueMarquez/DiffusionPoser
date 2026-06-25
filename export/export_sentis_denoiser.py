from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn


DIFFUSIONPOSER_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = Path(__file__).resolve().parent
if str(DIFFUSIONPOSER_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSIONPOSER_ROOT))
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))


from data_loaders.sensor_masking import (  # noqa: E402
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    get_schema_spec,
)
from schemas.base import SchemaSpec  # noqa: E402
from utils.model_util import create_model_and_diffusion  # noqa: E402
from write_unity_runtime_assets import default_unity_model_dir, validate_normalizer_metadata, write_runtime_assets  # noqa: E402


DEFAULT_MODEL_CONFIG = {
    "schema": DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    "input_feats": get_schema_spec(DEFAULT_REALTIME_POSE_SCHEMA_NAME).feature_dim,
    "seq_len": REALTIME_POSE_SEQ_LEN,
    "max_seq_len": REALTIME_POSE_SEQ_LEN,
    "layers": 8,
    "heads": 8,
    "latent_dim": 512,
    "dropout": 0.0,
    "zero_init": False,
    "noise_schedule": "cosine",
    "diffusion_steps": 50,
    "sigma_small": True,
    "predict_xstart": 1,
    "ts_respace": "",
    "model_arch": "full_feature_dit",
}


class SentisDenoiserWrapper(nn.Module):
    """Unity Sentis 固定调用合约：输入 `[1,C,61]`，输出 `pred_x0`。"""

    def __init__(self, denoiser: nn.Module):
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        inpaint_mask: torch.Tensor,
        valid_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(
            hidden_states=x_t,
            timestep=timestep,
            inpaint_cond=inpaint_mask,
            valid_frame_mask=valid_frame_mask,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a realtime_pose DiffusionPoser denoiser for Unity Sentis.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to model*.pt checkpoint.")
    parser.add_argument("--output_dir", default=str(default_unity_model_dir()), type=str)
    parser.add_argument("--normalizer_dir", default="", type=str)
    parser.add_argument("--normalize_input", default=True, type=str2bool)
    parser.add_argument("--strict_normalizer", action="store_true")
    parser.add_argument("--use_ema", default=True, type=str2bool)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--opset", default=17, type=int)
    parser.add_argument("--onnx_check_tolerance", default=2e-4, type=float)
    parser.add_argument("--strict_onnx_check", action="store_true")

    for key, value in DEFAULT_MODEL_CONFIG.items():
        arg_type = type(value)
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", default=None, type=str2bool)
        else:
            parser.add_argument(f"--{key}", default=None, type=arg_type)
    return parser


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


def load_checkpoint_args(model_path: Path) -> dict[str, Any]:
    args_path = model_path.with_name("args.json")
    if not args_path.exists():
        return {}
    with args_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def checkpoint_exact_schema_name(checkpoint_args: dict[str, Any]) -> str | None:
    names: dict[str, str] = {}
    for key in ("schema", "schema_name"):
        value = checkpoint_args.get(key)
        if value is None or str(value).strip() == "":
            continue
        names[key] = str(value)
    unique_names = set(names.values())
    if len(unique_names) > 1:
        raise ValueError(f"checkpoint args.json schema 元数据不一致: {names}")
    return next(iter(unique_names), None)


def resolve_export_schema(cli_schema: str | None, checkpoint_args: dict[str, Any]) -> SchemaSpec:
    checkpoint_schema = checkpoint_exact_schema_name(checkpoint_args)
    requested_schema = None if cli_schema is None or str(cli_schema).strip() == "" else str(cli_schema)
    if checkpoint_schema is not None:
        if requested_schema is not None and requested_schema != checkpoint_schema:
            raise ValueError(
                "Sentis export 要求 CLI --schema 与 checkpoint exact schema 完全一致："
                f"checkpoint schema={checkpoint_schema!r}, CLI --schema={requested_schema!r}。"
            )
        return get_schema_spec(checkpoint_schema)
    return get_schema_spec(requested_schema or DEFAULT_REALTIME_POSE_SCHEMA_NAME)


def validate_normalizer_export_contract(cli_args: argparse.Namespace, checkpoint_args: dict[str, Any]) -> None:
    schema = resolve_export_schema(getattr(cli_args, "schema", None), checkpoint_args)
    checkpoint_normalize_input = bool(checkpoint_args.get("normalize_input", True))
    export_normalize_input = bool(cli_args.normalize_input)
    normalizer_dir = Path(cli_args.normalizer_dir).resolve() if cli_args.normalizer_dir else None

    if checkpoint_normalize_input and not export_normalize_input:
        raise ValueError(
            "checkpoint args.json 显示 normalize_input=True，Sentis 导出不能关闭 normalize_input。"
        )
    if not export_normalize_input:
        return
    if normalizer_dir is None:
        raise FileNotFoundError(
            "导出 normalized checkpoint 时必须提供 --normalizer_dir，避免 Unity 端把 raw 特征喂给模型。"
        )
    missing = [path for path in (normalizer_dir / "mean.pt", normalizer_dir / "std.pt") if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "导出 normalized checkpoint 时缺少 normalizer 文件："
            + ", ".join(str(path) for path in missing)
        )
    validate_normalizer_metadata(normalizer_dir=normalizer_dir, schema_name=schema.name)


def build_model_config(
    cli_args: argparse.Namespace,
    checkpoint_args: dict[str, Any] | None = None,
) -> SimpleNamespace:
    config = dict(DEFAULT_MODEL_CONFIG)
    if checkpoint_args is None:
        checkpoint_args = load_checkpoint_args(Path(cli_args.model_path))
    schema = resolve_export_schema(getattr(cli_args, "schema", None), checkpoint_args)
    config.update({key: value for key, value in checkpoint_args.items() if key in config and key != "schema"})
    config["schema"] = schema.name
    for key in DEFAULT_MODEL_CONFIG:
        if key == "schema":
            continue
        value = getattr(cli_args, key)
        if value is not None:
            config[key] = value

    schema = get_schema_spec(config.get("schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME))
    if "input_feats" not in checkpoint_args and getattr(cli_args, "input_feats") is None:
        config["input_feats"] = schema.feature_dim
    if int(config["input_feats"]) != schema.feature_dim:
        raise ValueError(
            f"{schema.name} 只支持 input_feats={schema.feature_dim}，"
            f"checkpoint/CLI 给出 {config['input_feats']}。旧 X277/current277 checkpoint 不能导出。"
        )
    if int(config["seq_len"]) != REALTIME_POSE_SEQ_LEN or int(config["max_seq_len"]) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(
            f"{schema.name} 只支持 seq_len=max_seq_len={REALTIME_POSE_SEQ_LEN}，"
            f"实际 seq_len={config['seq_len']}, max_seq_len={config['max_seq_len']}。"
        )
    config["predict_xstart"] = int(config["predict_xstart"])
    if not bool(config["predict_xstart"]):
        raise ValueError("Unity Sentis export requires predict_xstart=true. Epsilon-prediction checkpoints are rejected.")
    config["schema_canonical_name"] = schema.canonical_name
    return SimpleNamespace(**config)


def load_export_model(model: nn.Module, model_path: Path, device: torch.device, use_ema: bool) -> tuple[nn.Module, str]:
    if use_ema:
        ema_path = model_path.with_name(model_path.name.replace("model", "ema", 1))
        if ema_path.exists():
            try:
                from ema_pytorch import EMA
            except ImportError as exc:
                raise ImportError("EMA checkpoint exists but ema_pytorch is not installed.") from exc

            ema = EMA(model, include_online_model=False)
            ema.load_state_dict(torch_load(ema_path))
            export_model = getattr(ema, "ema_model", ema)
            export_model.to(device)
            export_model.eval()
            return export_model, "ema"

    state_dict = unwrap_state_dict(torch_load(model_path))
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match realtime_pose model. missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model, "model"


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload is not a state_dict.")
    state_dict = dict(payload)
    if state_dict and all(str(key).startswith("module.") for key in state_dict):
        state_dict = {str(key)[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def build_dummy_inputs(
    feature_dim: int,
    sequence_length: int,
    device: torch.device,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    schema = get_schema_spec(schema_name)
    x_t = torch.randn(1, feature_dim, sequence_length, dtype=torch.float32, device=device)
    timestep = torch.tensor([0.0], dtype=torch.float32, device=device)
    inpaint_mask = torch.zeros(1, feature_dim, sequence_length, dtype=torch.float32, device=device)
    inpaint_mask[:, schema.target_slice(), REALTIME_POSE_TARGET_START] = 1.0
    valid_frame_mask = torch.ones(1, sequence_length, dtype=torch.float32, device=device)
    return x_t, timestep, inpaint_mask, valid_frame_mask


def export_onnx(
    wrapper: nn.Module,
    onnx_path: Path,
    feature_dim: int,
    sequence_length: int,
    opset: int,
    device: torch.device,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    wrapper.eval()
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)

    dummy_inputs = build_dummy_inputs(feature_dim, sequence_length, device, schema_name=schema_name)
    with torch.no_grad():
        reference = wrapper(*dummy_inputs).detach().cpu()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(onnx_path),
        input_names=["x_t", "timestep", "inpaint_mask", "valid_frame_mask"],
        output_names=["pred_x0"],
        opset_version=opset,
        do_constant_folding=True,
    )
    return reference, tuple(value.detach().cpu() for value in dummy_inputs)


def check_onnx_alignment(
    onnx_path: Path,
    reference: torch.Tensor,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tolerance: float,
    strict: bool,
) -> float | None:
    try:
        import onnxruntime as ort
    except ImportError:
        message = "onnxruntime is not installed; skipped PyTorch vs ONNXRuntime numeric check."
        if strict:
            raise ImportError(message)
        print(f"[export_sentis_denoiser] WARNING: {message}")
        return None

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_t, timestep, inpaint_mask, valid_frame_mask = inputs
    outputs = session.run(
        ["pred_x0"],
        {
            "x_t": x_t.numpy(),
            "timestep": timestep.numpy(),
            "inpaint_mask": inpaint_mask.numpy(),
            "valid_frame_mask": valid_frame_mask.numpy(),
        },
    )
    onnx_output = np.asarray(outputs[0], dtype=np.float32)
    reference_np = reference.numpy().astype(np.float32, copy=False)
    max_abs_error = float(np.max(np.abs(reference_np - onnx_output)))
    if max_abs_error > tolerance:
        message = f"PyTorch vs ONNXRuntime max_abs_error={max_abs_error:.6g} exceeds tolerance={tolerance:.6g}."
        if strict:
            raise RuntimeError(message)
        print(f"[export_sentis_denoiser] WARNING: {message}")
    return max_abs_error


def main(argv: list[str] | None = None) -> dict[str, Any]:
    cli_args = build_arg_parser().parse_args(argv)
    model_path = Path(cli_args.model_path).resolve()
    output_dir = Path(cli_args.output_dir).resolve()
    normalizer_dir = Path(cli_args.normalizer_dir).resolve() if cli_args.normalizer_dir else None
    checkpoint_args = load_checkpoint_args(model_path)
    validate_normalizer_export_contract(cli_args=cli_args, checkpoint_args=checkpoint_args)
    device = torch.device(cli_args.device)
    model_config = build_model_config(cli_args, checkpoint_args=checkpoint_args)
    schema = get_schema_spec(model_config.schema)

    model, _diffusion = create_model_and_diffusion(model_config)
    export_model, model_source = load_export_model(model, model_path, device=device, use_ema=cli_args.use_ema)
    wrapper = SentisDenoiserWrapper(export_model).to(device).eval()

    onnx_path = output_dir / "diffusionposer_denoiser.onnx"
    staging_onnx_path = output_dir / ".diffusionposer_denoiser.onnx.tmp"
    reference, onnx_inputs = export_onnx(
        wrapper=wrapper,
        onnx_path=staging_onnx_path,
        feature_dim=schema.feature_dim,
        sequence_length=REALTIME_POSE_SEQ_LEN,
        opset=int(cli_args.opset),
        device=device,
        schema_name=schema.name,
    )
    try:
        max_abs_error = check_onnx_alignment(
            onnx_path=staging_onnx_path,
            reference=reference,
            inputs=onnx_inputs,
            tolerance=float(cli_args.onnx_check_tolerance),
            strict=bool(cli_args.strict_onnx_check),
        )
    except Exception:
        if staging_onnx_path.exists():
            staging_onnx_path.unlink()
        raise
    staging_onnx_path.replace(onnx_path)

    runtime_assets = write_runtime_assets(
        output_dir=output_dir,
        feature_dim=schema.feature_dim,
        sequence_length=REALTIME_POSE_SEQ_LEN,
        diffusion_steps=int(model_config.diffusion_steps),
        noise_schedule=str(model_config.noise_schedule),
        predict_xstart=True,
        normalizer_dir=normalizer_dir,
        normalize_input=bool(cli_args.normalize_input),
        strict_normalizer=bool(cli_args.strict_normalizer or cli_args.normalize_input),
        schema_name=schema.name,
    )

    result = {
        "onnx_path": onnx_path,
        "runtime_assets": runtime_assets,
        "model_source": model_source,
        "max_abs_error": max_abs_error,
        "schema_name": schema.name,
        "schema_canonical_name": schema.canonical_name,
        "feature_dim": schema.feature_dim,
        "sequence_length": REALTIME_POSE_SEQ_LEN,
    }
    print(f"[export_sentis_denoiser] ONNX: {onnx_path}")
    print(f"[export_sentis_denoiser] weights: {model_source}")
    print(f"[export_sentis_denoiser] schema={schema.name} feature_dim={schema.feature_dim} seq_len={REALTIME_POSE_SEQ_LEN}")
    if max_abs_error is not None:
        print(f"[export_sentis_denoiser] PyTorch/ONNXRuntime max_abs_error={max_abs_error:.6g}")
    for name, path in runtime_assets.items():
        print(f"[export_sentis_denoiser] {name}: {path}")
    return result


if __name__ == "__main__":
    main()
