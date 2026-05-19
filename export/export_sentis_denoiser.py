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


from utils.model_util import create_model_and_diffusion  # noqa: E402
from write_unity_runtime_assets import (  # noqa: E402
    default_unity_model_dir,
    expected_x277_model_input_dim,
    write_runtime_assets,
)


DEFAULT_MODEL_CONFIG = {
    "input_feats": expected_x277_model_input_dim(),
    "seq_len": 150,
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
}


class SentisDenoiserWrapper(nn.Module):
    """Fixed ONNX/Sentis contract wrapper for the realtime denoiser."""

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
    parser = argparse.ArgumentParser(description="Export a DiffusionPoser denoiser for Unity Sentis.")
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
    parser.add_argument("--allow_feature_dim_mismatch", action="store_true")
    parser.add_argument(
        "--skip_runtime_assets",
        action="store_true",
        help="Export only diffusionposer_denoiser.onnx. Use this for checkpoints whose feature schema is not RealtimePose current277 v1.",
    )

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


def build_model_config(cli_args: argparse.Namespace) -> SimpleNamespace:
    config = dict(DEFAULT_MODEL_CONFIG)
    config.update({key: value for key, value in load_checkpoint_args(Path(cli_args.model_path)).items() if key in config})
    for key in DEFAULT_MODEL_CONFIG:
        value = getattr(cli_args, key)
        if value is not None:
            config[key] = value

    config["predict_xstart"] = int(config["predict_xstart"])
    if not bool(config["predict_xstart"]):
        raise ValueError("Unity Sentis export requires predict_xstart=true. Epsilon-prediction checkpoints are rejected.")

    expected_dim = expected_x277_model_input_dim()
    if int(config["input_feats"]) != expected_dim and not (
        cli_args.allow_feature_dim_mismatch or cli_args.skip_runtime_assets
    ):
        raise ValueError(
            f"Checkpoint input_feats={config['input_feats']} but RealtimePose X277 schema expects {expected_dim}. "
            "Pass --allow_feature_dim_mismatch only if you also update feature_schema.json explicitly."
        )

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
        raise RuntimeError(f"Checkpoint does not match model. missing={missing}, unexpected={unexpected}")
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


def export_onnx(
    wrapper: nn.Module,
    onnx_path: Path,
    feature_dim: int,
    sequence_length: int,
    opset: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    wrapper.eval()
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)

    x_t = torch.randn(1, feature_dim, sequence_length, dtype=torch.float32, device=device)
    timestep = torch.tensor([0.0], dtype=torch.float32, device=device)
    inpaint_mask = torch.ones(1, feature_dim, sequence_length, dtype=torch.float32, device=device)
    valid_frame_mask = torch.ones(1, sequence_length, dtype=torch.float32, device=device)

    with torch.no_grad():
        reference = wrapper(x_t, timestep, inpaint_mask, valid_frame_mask).detach().cpu()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (x_t, timestep, inpaint_mask, valid_frame_mask),
        str(onnx_path),
        input_names=["x_t", "timestep", "inpaint_mask", "valid_frame_mask"],
        output_names=["pred_x0"],
        opset_version=opset,
        do_constant_folding=True,
    )
    return reference, (x_t.detach().cpu(), timestep.detach().cpu(), inpaint_mask.detach().cpu(), valid_frame_mask.detach().cpu())


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
    device = torch.device(cli_args.device)
    model_config = build_model_config(cli_args)

    model, _diffusion = create_model_and_diffusion(model_config)
    export_model, model_source = load_export_model(model, model_path, device=device, use_ema=cli_args.use_ema)
    wrapper = SentisDenoiserWrapper(export_model).to(device).eval()

    onnx_path = output_dir / "diffusionposer_denoiser.onnx"
    staging_onnx_path = output_dir / ".diffusionposer_denoiser.onnx.tmp"
    reference, onnx_inputs = export_onnx(
        wrapper=wrapper,
        onnx_path=staging_onnx_path,
        feature_dim=int(model_config.input_feats),
        sequence_length=int(model_config.seq_len),
        opset=int(cli_args.opset),
        device=device,
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

    if cli_args.skip_runtime_assets:
        runtime_assets = {}
    else:
        runtime_assets = write_runtime_assets(
            output_dir=output_dir,
            feature_dim=int(model_config.input_feats),
            sequence_length=int(model_config.seq_len),
            diffusion_steps=int(model_config.diffusion_steps),
            noise_schedule=str(model_config.noise_schedule),
            predict_xstart=True,
            normalizer_dir=normalizer_dir,
            normalize_input=bool(cli_args.normalize_input),
            strict_normalizer=bool(cli_args.strict_normalizer),
            schema="current277",
        )

    result = {
        "onnx_path": onnx_path,
        "runtime_assets": runtime_assets,
        "model_source": model_source,
        "max_abs_error": max_abs_error,
        "feature_dim": int(model_config.input_feats),
        "sequence_length": int(model_config.seq_len),
    }
    print(f"[export_sentis_denoiser] ONNX: {onnx_path}")
    print(f"[export_sentis_denoiser] weights: {model_source}")
    print(f"[export_sentis_denoiser] feature_dim={result['feature_dim']} seq_len={result['sequence_length']}")
    if max_abs_error is not None:
        print(f"[export_sentis_denoiser] PyTorch/ONNXRuntime max_abs_error={max_abs_error:.6g}")
    for name, path in runtime_assets.items():
        print(f"[export_sentis_denoiser] {name}: {path}")
    return result


if __name__ == "__main__":
    main()
