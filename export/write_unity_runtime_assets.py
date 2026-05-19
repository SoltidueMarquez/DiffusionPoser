from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


DIFFUSIONPOSER_ROOT = Path(__file__).resolve().parents[1]
if str(DIFFUSIONPOSER_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSIONPOSER_ROOT))


SMPL24_BONE_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]


TRACKER_BINDINGS = [
    {"sensor": 0, "boneIndex": 15, "observePosition": True, "observeRotation": True, "observeVelocity": True},
    {"sensor": 1, "boneIndex": 20, "observePosition": True, "observeRotation": True, "observeVelocity": True},
    {"sensor": 2, "boneIndex": 21, "observePosition": True, "observeRotation": True, "observeVelocity": True},
    {"sensor": 3, "boneIndex": 0, "observePosition": True, "observeRotation": True, "observeVelocity": True},
    {"sensor": 4, "boneIndex": 10, "observePosition": True, "observeRotation": True, "observeVelocity": True},
    {"sensor": 5, "boneIndex": 11, "observePosition": True, "observeRotation": True, "observeVelocity": True},
]


X277_FEATURE_DIM = 277
SENSOR_LABEL_DIM = 6
X277_MODEL_INPUT_DIM = X277_FEATURE_DIM + SENSOR_LABEL_DIM


def default_unity_model_dir() -> Path:
    main_project = DIFFUSIONPOSER_ROOT.parent
    return (
        main_project
        / "SIGGRAPH2024Unity"
        / "Assets"
        / "Projects"
        / "RealtimePose"
        / "Models"
        / "DiffusionPoser"
    )


def expected_full_body_feature_dim(bone_count: int = 24) -> int:
    return 3 + bone_count * 3 + bone_count * 6 + bone_count * 3 + 4


def expected_x277_model_input_dim() -> int:
    return X277_MODEL_INPUT_DIM


def build_feature_schema(
    feature_dim: int,
    sequence_length: int,
    bone_names: list[str] | None = None,
    schema: str = "current277",
) -> dict[str, Any]:
    if schema in {"current277", "x277"}:
        return build_current277_feature_schema(
            feature_dim=feature_dim,
            sequence_length=sequence_length,
            bone_names=bone_names,
        )
    if schema == "full_body":
        return build_full_body_feature_schema(feature_dim=feature_dim, sequence_length=sequence_length, bone_names=bone_names)
    raise ValueError(f"Unsupported schema: {schema}")


def build_current277_feature_schema(
    feature_dim: int,
    sequence_length: int,
    bone_names: list[str] | None = None,
) -> dict[str, Any]:
    bone_names = list(bone_names or SMPL24_BONE_NAMES)
    bone_count = len(bone_names)
    if feature_dim < X277_MODEL_INPUT_DIM:
        raise ValueError(f"feature_dim={feature_dim} is smaller than X277 model input dim={X277_MODEL_INPUT_DIM}")

    return {
        "schemaVersion": 1,
        "schemaName": "current277_v1",
        "featureDim": int(feature_dim),
        "sequenceLength": int(sequence_length),
        "boneCount": int(bone_count),
        "trackerCount": 6,
        "boneNames": bone_names,
        "bodyRotation6dRootNow": {"name": "body_rot_root_now", "start": 0, "length": bone_count * 6},
        "bodyVelocityRootNow": {"name": "body_vel_root_now", "start": 144, "length": bone_count * 3},
        "trackerPositionRootNow": {"name": "tracker_pos_root_now", "start": 216, "length": 18},
        "trackerRotation6dRootNow": {"name": "tracker_rot_root_fwd_up_now", "start": 234, "length": 36},
        "waistDeltaXz": {"name": "waist_delta_xz", "start": 270, "length": 2},
        "waistYawDeltaDegree": {"name": "waist_yaw_delta_degree", "start": 272, "length": 1},
        "contacts": {"name": "contact_cur", "start": 273, "length": 4},
        "sensorMissingLabels": {"name": "sensor_missing_labels", "start": 277, "length": 6},
        "trackerBindings": TRACKER_BINDINGS,
    }


def build_full_body_feature_schema(feature_dim: int, sequence_length: int, bone_names: list[str] | None = None) -> dict[str, Any]:
    bone_names = list(bone_names or SMPL24_BONE_NAMES)
    bone_count = len(bone_names)
    expected_dim = expected_full_body_feature_dim(bone_count)
    if feature_dim < expected_dim:
        raise ValueError(f"feature_dim={feature_dim} is smaller than full-body schema dim={expected_dim}")

    root_start = 0
    pos_start = root_start + 3
    rot_start = pos_start + bone_count * 3
    vel_start = rot_start + bone_count * 6
    contact_start = vel_start + bone_count * 3

    return {
        "schemaVersion": 1,
        "schemaName": "full_body_root_local_v1",
        "featureDim": int(feature_dim),
        "sequenceLength": int(sequence_length),
        "boneCount": int(bone_count),
        "boneNames": bone_names,
        "rootDeltaXzYaw": {"name": "root_delta_xz_yaw", "start": root_start, "length": 3},
        "bonePositionRoot": {"name": "bone_position_root", "start": pos_start, "length": bone_count * 3},
        "boneRotation6dRoot": {"name": "bone_rotation_6d_root", "start": rot_start, "length": bone_count * 6},
        "boneVelocityRoot": {"name": "bone_velocity_root", "start": vel_start, "length": bone_count * 3},
        "contacts": {"name": "contacts", "start": contact_start, "length": 4},
        "trackerBindings": TRACKER_BINDINGS,
    }


def build_ddim_schedule(diffusion_steps: int, noise_schedule: str, predict_xstart: bool = True) -> dict[str, Any]:
    if not predict_xstart:
        raise ValueError("Unity RealtimePose DDIM sampler requires predict_xstart=True.")

    from diffusion import gaussian_diffusion as gd

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps, scale_betas=1.0)
    alphas = 1.0 - np.asarray(betas, dtype=np.float64)
    alphas_cumprod = np.cumprod(alphas).astype(np.float32)
    return {
        "schemaVersion": 1,
        "trainStepCount": int(diffusion_steps),
        "noiseSchedule": str(noise_schedule),
        "predictXStart": True,
        "alphasCumprod": [float(value) for value in alphas_cumprod.tolist()],
    }


def build_normalizer(
    feature_dim: int,
    normalizer_dir: Path | None,
    normalize_input: bool,
    strict: bool,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    if not normalize_input:
        return disabled_normalizer(feature_dim, epsilon)

    if normalizer_dir is None:
        if strict:
            raise FileNotFoundError("normalizer_dir is required when strict normalizer export is enabled.")
        return disabled_normalizer(feature_dim, epsilon)

    mean_path = normalizer_dir / "mean.pt"
    std_path = normalizer_dir / "std.pt"
    if not mean_path.exists() or not std_path.exists():
        if strict:
            raise FileNotFoundError(f"Missing normalizer tensors: {mean_path}, {std_path}")
        return disabled_normalizer(feature_dim, epsilon)

    import torch

    mean = torch_load(mean_path).float().flatten().cpu().numpy()
    std = torch_load(std_path).float().flatten().cpu().numpy()
    if mean.size == X277_FEATURE_DIM and std.size == X277_FEATURE_DIM and feature_dim == X277_MODEL_INPUT_DIM:
        # 训练 dataloader 在 normalize_input=True 时把缺失标签从 0/1 映射到 -1/+1。
        # Runtime 仍然更自然地输入 0/1，因此 normalizer 用 0.5/0.5 完成同一映射。
        label_mean = np.full(SENSOR_LABEL_DIM, 0.5, dtype=mean.dtype)
        label_std = np.full(SENSOR_LABEL_DIM, 0.5, dtype=std.dtype)
        mean = np.concatenate([mean, label_mean], axis=0)
        std = np.concatenate([std, label_std], axis=0)

    if mean.size != feature_dim or std.size != feature_dim:
        message = (
            f"Normalizer dim mismatch: mean={mean.size}, std={std.size}, "
            f"expected feature_dim={feature_dim}."
        )
        if strict:
            raise ValueError(message)
        print(f"[write_unity_runtime_assets] WARNING: {message} Writing disabled normalizer.")
        return disabled_normalizer(feature_dim, epsilon)

    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Normalizer mean/std contain NaN, Inf, or non-positive std values.")

    return {
        "enabled": True,
        "featureDim": int(feature_dim),
        "epsilon": float(epsilon),
        "mean": [float(value) for value in mean.astype(np.float32).tolist()],
        "std": [float(max(value, epsilon)) for value in std.astype(np.float32).tolist()],
    }


def disabled_normalizer(feature_dim: int, epsilon: float = 1e-8) -> dict[str, Any]:
    return {
        "enabled": False,
        "featureDim": int(feature_dim),
        "epsilon": float(epsilon),
        "mean": [0.0] * int(feature_dim),
        "std": [1.0] * int(feature_dim),
    }


def torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_runtime_assets(
    output_dir: Path,
    feature_dim: int,
    sequence_length: int,
    diffusion_steps: int,
    noise_schedule: str,
    predict_xstart: bool,
    normalizer_dir: Path | None,
    normalize_input: bool,
    strict_normalizer: bool,
    schema: str = "current277",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "feature_schema": output_dir / "feature_schema.json",
        "normalizer": output_dir / "normalizer.json",
        "ddim_schedule": output_dir / "ddim_schedule.json",
    }

    write_json(assets["feature_schema"], build_feature_schema(feature_dim, sequence_length, schema=schema))
    write_json(
        assets["normalizer"],
        build_normalizer(
            feature_dim=feature_dim,
            normalizer_dir=normalizer_dir,
            normalize_input=normalize_input,
            strict=strict_normalizer,
        ),
    )
    write_json(
        assets["ddim_schedule"],
        build_ddim_schedule(
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
            predict_xstart=predict_xstart,
        ),
    )
    return assets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write Unity RealtimePose runtime JSON assets.")
    parser.add_argument("--output_dir", default=str(default_unity_model_dir()), type=str)
    parser.add_argument("--schema", default="current277", choices=["current277", "x277", "full_body"])
    parser.add_argument("--feature_dim", default=expected_x277_model_input_dim(), type=int)
    parser.add_argument("--seq_len", default=150, type=int)
    parser.add_argument("--diffusion_steps", default=50, type=int)
    parser.add_argument("--noise_schedule", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--normalizer_dir", default="", type=str)
    parser.add_argument("--normalize_input", action="store_true")
    parser.add_argument("--strict_normalizer", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_arg_parser().parse_args(argv)
    normalizer_dir = Path(args.normalizer_dir).resolve() if args.normalizer_dir else None
    assets = write_runtime_assets(
        output_dir=Path(args.output_dir).resolve(),
        feature_dim=args.feature_dim,
        sequence_length=args.seq_len,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        predict_xstart=True,
        normalizer_dir=normalizer_dir,
        normalize_input=args.normalize_input,
        strict_normalizer=args.strict_normalizer,
        schema=args.schema,
    )
    for name, path in assets.items():
        print(f"[write_unity_runtime_assets] {name}: {path}")
    return assets


if __name__ == "__main__":
    main()
