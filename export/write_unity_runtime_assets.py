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


from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES, TRACKER_JOINT_INDICES  # noqa: E402
from data_loaders.sensor_masking import (  # noqa: E402
    BODY_POSE_DIM,
    BODY_POSE_START,
    HIP_TRACKER_INDEX,
    MIN_VALID_TRACKERS,
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_LENGTH,
    REALTIME_POSE_TARGET_START,
    ROOT_YAW_DELTA_DIM,
    ROOT_YAW_DELTA_START,
    SENSOR_VALID_DIM,
    SENSOR_VALID_START,
    TRACKER_COUNT,
    TRACKER_NAMES,
    TRACKER_POS_DIM,
    TRACKER_POS_REF_START,
    TRACKER_ROT_DIM,
    TRACKER_ROT_REF_START,
)
from utils.normalizer import enforce_realtime_pose_normalizer_contract  # noqa: E402


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


def expected_realtime_pose_input_dim() -> int:
    return REALTIME_POSE_INPUT_DIM


def build_realtime_pose_feature_schema(
    feature_dim: int = REALTIME_POSE_INPUT_DIM,
    sequence_length: int = REALTIME_POSE_SEQ_LEN,
) -> dict[str, Any]:
    if int(feature_dim) != REALTIME_POSE_INPUT_DIM:
        raise ValueError(f"realtime_pose_v1 featureDim 必须为 {REALTIME_POSE_INPUT_DIM}，实际为 {feature_dim}")
    if int(sequence_length) != REALTIME_POSE_SEQ_LEN:
        raise ValueError(f"realtime_pose_v1 sequenceLength 必须为 {REALTIME_POSE_SEQ_LEN}，实际为 {sequence_length}")

    return {
        "schemaVersion": 1,
        "schemaName": REALTIME_POSE_SCHEMA_NAME,
        "featureDim": REALTIME_POSE_INPUT_DIM,
        "sequenceLength": REALTIME_POSE_SEQ_LEN,
        "targetStart": REALTIME_POSE_TARGET_START,
        "targetLength": REALTIME_POSE_TARGET_LENGTH,
        "targetFeatureLength": REALTIME_POSE_TARGET_DIM,
        "boneCount": len(SMPL_JOINT_NAMES),
        "boneNames": list(SMPL_JOINT_NAMES),
        "trackerCount": TRACKER_COUNT,
        "trackerNames": list(TRACKER_NAMES),
        "trackerJointIndices": [int(value) for value in TRACKER_JOINT_INDICES.tolist()],
        "hipTrackerIndex": HIP_TRACKER_INDEX,
        "minValidTrackers": MIN_VALID_TRACKERS,
        "bodyPoseParent6d": {"name": "body_pose_parent_6d", "start": BODY_POSE_START, "length": BODY_POSE_DIM},
        "rootYawDeltaSinCos": {"name": "root_yaw_delta_sincos", "start": ROOT_YAW_DELTA_START, "length": ROOT_YAW_DELTA_DIM},
        "trackerPositionReference": {"name": "tracker_pos_ref", "start": TRACKER_POS_REF_START, "length": TRACKER_POS_DIM},
        "trackerRotation6dReference": {"name": "tracker_rot_ref_6d", "start": TRACKER_ROT_REF_START, "length": TRACKER_ROT_DIM},
        "sensorValid": {"name": "sensor_valid", "start": SENSOR_VALID_START, "length": SENSOR_VALID_DIM},
        "runtimeRules": {
            "requiresHipTracker": True,
            "requiresTotalValidTrackersAtLeast": MIN_VALID_TRACKERS,
            "trackerReferenceYaw": "previous_frame_root_yaw",
            "onnxDummyInputShape": [1, REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN],
            "failSafe": "hold_previous_frame_when_tracker_validity_fails",
        },
    }


def build_ddim_schedule(diffusion_steps: int, noise_schedule: str, predict_xstart: bool = True) -> dict[str, Any]:
    if not predict_xstart:
        raise ValueError("Unity realtime_pose_v1 DDIM sampler requires predict_xstart=True.")

    from diffusion import gaussian_diffusion as gd

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps, scale_betas=1.0)
    alphas = 1.0 - np.asarray(betas, dtype=np.float64)
    alphas_cumprod = np.cumprod(alphas).astype(np.float32)
    return {
        "schemaVersion": 1,
        "schemaName": REALTIME_POSE_SCHEMA_NAME,
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
    if int(feature_dim) != REALTIME_POSE_INPUT_DIM:
        raise ValueError(f"realtime_pose_v1 normalizer feature_dim 必须为 {REALTIME_POSE_INPUT_DIM}。")
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

    mean_tensor, std_tensor = enforce_realtime_pose_normalizer_contract(
        torch_load(mean_path).float().flatten(),
        torch_load(std_path).float().flatten(),
    )
    mean = mean_tensor.cpu().numpy()
    std = std_tensor.cpu().numpy()
    if mean.size != feature_dim or std.size != feature_dim:
        message = f"Normalizer dim mismatch: mean={mean.size}, std={std.size}, expected={feature_dim}."
        if strict:
            raise ValueError(message)
        print(f"[write_unity_runtime_assets] WARNING: {message} Writing disabled normalizer.")
        return disabled_normalizer(feature_dim, epsilon)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Normalizer mean/std contain NaN, Inf, or non-positive std values.")

    return {
        "enabled": True,
        "schemaName": REALTIME_POSE_SCHEMA_NAME,
        "featureDim": int(feature_dim),
        "epsilon": float(epsilon),
        "mean": [float(value) for value in mean.astype(np.float32).tolist()],
        "std": [float(max(value, epsilon)) for value in std.astype(np.float32).tolist()],
    }


def disabled_normalizer(feature_dim: int, epsilon: float = 1e-8) -> dict[str, Any]:
    return {
        "enabled": False,
        "schemaName": REALTIME_POSE_SCHEMA_NAME,
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
    feature_dim: int = REALTIME_POSE_INPUT_DIM,
    sequence_length: int = REALTIME_POSE_SEQ_LEN,
    diffusion_steps: int = 50,
    noise_schedule: str = "cosine",
    predict_xstart: bool = True,
    normalizer_dir: Path | None = None,
    normalize_input: bool = False,
    strict_normalizer: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "feature_schema": output_dir / "feature_schema.json",
        "normalizer": output_dir / "normalizer.json",
        "ddim_schedule": output_dir / "ddim_schedule.json",
    }
    write_json(assets["feature_schema"], build_realtime_pose_feature_schema(feature_dim, sequence_length))
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
    parser = argparse.ArgumentParser(description="Write Unity realtime_pose_v1 runtime JSON assets.")
    parser.add_argument("--output_dir", default=str(default_unity_model_dir()), type=str)
    parser.add_argument("--feature_dim", default=REALTIME_POSE_INPUT_DIM, type=int)
    parser.add_argument("--seq_len", default=REALTIME_POSE_SEQ_LEN, type=int)
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
        feature_dim=int(args.feature_dim),
        sequence_length=int(args.seq_len),
        diffusion_steps=int(args.diffusion_steps),
        noise_schedule=str(args.noise_schedule),
        predict_xstart=True,
        normalizer_dir=normalizer_dir,
        normalize_input=bool(args.normalize_input),
        strict_normalizer=bool(args.strict_normalizer),
    )
    for name, path in assets.items():
        print(f"[write_unity_runtime_assets] {name}: {path}")
    return assets


if __name__ == "__main__":
    main()
