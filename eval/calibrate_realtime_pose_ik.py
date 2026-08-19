from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_loaders.get_data import get_dataset_loader
from data_loaders.realtime_pose_config import IKInpaintingConfig
from data_loaders.realtime_pose_ik import DIRECTION_ONLY, build_current_ik, build_ik_joint_chain_length
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_torch
from train.training_loop import move_batch_to_device
from utils.model_util import load_realtime_pose_predictor
from utils.normalizer import RealtimePoseNormalizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用冻结 Predictor 初始化结果校准 IK confidence 参数。"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--normalizer_dir", required=True)
    parser.add_argument("--predictor_model_path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", default=20_000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--fabrik_iterations", default=2, type=int)
    return parser


@torch.no_grad()
def calibrate(args) -> dict:
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )
    normalizer = RealtimePoseNormalizer(args.normalizer_dir)
    predictor = load_realtime_pose_predictor(args.predictor_model_path, device)
    loader = get_dataset_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_feats=144,
        seq_len=11,
        split=args.split,
        normalizer_dir=args.normalizer_dir,
        normalize_input=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        seed=args.seed,
    )
    angle_values = []
    residual_ratios = []
    consumed = 0
    config = IKInpaintingConfig(
        fabrik_iterations=args.fabrik_iterations,
        direction_only_quality=0.8,
        residual_scale=1.0,
    )
    mean = normalizer.pose_mean.to(device)
    scale = normalizer.pose_scale.to(device)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        horizon = predictor(batch["motion_context"], batch["core_tracker_context"])
        initial = horizon[:, 0] * scale + mean
        result = build_current_ik(
            initial,
            batch["current_tracker_raw"],
            batch["joint_offsets_parent"],
            config,
        )
        target = batch["x"] * scale + mean
        predicted_rot = rotation_6d_to_matrix_torch(result.pose)
        target_rot = rotation_6d_to_matrix_torch(target.reshape(-1, 24, 6))
        angles = _rotation_angle(predicted_rot, target_rot)
        direction_mask = result.constraint_type == DIRECTION_ONLY
        angle_values.extend(angles[direction_mask].cpu().tolist())
        chain_length = build_ik_joint_chain_length(batch["joint_offsets_parent"])
        ratios = result.position_residual / chain_length.clamp_min(1e-8)
        residual_ratios.extend(ratios[direction_mask].cpu().tolist())
        consumed += int(target.shape[0])
        if consumed >= args.max_samples:
            break
    if not angle_values or not residual_ratios:
        raise RuntimeError("校准样本没有产生 DIRECTION_ONLY IK 约束。")
    median_angle = float(np.median(angle_values))
    direction_quality = float(np.clip(np.exp(-median_angle), 0.05, 0.99))
    residual_scale = float(max(np.median(residual_ratios), 1e-3))
    return {
        "sample_count": min(consumed, int(args.max_samples)),
        "predictor_model_path": str(Path(args.predictor_model_path).resolve()),
        "recommended_parameters": {
            "ik_direction_only_quality": direction_quality,
            "ik_residual_scale": residual_scale,
        },
        "diagnostics": {
            "median_direction_rotation_error_deg": float(np.degrees(median_angle)),
            "median_endpoint_residual_ratio": float(np.median(residual_ratios)),
        },
    }


def _rotation_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    relative = first.transpose(-1, -2) @ second
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def main(argv: list[str] | None = None) -> dict:
    args = build_arg_parser().parse_args(argv)
    report = calibrate(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ik-calibration] wrote {output}")
    return report


if __name__ == "__main__":
    main()
