"""隔离官方 RPM 的 import 路径，批量生成固定掉线协议下的姿态。"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


def generate_sequence(model, args, arrays, device, data_type, output_type):
    """官方滚动推理，输入 `[T,132]` motion 和 `[T,54]` sparse。"""
    motion = torch.as_tensor(arrays["rpm_motion_6d"], dtype=torch.float32)
    mean = torch.as_tensor(arrays["rpm_pose_mean"], dtype=torch.float32)
    std = torch.as_tensor(arrays["rpm_pose_std"], dtype=torch.float32)
    normalized = ((motion - mean) / (std + 1e-8)).to(device)[None]
    sparse = torch.as_tensor(arrays["rpm_sparse_54d"], device=device)[None]
    context = max(args.rolling_motion_ctx, args.rolling_sparse_ctx)
    if len(motion) < context + args.input_motion_length:
        raise ValueError("RPM 序列短于初始化窗口。")
    output = torch.zeros_like(normalized)
    output[:, :context] = normalized[:, :context]
    rolling = normalized[:, context:context + args.input_motion_length].clone()
    with torch.no_grad():
        for current in range(context, len(motion)):
            condition = {
                data_type.MOTION_CTX: output[:, current - args.rolling_motion_ctx:current],
                data_type.SPARSE: sparse[:, current - args.rolling_sparse_ctx:current + 1],
            }
            rolling = model(rolling, condition, x_start=rolling)[output_type.RELATIVE_ROTS]
            output[:, current:current + 1] = rolling[:, :1]
            rolling[:, :-1] = rolling.clone()[:, 1:]
            rolling[:, -1] = 0.0
    return (output[0].cpu() * std + mean).numpy().astype(np.float32)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rpm_repo", "human_body_prior_repo", "model_path", "input_dir", "output_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    cli = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in {project_root, project_root / "sample"}]
    sys.path[:0] = [str(cli.rpm_repo.resolve()), str(cli.human_body_prior_repo.resolve())]
    # 官方与本仓库的 model/utils 同名，必须在路径隔离后再加载。
    from utils.constants import (
        ConditionMasker, DataTypeGT, DatasetType, LossDistType, ModelOutputType,
        PredictionInputType, PredictionTargetType,
    )
    from utils.model_util import load_rpm_model

    payload = json.loads((cli.model_path.parent / "args.json").read_text())
    for key, enum in {
        "dataset": DatasetType, "masker": ConditionMasker, "loss_dist_type": LossDistType,
        "target_type": PredictionTargetType, "prediction_input_type": PredictionInputType,
    }.items():
        payload[key] = enum(payload[key])
    payload.update(model_path=cli.model_path.resolve(), support_dir=cli.rpm_repo.resolve() / "SMPL")
    args = Namespace(**payload)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{cli.device}" if cli.device >= 0 else "cpu")
    model, _ = load_rpm_model(args, device=device)
    model.eval()
    manifest = json.loads((cli.input_dir / "manifest.json").read_text())
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(manifest):
        with np.load(cli.input_dir / item["file"], allow_pickle=False) as arrays:
            predicted = generate_sequence(model, args, arrays, device, DataTypeGT, ModelOutputType)
            np.savez_compressed(
                cli.output_dir / item["file"], rpm_local_rotations_6d=predicted,
                source_frames=np.arange(1, len(predicted) + 1, dtype=np.int64),
                source_relative_path=arrays["source_relative_path"],
                tracker_available=arrays["tracker_available"][1:],
                checkpoint=np.asarray(str(cli.model_path.resolve())), seed=np.asarray(args.seed),
            )
        print(f"[rpm-dropout] {index + 1}/{len(manifest)} {item['source']}", flush=True)


if __name__ == "__main__":
    main()
