from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在隔离 import 路径中运行官方 RPM-P2 三点 checkpoint。"
    )
    parser.add_argument("--rpm_repo", required=True, type=Path)
    parser.add_argument("--human_body_prior_repo", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--input_npz", required=True, type=Path)
    parser.add_argument("--output_npz", required=True, type=Path)
    parser.add_argument("--device", default=0, type=int)
    return parser


def load_official_args(model_path: Path) -> dict:
    args_path = model_path.parent / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"RPM checkpoint 缺少 args.json：{args_path}")
    return json.loads(args_path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> Path:
    cli = build_arg_parser().parse_args(argv)
    rpm_repo = cli.rpm_repo.expanduser().resolve()
    hbp_repo = cli.human_body_prior_repo.expanduser().resolve()
    model_path = cli.model_path.expanduser().resolve()
    # 官方 RPM 与本项目都有 `model`、`utils` 等顶层包，因此必须在任何官方
    # import 之前移除本项目的脚本/根目录，再把它放到 sys.path 最前面。
    # RPM 的 `utils/` 是 namespace package；若保留 `sample/utils.py`，Python
    # 会优先把后者解析成普通模块，从而导致 `utils.constants` 无法导入。
    project_root = Path(__file__).resolve().parents[1]
    project_paths = {project_root, project_root / "sample"}
    sys.path[:] = [
        value
        for value in sys.path
        if Path(value or ".").resolve() not in project_paths
    ]
    sys.path.insert(0, str(hbp_repo))
    sys.path.insert(0, str(rpm_repo))

    from utils.constants import (  # type: ignore[import-not-found]
        ConditionMasker,
        DataTypeGT,
        DatasetType,
        LossDistType,
        ModelOutputType,
        PredictionInputType,
        PredictionTargetType,
    )
    from utils.model_util import load_rpm_model  # type: ignore[import-not-found]

    payload = load_official_args(model_path)
    payload["model_path"] = model_path
    payload["dataset"] = DatasetType(payload["dataset"])
    payload["masker"] = ConditionMasker(payload["masker"])
    payload["loss_dist_type"] = LossDistType(payload["loss_dist_type"])
    payload["target_type"] = PredictionTargetType(payload["target_type"])
    payload["prediction_input_type"] = PredictionInputType(
        payload["prediction_input_type"]
    )
    payload["support_dir"] = rpm_repo / "SMPL"
    args = Namespace(**payload)

    device = torch.device(
        f"cuda:{int(cli.device)}" if torch.cuda.is_available() else "cpu"
    )
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    model, _ = load_rpm_model(args, device=device)

    with np.load(cli.input_npz, allow_pickle=False) as data:
        motion = torch.as_tensor(data["rpm_motion_6d"], dtype=torch.float32)
        sparse = torch.as_tensor(data["rpm_sparse_54d"], dtype=torch.float32)
        mean = torch.as_tensor(data["rpm_pose_mean"], dtype=torch.float32)
        std = torch.as_tensor(data["rpm_pose_std"], dtype=torch.float32)
        source_offset = int(np.asarray(data["source_feature_frame_offset"]).item())
    if motion.shape[0] != sparse.shape[0] or motion.shape[1:] != (132,) or sparse.shape[1:] != (54,):
        raise ValueError(f"RPM 输入维度不匹配：motion={motion.shape}, sparse={sparse.shape}")

    normalized = ((motion - mean) / (std + 1e-8)).to(device)[None]
    sparse = sparse.to(device)[None]
    context = max(int(args.rolling_motion_ctx), int(args.rolling_sparse_ctx))
    if normalized.shape[1] < context + int(args.input_motion_length):
        raise ValueError("RPM 序列短于 context + rolling window。")
    output = torch.zeros_like(normalized)
    output[:, :context] = normalized[:, :context]
    current = context
    rolling = normalized[:, context : context + int(args.input_motion_length)].clone()
    with torch.no_grad():
        while current < normalized.shape[1]:
            condition = {
                DataTypeGT.MOTION_CTX: output[
                    :, current - int(args.rolling_motion_ctx) : current
                ],
                DataTypeGT.SPARSE: sparse[
                    :, current - int(args.rolling_sparse_ctx) : current + 1
                ],
            }
            result = model(rolling, condition, x_start=rolling)
            rolling = result[ModelOutputType.RELATIVE_ROTS]
            output[:, current : current + 1] = rolling[:, :1]
            rolling[:, :-1] = rolling.clone()[:, 1:]
            rolling[:, -1] = 0.0
            current += 1
    predicted = (output[0].cpu() * std + mean).numpy().astype(np.float32)

    output_path = cli.output_npz.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        rpm_local_rotations_6d=predicted,
        source_frame_offset=np.asarray(source_offset, dtype=np.int64),
        seed=np.asarray(int(args.seed), dtype=np.int64),
        checkpoint=np.asarray(str(model_path)),
    )
    print(f"[rpm-three-point] wrote: {output_path}", flush=True)
    return output_path


if __name__ == "__main__":
    main()
