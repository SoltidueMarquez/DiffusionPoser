from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data_loaders.realtime_pose_predictor_dataset import get_predictor_dataset_loader
from model.realtime_pose_predictor import RealtimePosePredictor
from train.predictor_training_loop import PredictorTrainLoop
from utils.fixseed import fixseed
from utils.training_precision import TRAINING_PRECISIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 realtime pose Predictor Transformer。")
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--split_dir", default="")
    parser.add_argument("--normalizer_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--data_split", default="train")
    parser.add_argument("--windows_per_source", default=128, type=int)
    parser.add_argument("--source_limit", default=0, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--precision", default="fp32", choices=TRAINING_PRECISIONS
    )
    parser.add_argument("--num_steps", default=100_000, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--lr_drop_step", default=50_000, type=int)
    parser.add_argument("--lr_drop_factor", default=30.0, type=float)
    parser.add_argument("--gradient_clip_norm", default=1.0, type=float)
    parser.add_argument("--ema_decay", default=0.995, type=float)
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--save_interval", default=5_000, type=int)
    parser.add_argument(
        "--checkpoint_max_keep",
        default=3,
        type=int,
        help="最多保留最近 N 组编号 checkpoint；设为 0 表示不清理。",
    )
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--layers", default=4, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--feedforward_dim", default=1024, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--device", default=0, type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    fixseed(args.seed)
    args.normalizer_dir = str(Path(args.normalizer_dir).resolve())
    save_dir = Path(args.save_dir).resolve()
    _prepare_save_dir(save_dir, args.resume_checkpoint)
    args.save_dir = str(save_dir)
    if args.resume_checkpoint.lower() not in {"latest", "auto"}:
        args.resume_checkpoint = (
            str(Path(args.resume_checkpoint).resolve()) if args.resume_checkpoint else ""
        )
    args.free_running_max_steps = 30
    args_path = save_dir / ("resume_args.json" if args.resume_checkpoint else "args.json")
    with args_path.open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, sort_keys=True, ensure_ascii=False)
    canonical_args_path = save_dir / "args.json"
    if not canonical_args_path.is_file():
        canonical_args_path.write_text(
            json.dumps(vars(args), indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu"
    )
    common = dict(
        source_dir=args.source_dir,
        split_dir=args.split_dir or None,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        limit=args.source_limit,
    )
    train_data = get_predictor_dataset_loader(
        split=args.data_split,
        windows_per_source=args.windows_per_source,
        shuffle=True,
        **common,
    )
    model = RealtimePosePredictor(
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
    ).to(device)
    print(
        f"Predictor params: {model.num_parameters() / 1_000_000.0:.2f}M, "
        f"device={device}, precision={args.precision}"
    )
    PredictorTrainLoop(args, model, train_data, device).run()


def _prepare_save_dir(save_dir: Path, resume_checkpoint: str) -> None:
    """新训练不混入旧产物；恢复训练允许复用同一目录。"""

    if save_dir.is_dir() and any(save_dir.iterdir()) and not str(resume_checkpoint).strip():
        raise FileExistsError(
            f"Predictor save_dir 非空：{save_dir}；请指定 --resume_checkpoint。"
        )
    save_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
