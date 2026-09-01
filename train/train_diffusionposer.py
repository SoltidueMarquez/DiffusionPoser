import json
import os
import re
from datetime import datetime
from pathlib import Path

import torch

from data_loaders.get_data import get_dataset_loader
from diffusion import logger
from train.train_platforms import NoPlatform, TensorboardPlatform
from train.training_loop import TrainLoop, find_resume_checkpoint
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args


TRAIN_PLATFORMS = {
    "NoPlatform": NoPlatform,
    "TensorboardPlatform": TensorboardPlatform,
}


def main():
    args = train_args()
    args.predictor_model_path = str(Path(args.predictor_model_path).resolve())
    fixseed(args.seed)
    resolve_save_dir(args)
    prepare_save_dir(args)
    dist_util.setup_dist(args.device if args.cuda else -1)
    logger.configure(dir=args.save_dir)
    torch.backends.cudnn.benchmark = True

    train_platform = TRAIN_PLATFORMS[args.train_platform_type](args.save_dir)
    try:
        train_platform.report_args(args, name="Args")
        save_args(args)
        print("creating data loader...")
        data = get_dataset_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            input_feats=args.input_feats,
            seq_len=args.seq_len,
            split=args.data_split,
            normalizer_dir=args.normalizer_dir,
            normalize_input=args.normalize_input,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
            seed=args.seed,
            rpm_hand_dropout=args.rpm_hand_dropout,
            rpm_hand_dropout_seed=args.rpm_hand_dropout_seed,
        )
        eval_data = None
        if args.eval_during_training:
            print("creating eval data loader...")
            eval_data = get_dataset_loader(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                input_feats=args.input_feats,
                seq_len=args.seq_len,
                split=args.eval_split,
                normalizer_dir=args.normalizer_dir,
                normalize_input=args.normalize_input,
                num_workers=args.num_workers,
                pin_memory=args.cuda,
                seed=args.seed,
                rpm_hand_dropout=args.rpm_hand_dropout,
                rpm_hand_dropout_seed=args.rpm_hand_dropout_seed + 1,
            )

        print("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args)
        model.to(dist_util.dev())
        print(f"Total params: {model.num_parameters() / 1_000_000.0:.2f}M")

        print(
            f"training DiffusionPoser model, task_mode={args.task_mode}, "
            f"precision={args.precision}..."
        )
        TrainLoop(args, train_platform, model, diffusion, data, eval_data=eval_data).run_loop()
    finally:
        train_platform.close()


def prepare_save_dir(args):
    save_dir = Path(args.save_dir)
    if save_dir.exists() and not args.overwrite and not args.resume_checkpoint:
        raise FileExistsError(
            f"save_dir [{save_dir}] already exists. "
            "For a fresh run, choose a new --save_dir or pass --overwrite to reuse it. "
            "To continue training, pass --resume_checkpoint latest."
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_checkpoint:
        write_latest_run_pointer(args)


def resolve_save_dir(args):
    """把用户给的 save_dir 解析成本次训练实际写入的 run 目录。"""

    if args.resume_checkpoint:
        args.resume_checkpoint = find_resume_checkpoint(
            save_dir=args.save_dir,
            requested_checkpoint=args.resume_checkpoint,
        )
        args.save_dir = str(Path(args.resume_checkpoint).resolve().parent)
        return

    run_root = Path(args.save_dir).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = resolve_run_label(args)
    candidate = run_root / f"{run_id}_{run_label}"
    suffix = 2
    while candidate.exists():
        candidate = run_root / f"{run_id}_{run_label}_{suffix:02d}"
        suffix += 1
    args.run_root = str(run_root)
    args.run_id = run_id
    args.save_dir = str(candidate)


def resolve_run_label(args) -> str:
    run_name = str(getattr(args, "run_name", "auto") or "auto").strip()
    if run_name.lower() in {"auto", ""}:
        run_name = f"{getattr(args, 'model_arch', 'model')}_seed{getattr(args, 'seed', 0)}"
    run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("._-")
    return run_name or "run"


def write_latest_run_pointer(args):
    """在 run 根目录留下稳定指针，方便脚本和 AI 快速定位最近一次训练。"""

    run_root = Path(getattr(args, "run_root", Path(args.save_dir).parent)).resolve()
    save_dir = Path(args.save_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "save_dir": str(save_dir),
        "run_root": str(run_root),
        "run_id": getattr(args, "run_id", ""),
        "run_name": getattr(args, "run_name", "auto"),
        "model_arch": getattr(args, "model_arch", ""),
        "seed": getattr(args, "seed", None),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_root / "latest_run.txt").write_text(str(save_dir), encoding="utf-8")
    with (run_root / "latest_run.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)


def save_args(args):
    args_file = "resume_args.json" if args.resume_checkpoint else "args.json"
    args_path = os.path.join(args.save_dir, args_file)
    payload = vars(args).copy()
    with open(args_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    main()
