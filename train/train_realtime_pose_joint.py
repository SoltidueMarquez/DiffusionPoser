from __future__ import annotations

from pathlib import Path

import torch

from data_loaders.get_data import get_dataset_loader
from diffusion import logger
from train.train_diffusionposer import (
    TRAIN_PLATFORMS,
    prepare_save_dir,
    resolve_save_dir,
    save_args,
)
from train.training_loop import TrainLoop
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import (
    create_model_and_diffusion,
    load_realtime_pose_predictor_architecture,
)
from utils.parser_util import joint_finetune_args


def main() -> None:
    args = joint_finetune_args()
    args.predictor_model_path = str(Path(args.predictor_model_path).resolve())
    args.dit_model_path = str(Path(args.dit_model_path).resolve())
    args.predictor_architecture = load_realtime_pose_predictor_architecture(
        args.predictor_model_path
    )
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
        data = _create_data_loader(args, args.data_split)
        eval_data = (
            _create_data_loader(args, args.eval_split)
            if args.eval_during_training
            else None
        )

        model, diffusion = create_model_and_diffusion(args)
        model.to(dist_util.dev())
        if not args.resume_checkpoint:
            model.load_state_dict(
                torch.load(
                    args.dit_model_path,
                    map_location=dist_util.dev(),
                    weights_only=True,
                )
            )
        print(
            "joint finetuning Predictor + DiT, "
            f"dit_params={model.num_parameters() / 1_000_000.0:.2f}M, "
            f"precision={args.precision}...",
            flush=True,
        )
        TrainLoop(
            args,
            train_platform,
            model,
            diffusion,
            data,
            eval_data=eval_data,
        ).run_loop()
    finally:
        train_platform.close()


def _create_data_loader(args, split: str):
    return get_dataset_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_feats=args.input_feats,
        seq_len=args.seq_len,
        split=split,
        normalizer_dir=args.normalizer_dir,
        normalize_input=args.normalize_input,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
        seed=args.seed,
        rpm_hand_dropout=args.rpm_hand_dropout,
        rpm_hand_dropout_seed=(
            args.rpm_hand_dropout_seed
            if str(split) == str(args.data_split)
            else args.rpm_hand_dropout_seed + 1
        ),
    )


if __name__ == "__main__":
    main()
