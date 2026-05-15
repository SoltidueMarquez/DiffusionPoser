import json
import os
from pathlib import Path

import torch

from data_loaders.get_data import get_dataset_loader
from diffusion import logger
from train.train_platforms import NoPlatform, TensorboardPlatform
from train.training_loop import TrainLoop
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
    fixseed(args.seed)
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
        )

        print("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args)
        model.to(dist_util.dev())
        print(f"Total params: {model.num_parameters() / 1_000_000.0:.2f}M")

        print("training DiffusionPoser fix-only model...")
        TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    finally:
        train_platform.close()


def prepare_save_dir(args):
    save_dir = Path(args.save_dir)
    if save_dir.exists() and not args.overwrite:
        raise FileExistsError(f"save_dir [{save_dir}] already exists. Use --overwrite to reuse it.")
    save_dir.mkdir(parents=True, exist_ok=True)


def save_args(args):
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    main()
