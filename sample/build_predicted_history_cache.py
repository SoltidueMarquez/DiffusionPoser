from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from sample.reconstruct_stream import reconstruct_batch, tensor_bct_to_numpy_btc
from sample.utils import load_checkpoint_model
from utils import dist_util
from utils.model_util import create_model_and_diffusion
from utils.parser_util import (
    add_base_options,
    add_data_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    parse_and_load_runtime_schema_from_model,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build predicted-history cache from a realtime_pose checkpoint.",
        allow_abbrev=False,
    )
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    parser.add_argument("--limit", default=0, type=int)
    return parser


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = parse_and_load_runtime_schema_from_model(build_arg_parser(), argv=argv)
    output_dir = Path(args.output_dir or "output/pred_history_cache").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    dataset = RealtimePoseTaskDataset(
        data_dir=args.data_dir,
        split=args.data_split,
        seq_len=args.seq_len,
        normalizer_dir=args.normalizer_dir,
        normalize_input=args.normalize_input,
        schema_name=args.schema,
        tracker_mask_policy="task",
    )
    model, diffusion = create_model_and_diffusion(args)
    model, _source = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    written = 0
    max_count = len(dataset) if int(args.limit) <= 0 else min(len(dataset), int(args.limit))
    for index in range(max_count):
        item = dataset[index]
        batch = {key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value for key, value in item.items()}
        pred = reconstruct_batch(
            model=model,
            diffusion=diffusion,
            batch=batch,
            device=device,
            use_ddim=str(args.ts_respace).startswith("ddim"),
            schema_name=args.schema,
        )
        pred_np = tensor_bct_to_numpy_btc(pred)
        task_id = str(dataset.entries[index].get("task_id", index))
        np.savez(
            output_dir / f"{task_id}.npz",
            predicted_features_normalized=pred_np,
            schema_name=np.asarray(args.schema),
            feature_space=np.asarray("normalized"),
        )
        written += 1
    print(f"[build_predicted_history_cache] written={written} output_dir={output_dir}")
    return {"written": written}


if __name__ == "__main__":
    main()
