from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark source-reference realtime pose DataLoader throughput.")
    parser.add_argument("--task_dir", required=True, type=Path)
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--workers", nargs="+", default=[0, 2, 4], type=int)
    parser.add_argument("--rollout_steps", nargs="+", default=[1, 9], type=int)
    parser.add_argument("--warmup_batches", default=2, type=int)
    parser.add_argument("--timed_batches", default=10, type=int)
    parser.add_argument("--source_cache_max_mib", default=512, type=int)
    return parser


def benchmark_loader(args: argparse.Namespace) -> list[dict[str, float | int]]:
    results: list[dict[str, float | int]] = []
    for rollout_steps in args.rollout_steps:
        for workers in args.workers:
            dataset = RealtimePoseTaskDataset(
                args.task_dir,
                split=args.split,
                normalize_input=False,
                source_cache_max_mib=args.source_cache_max_mib,
                enable_rollout=int(rollout_steps) > 1,
                rollout_steps=int(rollout_steps),
            )
            loader = DataLoader(
                dataset,
                batch_size=int(args.batch_size),
                shuffle=False,
                drop_last=True,
                num_workers=int(workers),
                pin_memory=False,
                persistent_workers=False,
                **({"prefetch_factor": 2} if int(workers) > 0 else {}),
            )
            iterator = iter(loader)
            measured: list[float] = []
            total_batches = int(args.warmup_batches) + int(args.timed_batches)
            for batch_index in range(total_batches):
                started = time.perf_counter()
                try:
                    next(iterator)
                except StopIteration:
                    break
                elapsed = time.perf_counter() - started
                if batch_index >= int(args.warmup_batches):
                    measured.append(elapsed)
            if not measured:
                raise RuntimeError("No timed DataLoader batches were produced.")
            timings = torch.tensor(measured, dtype=torch.float64)
            results.append(
                {
                    "rollout_steps": int(rollout_steps),
                    "num_workers": int(workers),
                    "timed_batches": len(measured),
                    "mean_batch_seconds": float(timings.mean().item()),
                    "p95_batch_seconds": float(torch.quantile(timings, 0.95).item()),
                }
            )
    return results


def main(argv: list[str] | None = None) -> list[dict[str, float | int]]:
    args = build_argument_parser().parse_args(argv)
    results = benchmark_loader(args)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
