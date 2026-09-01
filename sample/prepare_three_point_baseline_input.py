from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sample.three_point_baseline_data import prepare_baseline_sequence_input


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把同一条 realtime source 转为 RPM/AGRoL 官方三点输入。"
    )
    parser.add_argument("--source_npz", required=True, type=Path)
    parser.add_argument("--amass_npz", required=True, type=Path)
    parser.add_argument("--smpl_model_dir", required=True, type=Path)
    parser.add_argument("--rpm_stats_npz", required=True, type=Path)
    parser.add_argument("--agrol_stats_npz", required=True, type=Path)
    parser.add_argument("--output_npz", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    arrays = prepare_baseline_sequence_input(
        source_npz=args.source_npz.expanduser().resolve(),
        amass_npz=args.amass_npz.expanduser().resolve(),
        smpl_model_dir=args.smpl_model_dir.expanduser().resolve(),
        rpm_stats_npz=args.rpm_stats_npz.expanduser().resolve(),
        agrol_stats_npz=args.agrol_stats_npz.expanduser().resolve(),
    )
    output = args.output_npz.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"[baseline-input] wrote: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
