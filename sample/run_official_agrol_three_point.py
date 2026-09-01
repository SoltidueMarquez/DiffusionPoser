from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


AGROL_WINDOW = 196


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行官方 AGRoL checkpoint，并导出指定 30 Hz 展示片段对应的原始旋转。"
    )
    parser.add_argument("--agrol_repo", required=True, type=Path)
    parser.add_argument("--human_body_prior_repo", required=True, type=Path)
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--input_npz", required=True, type=Path)
    parser.add_argument("--output_npz", required=True, type=Path)
    parser.add_argument("--source_frame_start", required=True, type=int)
    parser.add_argument("--source_frame_end_exclusive", required=True, type=int)
    parser.add_argument("--device", default=0, type=int)
    return parser


def choose_window_start(
    *,
    feature_count: int,
    source_frame_start: int,
    source_frame_end_exclusive: int,
) -> int:
    """让目标 30 Hz 片段尽量位于 AGRoL 196 帧条件窗中央。"""

    # AGRoL 60 Hz feature index 0 对应 1/60 s；source frame n 对应 2n/60 s，
    # 因此在 feature 数组中的索引为 2*n-1。
    target_start = 2 * int(source_frame_start) - 1
    target_end = 2 * int(source_frame_end_exclusive) - 2
    target_length = target_end - target_start
    if target_start < 0 or target_end > int(feature_count):
        raise ValueError(
            f"展示帧超出 AGRoL 输入：feature range [{target_start},{target_end}) / {feature_count}"
        )
    if target_length > AGROL_WINDOW:
        raise ValueError(
            f"AGRoL 单窗最多覆盖 {AGROL_WINDOW / 2:.1f} 个 30 Hz 帧，"
            f"当前展示片段需要 {target_length / 2:.1f} 帧。"
        )
    centered = target_start - (AGROL_WINDOW - target_length) // 2
    return int(np.clip(centered, 0, max(0, feature_count - AGROL_WINDOW)))


def main(argv: list[str] | None = None) -> Path:
    cli = build_arg_parser().parse_args(argv)
    agrol_repo = cli.agrol_repo.expanduser().resolve()
    hbp_repo = cli.human_body_prior_repo.expanduser().resolve()
    model_path = cli.model_path.expanduser().resolve()
    project_root = Path(__file__).resolve().parents[1]
    project_paths = {project_root, project_root / "sample"}
    sys.path[:] = [
        value
        for value in sys.path
        if Path(value or ".").resolve() not in project_paths
    ]
    sys.path.insert(0, str(hbp_repo))
    sys.path.insert(0, str(agrol_repo))

    from utils.model_util import (  # type: ignore[import-not-found]
        create_model_and_diffusion,
        load_model_wo_clip,
    )

    args_path = model_path.parent / "args.json"
    payload = json.loads(args_path.read_text(encoding="utf-8"))
    payload["arch"] = str(payload["arch"]).removeprefix("diffusion_")
    payload["timestep_respacing"] = "ddim5"
    args = Namespace(**payload)

    device = torch.device(
        f"cuda:{int(cli.device)}" if torch.cuda.is_available() else "cpu"
    )
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    model, diffusion = create_model_and_diffusion(args)
    state_dict = torch.load(model_path, map_location="cpu")
    load_model_wo_clip(model, state_dict)
    model.to(device).eval()

    with np.load(cli.input_npz, allow_pickle=False) as data:
        sparse = np.asarray(data["agrol_sparse_60hz_54d"], dtype=np.float32)
        mean = np.asarray(data["agrol_pose_mean"], dtype=np.float32)
        std = np.asarray(data["agrol_pose_std"], dtype=np.float32)
    window_start = choose_window_start(
        feature_count=sparse.shape[0],
        source_frame_start=int(cli.source_frame_start),
        source_frame_end_exclusive=int(cli.source_frame_end_exclusive),
    )
    sparse_window = torch.as_tensor(
        sparse[window_start : window_start + AGROL_WINDOW],
        dtype=torch.float32,
        device=device,
    )[None]
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model,
            (1, AGROL_WINDOW, int(args.motion_nfeat)),
            sparse=sparse_window,
            clip_denoised=False,
            model_kwargs=None,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    predicted = sample[0].cpu().numpy().astype(np.float32)
    predicted = predicted * std[None] + mean[None]

    output_path = cli.output_npz.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        agrol_local_rotations_6d_60hz=predicted,
        agrol_feature_window_start=np.asarray(window_start, dtype=np.int64),
        source_frame_start=np.asarray(cli.source_frame_start, dtype=np.int64),
        source_frame_end_exclusive=np.asarray(
            cli.source_frame_end_exclusive, dtype=np.int64
        ),
        seed=np.asarray(int(args.seed), dtype=np.int64),
        checkpoint=np.asarray(str(model_path)),
    )
    print(
        f"[agrol-three-point] wrote: {output_path} "
        f"(60 Hz feature window [{window_start},{window_start + AGROL_WINDOW}))",
        flush=True,
    )
    return output_path


if __name__ == "__main__":
    main()
