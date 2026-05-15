from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from data_loaders.get_data import get_dataset_loader
from data_loaders.sensor_masking import (
    X277_FEATURE_DIM,
    build_inpaint_mask_from_sensor_missing_labels,
)
from diffusion import logger
from sample.utils import build_output_dir, choose_sampler, load_checkpoint_model, sanitize_path_token
from sample.visualization import HAS_VISUALIZATION_BACKEND, render_fix_visualization
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.parser_util import (
    add_base_options,
    add_data_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    parse_and_load_from_model,
)


# 这个入口只负责“修复模式”的评估和可视化，因此注释会尽量把每一步为什么要这样做讲清楚。


def build_argument_parser() -> ArgumentParser:
    # 默认固定为 test split，避免修复入口误跑到训练集上。
    parser = ArgumentParser(description="Run DiffusionPoser fix-only test / visualization pipeline.")
    add_base_options(parser)
    add_data_options(parser)
    parser.set_defaults(data_split="test")
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    return parser


def move_batch_to_device(batch: dict, device) -> dict:
    # batch 里既有 tensor，也有字符串 / 标量元数据；这里只迁移 tensor，其他字段原样保留。
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def get_batch_string_item(batch: dict, key: str, index: int) -> str:
    # collate 之后同一个字段可能会变成 list、numpy.ndarray 或 tensor，
    # 这里统一转成字符串，方便写日志和文件名。
    value = batch.get(key, "")
    if isinstance(value, (list, tuple)):
        return str(value[index])
    if isinstance(value, np.ndarray):
        return str(value[index])
    if torch.is_tensor(value):
        return str(value[index].item())
    return str(value)


def denormalize_x277_batch(batch_x: torch.Tensor, normalizer) -> np.ndarray:
    """把 `[B, 283, T]` 里的 X277 主体部分还原成 `[B, T, 277]`，用于保存和可视化。"""

    x277 = batch_x[:, :X277_FEATURE_DIM, :].transpose(1, 2).contiguous()
    x277 = x277.cpu()
    if normalizer is None:
        return x277.numpy()
    return np.asarray(normalizer.inverse(x277))


def build_conditioned_motion(
    batch_x: torch.Tensor,
    inpaint_mask: torch.Tensor,
) -> torch.Tensor:
    """
    生成喂给模型的条件输入。
    这里会把“需要修复”的 tracker 区域先清零，
    这样 diffusion 只能依赖未损坏上下文做补全，避免直接偷看答案。
    """

    conditioned = batch_x.clone()
    tracker_motion = conditioned[:, :X277_FEATURE_DIM, :]
    tracker_mask = inpaint_mask[:, :X277_FEATURE_DIM, :]
    tracker_motion = torch.where(tracker_mask, torch.zeros_like(tracker_motion), tracker_motion)
    conditioned[:, :X277_FEATURE_DIM, :] = tracker_motion
    return conditioned


def build_model_kwargs(
    *,
    conditioned_motion: torch.Tensor,
    inpaint_mask: torch.Tensor,
    valid_frame_mask: torch.Tensor,
) -> dict:
    # 这里沿用训练时的条件组织方式：
    # - inpaint_cond / mask 负责告诉模型哪些位置要补；
    # - inpainted_motion 负责提供已经清零后的条件输入；
    # - valid_frame_mask / attention_mask 负责屏蔽 padding 帧。
    return {
        "inpaint_cond": inpaint_mask,
        "valid_frame_mask": valid_frame_mask,
        "attention_mask": valid_frame_mask,
        "y": {
            "mask": inpaint_mask,
            "inpainted_motion": conditioned_motion,
        },
    }


def save_sample_artifacts(
    *,
    sample_dir: Path,
    sample_name: str,
    reference_motion: np.ndarray,
    corrupted_motion: np.ndarray,
    repaired_motion: np.ndarray,
    sensor_missing_labels: np.ndarray,
    inpaint_mask: np.ndarray,
    valid_frame_mask: np.ndarray,
    rendered: bool,
    render_meta: dict | None,
    render_error: str | None,
    args,
    keyid: str,
    source_path: str,
) -> None:
    # 每个样本单独落盘，后面无论是人工复查、补指标还是接 PostEdit 都会更方便。
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        sample_dir / "sample_outputs.npz",
        reference_motion=reference_motion,
        corrupted_motion=corrupted_motion,
        repaired_motion=repaired_motion,
        sensor_missing_labels=sensor_missing_labels,
        inpaint_mask=inpaint_mask,
        valid_frame_mask=valid_frame_mask,
    )

    metadata = {
        "sample_name": sample_name,
        "task_id": keyid,
        "source_path": source_path,
        "rendered": rendered,
        "render_meta": render_meta or {},
        "render_error": render_error or "",
        "visualize_num": int(args.visualize_num),
        "model_path": str(args.model_path),
        "output_dir": str(sample_dir),
        "fps": float(args.visualize_fps),
        "x277_fps": float(args.x277_fps),
    }
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, sort_keys=True)


def sample_fix_batch(
    *,
    model,
    diffusion,
    args,
    batch: dict,
) -> torch.Tensor:
    # 先选采样器，再把损坏后的输入作为 init_image 交给扩散过程。
    # 对 inpainting 任务来说，这等价于在已有上下文上做局部修复。
    sample_fn = choose_sampler(diffusion, bool(args.ts_respace))
    corrupted_motion = batch["corrupted_motion"]
    model_kwargs = build_model_kwargs(
        conditioned_motion=corrupted_motion,
        inpaint_mask=batch["inpaint_mask"],
        valid_frame_mask=batch["valid_frame_mask"],
    )
    repaired = sample_fn(
        model,
        tuple(corrupted_motion.shape),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=corrupted_motion,
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    # 采样结束后再按 mask 兜底一次，确保未损坏区域不会被模型改写。
    repaired = torch.where(batch["inpaint_mask"], repaired, corrupted_motion)
    return repaired


def main(argv: list[str] | None = None) -> None:
    # 先把命令行参数和 checkpoint 同目录里的 args.json 合并，
    # 这样测试入口会自动继承训练阶段配置，只覆盖修复流程真正需要变更的部分。
    parser = build_argument_parser()
    args = parse_and_load_from_model(parser, argv=argv, ignore_keys={"data_split"})

    fixseed(args.seed)
    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()

    # 输出目录按 checkpoint / seed / folder_path 组织，避免多次测试互相覆盖。
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False, sort_keys=True)

    logger.configure(dir=str(output_dir))

    data = get_dataset_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_feats=args.input_feats,
        seq_len=args.seq_len,
        split=args.data_split,
        normalizer_dir=args.normalizer_dir,
        normalize_input=args.normalize_input,
        preload_data=args.preload_data,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
        folder_path=args.folder_path or None,
    )
    motion_normalizer = data.dataset.normalizer

    model, diffusion = create_model_and_diffusion(args)
    model, model_source = load_checkpoint_model(
        model=model,
        model_path=args.model_path,
        device=device,
        use_ema=bool(args.use_ema),
    )

    # summary.json 记录整次测试的全局信息；单个样本的细节则写进各自目录。
    summary = {
        "model_path": str(args.model_path),
        "model_source": model_source,
        "data_dir": str(args.data_dir),
        "data_split": str(args.data_split),
        "folder_path": str(args.folder_path),
        "visualize_num": int(args.visualize_num),
        "x277_fps": float(args.x277_fps),
        "samples": [],
    }

    visualized_count = 0
    render_failed_count = 0
    processed_count = 0
    render_all = args.visualize_num < 0
    render_enabled = args.visualize_num != 0
    render_disabled_reason = ""
    if render_enabled and not HAS_VISUALIZATION_BACKEND:
        # 视频导出依赖 matplotlib + moviepy/imageio/ffmpeg。缺少后端时不要中断测试，
        # 因为 `.npz` 和 metadata 仍然是后续算指标、接 PostEdit 的有效结果。
        render_enabled = False
        render_disabled_reason = "缺少 matplotlib 或可用的 MP4 导出后端，本次仅保存数值结果。"
        logger.log(render_disabled_reason)

    for batch_index, raw_batch in enumerate(tqdm(data, desc="Fix test", unit="batch")):
        batch = move_batch_to_device(raw_batch, device)
        valid_frame_mask = batch["valid_frame_mask"].bool()

        # 数据层给的是 `[B, T, 6]` 的传感器损坏标签；
        # 这里先转成 numpy，再重建 `[B, T, 283]` 的 inpaint mask，保证和训练语义一致。
        sensor_missing_labels_bt6 = batch["sensor_missing_labels"].permute(0, 2, 1).cpu().numpy()
        valid_frame_mask_np = valid_frame_mask.cpu().numpy()
        inpaint_mask_np = build_inpaint_mask_from_sensor_missing_labels(
            sensor_missing_labels=sensor_missing_labels_bt6,
            valid_frame_mask=valid_frame_mask_np,
        )

        # 模型内部还是 `[B, 283, T]` 的通道优先布局，因此这里再转回 tensor 并转置。
        inpaint_mask = torch.from_numpy(inpaint_mask_np).permute(0, 2, 1).to(device=device, dtype=torch.bool)

        # 先把损坏 tracker 区域抹零，再交给 diffusion 修复，避免输入里直接泄露答案。
        conditioned_motion = build_conditioned_motion(batch["x"], inpaint_mask)
        batch_for_sample = {
            **batch,
            "corrupted_motion": conditioned_motion,
            "inpaint_mask": inpaint_mask,
            "valid_frame_mask": valid_frame_mask,
        }
        with torch.no_grad():
            repaired = sample_fix_batch(
                model=model,
                diffusion=diffusion,
                args=args,
                batch=batch_for_sample,
            )

        reference_motion = denormalize_x277_batch(batch["x"], motion_normalizer)
        corrupted_motion = denormalize_x277_batch(conditioned_motion, motion_normalizer)
        repaired_motion = denormalize_x277_batch(repaired, motion_normalizer)
        valid_length_np = valid_frame_mask.sum(dim=1).cpu().numpy().astype(int)
        sensor_missing_labels_np = batch["sensor_missing_labels"].permute(0, 2, 1).cpu().numpy()
        inpaint_mask_np_bt = inpaint_mask.permute(0, 2, 1).cpu().numpy()

        for item_index in range(batch["x"].shape[0]):
            keyid = get_batch_string_item(batch, "keyid", item_index)
            source_path = get_batch_string_item(batch, "source_path", item_index)

            # 每个样本一个独立子目录，既不会覆盖，也方便只看某一条轨迹的效果。
            sample_name = f"{processed_count:05d}_{sanitize_path_token(keyid or f'batch{batch_index}_item{item_index}')}"
            sample_dir = output_dir / sample_name

            render_meta = None
            rendered = False
            render_error = render_disabled_reason
            if render_enabled and (render_all or visualized_count < int(args.visualize_num)):
                # 可视化只解释结果，不参与模型计算；这里把帧级损坏信息写进视频标题。
                try:
                    render_meta = render_fix_visualization(
                        reference_motion=reference_motion[item_index],
                        corrupted_motion=corrupted_motion[item_index],
                        repaired_motion=repaired_motion[item_index],
                        sensor_missing_labels=sensor_missing_labels_np[item_index],
                        output_path=sample_dir / "repair.mp4",
                        fps=float(args.visualize_fps),
                        x277_fps=float(args.x277_fps),
                        title=keyid or sample_name,
                        valid_length=int(valid_length_np[item_index]),
                    )
                    rendered = True
                    render_error = ""
                    visualized_count += 1
                except Exception as exc:
                    # 单条视频失败不应让整轮测试失效；保留数值结果和错误信息，方便之后补渲染或排查。
                    render_failed_count += 1
                    render_error = f"{type(exc).__name__}: {exc}"
                    logger.log(f"render failed for {keyid or sample_name}: {render_error}")

            save_sample_artifacts(
                sample_dir=sample_dir,
                sample_name=sample_name,
                reference_motion=reference_motion[item_index],
                corrupted_motion=corrupted_motion[item_index],
                repaired_motion=repaired_motion[item_index],
                sensor_missing_labels=sensor_missing_labels_np[item_index],
                inpaint_mask=inpaint_mask_np_bt[item_index],
                valid_frame_mask=valid_frame_mask_np[item_index],
                rendered=rendered,
                render_meta=render_meta,
                render_error=render_error,
                args=args,
                keyid=keyid,
                source_path=source_path,
            )

            summary["samples"].append(
                {
                    "sample_name": sample_name,
                    "task_id": keyid,
                    "source_path": source_path,
                    "rendered": rendered,
                    "render_error": render_error,
                    "sample_dir": str(sample_dir),
                }
            )
            processed_count += 1

    summary["processed_count"] = processed_count
    summary["visualized_count"] = visualized_count
    summary["render_failed_count"] = render_failed_count
    summary["render_disabled_reason"] = render_disabled_reason
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
