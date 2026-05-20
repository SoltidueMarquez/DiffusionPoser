from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from data_loaders.get_data import get_dataset_loader
from data_loaders.sensor_masking import MODEL_INPUT_DIM, X277_FEATURE_DIM
from diffusion import logger
from sample.utils import build_output_dir, choose_sampler, load_checkpoint_model, sanitize_path_token
from sample.visualization import (
    HAS_VISUALIZATION_BACKEND,
    derive_tracker_features_from_body,
    render_full_reconstruction_visualization,
)
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


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run current277 offline streaming full-body reconstruction.")
    add_base_options(parser)
    add_data_options(parser)
    parser.set_defaults(data_split="test", batch_size=1)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    return parser


def move_batch_to_device(batch: dict, device) -> dict:
    return {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }


def get_batch_string_item(batch: dict, key: str, index: int) -> str:
    value = batch.get(key, "")
    if isinstance(value, (list, tuple)):
        return str(value[index])
    if isinstance(value, np.ndarray):
        return str(value[index])
    if torch.is_tensor(value):
        return str(value[index].item())
    return str(value)


def get_batch_int_item(batch: dict, key: str, index: int, default: int) -> int:
    value = batch.get(key)
    if value is None:
        return int(default)
    if isinstance(value, (list, tuple)):
        return int(value[index])
    if isinstance(value, np.ndarray):
        return int(value[index])
    if torch.is_tensor(value):
        return int(value[index].item())
    return int(value)


def denormalize_x277_sequence(sequence_x: torch.Tensor, normalizer) -> np.ndarray:
    """Convert one normalized `[283, T]` tensor into denormalized `[T, 277]`."""

    x277 = sequence_x[:X277_FEATURE_DIM].transpose(0, 1).detach().cpu()
    if normalizer is None:
        return x277.numpy()
    return np.asarray(normalizer.inverse(x277))


def build_model_kwargs(
    *,
    conditioned_motion: torch.Tensor,
    inpaint_mask: torch.Tensor,
    valid_frame_mask: torch.Tensor,
) -> dict:
    return {
        "inpaint_cond": inpaint_mask,
        "valid_frame_mask": valid_frame_mask,
        "attention_mask": valid_frame_mask,
        "y": {
            "mask": inpaint_mask,
            "inpainted_motion": conditioned_motion,
        },
    }


def seed_missing_features_from_previous_frame(
    *,
    current_frame: torch.Tensor,
    current_inpaint_mask: torch.Tensor,
    previous_frame: torch.Tensor | None,
) -> torch.Tensor:
    """
    用上一帧重建结果初始化当前帧待补全维度。

    DiffusionPoser 的自回归 inpainting 推理不是把未知维度置零，而是让未知
    body/root/contact 和离线 tracker 从上一帧的合理动作状态开始去噪。mask 仍然为
    True，表示这些位置不会被当作观测条件强制保留。
    """

    if current_frame.shape != current_inpaint_mask.shape:
        raise ValueError(
            f"current_frame 和 current_inpaint_mask 形状必须一致，实际为 "
            f"{tuple(current_frame.shape)} vs {tuple(current_inpaint_mask.shape)}"
        )
    if previous_frame is None:
        # 第一帧没有历史重建值；保持冷启动零初始化，避免使用当前帧 GT 泄漏未知信息。
        previous_frame = torch.zeros_like(current_frame)
    elif previous_frame.shape != current_frame.shape:
        raise ValueError(
            f"previous_frame 应与 current_frame 同形，实际为 "
            f"{tuple(previous_frame.shape)} vs {tuple(current_frame.shape)}"
        )
    return torch.where(current_inpaint_mask, previous_frame, current_frame)


def build_stream_window(
    *,
    reference: torch.Tensor,
    conditioned_reference: torch.Tensor | None = None,
    reconstructed: torch.Tensor,
    task_inpaint_mask: torch.Tensor,
    valid_frame_mask: torch.Tensor,
    frame_index: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build a left-padded sliding window ending at ``frame_index``.

    History frames come from ``reconstructed`` and are always hard conditions.
    The current observed trackers remain hard conditions, while current unknown
    body/root/contact and offline trackers are initialized from the previous
    reconstructed frame before diffusion sampling.
    """

    channels, total_frames = reference.shape
    if channels != MODEL_INPUT_DIM:
        raise ValueError(f"reference must be [283, T], got {tuple(reference.shape)}")
    condition_source = reference if conditioned_reference is None else conditioned_reference
    if condition_source.shape != reference.shape:
        raise ValueError(f"conditioned_reference must match reference, got {tuple(condition_source.shape)}")
    if reconstructed.shape != reference.shape:
        raise ValueError(f"reconstructed must match reference, got {tuple(reconstructed.shape)}")
    if task_inpaint_mask.shape != reference.shape:
        raise ValueError(f"task_inpaint_mask must match reference, got {tuple(task_inpaint_mask.shape)}")
    if valid_frame_mask.shape != (total_frames,):
        raise ValueError(f"valid_frame_mask must be [T], got {tuple(valid_frame_mask.shape)}")
    if frame_index < 0 or frame_index >= total_frames:
        raise ValueError(f"frame_index out of range: {frame_index}, T={total_frames}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    source_start = max(0, frame_index - seq_len + 1)
    source_end = frame_index + 1
    window_length = source_end - source_start
    offset = seq_len - window_length

    conditioned = torch.zeros((1, channels, seq_len), dtype=reference.dtype, device=reference.device)
    inpaint_mask = torch.zeros((1, channels, seq_len), dtype=torch.bool, device=reference.device)
    window_valid = torch.zeros((1, seq_len), dtype=torch.bool, device=reference.device)

    if window_length > 1:
        history_slice = slice(source_start, frame_index)
        history_dest = slice(offset, offset + window_length - 1)
        conditioned[0, :, history_dest] = reconstructed[:, history_slice]

    current_dest = offset + window_length - 1
    current_mask = task_inpaint_mask[:, frame_index]
    previous_frame = reconstructed[:, frame_index - 1] if frame_index > 0 else None
    conditioned[0, :, current_dest] = seed_missing_features_from_previous_frame(
        current_frame=condition_source[:, frame_index],
        current_inpaint_mask=current_mask,
        previous_frame=previous_frame,
    )
    inpaint_mask[0, :, current_dest] = current_mask
    window_valid[0, offset : offset + window_length] = valid_frame_mask[source_start:source_end]
    return conditioned, inpaint_mask, window_valid, current_dest


def build_previous_frame_conditioned_sequence(
    *,
    reference: torch.Tensor,
    conditioned_reference: torch.Tensor | None = None,
    reconstructed: torch.Tensor,
    task_inpaint_mask: torch.Tensor,
    valid_frame_mask: torch.Tensor,
) -> torch.Tensor:
    """
    构造与流式推理一致的条件轨迹，主要用于保存和可视化。

    `reference` 和 `reconstructed` 都是 `[283, T]`。待补全维度记录上一帧重建值，
    当前真实可观测 tracker 和 6 维缺失标签仍然来自 `reference`。
    """

    if reconstructed.shape != reference.shape:
        raise ValueError(f"reconstructed must match reference, got {tuple(reconstructed.shape)}")
    condition_source = reference if conditioned_reference is None else conditioned_reference
    if condition_source.shape != reference.shape:
        raise ValueError(f"conditioned_reference must match reference, got {tuple(condition_source.shape)}")
    if task_inpaint_mask.shape != reference.shape:
        raise ValueError(f"task_inpaint_mask must match reference, got {tuple(task_inpaint_mask.shape)}")
    total_frames = reference.shape[1]
    if valid_frame_mask.shape != (total_frames,):
        raise ValueError(f"valid_frame_mask must be [T], got {tuple(valid_frame_mask.shape)}")

    conditioned = condition_source.clone()
    valid_length = int(valid_frame_mask.sum().item())
    if valid_length <= 0:
        raise ValueError("valid_frame_mask 没有有效帧，无法构造当前帧条件。")
    frame_index = valid_length - 1
    frame_mask = task_inpaint_mask[:, frame_index]
    if not frame_mask.any():
        raise ValueError("当前窗口最后一帧没有任何待补全维度，请检查 inpaint_mask。")
    previous_frame = reconstructed[:, frame_index - 1] if frame_index > 0 else None
    conditioned[:, frame_index] = seed_missing_features_from_previous_frame(
        current_frame=condition_source[:, frame_index],
        current_inpaint_mask=frame_mask,
        previous_frame=previous_frame,
    )
    conditioned[:, valid_length:] = 0.0
    return conditioned


def sample_streaming_sequence(
    *,
    model,
    diffusion,
    args,
    reference: torch.Tensor,
    conditioned_reference: torch.Tensor | None = None,
    task_inpaint_mask: torch.Tensor,
    valid_frame_mask: torch.Tensor,
) -> torch.Tensor:
    reconstructed = reference.clone()
    sample_fn = choose_sampler(diffusion, bool(args.ts_respace))
    valid_length = int(valid_frame_mask.sum().item())
    if valid_length <= 0:
        raise ValueError("valid_frame_mask 没有有效帧，无法执行流式重建。")

    frame_index = valid_length - 1
    frame_mask = task_inpaint_mask[:, frame_index]
    if not frame_mask.any():
        raise ValueError("当前窗口最后一帧没有任何待补全维度，请检查 inpaint_mask。")

    conditioned, inpaint_mask, window_valid, current_dest = build_stream_window(
        reference=reference,
        conditioned_reference=conditioned_reference,
        reconstructed=reconstructed,
        task_inpaint_mask=task_inpaint_mask,
        valid_frame_mask=valid_frame_mask,
        frame_index=frame_index,
        seq_len=int(args.seq_len),
    )
    model_kwargs = build_model_kwargs(
        conditioned_motion=conditioned,
        inpaint_mask=inpaint_mask,
        valid_frame_mask=window_valid,
    )
    sampled_window = sample_fn(
        model,
        tuple(conditioned.shape),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=conditioned,
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    sampled_frame = sampled_window[0, :, current_dest]
    conditioned_frame = conditioned[0, :, current_dest]
    reconstructed[:, frame_index] = torch.where(frame_mask, sampled_frame, conditioned_frame)

    reconstructed[:, valid_length:] = 0.0
    return reconstructed


def save_stream_artifacts(
    *,
    sample_dir: Path,
    reference_motion: np.ndarray,
    conditioned_motion: np.ndarray,
    reconstructed_motion: np.ndarray,
    sensor_missing_labels: np.ndarray,
    inpaint_mask: np.ndarray,
    valid_frame_mask: np.ndarray,
    metadata: dict,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        sample_dir / "stream_outputs.npz",
        reference_motion=reference_motion,
        conditioned_motion=conditioned_motion,
        reconstructed_motion=reconstructed_motion,
        sensor_missing_labels=sensor_missing_labels,
        inpaint_mask=inpaint_mask,
        valid_frame_mask=valid_frame_mask,
    )
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, sort_keys=True)


def main(argv: list[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parse_and_load_from_model(parser, argv=argv, ignore_keys={"batch_size", "data_split", "seq_len"})
    if int(args.batch_size) != 1:
        raise ValueError("reconstruct_stream 当前按单样本自回归运行，请使用 --batch_size 1。")

    fixseed(args.seed)
    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()

    output_dir = build_output_dir(args)
    output_dir = output_dir.with_name(output_dir.name.replace("FixTest", "StreamReconstruct", 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False, sort_keys=True)
    logger.configure(dir=str(output_dir))

    data = get_dataset_loader(
        data_dir=args.data_dir,
        batch_size=1,
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

    summary = {
        "model_path": str(args.model_path),
        "model_source": model_source,
        "data_dir": str(args.data_dir),
        "data_split": str(args.data_split),
        "task_mode": str(args.task_mode),
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
        # 完整补全的数值输出仍然有效；缺视频后端时只跳过 MP4，避免影响离线指标计算。
        render_enabled = False
        render_disabled_reason = "缺少 matplotlib 或可用的 MP4 导出后端，本次仅保存 stream_outputs.npz。"
        logger.log(render_disabled_reason)

    for batch_index, raw_batch in enumerate(tqdm(data, desc="Stream reconstruct", unit="sample")):
        batch = move_batch_to_device(raw_batch, device)
        reference = batch["x"][0]
        conditioned_input = batch.get("conditioned_x", batch["x"])[0]
        task_inpaint_mask = batch["inpaint_mask"][0].bool()
        valid_frame_mask = batch["valid_frame_mask"][0].bool()

        with torch.no_grad():
            reconstructed = sample_streaming_sequence(
                model=model,
                diffusion=diffusion,
                args=args,
                reference=reference,
                conditioned_reference=conditioned_input,
                task_inpaint_mask=task_inpaint_mask,
                valid_frame_mask=valid_frame_mask,
            )
            conditioned_reference = build_previous_frame_conditioned_sequence(
                reference=reference,
                conditioned_reference=conditioned_input,
                reconstructed=reconstructed,
                task_inpaint_mask=task_inpaint_mask,
                valid_frame_mask=valid_frame_mask,
            )

        keyid = get_batch_string_item(batch, "keyid", 0)
        sample_name = f"{processed_count:05d}_{sanitize_path_token(keyid or f'batch{batch_index}')}"
        sample_dir = output_dir / sample_name

        reference_motion = denormalize_x277_sequence(reference, motion_normalizer)
        conditioned_motion = denormalize_x277_sequence(conditioned_reference, motion_normalizer)
        reconstructed_motion = denormalize_x277_sequence(reconstructed, motion_normalizer)
        reconstructed_motion = derive_tracker_features_from_body(
            reconstructed_motion,
            target_fps=float(args.x277_fps),
        )
        sensor_missing_labels = batch["sensor_missing_labels"][0].transpose(0, 1).cpu().numpy()
        inpaint_mask = task_inpaint_mask.transpose(0, 1).cpu().numpy()
        valid_frame_mask_np = valid_frame_mask.cpu().numpy()
        valid_length = int(valid_frame_mask.sum().item())
        target_start = get_batch_int_item(batch, "target_start", 0, valid_length - 1)
        target_length = get_batch_int_item(batch, "target_length", 0, 1)

        render_meta = None
        rendered = False
        render_error = render_disabled_reason
        if render_enabled and (render_all or visualized_count < int(args.visualize_num)):
            try:
                render_meta = render_full_reconstruction_visualization(
                    reference_motion=reference_motion,
                    conditioned_motion=conditioned_motion,
                    reconstructed_motion=reconstructed_motion,
                    sensor_missing_labels=sensor_missing_labels,
                    inpaint_mask=inpaint_mask,
                    output_path=sample_dir / "reconstruction.mp4",
                    fps=float(args.visualize_fps),
                    x277_fps=float(args.x277_fps),
                    title=keyid or sample_name,
                    valid_length=int(valid_frame_mask.sum().item()),
                )
                rendered = True
                render_error = ""
                visualized_count += 1
            except Exception as exc:
                # 单条视频失败不应吞掉完整补全结果；保留错误信息，方便之后单独补渲染。
                render_failed_count += 1
                render_error = f"{type(exc).__name__}: {exc}"
                logger.log(f"render failed for {keyid or sample_name}: {render_error}")

        metadata = {
            "sample_name": sample_name,
            "task_id": keyid,
            "source_path": get_batch_string_item(batch, "source_path", 0),
            "task_mode": get_batch_string_item(batch, "task_mode", 0),
            "schema_name": get_batch_string_item(batch, "schema_name", 0),
            "valid_length": valid_length,
            "target_start": target_start,
            "target_length": target_length,
            "model_path": str(args.model_path),
            "model_source": model_source,
            "output_dir": str(sample_dir),
            "rendered": rendered,
            "render_meta": render_meta or {},
            "render_error": render_error,
            "visualize_num": int(args.visualize_num),
            "fps": float(args.visualize_fps),
            "x277_fps": float(args.x277_fps),
            "tracker_target_mode": "derived_from_body",
        }
        save_stream_artifacts(
            sample_dir=sample_dir,
            reference_motion=reference_motion,
            conditioned_motion=conditioned_motion,
            reconstructed_motion=reconstructed_motion,
            sensor_missing_labels=sensor_missing_labels,
            inpaint_mask=inpaint_mask,
            valid_frame_mask=valid_frame_mask_np,
            metadata=metadata,
        )
        summary["samples"].append(
            {
                "sample_name": sample_name,
                "task_id": keyid,
                "sample_dir": str(sample_dir),
                "rendered": rendered,
                "render_error": render_error,
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
