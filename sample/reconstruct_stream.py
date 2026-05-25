from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.sensor_masking import (
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    TRACKER_MASK_POLICY_AUTO,
    TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
    TRACKER_MASK_POLICY_TASK,
    get_schema_spec,
)
from sample.utils import choose_sampler, load_checkpoint_model
from utils import dist_util
from utils.model_util import create_model_and_diffusion
from utils.parser_util import add_base_options, add_data_options, add_diffusion_options, add_model_options, add_sampling_options, parse_and_load_from_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample realtime_pose_v2 single-frame reconstruction tasks.")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    return parser


def build_realtime_inpaint_mask(
    batch_size: int,
    device: torch.device,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> torch.Tensor:
    schema = get_schema_spec(schema_name)
    mask = torch.zeros(batch_size, schema.feature_dim, REALTIME_POSE_SEQ_LEN, dtype=torch.bool, device=device)
    mask[:, schema.target_slice(), REALTIME_POSE_TARGET_START] = True
    return mask


def reconstruct_batch(
    model,
    diffusion,
    batch: dict,
    device: torch.device,
    use_ddim: bool = True,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> torch.Tensor:
    schema = get_schema_spec(schema_name)
    sample = batch["conditioned_x"].to(device)
    batch_size = sample.shape[0]
    if tuple(sample.shape[1:]) != (schema.feature_dim, REALTIME_POSE_SEQ_LEN):
        raise ValueError(f"{schema.name} sample 应为 [B,{schema.feature_dim},61]，实际为 {tuple(sample.shape)}")
    inpaint_mask = build_realtime_inpaint_mask(batch_size, device, schema_name=schema.name)
    valid_frame_mask = batch["valid_frame_mask"].to(device)
    model_kwargs = {
        "inpaint_cond": inpaint_mask,
        "valid_frame_mask": valid_frame_mask,
        "attention_mask": valid_frame_mask,
        "y": {
            "mask": inpaint_mask,
            "inpainted_motion": sample,
            "schema_name": schema.name,
        },
    }
    sampler = choose_sampler(diffusion, use_ddim=use_ddim)
    with torch.no_grad():
        reconstructed = sampler(
            model,
            shape=sample.shape,
            noise=None,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
    return torch.where(inpaint_mask, reconstructed, sample)


def tensor_bct_to_numpy_btc(tensor: torch.Tensor) -> np.ndarray:
    """把模型内部 `[B, C, T]` 特征转成文件里使用的 `[B, T, C]`。"""

    return tensor.detach().cpu().numpy().transpose(0, 2, 1).astype(np.float32, copy=False)


def inverse_normalized_features(features: np.ndarray, normalizer) -> np.ndarray:
    if normalizer is None:
        return features
    return np.asarray(normalizer.inverse(features), dtype=np.float32)


def save_reconstruction(
    path: Path,
    reference: torch.Tensor,
    conditioned: torch.Tensor,
    reconstructed: torch.Tensor,
    inpaint_mask: torch.Tensor,
    normalizer=None,
    schema_name: str = REALTIME_POSE_SCHEMA_NAME,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_normalizer = normalizer is not None
    reference_input = tensor_bct_to_numpy_btc(reference)
    conditioned_input = tensor_bct_to_numpy_btc(conditioned)
    reconstructed_input = tensor_bct_to_numpy_btc(reconstructed)
    reference_raw = inverse_normalized_features(reference_input, normalizer=normalizer)
    conditioned_raw = inverse_normalized_features(conditioned_input, normalizer=normalizer)
    reconstructed_raw = inverse_normalized_features(reconstructed_input, normalizer=normalizer)
    payload = {
        "schema_name": np.asarray(schema_name),
        "feature_space": np.asarray("raw"),
        "input_feature_space": np.asarray("normalized" if has_normalizer else "raw"),
        # 兼容已有评估读取字段，但现在明确写 raw，避免评估默认落在归一化空间。
        "reference_features": reference_raw,
        "conditioned_features": conditioned_raw,
        "reconstructed_features": reconstructed_raw,
        "reference_features_raw": reference_raw,
        "conditioned_features_raw": conditioned_raw,
        "reconstructed_features_raw": reconstructed_raw,
        "inpaint_mask": inpaint_mask.detach().cpu().numpy().transpose(0, 2, 1),
    }
    if has_normalizer:
        payload.update(
            reference_features_normalized=reference_input,
            conditioned_features_normalized=conditioned_input,
            reconstructed_features_normalized=reconstructed_input,
        )
    np.savez(path, **payload)


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = build_arg_parser()
    args = parse_and_load_from_model(parser, argv=argv)
    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()

    sample_mask_policy = (
        TRACKER_MASK_POLICY_TASK
        if args.tracker_mask_policy in {TRACKER_MASK_POLICY_AUTO, TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES}
        else args.tracker_mask_policy
    )
    dataset = RealtimePoseTaskDataset(
        data_dir=args.data_dir,
        split=args.data_split,
        seq_len=args.seq_len,
        normalizer_dir=args.normalizer_dir,
        normalize_input=args.normalize_input,
        folder_path=getattr(args, "folder_path", "") or None,
        schema_name=args.schema,
        tracker_mask_policy=sample_mask_policy,
        tracker_mask_seed=args.tracker_mask_seed,
        tracker_mask_fill=args.tracker_mask_fill,
        tracker_mask_categories=args.tracker_mask_categories,
    )
    batch = dataset[0]
    batch = {key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

    model, diffusion = create_model_and_diffusion(args)
    model, source = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    reconstructed = reconstruct_batch(
        model,
        diffusion,
        batch,
        device=device,
        use_ddim=str(args.ts_respace).startswith("ddim"),
        schema_name=args.schema,
    )
    inpaint_mask = build_realtime_inpaint_mask(1, device, schema_name=args.schema)
    output_dir = Path(args.output_dir or Path(args.model_path).with_suffix("").name).resolve()
    output_path = output_dir / "realtime_pose_reconstruction.npz"
    save_reconstruction(
        output_path,
        batch["x"],
        batch["conditioned_x"],
        reconstructed,
        inpaint_mask,
        normalizer=getattr(dataset, "normalizer", None),
        schema_name=args.schema,
    )
    print(f"[reconstruct_stream] weights={source} output={output_path}")
    return {"output_path": output_path}


if __name__ == "__main__":
    main()
