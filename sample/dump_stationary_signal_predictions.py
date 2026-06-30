from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.sensor_masking import (
    REALTIME_POSE_SCHEMA_NAME,
    TRACKER_MASK_POLICY_AUTO,
    TRACKER_MASK_POLICY_DYNAMIC_CATEGORIES,
    TRACKER_MASK_POLICY_TASK,
    get_schema_spec,
)
from sample.ik_initializer import resolve_ik_init_timestep
from sample.reconstruct_stream import (
    build_ik_init_image_for_batch,
    inverse_normalized_features,
    reconstruct_batch,
    tensor_bct_to_numpy_btc,
)
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
    str2bool,
)


class StationaryHeadCaptureModel:
    """采样时对模型做薄包装：sampler 仍只看 motion，同时记录最后一次 head 概率。"""

    def __init__(self, model):
        self.model = model
        self.last_stationary_prob_5: torch.Tensor | None = None

    def __getattr__(self, name: str):
        return getattr(self.model, name)

    def __call__(self, x, timesteps, **kwargs):
        call_kwargs = dict(kwargs)
        call_kwargs["return_stationary_head"] = True
        raw_output = self.model(x, timesteps, **call_kwargs)
        if not isinstance(raw_output, dict):
            self.last_stationary_prob_5 = None
            return raw_output
        logits = raw_output.get("stationary_logits")
        if logits is None:
            self.last_stationary_prob_5 = None
        else:
            self.last_stationary_prob_5 = torch.sigmoid(logits.detach()).clamp(0.0, 1.0)
        return raw_output["motion"]


def save_stationary_prediction_payload(
    *,
    path: Path,
    schema_name: str,
    reference_features_raw: np.ndarray,
    reconstructed_features_raw: np.ndarray,
    reference_stationary_prob_5: np.ndarray,
    feature_stationary_prob_5: np.ndarray,
    head_stationary_prob_5: np.ndarray | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": np.asarray(schema_name),
        "reference_features_raw": np.asarray(reference_features_raw, dtype=np.float32),
        "reconstructed_features_raw": np.asarray(reconstructed_features_raw, dtype=np.float32),
        "reference_stationary_prob_5": np.asarray(reference_stationary_prob_5, dtype=np.float32),
        "feature_stationary_prob_5": np.asarray(feature_stationary_prob_5, dtype=np.float32),
    }
    if head_stationary_prob_5 is not None:
        payload["head_stationary_prob_5"] = np.asarray(head_stationary_prob_5, dtype=np.float32)
    np.savez(path, **payload)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump stationary signal predictions for feature/head comparison.",
        allow_abbrev=False,
    )
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    parser.add_argument("--max_batches", default=16, type=int)
    parser.add_argument("--capture_stationary_head", default=True, type=str2bool)
    return parser


def _to_single_item_batch(item: dict, device: torch.device) -> dict:
    return {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in item.items()
    }


def _stationary_head_sequence(
    capture_model: StationaryHeadCaptureModel | None,
    feature_stationary_prob_5: np.ndarray,
    target_frame: int,
) -> np.ndarray | None:
    if capture_model is None or capture_model.last_stationary_prob_5 is None:
        return None
    head = capture_model.last_stationary_prob_5.detach().cpu().numpy().astype(np.float32)
    if head.shape != (feature_stationary_prob_5.shape[0], feature_stationary_prob_5.shape[2]):
        raise ValueError(
            "stationary head output must be [B,5], "
            f"got {head.shape} for feature stationary shape {feature_stationary_prob_5.shape}"
        )
    head_stationary = np.zeros_like(feature_stationary_prob_5)
    head_stationary[:, target_frame, :] = head
    return head_stationary


def main(argv: list[str] | None = None) -> dict[str, Path | list[Path]]:
    parser = build_arg_parser()
    args = parse_and_load_runtime_schema_from_model(parser, argv=argv)
    if not str(getattr(args, "output_dir", "") or "").strip():
        raise ValueError("--output_dir is required")

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
    model, diffusion = create_model_and_diffusion(args)
    model, source = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    capture_model = StationaryHeadCaptureModel(model) if args.capture_stationary_head else None
    sampling_model = capture_model if capture_model is not None else model

    schema = get_schema_spec(args.schema)
    stationary_slice = schema.stationary_prob_slice()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = getattr(dataset, "normalizer", None)
    written_paths: list[Path] = []
    max_batches = min(max(0, int(args.max_batches)), len(dataset))

    for index in range(max_batches):
        batch = _to_single_item_batch(dataset[index], device=device)
        ik_init_image = build_ik_init_image_for_batch(
            batch,
            device=device,
            schema_name=args.schema,
            normalizer=normalizer,
            ik_init_mode=args.ik_init_mode,
            ik_init_iterations=args.ik_init_iterations,
            ik_init_lr=args.ik_init_lr,
            ik_init_pos_weight=args.ik_init_pos_weight,
            ik_init_rot_weight=args.ik_init_rot_weight,
            ik_init_reg_weight=args.ik_init_reg_weight,
            ik_init_delta_limit=args.ik_init_delta_limit,
        )
        ik_start_timestep = (
            resolve_ik_init_timestep(diffusion, args.ik_init_timestep)
            if ik_init_image is not None
            else None
        )
        reconstructed = reconstruct_batch(
            sampling_model,
            diffusion,
            batch,
            device=device,
            use_ddim=str(args.ts_respace).startswith("ddim"),
            schema_name=args.schema,
            init_image=ik_init_image,
            start_timestep=ik_start_timestep,
        )
        reference_raw = inverse_normalized_features(tensor_bct_to_numpy_btc(batch["x"]), normalizer=normalizer)
        reconstructed_raw = inverse_normalized_features(tensor_bct_to_numpy_btc(reconstructed), normalizer=normalizer)
        reference_stationary = reference_raw[:, :, stationary_slice]
        feature_stationary = reconstructed_raw[:, :, stationary_slice]
        head_stationary = _stationary_head_sequence(
            capture_model,
            feature_stationary_prob_5=feature_stationary,
            target_frame=schema.target_start,
        )
        output_path = output_dir / f"stationary_predictions_{index:06d}.npz"
        save_stationary_prediction_payload(
            path=output_path,
            schema_name=schema.name,
            reference_features_raw=reference_raw,
            reconstructed_features_raw=reconstructed_raw,
            reference_stationary_prob_5=reference_stationary,
            feature_stationary_prob_5=feature_stationary,
            head_stationary_prob_5=head_stationary,
        )
        written_paths.append(output_path)

    print(
        f"[dump_stationary_signal_predictions] wrote {len(written_paths)} files "
        f"from weights={source} to {output_dir}"
    )
    return {"output_dir": output_dir, "output_paths": written_paths}


if __name__ == "__main__":
    main()
