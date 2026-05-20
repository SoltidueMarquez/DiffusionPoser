from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_unity_sentis_dump import (
    DEFAULT_UNITY_MODEL_DIR,
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    body6d_summary,
    build_unity_runtime_steps,
    load_alphas,
    load_normalizer,
)
from data_loaders.x277_dataset import X277MissingTaskDataset  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exported ONNX on a real current277 materialized task window.")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_ROOT / "dataset" / "AMASS_current277_60hz_missing_tasks")
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--normalizer_dir", type=Path, default=PROJECT_ROOT / "dataset" / "meta_AMASS_current277_60hz")
    parser.add_argument("--onnx_path", type=Path, default=DEFAULT_UNITY_MODEL_DIR / "diffusionposer_denoiser.onnx")
    parser.add_argument("--normalizer_json", type=Path, default=DEFAULT_UNITY_MODEL_DIR / "normalizer.json")
    parser.add_argument("--ddim_schedule_json", type=Path, default=DEFAULT_UNITY_MODEL_DIR / "ddim_schedule.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sample_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=202405)
    return parser


def denormalize_last(window: np.ndarray, normalizer: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    frame = window.reshape(FEATURE_DIM, SEQUENCE_LENGTH)[:, -1].astype(np.float32, copy=True)
    if normalizer is None:
        return frame
    mean, std = normalizer
    return frame * std + mean


def print_summary(name: str, window: np.ndarray, normalizer: tuple[np.ndarray, np.ndarray] | None) -> None:
    print(f"{name}: {body6d_summary(denormalize_last(window, normalizer))}")


def apply_constraints(sample: np.ndarray, conditioned: np.ndarray, inpaint_mask: np.ndarray, valid_frame_mask: np.ndarray) -> np.ndarray:
    valid = valid_frame_mask.reshape(1, 1, SEQUENCE_LENGTH) > 0.5
    return np.where(valid, np.where(inpaint_mask > 0.5, sample, conditioned), 0.0).astype(np.float32)


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = X277MissingTaskDataset(
        data_dir=args.data_dir,
        split=args.data_split,
        seq_len=SEQUENCE_LENGTH,
        normalizer_dir=args.normalizer_dir,
        normalize_input=True,
        preload_data=False,
    )
    item = dataset[int(args.index)]
    conditioned = item["conditioned_x"].numpy()[None].astype(np.float32)
    inpaint_mask = item["inpaint_mask"].numpy()[None].astype(np.float32)
    valid_frame_mask = item["valid_frame_mask"].numpy()[None].astype(np.float32)
    normalizer = load_normalizer(args.normalizer_json)
    alphas = load_alphas(args.ddim_schedule_json)

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed in this environment.") from exc

    session = ort.InferenceSession(str(args.onnx_path), providers=["CPUExecutionProvider"])
    runtime_steps = build_unity_runtime_steps(args.sample_steps, alphas)
    first_timestep, first_alpha, _ = runtime_steps[-1]
    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(conditioned.shape).astype(np.float32)
    sample = np.sqrt(first_alpha) * conditioned + np.sqrt(max(1.0 - first_alpha, 0.0)) * noise
    sample = apply_constraints(sample, conditioned, inpaint_mask, valid_frame_mask)

    print(
        json.dumps(
            {
                "index": int(args.index),
                "source_path": item.get("source_path", ""),
                "keyid": item.get("keyid", ""),
                "target_start": int(item["target_start"]),
                "mask_last_count": int(inpaint_mask[0, :, -1].sum()),
                "label_last": item["sensor_missing_labels"].numpy()[:, -1].astype(int).tolist(),
                "timesteps": [int(step[0]) for step in runtime_steps],
            },
            ensure_ascii=False,
        )
    )
    print_summary("dataset_conditioned", conditioned[0], normalizer)
    print_summary("dataset_initial_x_t", sample[0], normalizer)

    pred = None
    for step_index in range(len(runtime_steps) - 1, -1, -1):
        timestep, alpha, previous_alpha_from_schedule = runtime_steps[step_index]
        pred = session.run(
            None,
            {
                "x_t": sample,
                "timestep": np.asarray([float(timestep)], dtype=np.float32),
                "inpaint_mask": inpaint_mask,
                "valid_frame_mask": valid_frame_mask,
            },
        )[0].astype(np.float32)
        previous_alpha = 1.0 if step_index == 0 else previous_alpha_from_schedule
        eps = (sample - np.sqrt(max(alpha, 1e-8)) * pred) / np.sqrt(max(1.0 - alpha, 1e-8))
        sample = np.sqrt(previous_alpha) * pred + np.sqrt(max(1.0 - previous_alpha, 0.0)) * eps
        sample = apply_constraints(sample, conditioned, inpaint_mask, valid_frame_mask)

    if pred is not None:
        print_summary("dataset_last_pred_x0", pred[0], normalizer)
    print_summary("dataset_final_sample", sample[0], normalizer)


if __name__ == "__main__":
    main()
