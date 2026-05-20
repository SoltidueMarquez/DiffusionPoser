from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNITY_MODEL_DIR = PROJECT_ROOT.parent / "SIGGRAPH2024Unity" / "Assets" / "Projects" / "RealtimePose" / "Models" / "DiffusionPoser"
BODY_ROTATION_DIM = 144
FEATURE_DIM = 283
SEQUENCE_LENGTH = 11
TRAIN_STEP_COUNT = 50


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a Unity Sentis IO dump against ONNXRuntime.")
    parser.add_argument("dump_dir", type=Path, help="Unity dump directory printed by RealtimePose.")
    parser.add_argument("--onnx_path", type=Path, default=DEFAULT_UNITY_MODEL_DIR / "diffusionposer_denoiser.onnx")
    parser.add_argument("--normalizer_json", type=Path, default=DEFAULT_UNITY_MODEL_DIR / "normalizer.json")
    parser.add_argument("--ddim_schedule_json", type=Path, default=DEFAULT_UNITY_MODEL_DIR / "ddim_schedule.json")
    parser.add_argument("--simulate_steps", type=int, nargs="*", default=[5, 10])
    parser.add_argument("--dataset_dir", type=Path, default=PROJECT_ROOT / "dataset" / "AMASS_current277_60hz")
    parser.add_argument("--dataset_files", type=int, default=200)
    return parser


def read_float32(path: Path, count: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size != count:
        raise ValueError(f"{path} should contain {count} float32 values, got {values.size}")
    return values.astype(np.float32, copy=False)


def load_normalizer(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("enabled", False):
        return None
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,):
        raise ValueError(f"normalizer shape mismatch: mean={mean.shape}, std={std.shape}")
    return mean, std


def denormalize_last_frame(window: np.ndarray, normalizer: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    frame = window.reshape(FEATURE_DIM, SEQUENCE_LENGTH)[:, -1].astype(np.float32, copy=True)
    if normalizer is None:
        return frame
    mean, std = normalizer
    return frame * std + mean


def body6d_summary(frame: np.ndarray) -> str:
    # body rotation 采用每骨骼 6D forward/up 表示；这里检查长度和正交性来定位模型输出是否物理可解。
    rotations = frame[:BODY_ROTATION_DIM].reshape(24, 6)
    forward = rotations[:, :3]
    up = rotations[:, 3:6]
    forward_norm = np.linalg.norm(forward, axis=1)
    up_norm = np.linalg.norm(up, axis=1)
    denom = np.maximum(forward_norm * up_norm, 1e-8)
    abs_dot = np.abs(np.sum(forward * up, axis=1) / denom)
    worst = int(abs_dot.argmax())
    return (
        f"forwardNorm=[{forward_norm.min():.4g},{forward_norm.max():.4g}] "
        f"upNorm=[{up_norm.min():.4g},{up_norm.max():.4g}] "
        f"maxAbsDot={abs_dot[worst]:.4g}(bone#{worst})"
    )


def print_window_summary(name: str, window: np.ndarray, normalizer: tuple[np.ndarray, np.ndarray] | None) -> None:
    frame = denormalize_last_frame(window, normalizer)
    print(f"{name}: {body6d_summary(frame)}")


def print_feature_distribution_comparison(
    *,
    conditioned: np.ndarray,
    normalizer: tuple[np.ndarray, np.ndarray] | None,
    dataset_dir: Path,
    dataset_files: int,
) -> None:
    if normalizer is None:
        unity_denorm = conditioned[0]
    else:
        mean, std = normalizer
        unity_denorm = conditioned[0] * std[:, None] + mean[:, None]

    groups = {
        "body_rot_last": slice(0, 144),
        "body_vel_last": slice(144, 216),
        "tracker_pos_last": slice(216, 234),
        "tracker_rot_last": slice(234, 270),
        "root_last": slice(270, 273),
        "contact_last": slice(273, 277),
    }
    print("unity_conditioned_abs_mean_max:")
    for name, feature_slice in groups.items():
        values = np.abs(unity_denorm[feature_slice, -1])
        print(f"  {name}: mean={values.mean():.5g}, max={values.max():.5g}")

    if not dataset_dir.exists():
        print(f"dataset_distribution: skipped, missing {dataset_dir}")
        return

    files = sorted(dataset_dir.rglob("*.npz"))[: max(0, int(dataset_files))]
    if not files:
        print(f"dataset_distribution: skipped, no npz files under {dataset_dir}")
        return

    collected: dict[str, list[np.ndarray]] = {name: [] for name in groups}
    for path in files:
        with np.load(path) as data:
            x277 = np.asarray(data["x"], dtype=np.float32)
        if x277.ndim != 2 or x277.shape[1] < 277 or x277.shape[0] <= 0:
            continue
        frame_indices = np.linspace(0, x277.shape[0] - 1, min(20, x277.shape[0])).astype(np.int64)
        sampled = x277[frame_indices]
        for name, feature_slice in groups.items():
            collected[name].append(np.abs(sampled[:, feature_slice]).reshape(-1))

    print("dataset_abs_p50_p95_p99_max:")
    for name in groups:
        if not collected[name]:
            print(f"  {name}: no-data")
            continue
        values = np.concatenate(collected[name])
        print(
            f"  {name}: p50={np.percentile(values, 50):.5g}, "
            f"p95={np.percentile(values, 95):.5g}, "
            f"p99={np.percentile(values, 99):.5g}, max={values.max():.5g}"
        )


def load_alphas(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = np.asarray(payload["alphasCumprod"], dtype=np.float32)
    if values.shape[0] < TRAIN_STEP_COUNT:
        raise ValueError(f"DDIM schedule has too few alpha values: {values.shape[0]}")
    return values


def build_unity_runtime_steps(sample_steps: int, alphas: np.ndarray) -> list[tuple[int, float, float]]:
    # 这里复现 Unity DdimSchedule.BuildRuntimeSteps，用于判断 Unity dump 的输入在 Python ONNXRuntime 下是否也会坏。
    step_count = max(1, min(int(sample_steps), TRAIN_STEP_COUNT))
    stride = TRAIN_STEP_COUNT / float(step_count)
    previous_timestep = -1
    steps: list[tuple[int, float, float]] = []
    for i in range(step_count):
        timestep = round((i + 1) * stride) - 1
        timestep = max(0, min(TRAIN_STEP_COUNT - 1, timestep))
        if timestep <= previous_timestep:
            timestep = min(previous_timestep + 1, TRAIN_STEP_COUNT - 1)
        previous_alpha = 1.0 if i == 0 else float(alphas[steps[i - 1][0]])
        steps.append((timestep, float(alphas[timestep]), previous_alpha))
        previous_timestep = timestep
    return steps


def apply_unity_inpaint_constraints(
    sample: np.ndarray,
    conditioned: np.ndarray,
    inpaint_mask: np.ndarray,
    valid_frame_mask: np.ndarray,
) -> np.ndarray:
    valid = valid_frame_mask.reshape(1, 1, SEQUENCE_LENGTH) > 0.5
    return np.where(valid, np.where(inpaint_mask > 0.5, sample, conditioned), 0.0).astype(np.float32)


def simulate_unity_ddim_with_onnxruntime(
    session,
    *,
    initial_sample: np.ndarray,
    conditioned: np.ndarray,
    inpaint_mask: np.ndarray,
    valid_frame_mask: np.ndarray,
    alphas: np.ndarray,
    sample_steps: int,
    clip_denoised: bool,
) -> np.ndarray:
    sample = initial_sample.astype(np.float32, copy=True)
    runtime_steps = build_unity_runtime_steps(sample_steps=sample_steps, alphas=alphas)
    print(f"simulate_steps={sample_steps} timesteps={[step[0] for step in runtime_steps]} clip_denoised={clip_denoised}")
    for step_index in range(len(runtime_steps) - 1, -1, -1):
        timestep, alpha, previous_alpha_from_schedule = runtime_steps[step_index]
        pred_x0 = session.run(
            None,
            {
                "x_t": sample,
                "timestep": np.asarray([float(timestep)], dtype=np.float32),
                "inpaint_mask": inpaint_mask,
                "valid_frame_mask": valid_frame_mask,
            },
        )[0].astype(np.float32)
        if clip_denoised:
            pred_x0 = np.clip(pred_x0, -1.0, 1.0)
        previous_alpha = 1.0 if step_index == 0 else previous_alpha_from_schedule
        sqrt_alpha = np.sqrt(max(alpha, 1e-8))
        sqrt_one_minus_alpha = np.sqrt(max(1.0 - alpha, 1e-8))
        eps = (sample - sqrt_alpha * pred_x0) / sqrt_one_minus_alpha
        sample = np.sqrt(previous_alpha) * pred_x0 + np.sqrt(max(1.0 - previous_alpha, 0.0)) * eps
        sample = apply_unity_inpaint_constraints(
            sample=sample,
            conditioned=conditioned,
            inpaint_mask=inpaint_mask,
            valid_frame_mask=valid_frame_mask,
        )
    return sample.astype(np.float32)


def main() -> None:
    args = build_arg_parser().parse_args()
    dump_dir = args.dump_dir
    meta = json.loads((dump_dir / "meta.json").read_text(encoding="utf-8"))
    feature_dim = int(meta.get("featureDim", FEATURE_DIM))
    sequence_length = int(meta.get("sequenceLength", SEQUENCE_LENGTH))
    if feature_dim != FEATURE_DIM or sequence_length != SEQUENCE_LENGTH:
        raise ValueError(f"unexpected dump shape: C={feature_dim}, T={sequence_length}")

    count = FEATURE_DIM * SEQUENCE_LENGTH
    x_t = read_float32(dump_dir / "x_t_feature_major.bin", count).reshape(1, FEATURE_DIM, SEQUENCE_LENGTH)
    inpaint_mask = read_float32(dump_dir / "inpaint_mask_feature_major.bin", count).reshape(1, FEATURE_DIM, SEQUENCE_LENGTH)
    valid_frame_mask = read_float32(dump_dir / "valid_frame_mask.bin", SEQUENCE_LENGTH).reshape(1, SEQUENCE_LENGTH)
    unity_pred = read_float32(dump_dir / "pred_x0_feature_major.bin", count).reshape(1, FEATURE_DIM, SEQUENCE_LENGTH)
    conditioned = read_float32(dump_dir / "conditioned_window_feature_major.bin", count).reshape(1, FEATURE_DIM, SEQUENCE_LENGTH)
    normalizer = load_normalizer(args.normalizer_json)
    alphas = load_alphas(args.ddim_schedule_json)

    print(f"dump_dir={dump_dir}")
    print(f"timestep={float(meta['timestep'])}")
    print(f"valid_frame_mask={valid_frame_mask.astype(int).tolist()[0]}")
    last_mask = inpaint_mask[0, :, -1] > 0.5
    print(
        "last_frame_mask_counts: "
        f"total={int(last_mask.sum())}, body_vel={int(last_mask[:216].sum())}, "
        f"root={int(last_mask[270:273].sum())}, contact={int(last_mask[273:277].sum())}, "
        f"labels={int(last_mask[277:283].sum())}"
    )
    print_window_summary("unity_conditioned", conditioned[0], normalizer)
    print_window_summary("unity_x_t", x_t[0], normalizer)
    print_window_summary("unity_sentis_pred_x0", unity_pred[0], normalizer)
    print_feature_distribution_comparison(
        conditioned=conditioned,
        normalizer=normalizer,
        dataset_dir=args.dataset_dir,
        dataset_files=args.dataset_files,
    )

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed in this environment.") from exc

    session = ort.InferenceSession(str(args.onnx_path), providers=["CPUExecutionProvider"])
    ort_pred = session.run(
        None,
        {
            "x_t": x_t,
            "timestep": np.asarray([float(meta["timestep"])], dtype=np.float32),
            "inpaint_mask": inpaint_mask,
            "valid_frame_mask": valid_frame_mask,
        },
    )[0].astype(np.float32)

    diff = np.abs(ort_pred - unity_pred)
    print_window_summary("onnxruntime_pred_x0", ort_pred[0], normalizer)
    print(f"sentis_vs_onnx: max_abs={diff.max():.6g}, mean_abs={diff.mean():.6g}, p99={np.percentile(diff, 99):.6g}")
    flat = diff.reshape(-1)
    top_indices = np.argsort(flat)[-8:][::-1]
    print("top_abs_diff_indices=" + ", ".join(f"{int(i)}:{flat[i]:.6g}" for i in top_indices))

    for sample_steps in args.simulate_steps:
        for clip_denoised in (False, True):
            simulated = simulate_unity_ddim_with_onnxruntime(
                session,
                initial_sample=x_t,
                conditioned=conditioned,
                inpaint_mask=inpaint_mask,
                valid_frame_mask=valid_frame_mask,
                alphas=alphas,
                sample_steps=sample_steps,
                clip_denoised=clip_denoised,
            )
            print_window_summary(f"onnxruntime_unity_ddim{sample_steps}_clip{clip_denoised}", simulated[0], normalizer)


if __name__ == "__main__":
    main()
