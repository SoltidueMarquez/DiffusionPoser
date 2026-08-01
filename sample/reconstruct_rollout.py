from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_np,
    global_head_rotations_to_local_delta_6d_np,
    reexpress_pose_target_between_head_yaws_torch,
)
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np, rotation_6d_to_matrix_np
from sample.reconstruct_stream import inverse_normalized_target, reconstruct_batch
from sample.realtime_pose_runtime import decode_and_resolve_pose
from sample.utils import load_checkpoint_model
from utils import dist_util
from utils.model_util import create_model_and_diffusion
from utils.parser_util import (
    add_base_options,
    add_data_options,
    add_diffusion_options,
    add_model_options,
    add_sampling_options,
    parse_and_load_from_model,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行 140D Head-anchor 显式 rollout。")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    parser.add_argument("--rollout_limit", default=0, type=int)
    return parser


def _batch_item(item: dict, device: torch.device) -> dict:
    return {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in item.items()
        if key != "rollout"
    }


def _inject_prediction_into_next_history(
    predicted_normalized: torch.Tensor,
    current_batch: dict,
    next_batch: dict,
    normalizer,
) -> None:
    if normalizer is None:
        predicted_raw = predicted_normalized
        mean = std = None
    else:
        mean = normalizer.pose_mean.to(device=predicted_normalized.device, dtype=predicted_normalized.dtype)
        std = normalizer.pose_std.to(device=predicted_normalized.device, dtype=predicted_normalized.dtype)
        predicted_raw = predicted_normalized * std + mean
    reexpressed = reexpress_pose_target_between_head_yaws_torch(
        predicted_raw,
        current_batch["current_head_yaw_world"],
        next_batch["current_head_yaw_world"],
    )
    if mean is not None and std is not None:
        reexpressed = (reexpressed - mean) / std
    next_batch["pose_history"][:, -1] = reexpressed


def rollout_dataset_item(
    model,
    diffusion,
    dataset: RealtimePoseTaskDataset,
    index: int,
    device: torch.device,
    use_ddim: bool,
) -> dict[str, np.ndarray]:
    item = dataset[index]
    sequence = [item, *item.get("rollout", [])]
    reconstructed_raw: list[np.ndarray] = []
    reference_raw: list[np.ndarray] = []
    predicted_local_delta: list[np.ndarray] = []
    reference_local_delta: list[np.ndarray] = []
    predicted_roots_world: list[np.ndarray] = []
    reference_roots_world: list[np.ndarray] = []
    root_yaws: list[float] = []
    reference_root_yaws: list[float] = []
    hip_heights: list[float] = []
    reference_hip_heights: list[float] = []
    predicted_joints_world: list[np.ndarray] = []
    reference_joints_world: list[np.ndarray] = []
    known_masks: list[np.ndarray] = []
    tracker_positions_world: list[np.ndarray] = []
    configured_values: list[np.ndarray] = []
    measured_values: list[np.ndarray] = []
    missing_ages: list[np.ndarray] = []
    scenarios: list[str] = []
    known_errors: list[float] = []
    previous_prediction: torch.Tensor | None = None
    previous_batch: dict | None = None

    for step_item in sequence:
        batch = _batch_item(step_item, device)
        if previous_prediction is not None and previous_batch is not None:
            _inject_prediction_into_next_history(
                previous_prediction,
                previous_batch,
                batch,
                dataset.normalizer,
            )
        predicted = reconstruct_batch(model, diffusion, batch, device, use_ddim=use_ddim)
        pred_raw = inverse_normalized_target(predicted, dataset.normalizer)[0]
        ref_raw = inverse_normalized_target(batch["x"], dataset.normalizer)[0]
        tracker_current = np.concatenate(
            [
                batch["current_tracker_pos_head_ref"][0].cpu().numpy(),
                batch["current_tracker_rot_head_ref_6d"][0].cpu().numpy(),
                batch["configured"][0, -1, :, None].cpu().numpy(),
                batch["measured_valid"][0, -1, :, None].cpu().numpy(),
                batch["missing_age_norm"][0, -1, :, None].cpu().numpy(),
            ],
            axis=-1,
        )
        resolved = decode_and_resolve_pose(
            pred_raw,
            tracker_current,
            float(batch["current_head_yaw_world"][0].item()),
            batch["current_head_position_world"][0].cpu().numpy(),
            float(batch["floor_y"][0].item()),
            batch["joint_offsets_parent"][0].cpu().numpy(),
            batch["joint_rest_local_rotations_6d"][0].cpu().numpy(),
        )
        reconstructed_raw.append(pred_raw)
        reference_raw.append(ref_raw)
        head_yaw = float(batch["current_head_yaw_world"][0].item())
        head_position = batch["current_head_position_world"][0].cpu().numpy()
        floor_y = float(batch["floor_y"][0].item())
        origin = np.asarray([head_position[0], floor_y, head_position[2]], dtype=np.float32)
        yaw_rotation = make_yaw_rotation_np(np.asarray([head_yaw], dtype=np.float32))[0]
        predicted_roots_world.append(resolved.root_position_world)
        reference_roots_world.append(
            origin + yaw_rotation @ batch["target_root_position_head_ref"][0].cpu().numpy()
        )
        root_yaws.append(resolved.root_yaw_world)
        reference_root_yaws.append(float(batch["target_root_yaw_world"][0].item()))
        hip_heights.append(resolved.hip_height)
        reference_hip_heights.append(float(batch["target_hip_height"][0].item()))
        predicted_joints_world.append(resolved.joints_world)
        reference_joints_world.append(
            origin[None]
            + np.einsum(
                "ij,aj->ai",
                yaw_rotation,
                batch["target_joints_head_ref"][0].cpu().numpy(),
            )
        )
        predicted_local_delta.append(resolved.body_local_delta_6d)
        rest_rotations = rotation_6d_to_matrix_np(
            batch["joint_rest_local_rotations_6d"][0].cpu().numpy()
        )
        reference_head_rotations, _ = decode_target_head_rotations_np(ref_raw, rest_rotations)
        reference_local_delta.append(
            global_head_rotations_to_local_delta_6d_np(reference_head_rotations, rest_rotations)
        )
        known_masks.append(batch["known_mask"][0].cpu().numpy())
        tracker_positions_world.append(
            origin[None]
            + np.einsum(
                "ij,aj->ai",
                yaw_rotation,
                batch["current_tracker_pos_head_ref"][0].cpu().numpy(),
            )
        )
        configured_values.append(batch["configured"][0, -1].cpu().numpy())
        measured_values.append(batch["measured_valid"][0, -1].cpu().numpy())
        missing_ages.append(batch["missing_age"][0, -1].cpu().numpy())
        scenarios.append(str(step_item.get("scenario", "")))
        known_errors.append(resolved.known_rotation_max_error)
        previous_prediction = predicted
        previous_batch = batch

    return {
        "reference_target_raw": np.asarray(reference_raw, dtype=np.float32),
        "reconstructed_target_raw": np.asarray(reconstructed_raw, dtype=np.float32),
        "reference_body_local_delta_6d": np.asarray(reference_local_delta, dtype=np.float32),
        "predicted_body_local_delta_6d": np.asarray(predicted_local_delta, dtype=np.float32),
        "reference_joints_world": np.asarray(reference_joints_world, dtype=np.float32),
        "predicted_joints_world": np.asarray(predicted_joints_world, dtype=np.float32),
        "reference_root_position_world": np.asarray(reference_roots_world, dtype=np.float32),
        "predicted_root_position_world": np.asarray(predicted_roots_world, dtype=np.float32),
        "reference_root_yaw_world": np.asarray(reference_root_yaws, dtype=np.float32),
        "predicted_root_yaw_world": np.asarray(root_yaws, dtype=np.float32),
        "reference_hip_height": np.asarray(reference_hip_heights, dtype=np.float32),
        "predicted_hip_height": np.asarray(hip_heights, dtype=np.float32),
        "known_mask": np.asarray(known_masks, dtype=bool),
        "tracker_pos_world": np.asarray(tracker_positions_world, dtype=np.float32),
        "configured": np.asarray(configured_values, dtype=bool),
        "measured_valid": np.asarray(measured_values, dtype=bool),
        "missing_age": np.asarray(missing_ages, dtype=np.int64),
        "scenario": np.asarray(scenarios),
        "eval_frame_mask": np.ones(len(sequence), dtype=bool),
        "known_rotation_max_error": np.asarray(known_errors, dtype=np.float32),
    }


def save_rollout(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = parse_and_load_from_model(build_arg_parser(), argv=argv)
    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    dataset = RealtimePoseTaskDataset(
        data_dir=args.data_dir,
        split=args.data_split,
        seq_len=args.seq_len,
        normalizer_dir=args.normalizer_dir,
        normalize_input=args.normalize_input,
        folder_path=getattr(args, "folder_path", "") or None,
        enable_rollout=args.rollout_steps > 1,
        rollout_steps=args.rollout_steps,
    )
    model, diffusion = create_model_and_diffusion(args)
    model, source = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    limit = len(dataset) if int(args.rollout_limit) <= 0 else min(len(dataset), int(args.rollout_limit))
    payloads = [rollout_dataset_item(model, diffusion, dataset, index, device, True) for index in range(limit)]
    payload = {
        key: np.asarray([item[key] for item in payloads])
        for key in payloads[0]
    }
    payload["fps"] = np.float32(60.0)
    output_dir = Path(args.output_dir or "output/realtime_pose_140d_rollout").resolve()
    output_path = output_dir / "rollout_result.npz"
    save_rollout(output_path, payload)
    print(f"[reconstruct_rollout] weights={source} output={output_path}")
    return {"output_path": output_path}


if __name__ == "__main__":
    main()
