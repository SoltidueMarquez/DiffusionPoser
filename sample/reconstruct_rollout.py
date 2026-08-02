from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset, TaskRequest
from data_loaders.realtime_pose_geometry import (
    advance_rollout_pose_history_torch,
    decode_target_head_rotations_np,
    global_head_rotations_to_local_delta_6d_np,
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
    parser = argparse.ArgumentParser(description="执行 144D deployed-history 显式 rollout。")
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


def rollout_dataset_item(
    model,
    diffusion,
    dataset: RealtimePoseTaskDataset,
    index: int,
    device: torch.device,
    rollout_steps: int = 4,
    config_index: int = 0,
    projection_mode: str = "all_steps",
    late_steps: int = 5,
) -> dict[str, np.ndarray]:
    item = dataset[TaskRequest(index, config_index, rollout_steps)]
    sequence = [item, *item.get("rollout", [])]
    result_lists: dict[str, list] = {
        name: []
        for name in (
            "reference_target_raw",
            "raw_pred_target_raw",
            "deployed_pred_target_raw",
            "reference_body_local_delta_6d",
            "predicted_body_local_delta_6d",
            "reference_joints_world",
            "predicted_joints_world",
            "reference_root_position_world",
            "predicted_root_position_world",
            "reference_root_yaw_world",
            "predicted_root_yaw_world",
            "reference_hip_height",
            "predicted_hip_height",
            "tracker_pos_world",
            "current_tracker_raw",
            "configured",
            "measured_valid",
            "d_off",
            "d_on",
            "hard_rotation_state",
            "current_trajectory",
            "contact_target",
            "contact_logits",
            "future_leg_prediction",
            "future_leg_target",
            "scenario",
            "hard_rotation_max_error",
        )
    }
    pose_history: torch.Tensor | None = None
    previous_deployed: torch.Tensor | None = None
    previous_batch: dict | None = None

    for step_item in sequence:
        batch = _batch_item(step_item, device)
        if pose_history is None:
            pose_history = batch["pose_history"]
        else:
            assert previous_deployed is not None and previous_batch is not None
            mean = None if dataset.normalizer is None else dataset.normalizer.pose_mean
            std = None if dataset.normalizer is None else dataset.normalizer.pose_std
            pose_history = advance_rollout_pose_history_torch(
                pose_history,
                previous_deployed,
                previous_batch["history_head_yaw_world"],
                batch["history_head_yaw_world"],
                mean,
                std,
                detach_prediction=True,
            )
            batch["pose_history"] = pose_history

        reconstruction = reconstruct_batch(
            model,
            diffusion,
            batch,
            device,
            normalizer=dataset.normalizer,
            projection_mode=projection_mode,
            late_steps=late_steps,
        )
        raw_prediction = inverse_normalized_target(
            reconstruction["raw_pred_xstart"], dataset.normalizer
        )[0]
        deployed_prediction = inverse_normalized_target(
            reconstruction["deployed_pred_xstart"], dataset.normalizer
        )[0]
        reference = inverse_normalized_target(batch["x"], dataset.normalizer)[0]
        current_tracker_raw = batch["current_tracker_raw"][0].detach().cpu().numpy()
        hard = batch["hard_rotation_state"][0].detach().cpu().numpy()
        resolved = decode_and_resolve_pose(
            deployed_prediction,
            current_tracker_raw,
            hard,
            float(batch["current_head_yaw_world"][0].item()),
            batch["current_head_position_world"][0].detach().cpu().numpy(),
            float(batch["floor_y"][0].item()),
            batch["joint_offsets_parent"][0].detach().cpu().numpy(),
            batch["joint_rest_local_rotations_6d"][0].detach().cpu().numpy(),
        )
        head_yaw = float(batch["current_head_yaw_world"][0].item())
        head_position = batch["current_head_position_world"][0].detach().cpu().numpy()
        floor_y = float(batch["floor_y"][0].item())
        origin = np.asarray([head_position[0], floor_y, head_position[2]], dtype=np.float32)
        yaw_rotation = make_yaw_rotation_np(np.asarray([head_yaw], dtype=np.float32))[0]
        reference_joints_world = origin[None] + np.einsum(
            "ij,aj->ai",
            yaw_rotation,
            batch["target_joints_head_ref"][0].detach().cpu().numpy(),
        )
        reference_root_world = origin + yaw_rotation @ batch[
            "target_root_position_head_ref"
        ][0].detach().cpu().numpy()
        tracker_positions_world = origin[None] + np.einsum(
            "ij,aj->ai", yaw_rotation, current_tracker_raw[:, :3]
        )
        reference_rotations, reference_heading = decode_target_head_rotations_np(reference)
        rest_rotations = rotation_6d_to_matrix_np(
            batch["joint_rest_local_rotations_6d"][0].detach().cpu().numpy()
        )

        result_lists["reference_target_raw"].append(reference)
        result_lists["raw_pred_target_raw"].append(raw_prediction)
        result_lists["deployed_pred_target_raw"].append(deployed_prediction)
        result_lists["reference_body_local_delta_6d"].append(
            global_head_rotations_to_local_delta_6d_np(
                reference_rotations,
                root_heading_head=reference_heading,
                rest_local_rotations=rest_rotations,
            )
        )
        result_lists["predicted_body_local_delta_6d"].append(resolved.body_local_delta_6d)
        result_lists["reference_joints_world"].append(reference_joints_world)
        result_lists["predicted_joints_world"].append(resolved.joints_world)
        result_lists["reference_root_position_world"].append(reference_root_world)
        result_lists["predicted_root_position_world"].append(resolved.root_position_world)
        result_lists["reference_root_yaw_world"].append(
            float(batch["target_root_yaw_world"][0].item())
        )
        result_lists["predicted_root_yaw_world"].append(resolved.root_yaw_world)
        result_lists["reference_hip_height"].append(float(batch["target_hip_height"][0].item()))
        result_lists["predicted_hip_height"].append(resolved.hip_height)
        result_lists["tracker_pos_world"].append(tracker_positions_world)
        result_lists["current_tracker_raw"].append(current_tracker_raw)
        for name in (
            "configured",
            "measured_valid",
            "d_off",
            "d_on",
        ):
            result_lists[name].append(batch[name][0, -1].detach().cpu().numpy())
        result_lists["hard_rotation_state"].append(hard)
        result_lists["current_trajectory"].append(
            batch["current_trajectory"][0, 0].detach().cpu().numpy()
        )
        result_lists["contact_target"].append(batch["contact_target"][0].detach().cpu().numpy())
        result_lists["contact_logits"].append(
            reconstruction.get(
                "contact_logits",
                torch.full_like(batch["contact_target"], float("nan")),
            )[0].detach().cpu().numpy()
        )
        result_lists["future_leg_prediction"].append(
            reconstruction.get(
                "future_leg",
                torch.full_like(batch["future_leg_target"], float("nan")),
            )[0].detach().cpu().numpy()
        )
        result_lists["future_leg_target"].append(
            batch["future_leg_target"][0].detach().cpu().numpy()
        )
        result_lists["scenario"].append(str(step_item.get("scenario", "")))
        result_lists["hard_rotation_max_error"].append(resolved.hard_rotation_max_error)
        # 只有 deployed 输出可以进入下一帧历史，且显式断开采样图。
        previous_deployed = reconstruction["deployed_pred_xstart"].detach()
        previous_batch = batch

    payload = {name: np.asarray(values) for name, values in result_lists.items()}
    payload["eval_frame_mask"] = np.ones(len(sequence), dtype=bool)
    return payload


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
    )
    model, diffusion = create_model_and_diffusion(args)
    model, source = load_checkpoint_model(
        model, args.model_path, device=device, use_ema=args.use_ema
    )
    limit = len(dataset) if int(args.rollout_limit) <= 0 else min(
        len(dataset), int(args.rollout_limit)
    )
    payloads = [
        rollout_dataset_item(
            model,
            diffusion,
            dataset,
            index,
            device,
            rollout_steps=args.rollout_steps,
            projection_mode=args.projected_ddim_mode,
            late_steps=args.projected_ddim_late_steps,
        )
        for index in range(limit)
    ]
    if not payloads:
        raise ValueError("rollout 数据集为空。")
    payload = {key: np.asarray([item[key] for item in payloads]) for key in payloads[0]}
    payload["fps"] = np.float32(60.0)
    output_dir = Path(args.output_dir or "output/realtime_pose_144d_rollout").resolve()
    output_path = output_dir / "rollout_result.npz"
    save_rollout(output_path, payload)
    print(f"[reconstruct_rollout] weights={source} output={output_path}")
    return {"output_path": output_path}


if __name__ == "__main__":
    main()
