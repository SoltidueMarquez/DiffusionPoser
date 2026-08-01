from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_np,
    global_head_rotations_to_local_delta_6d_np,
)
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM
from sample.realtime_pose_runtime import decode_and_resolve_pose
from sample.utils import choose_sampler, load_checkpoint_model
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
    parser = argparse.ArgumentParser(description="采样 144D 动态 Tracker 单帧关节补全任务。")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    return parser


def build_realtime_inpaint_mask(known_mask: torch.Tensor) -> torch.Tensor:
    if known_mask.ndim != 2 or known_mask.shape[1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError("known_mask 必须与 144D sample 同形。")
    return ~known_mask.bool()


def build_sampling_model_kwargs(batch: dict, device: torch.device) -> dict:
    known_mask = batch["known_mask"].to(device).bool()
    known_target = batch["known_target"].to(device)
    pose_history = batch["pose_history"].to(device)
    tracker_window = batch["tracker_window"].to(device)
    valid_frame_mask = batch["valid_frame_mask"].to(device).bool()
    unknown_mask = build_realtime_inpaint_mask(known_mask)
    return {
        "inpaint_cond": unknown_mask,
        "known_mask": known_mask,
        "pose_history": pose_history,
        "tracker_window": tracker_window,
        "valid_frame_mask": valid_frame_mask,
        "attention_mask": valid_frame_mask,
        "y": {
            "mask": unknown_mask,
            "inpainted_motion": known_target,
        },
    }


def reconstruct_batch(
    model,
    diffusion,
    batch: dict,
    device: torch.device,
    use_ddim: bool = True,
    init_image: torch.Tensor | None = None,
    start_timestep: int | None = None,
) -> torch.Tensor:
    del start_timestep
    if init_image is not None:
        raise ValueError("当前 144D 重构已关闭 local-delta IK initializer。")
    reference = batch["x"].to(device)
    if reference.ndim != 2 or reference.shape[1] != REALTIME_POSE_TARGET_DIM:
        raise ValueError(f"sample 应为 [B,144]，实际为 {tuple(reference.shape)}")
    model_kwargs = build_sampling_model_kwargs(batch, device)
    sampler = choose_sampler(diffusion, use_ddim=use_ddim)
    with torch.no_grad():
        reconstructed = sampler(
            model,
            shape=tuple(reference.shape),
            noise=None,
            clip_denoised=False,
            model_kwargs=model_kwargs,
        )
    known_mask = model_kwargs["known_mask"]
    return torch.where(known_mask, model_kwargs["y"]["inpainted_motion"], reconstructed)


def inverse_normalized_target(target: torch.Tensor, normalizer) -> np.ndarray:
    value = target.detach().cpu()
    if normalizer is not None:
        value = normalizer.inverse_pose(value)
    return np.asarray(value, dtype=np.float32)


def save_reconstruction(
    path: Path,
    reference: torch.Tensor,
    reconstructed: torch.Tensor,
    known_mask: torch.Tensor,
    batch: dict,
    normalizer=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reference_raw = inverse_normalized_target(reference, normalizer)
    reconstructed_raw = inverse_normalized_target(reconstructed, normalizer)
    resolved = []
    reference_local_delta = []
    reference_joints_world = []
    reference_root_world = []
    tracker_positions_world = []
    for batch_index in range(reconstructed_raw.shape[0]):
        value = decode_and_resolve_pose(
            reconstructed_raw[batch_index],
            np.concatenate(
                [
                    batch["current_tracker_pos_head_ref"][batch_index].detach().cpu().numpy(),
                    batch["current_tracker_rot_head_ref_6d"][batch_index].detach().cpu().numpy(),
                    batch["configured"][batch_index, -1, :, None].detach().cpu().numpy(),
                    batch["measured_valid"][batch_index, -1, :, None].detach().cpu().numpy(),
                    batch["missing_age_norm"][batch_index, -1, :, None].detach().cpu().numpy(),
                ],
                axis=-1,
            ),
            float(batch["current_head_yaw_world"][batch_index].item()),
            batch["current_head_position_world"][batch_index].detach().cpu().numpy(),
            float(batch["floor_y"][batch_index].item()),
            batch["joint_offsets_parent"][batch_index].detach().cpu().numpy(),
            batch["joint_rest_local_rotations_6d"][batch_index].detach().cpu().numpy(),
        )
        resolved.append(value)
        head_yaw = float(batch["current_head_yaw_world"][batch_index].item())
        head_position = batch["current_head_position_world"][batch_index].detach().cpu().numpy()
        floor_y = float(batch["floor_y"][batch_index].item())
        origin = np.asarray([head_position[0], floor_y, head_position[2]], dtype=np.float32)
        yaw_rotation = make_yaw_rotation_np(np.asarray([head_yaw], dtype=np.float32))[0]
        reference_joints_world.append(
            origin[None]
            + np.einsum(
                "ij,aj->ai",
                yaw_rotation,
                batch["target_joints_head_ref"][batch_index].detach().cpu().numpy(),
            )
        )
        reference_root_world.append(
            origin
            + yaw_rotation @ batch["target_root_position_head_ref"][batch_index].detach().cpu().numpy()
        )
        tracker_positions_world.append(
            origin[None]
            + np.einsum(
                "ij,aj->ai",
                yaw_rotation,
                batch["current_tracker_pos_head_ref"][batch_index].detach().cpu().numpy(),
            )
        )
        rest_rotations = rotation_6d_to_matrix_np(
            batch["joint_rest_local_rotations_6d"][batch_index].detach().cpu().numpy()
        )
        reference_head_rotations, reference_heading_head = decode_target_head_rotations_np(
            reference_raw[batch_index]
        )
        reference_local_delta.append(
            global_head_rotations_to_local_delta_6d_np(
                reference_head_rotations,
                root_heading_head=reference_heading_head,
                rest_local_rotations=rest_rotations,
            )
        )
    scenario = batch.get("scenario", "")
    if isinstance(scenario, str):
        scenario_values = np.asarray([scenario] * reconstructed_raw.shape[0])
    else:
        scenario_values = np.asarray(scenario)
    add_time = lambda value: np.asarray(value)[:, None]
    np.savez(
        path,
        fps=np.float32(60.0),
        reference_target_raw=add_time(reference_raw.astype(np.float32)),
        reconstructed_target_raw=add_time(reconstructed_raw.astype(np.float32)),
        known_mask=add_time(known_mask.detach().cpu().numpy()),
        reference_body_local_delta_6d=add_time(np.asarray(reference_local_delta, dtype=np.float32)),
        predicted_body_local_delta_6d=add_time(
            np.stack([value.body_local_delta_6d for value in resolved]).astype(np.float32)
        ),
        reference_joints_world=add_time(np.asarray(reference_joints_world, dtype=np.float32)),
        predicted_joints_world=add_time(
            np.stack([value.joints_world for value in resolved]).astype(np.float32)
        ),
        reference_root_position_world=add_time(np.asarray(reference_root_world, dtype=np.float32)),
        predicted_root_position_world=add_time(
            np.stack([value.root_position_world for value in resolved]).astype(np.float32)
        ),
        reference_root_yaw_world=add_time(
            batch["target_root_yaw_world"].detach().cpu().numpy().astype(np.float32)
        ),
        predicted_root_yaw_world=add_time(
            np.asarray([value.root_yaw_world for value in resolved], dtype=np.float32)
        ),
        reference_hip_height=add_time(
            batch["target_hip_height"].detach().cpu().numpy().astype(np.float32)
        ),
        predicted_hip_height=add_time(
            np.asarray([value.hip_height for value in resolved], dtype=np.float32)
        ),
        tracker_pos_world=add_time(np.asarray(tracker_positions_world, dtype=np.float32)),
        configured=add_time(batch["configured"][:, -1].detach().cpu().numpy()),
        measured_valid=add_time(batch["measured_valid"][:, -1].detach().cpu().numpy()),
        missing_age=add_time(batch["missing_age"][:, -1].detach().cpu().numpy()),
        scenario=add_time(scenario_values),
        eval_frame_mask=np.ones((reconstructed_raw.shape[0], 1), dtype=bool),
        known_rotation_max_error=add_time(np.asarray(
            [value.known_rotation_max_error for value in resolved], dtype=np.float32
        )),
    )


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
    item = dataset[0]
    batch = {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in item.items()
    }
    model, diffusion = create_model_and_diffusion(args)
    model, source = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    reconstructed = reconstruct_batch(
        model,
        diffusion,
        batch,
        device=device,
        use_ddim=str(args.ts_respace).startswith("ddim"),
    )
    output_dir = Path(args.output_dir or "output/realtime_pose_144d").resolve()
    output_path = output_dir / "realtime_pose_reconstruction.npz"
    save_reconstruction(
        output_path,
        batch["x"],
        reconstructed,
        batch["known_mask"],
        batch,
        normalizer=dataset.normalizer,
    )
    print(f"[reconstruct_stream] weights={source} output={output_path}")
    return {"output_path": output_path}


if __name__ == "__main__":
    main()
