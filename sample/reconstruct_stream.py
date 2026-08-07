from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_np,
    global_head_rotations_to_local_delta_6d_np,
)
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np, rotation_6d_to_matrix_np
from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM, REALTIME_POSE_WINDOW_LENGTH
from diffusion.realtime_pose_projection import project_realtime_pose_xstart
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
    parser = argparse.ArgumentParser(description="采样 144D 动态 Tracker 单帧姿态任务。")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    return parser


def build_sampling_model_kwargs(model, batch: dict, device: torch.device) -> dict:
    """每个目标帧只编码一次历史条件，后续 DDIM step 直接复用。"""

    values = {
        name: batch[name].to(device)
        for name in (
            "history_pose_observation",
            "tracker_window",
            "head_path_window",
            "history_region_confidence",
            "window_valid_mask",
            "frame_offsets",
        )
    }
    model_impl = getattr(model, "module", model)
    with torch.no_grad():
        prepared = model_impl.prepare_conditioning(
            values["history_pose_observation"],
            values["tracker_window"],
            values["head_path_window"],
            values["history_region_confidence"],
            values["window_valid_mask"].bool(),
            values["frame_offsets"],
        )
    return {"prepared_conditioning": prepared}


def build_projection_fn(
    batch: dict,
    device: torch.device,
    normalizer=None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    current_tracker_raw = batch["tracker_window_raw"][:, -1].to(device)
    hard_rotation_state = batch["hard_rotation_state_window"][:, -1].to(device).bool()
    if normalizer is None:
        mean = scale = None
    else:
        if normalizer.pose_mean is None or normalizer.pose_scale is None:
            normalizer.load()
        mean = normalizer.pose_mean.to(device)
        scale = normalizer.pose_scale.to(device)
    return lambda value: project_realtime_pose_xstart(
        value,
        current_tracker_raw,
        hard_rotation_state,
        mean,
        scale,
    )


def reconstruct_batch(
    model,
    diffusion,
    batch: dict,
    device: torch.device,
    normalizer=None,
    projection_mode: str = "all_steps",
    late_steps: int = 5,
) -> dict[str, torch.Tensor]:
    reference = batch["x"].to(device)
    if reference.ndim != 3 or tuple(reference.shape[1:]) != (
        REALTIME_POSE_WINDOW_LENGTH,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError(f"sample 应为 [B,11,144]，实际为 {tuple(reference.shape)}")
    model_kwargs = build_sampling_model_kwargs(model, batch, device)
    projection_fn = build_projection_fn(batch, device, normalizer)
    with torch.no_grad():
        result = diffusion.projected_ddim_sample_loop(
            model,
            shape=(reference.shape[0], REALTIME_POSE_TARGET_DIM),
            projection_fn=projection_fn,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=device,
            projection_mode=projection_mode,
            late_steps=late_steps,
        )
    output = {
        "sample": result["sample"],
        "raw_pred_xstart": result["raw_pred_xstart"],
        "deployed_pred_xstart": result["deployed_pred_xstart"],
    }
    if "auxiliary_outputs" in result:
        output.update(result["auxiliary_outputs"])
    return output


def inverse_normalized_target(target: torch.Tensor, normalizer) -> np.ndarray:
    value = target.detach().cpu()
    if normalizer is not None:
        value = normalizer.inverse_pose(value)
    return np.asarray(value, dtype=np.float32)


def save_reconstruction(
    path: Path,
    reference: torch.Tensor,
    reconstruction: dict[str, torch.Tensor],
    batch: dict,
    normalizer=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reference.ndim == 3:
        reference = reference[:, -1]
    reference_raw = inverse_normalized_target(reference, normalizer)
    raw_prediction = inverse_normalized_target(reconstruction["raw_pred_xstart"], normalizer)
    deployed_prediction = inverse_normalized_target(
        reconstruction["deployed_pred_xstart"], normalizer
    )
    resolved = []
    reference_local_delta = []
    reference_joints_world = []
    reference_root_world = []
    tracker_positions_world = []
    for batch_index in range(deployed_prediction.shape[0]):
        current_tracker_raw = batch["tracker_window_raw"][batch_index, -1].detach().cpu().numpy()
        hard_rotation = batch["hard_rotation_state_window"][batch_index, -1].detach().cpu().numpy()
        value = decode_and_resolve_pose(
            deployed_prediction[batch_index],
            current_tracker_raw,
            hard_rotation,
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
            + yaw_rotation
            @ batch["target_root_position_head_ref"][batch_index].detach().cpu().numpy()
        )
        tracker_positions_world.append(
            origin[None] + np.einsum("ij,aj->ai", yaw_rotation, current_tracker_raw[:, :3])
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
    scenario_values = (
        np.asarray([scenario] * deployed_prediction.shape[0])
        if isinstance(scenario, str)
        else np.asarray(scenario)
    )
    add_time = lambda value: np.asarray(value)[:, None]
    np.savez(
        path,
        fps=np.float32(60.0),
        reference_target_raw=add_time(reference_raw.astype(np.float32)),
        raw_pred_target_raw=add_time(raw_prediction.astype(np.float32)),
        deployed_pred_target_raw=add_time(deployed_prediction.astype(np.float32)),
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
        current_tracker_raw=add_time(batch["tracker_window_raw"][:, -1].detach().cpu().numpy()),
        configured=add_time(batch["configured"][:, -1].detach().cpu().numpy()),
        measured_valid=add_time(batch["measured_valid"][:, -1].detach().cpu().numpy()),
        d_off=add_time(batch["d_off"][:, -1].detach().cpu().numpy()),
        d_on=add_time(batch["d_on"][:, -1].detach().cpu().numpy()),
        hard_rotation_state=add_time(
            batch["hard_rotation_state_window"][:, -1].detach().cpu().numpy()
        ),
        contact_target=add_time(batch["contact_target"].detach().cpu().numpy()),
        contact_logits=add_time(
            reconstruction.get(
                "contact_logits",
                torch.full_like(batch["contact_target"], float("nan")),
            ).detach().cpu().numpy()
        ),
        future_leg_prediction=add_time(
            reconstruction.get(
                "future_leg",
                torch.full_like(batch["future_leg_target"], float("nan")),
            ).detach().cpu().numpy()
        ),
        future_leg_target=add_time(batch["future_leg_target"].detach().cpu().numpy()),
        scenario=add_time(scenario_values),
        eval_frame_mask=np.ones((deployed_prediction.shape[0], 1), dtype=bool),
        hard_rotation_max_error=add_time(
            np.asarray([value.hard_rotation_max_error for value in resolved], dtype=np.float32)
        ),
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
    model, source = load_checkpoint_model(
        model, args.model_path, device=device, use_ema=args.use_ema
    )
    reconstruction = reconstruct_batch(
        model,
        diffusion,
        batch,
        device=device,
        normalizer=dataset.normalizer,
        projection_mode=args.projected_ddim_mode,
        late_steps=args.projected_ddim_late_steps,
    )
    output_dir = Path(args.output_dir or "output/realtime_pose_144d").resolve()
    output_path = output_dir / "realtime_pose_reconstruction.npz"
    save_reconstruction(
        output_path,
        batch["x"],
        reconstruction,
        batch,
        normalizer=dataset.normalizer,
    )
    print(f"[reconstruct_stream] weights={source} output={output_path}")
    return {"output_path": output_path}


if __name__ == "__main__":
    main()
