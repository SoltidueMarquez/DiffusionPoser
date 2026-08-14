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
from data_loaders.realtime_pose_ik import build_current_ik_pose
from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_TARGET_LENGTH,
    TRACKER_CONFIGURED_OFFSET,
    TRACKER_D_ON_OFFSET,
    TRACKER_MEASURED_VALID_OFFSET,
)
from data_loaders.tracker_reliability import (
    compute_tracker_online_confidence_torch,
    map_tracker_confidence_to_joints_torch,
)
from diffusion.realtime_pose_inpainting import (
    RealtimePoseInpaintingCondition,
    build_realtime_pose_inpainting_condition,
)
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
            values["head_path_window"],
            values["history_region_confidence"],
            values["window_valid_mask"].bool(),
            values["frame_offsets"],
        )
    return {"prepared_conditioning": prepared}


def build_sampling_inpainting_condition(
    model,
    batch: dict,
    device: torch.device,
    normalizer=None,
    tracker_confidence_warmup: int = 15,
    future_confidence_decay: float = 0.9,
    fabrik_iterations: int = 2,
) -> RealtimePoseInpaintingCondition:
    """离线单 batch 按首次 runtime 语义构造条件：当前做 IK，未来不偷看 GT。"""

    del model
    previous_pose = batch["history_pose_observation"][:, -1].to(device).float()
    mean = scale = None
    if normalizer is not None:
        if normalizer.pose_mean is None or normalizer.pose_scale is None:
            normalizer.load()
        mean = normalizer.pose_mean.to(device).float()
        scale = normalizer.pose_scale.to(device).float()
        previous_pose = previous_pose * scale + mean
    current_tracker = batch["tracker_window_raw"][:, -1].to(device).float()
    configured = current_tracker[..., TRACKER_CONFIGURED_OFFSET] > 0.5
    measured = current_tracker[..., TRACKER_MEASURED_VALID_OFFSET] > 0.5
    tracker_valid = configured & measured
    d_on = current_tracker[..., TRACKER_D_ON_OFFSET] * 60.0
    tracker_confidence = compute_tracker_online_confidence_torch(
        tracker_valid=tracker_valid,
        d_on=d_on,
        warmup_frames=tracker_confidence_warmup,
    )
    current_pose_raw = build_current_ik_pose(
        previous_pose_raw=previous_pose,
        previous_pose_valid=batch["window_valid_mask"][:, -2].to(device).bool(),
        current_tracker_raw=current_tracker,
        joint_offsets_parent=batch["joint_offsets_parent"].to(device).float(),
        joint_rest_local_rotations_6d=batch[
            "joint_rest_local_rotations_6d"
        ].to(device).float(),
        fabrik_iterations=fabrik_iterations,
    )
    current_confidence = map_tracker_confidence_to_joints_torch(
        tracker_confidence
    )
    return build_realtime_pose_inpainting_condition(
        current_pose_raw=current_pose_raw,
        current_confidence=current_confidence,
        future_prior_raw=torch.zeros(
            previous_pose.shape[0],
            REALTIME_POSE_TARGET_LENGTH - 1,
            REALTIME_POSE_TARGET_DIM,
            device=device,
            dtype=torch.float32,
        ),
        future_prior_valid=torch.zeros(
            previous_pose.shape[0], device=device, dtype=torch.bool
        ),
        pose_mean=mean,
        pose_scale=scale,
        future_confidence_decay=future_confidence_decay,
    )


def build_projection_fn(
    batch: dict,
    device: torch.device,
    normalizer=None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    del batch
    if normalizer is None:
        mean = scale = None
    else:
        if normalizer.pose_mean is None or normalizer.pose_scale is None:
            normalizer.load()
        mean = normalizer.pose_mean.to(device)
        scale = normalizer.pose_scale.to(device)
    return lambda value: project_realtime_pose_xstart(
        value,
        mean,
        scale,
    )


def reconstruct_batch(
    model,
    diffusion,
    batch: dict,
    device: torch.device,
    normalizer=None,
    tracker_confidence_warmup: int = 15,
    future_confidence_decay: float = 0.9,
    fabrik_iterations: int = 2,
) -> dict[str, torch.Tensor]:
    reference = batch["x"].to(device)
    if reference.ndim != 3 or tuple(reference.shape[1:]) != (
        REALTIME_POSE_TARGET_LENGTH,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError(f"sample 应为 [B,11,144]，实际为 {tuple(reference.shape)}")
    model_kwargs = build_sampling_model_kwargs(model, batch, device)
    inpainting = build_sampling_inpainting_condition(
        model,
        batch,
        device,
        normalizer=normalizer,
        tracker_confidence_warmup=tracker_confidence_warmup,
        future_confidence_decay=future_confidence_decay,
        fabrik_iterations=fabrik_iterations,
    )
    projection_fn = build_projection_fn(batch, device, normalizer)
    # 离线入口也显式创建条件噪声；完整 DDIM 轨迹由 diffusion 复用同一张量。
    known_noise = torch.randn_like(reference)
    with torch.no_grad():
        result = diffusion.projected_ddim_sample_loop(
            model,
            shape=(
                reference.shape[0],
                REALTIME_POSE_TARGET_LENGTH,
                REALTIME_POSE_TARGET_DIM,
            ),
            projection_fn=projection_fn,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            device=device,
            inpaint_condition=inpainting,
            known_noise=known_noise,
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
    if reference.ndim != 3 or tuple(reference.shape[1:]) != (
        REALTIME_POSE_TARGET_LENGTH,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError(f"reference 应为 [B,11,144]，实际为 {tuple(reference.shape)}")
    reference_horizon = inverse_normalized_target(reference, normalizer)
    raw_prediction_horizon = inverse_normalized_target(
        reconstruction["raw_pred_xstart"], normalizer
    )
    deployed_prediction_horizon = inverse_normalized_target(
        reconstruction["deployed_pred_xstart"], normalizer
    )
    reference_raw = reference_horizon[:, 0]
    raw_prediction = raw_prediction_horizon[:, 0]
    deployed_prediction = deployed_prediction_horizon[:, 0]
    resolved = []
    reference_local_delta = []
    reference_joints_world = []
    reference_root_world = []
    tracker_positions_world = []
    for batch_index in range(deployed_prediction.shape[0]):
        current_tracker_raw = batch["tracker_window_raw"][batch_index, -1].detach().cpu().numpy()
        value = decode_and_resolve_pose(
            deployed_prediction[batch_index],
            current_tracker_raw,
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
        reference_pose_horizon_raw=add_time(reference_horizon.astype(np.float32)),
        raw_pred_pose_horizon_raw=add_time(raw_prediction_horizon.astype(np.float32)),
        deployed_pred_pose_horizon_raw=add_time(deployed_prediction_horizon.astype(np.float32)),
        pose_horizon_valid_mask=np.ones(
            (deployed_prediction.shape[0], 1, REALTIME_POSE_TARGET_LENGTH), dtype=bool
        ),
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
        scenario=add_time(scenario_values),
        eval_frame_mask=np.ones((deployed_prediction.shape[0], 1), dtype=bool),
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
        tracker_confidence_warmup=args.tracker_confidence_warmup,
        future_confidence_decay=args.future_confidence_decay,
        fabrik_iterations=args.fabrik_iterations,
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
