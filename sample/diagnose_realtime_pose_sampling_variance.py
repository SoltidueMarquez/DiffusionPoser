from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from data_loaders.build_realtime_longseq_eval_set import (
    read_longseq_manifest,
    resolve_longseq_eval_dir,
    resolve_manifest_source_path,
)
from data_loaders.generate_realtime_pose_tasks import (
    compute_source_joint_rotations_world,
    load_realtime_source,
)
from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES
from data_loaders.sensor_masking import REALTIME_POSE_TARGET_DIM
from data_loaders.tracker_timeline import (
    build_isolated_condition_timeline,
    stable_context_seed,
)
from sample.evaluate_longseq_eval_set import build_arg_parser
from sample.realtime_pose_runtime import RealtimePoseRuntime, step_realtime_pose_batch
from sample.utils import load_checkpoint_model
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.normalizer import RealtimePoseNormalizer
from utils.parser_util import parse_and_load_from_model


def _clone_runtime_state(runtime: RealtimePoseRuntime) -> RealtimePoseRuntime:
    """复制轻量状态但共享模型，避免重复采样诊断复制整套网络参数。"""

    cloned = RealtimePoseRuntime(
        runtime.model,
        runtime.diffusion,
        runtime.device,
        runtime.joint_offsets_parent,
        runtime.joint_rest_local_rotations_6d,
        normalizer=runtime.normalizer,
        projected_ddim_mode=runtime.projected_ddim_mode,
        projected_ddim_late_steps=runtime.projected_ddim_late_steps,
    )
    cloned.pose_history = list(runtime.pose_history)
    cloned.tracker_history = list(runtime.tracker_history)
    cloned.previous_d_off = runtime.previous_d_off.copy()
    cloned.previous_d_on = runtime.previous_d_on.copy()
    cloned.previous_head_yaw = runtime.previous_head_yaw
    return cloned


def _pairwise_rotation_angles(rotations: np.ndarray) -> np.ndarray:
    """返回所有重复采样两两之间的 SO(3) 夹角，形状为 `[R,R,24]`。"""

    relative = np.matmul(
        np.swapaxes(rotations[:, None], -1, -2),
        rotations[None],
    )
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    return np.arccos(cosine)


def _repeat_summary(
    rotations: np.ndarray,
    joints: np.ndarray,
    root_yaw: np.ndarray,
    root_position: np.ndarray,
    hip_height: np.ndarray,
    reference_rotations: np.ndarray,
    reference_joints: np.ndarray,
) -> dict[str, Any]:
    repeat_count = int(rotations.shape[0])
    pair_rows, pair_cols = np.triu_indices(repeat_count, k=1)
    pair_rotation = _pairwise_rotation_angles(rotations)[pair_rows, pair_cols]
    pair_position = np.linalg.norm(
        joints[:, None] - joints[None], axis=-1
    )[pair_rows, pair_cols]
    root_yaw_pairwise = np.abs(
        (root_yaw[:, None] - root_yaw[None] + np.pi) % (2.0 * np.pi) - np.pi
    )[pair_rows, pair_cols]
    root_xz_pairwise = np.linalg.norm(
        root_position[:, None, [0, 2]] - root_position[None, :, [0, 2]],
        axis=-1,
    )[pair_rows, pair_cols]

    reference_relative = np.matmul(
        np.swapaxes(reference_rotations[None], -1, -2),
        rotations,
    )
    reference_cosine = np.clip(
        (np.trace(reference_relative, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    reference_rotation_error = np.degrees(np.arccos(reference_cosine)).mean(axis=1)
    reference_position_error = np.linalg.norm(
        joints - reference_joints[None], axis=-1
    ).mean(axis=1) * 100.0

    per_joint_rotation = np.degrees(pair_rotation.mean(axis=0))
    per_joint_position = pair_position.mean(axis=0) * 100.0
    top_position = np.argsort(per_joint_position)[::-1][:8]
    top_rotation = np.argsort(per_joint_rotation)[::-1][:8]
    return {
        "repeat_count": repeat_count,
        "pair_count": int(len(pair_rows)),
        "pairwise_mpjre_deg_mean": float(np.degrees(pair_rotation).mean()),
        "pairwise_mpjre_deg_max_pair_mean": float(
            np.degrees(pair_rotation).mean(axis=1).max()
        ),
        "pairwise_mpjpe_cm_mean": float(pair_position.mean() * 100.0),
        "pairwise_mpjpe_cm_max_pair_mean": float(
            pair_position.mean(axis=1).max() * 100.0
        ),
        "pairwise_root_yaw_deg_mean": float(np.degrees(root_yaw_pairwise).mean()),
        "pairwise_root_yaw_deg_max": float(np.degrees(root_yaw_pairwise).max()),
        "pairwise_root_xz_cm_mean": float(root_xz_pairwise.mean() * 100.0),
        "pairwise_root_xz_cm_max": float(root_xz_pairwise.max() * 100.0),
        "hip_height_range_cm": float(np.ptp(hip_height) * 100.0),
        "reference_mpjre_deg_mean": float(reference_rotation_error.mean()),
        "reference_mpjre_deg_std": float(reference_rotation_error.std()),
        "reference_mpjpe_cm_mean": float(reference_position_error.mean()),
        "reference_mpjpe_cm_std": float(reference_position_error.std()),
        "top_position_variability_joints": [
            {
                "joint": str(SMPL_JOINT_NAMES[index]),
                "pairwise_position_cm": float(per_joint_position[index]),
            }
            for index in top_position
        ],
        "top_rotation_variability_joints": [
            {
                "joint": str(SMPL_JOINT_NAMES[index]),
                "pairwise_rotation_deg": float(per_joint_rotation[index]),
            }
            for index in top_rotation
        ],
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    diagnostic = parser.add_argument_group("sampling_variance")
    diagnostic.add_argument("--diagnostic_frame", default=600, type=int)
    diagnostic.add_argument("--repeat_count", default=32, type=int)
    args = parse_and_load_from_model(parser, argv=argv, ignore_keys={"ts_respace"})
    if len(args.conditions) != 1:
        raise ValueError("重复采样诊断必须只指定一种 conditions。")
    if int(args.repeat_count) < 2:
        raise ValueError("repeat_count 必须至少为 2。")
    if not str(args.output_dir).strip():
        raise ValueError("重复采样诊断必须显式指定 output_dir。")

    args.ts_respace = f"ddim{int(args.inference_steps)}"
    fixseed(int(args.seed))
    eval_set_dir = resolve_longseq_eval_dir(eval_root=args.eval_root, eval_set=args.eval_set)
    entries = read_longseq_manifest(eval_set_dir)
    selected = entries[: int(args.limit)] if int(args.limit) > 0 else entries
    if len(selected) != 1:
        raise ValueError("重复采样诊断必须通过 limit/eval_set 恰好选择一条序列。")
    entry = selected[0]
    source = load_realtime_source(resolve_manifest_source_path(eval_set_dir, entry))
    frame_count = int(source["tracker_pos_world"].shape[0])
    frame_index = int(args.diagnostic_frame)
    if not 60 <= frame_index < frame_count:
        raise ValueError(f"diagnostic_frame 必须位于 [60,{frame_count - 1}]。")
    timeline = build_isolated_condition_timeline(
        source_id=str(entry["sequence_id"]),
        frame_count=frame_count,
        condition=str(args.conditions[0]),
        global_seed=int(args.timeline_seed),
    )
    normalizer = (
        RealtimePoseNormalizer(args.normalizer_dir) if bool(args.normalize_input) else None
    )

    if bool(args.require_cuda) and (not bool(args.cuda) or not torch.cuda.is_available()):
        raise RuntimeError("重复采样诊断要求可用的 CUDA GPU。")
    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    model, diffusion = create_model_and_diffusion(args)
    model, weights = load_checkpoint_model(
        model, args.model_path, device=device, use_ema=args.use_ema
    )
    model.eval()
    runtime = RealtimePoseRuntime(
        model,
        diffusion,
        device,
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
        normalizer=normalizer,
        projected_ddim_mode=str(args.projected_ddim_mode),
        projected_ddim_late_steps=int(args.projected_ddim_late_steps),
    )

    warmup_seed = int(
        stable_context_seed(
            int(args.timeline_seed), str(entry["sequence_id"]), "repeat_warmup"
        )
        % (2**63)
    )
    warmup_generator = torch.Generator(device=device).manual_seed(warmup_seed)
    for index in tqdm(range(frame_index), desc="repeat diagnostic warmup", unit="frame"):
        noise = torch.randn(
            (1, REALTIME_POSE_TARGET_DIM), generator=warmup_generator, device=device
        )
        step_realtime_pose_batch(
            [runtime],
            source["tracker_pos_world"][index : index + 1],
            source["tracker_rot_world_6d"][index : index + 1],
            timeline.configured[index : index + 1],
            timeline.measured_valid[index : index + 1],
            np.asarray([source["root_pos_world"][index, 1]], dtype=np.float32),
            noise=noise,
        )

    repeat_count = int(args.repeat_count)
    clones = [_clone_runtime_state(runtime) for _ in range(repeat_count)]
    repeat_generator = torch.Generator(device=device).manual_seed(int(args.seed) + 9137)
    repeat_noise = torch.randn(
        (repeat_count, REALTIME_POSE_TARGET_DIM),
        generator=repeat_generator,
        device=device,
    )
    steps = step_realtime_pose_batch(
        clones,
        np.repeat(
            source["tracker_pos_world"][frame_index : frame_index + 1],
            repeat_count,
            axis=0,
        ),
        np.repeat(
            source["tracker_rot_world_6d"][frame_index : frame_index + 1],
            repeat_count,
            axis=0,
        ),
        np.repeat(
            timeline.configured[frame_index : frame_index + 1], repeat_count, axis=0
        ),
        np.repeat(
            timeline.measured_valid[frame_index : frame_index + 1],
            repeat_count,
            axis=0,
        ),
        np.full(repeat_count, source["root_pos_world"][frame_index, 1], dtype=np.float32),
        noise=repeat_noise,
    )

    rotations = np.stack([step.resolved_pose.joint_rotations_world for step in steps])
    joints = np.stack([step.resolved_pose.joints_world for step in steps])
    root_yaw = np.asarray([step.resolved_pose.root_yaw_world for step in steps])
    root_position = np.stack([step.resolved_pose.root_position_world for step in steps])
    hip_height = np.asarray([step.resolved_pose.hip_height for step in steps])
    raw_target = np.stack([step.raw_pred_xstart for step in steps])
    deployed_target = np.stack([step.deployed_pred_xstart for step in steps])
    reference_rotations = compute_source_joint_rotations_world(source)[frame_index]
    summary = _repeat_summary(
        rotations,
        joints,
        root_yaw,
        root_position,
        hip_height,
        reference_rotations,
        source["joints_world"][frame_index],
    )
    summary.update(
        {
            "sequence_id": str(entry["sequence_id"]),
            "condition": str(args.conditions[0]),
            "frame_index": frame_index,
            "history_length": len(runtime.pose_history),
            "model_path": str(Path(args.model_path).resolve()),
            "weights": str(weights),
            "inference_steps": int(diffusion.num_timesteps),
            "warmup_seed": warmup_seed,
            "repeat_seed": int(args.seed) + 9137,
        }
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "repeat_samples.npz",
        noise=repeat_noise.detach().cpu().numpy(),
        raw_pred_target_raw=raw_target,
        deployed_pred_target_raw=deployed_target,
        predicted_joint_rotations_world=rotations,
        predicted_joints_world=joints,
        predicted_root_yaw_world=root_yaw,
        predicted_root_position_world=root_position,
        predicted_hip_height=hip_height,
        reference_joint_rotations_world=reference_rotations,
        reference_joints_world=source["joints_world"][frame_index],
    )
    summary_path = output_dir / "repeat_sampling_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"[sampling_variance] wrote {summary_path}")
    return summary


if __name__ == "__main__":
    main()
