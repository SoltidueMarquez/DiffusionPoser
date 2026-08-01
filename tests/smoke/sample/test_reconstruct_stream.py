from __future__ import annotations

import numpy as np
import torch

from data_loaders.generate_realtime_pose_tasks import (
    build_task_bundle_row,
    compute_source_joint_rotations_world,
)
from data_loaders.realtime_pose_geometry import build_known_mask_from_measured_np, extract_forward_yaw_np
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from data_loaders.tracker_timeline import build_task_config_plan
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from model.realtime_pose_target_dit import RealtimePoseTargetDiT
from eval.evaluate_realtime_pose import evaluate_file
from sample.reconstruct_stream import reconstruct_batch, save_reconstruction
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source
from tests.smoke.train.test_realtime_pose_140d_training import _make_batch


def test_reconstruct_batch_preserves_every_known_channel():
    target, known_target, known, kwargs = _make_batch(batch_size=1)
    y = kwargs["y"]
    batch = {
        "x": target,
        "known_target": known_target,
        "known_mask": known,
        "pose_history": kwargs["pose_history"],
        "tracker_window": kwargs["tracker_window"],
        "valid_frame_mask": kwargs["valid_frame_mask"],
    }
    model = RealtimePoseTargetDiT(input_feats=140, latent_dim=32, num_layers=1, num_heads=4)
    diffusion = GaussianDiffusion(
        betas=get_named_beta_schedule("cosine", 6),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    reconstructed = reconstruct_batch(model, diffusion, batch, torch.device("cpu"), use_ddim=True)
    assert reconstructed.shape == (1, 140)
    torch.testing.assert_close(reconstructed[known], known_target[known], atol=1e-6, rtol=0.0)


def test_save_reconstruction_writes_unified_evaluation_fields(tmp_path):
    source = build_toy_realtime_source(frame_count=70)
    rotations_world = compute_source_joint_rotations_world(source)
    head_rotations = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"][:, 0])
    head_yaws = extract_forward_yaw_np(head_rotations, initial_yaw=0.0)
    row = build_task_bundle_row(
        source=source,
        joint_rotations_world=rotations_world,
        head_yaws=head_yaws,
        start_frame=0,
        source_index=0,
        config_plans=build_task_config_plan("toy", global_seed=10, max_rollout_steps=4),
        max_rollout_steps=4,
    )
    configured = row["configured"][0, :61].astype(bool)
    measured_valid = row["measured_valid"][0, :61].astype(bool)
    missing_age = row["missing_age"][0, :61]
    tracker = np.concatenate(
        [
            row["tracker_continuous"][0],
            configured[..., None],
            measured_valid[..., None],
            (missing_age.astype(np.float32) / 60.0)[..., None],
        ],
        axis=-1,
    ).astype(np.float32)
    tracker[..., :9] *= measured_valid[..., None]
    known_mask_np = build_known_mask_from_measured_np(measured_valid[-1])
    scenario = "fixed_six"
    batch = {
        "current_tracker_pos_head_ref": torch.from_numpy(tracker[-1:, :, :3]),
        "current_tracker_rot_head_ref_6d": torch.from_numpy(tracker[-1:, :, 3:9]),
        "configured": torch.from_numpy(configured[None]),
        "measured_valid": torch.from_numpy(measured_valid[None]),
        "missing_age": torch.from_numpy(missing_age[None].astype(np.int64)),
        "missing_age_norm": torch.from_numpy(tracker[None, :, :, 11]),
        "current_head_yaw_world": torch.from_numpy(row["current_head_yaw_world"][0:1]),
        "current_head_position_world": torch.from_numpy(row["current_head_position_world"][0:1]),
        "floor_y": torch.from_numpy(row["floor_y"][0:1]),
        "joint_offsets_parent": torch.from_numpy(source["joint_offsets_parent"][None]),
        "joint_rest_local_rotations_6d": torch.from_numpy(source["joint_rest_local_rotations_6d"][None]),
        "target_joints_head_ref": torch.from_numpy(row["target_joints_head_ref"][0:1]),
        "target_root_position_head_ref": torch.from_numpy(row["target_root_position_head_ref"][0:1]),
        "target_root_yaw_world": torch.from_numpy(row["target_root_yaw_world"][0:1]),
        "target_hip_height": torch.from_numpy(row["target_hip_height"][0:1]),
        "scenario": str(scenario),
    }
    reference = torch.from_numpy(row["current_target"][0:1])
    known_mask = torch.from_numpy(known_mask_np[None])
    result_path = tmp_path / "reconstruction.npz"
    save_reconstruction(
        path=result_path,
        reference=reference,
        reconstructed=reference.clone(),
        known_mask=known_mask,
        batch=batch,
        normalizer=None,
    )

    with np.load(result_path, allow_pickle=False) as data:
        assert data["reference_target_raw"].shape == (1, 1, 140)
        assert data["reference_body_local_delta_6d"].shape == (1, 1, 144)
        assert data["predicted_joints_world"].shape == (1, 1, 24, 3)
    metrics = evaluate_file(result_path)
    assert metrics["mpjre_deg"] < 1e-4
    assert metrics["mpjpe_cm"] < 1e-3
