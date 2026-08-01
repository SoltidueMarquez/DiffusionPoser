from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data_loaders.generate_realtime_pose_tasks import build_task_arrays, compute_source_joint_rotations_world
from data_loaders.realtime_pose_geometry import extract_forward_yaw_np
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from data_loaders.tracker_timeline import build_tracker_timeline, classify_tracker_window
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
    timeline_window = build_tracker_timeline("toy", 70, global_seed=10).window(0)
    scenario = classify_tracker_window(timeline_window.configured, timeline_window.measured_valid)
    task = build_task_arrays(
        source=source,
        source_path=Path("toy.npz"),
        source_frames=70,
        joint_rotations_world=rotations_world,
        head_yaws=head_yaws,
        timeline_window=timeline_window,
        start_frame=0,
        scenario=str(scenario),
    )
    tracker = task["tracker_window"]
    batch = {
        "current_tracker_pos_head_ref": torch.from_numpy(tracker[-1:, :, :3]),
        "current_tracker_rot_head_ref_6d": torch.from_numpy(tracker[-1:, :, 3:9]),
        "configured": torch.from_numpy(task["configured"][None]),
        "measured_valid": torch.from_numpy(task["measured_valid"][None]),
        "missing_age": torch.from_numpy(task["missing_age"][None]),
        "missing_age_norm": torch.from_numpy(tracker[None, :, :, 11]),
        "current_head_yaw_world": torch.from_numpy(np.asarray(task["current_head_yaw_world"]).reshape(1)),
        "current_head_position_world": torch.from_numpy(task["current_head_position_world"][None]),
        "floor_y": torch.from_numpy(np.asarray(task["floor_y"]).reshape(1)),
        "joint_offsets_parent": torch.from_numpy(task["joint_offsets_parent"][None]),
        "joint_rest_local_rotations_6d": torch.from_numpy(task["joint_rest_local_rotations_6d"][None]),
        "target_joints_head_ref": torch.from_numpy(task["target_joints_head_ref"][None]),
        "target_root_position_head_ref": torch.from_numpy(task["target_root_position_head_ref"][None]),
        "target_root_yaw_world": torch.from_numpy(np.asarray(task["target_root_yaw_world"]).reshape(1)),
        "target_hip_height": torch.from_numpy(np.asarray(task["target_hip_height"]).reshape(1)),
        "scenario": str(scenario),
    }
    reference = torch.from_numpy(task["current_target"][None])
    known_mask = torch.from_numpy(task["known_mask"][None])
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
