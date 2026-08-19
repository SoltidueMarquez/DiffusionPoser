from __future__ import annotations

import numpy as np
import torch

from data_loaders.generate_realtime_pose_tasks import compute_source_joint_rotations_world
from data_loaders.sensor_masking import STATIC_OPTIONAL_TRACKER_MASKS
from diffusion.gaussian_diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from model.realtime_pose_current_dit import RealtimePoseCurrentDiT
from model.realtime_pose_predictor import RealtimePosePredictor
from sample.realtime_pose_runtime import RealtimePoseRuntime, WorldPoseState, step_realtime_pose_batch
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def _models():
    predictor = RealtimePosePredictor(
        latent_dim=32, num_layers=1, num_heads=4, feedforward_dim=64, dropout=0.0
    ).eval()
    dit = RealtimePoseCurrentDiT(
        latent_dim=32, num_layers=1, num_heads=4, dropout=0.0
    ).eval()
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    return predictor, dit, diffusion


def _runtime(source, rotations, models):
    predictor, dit, diffusion = models
    runtime = RealtimePoseRuntime(
        predictor,
        dit,
        diffusion,
        torch.device("cpu"),
        source["joint_offsets_parent"],
        source["joint_rest_local_rotations_6d"],
        normalizer=None,
        fabrik_iterations=1,
        ik_direction_only_quality=0.8,
        ik_residual_scale=0.5,
    )
    runtime.initialize_history(
        [
            WorldPoseState(
                rotations[index],
                float(source["root_yaw"][index]),
                float(source["pelvis_height"][index, 0]),
                source["root_pos_world"][index],
            )
            for index in range(1, 11)
        ],
        source["tracker_pos_world"][:11],
        source["tracker_rot_world_6d"][:11],
        source["root_pos_world"][:11, 1],
    )
    return runtime


def test_runtime_requires_complete_history():
    source = build_toy_realtime_source(24)
    rotations = compute_source_joint_rotations_world(source)
    runtime = _runtime(source, rotations, _models())
    runtime.pose_history.pop()
    with np.testing.assert_raises(RuntimeError):
        runtime.step(
            source["tracker_pos_world"][11],
            source["tracker_rot_world_6d"][11],
            STATIC_OPTIONAL_TRACKER_MASKS[0],
            0.0,
        )


def test_stationary_tracker_values_still_advance_runtime_history():
    source = build_toy_realtime_source(24)
    rotations = compute_source_joint_rotations_world(source)
    runtime = _runtime(source, rotations, _models())
    repeated_position = source["tracker_pos_world"][10].copy()
    repeated_rotation = source["tracker_rot_world_6d"][10].copy()
    runtime.step(
        repeated_position,
        repeated_rotation,
        STATIC_OPTIONAL_TRACKER_MASKS[0],
        0.0,
        noise=torch.zeros(1, 144),
        known_noise=torch.zeros(1, 144),
    )
    assert len(runtime.tracker_history) == 12
    assert runtime._preloaded_current_pending is False


def test_single_and_batch_runtime_match_and_output_contract():
    torch.manual_seed(4)
    source = build_toy_realtime_source(24)
    rotations = compute_source_joint_rotations_world(source)
    models = _models()
    single = _runtime(source, rotations, models)
    batch_runtime = _runtime(source, rotations, models)
    noise = torch.randn(1, 144)
    known = torch.randn(1, 144)
    single_result = single.step(
        source["tracker_pos_world"][11],
        source["tracker_rot_world_6d"][11],
        STATIC_OPTIONAL_TRACKER_MASKS[-1],
        0.0,
        noise=noise,
        known_noise=known,
    )
    batch_result = step_realtime_pose_batch(
        [batch_runtime],
        source["tracker_pos_world"][11:12],
        source["tracker_rot_world_6d"][11:12],
        np.asarray(STATIC_OPTIONAL_TRACKER_MASKS[-1])[None],
        np.asarray([0.0]),
        noise=noise,
        known_noise=known,
    )[0]
    np.testing.assert_allclose(single_result.deployed_pred_pose, batch_result.deployed_pred_pose)
    assert single_result.predictor_pose_horizon.shape == (11, 144)
    assert single_result.raw_pred_pose.shape == (144,)
    assert single_result.inpaint_confidence.shape == (24,)
    assert single_result.contact_logits.shape == (2,)
    assert np.isfinite(single_result.current_head_yaw_world)


def test_all_eight_static_tracker_combinations_run():
    source = build_toy_realtime_source(24)
    rotations = compute_source_joint_rotations_world(source)
    models = _models()
    runtimes = [_runtime(source, rotations, models) for _ in STATIC_OPTIONAL_TRACKER_MASKS]
    count = len(runtimes)
    results = step_realtime_pose_batch(
        runtimes,
        np.repeat(source["tracker_pos_world"][11:12], count, axis=0),
        np.repeat(source["tracker_rot_world_6d"][11:12], count, axis=0),
        np.asarray(STATIC_OPTIONAL_TRACKER_MASKS),
        np.zeros(count, dtype=np.float32),
        noise=torch.zeros(count, 144),
        known_noise=torch.zeros(count, 144),
    )
    assert len(results) == 8
    assert all(np.isfinite(result.deployed_pred_pose).all() for result in results)
