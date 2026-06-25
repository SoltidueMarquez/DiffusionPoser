from __future__ import annotations

import numpy as np
import pytest
import torch

from data_loaders.realtime_pose_dataset import encode_realtime_pose_features
from data_loaders.realtime_pose_kinematics import make_yaw_rotation_np
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    HIP_TRACKER_INDEX,
    LEFT_HAND_TRACKER_INDEX,
    REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_TARGET_START,
    REALTIME_POSE_SEQ_LEN,
    SMPL_JOINT_COUNT,
    get_schema_spec,
)
from sample.ik_initializer import build_tracker_pose_init_image
from sample.simulate_unity_stream import (
    IDENTITY_6D,
    apply_tracker_position_ik,
    clamp_body_pose_delta,
    encode_unity_tracker_frame,
    estimate_root_pos_from_hip_tracker,
    fk_tracker_positions_from_target,
    initial_target_feature,
    simulate_unity_stream,
    smooth_tracker_positions_for_ik,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


class FixedV2Diffusion:
    def __init__(self, yaw_delta: float):
        self.yaw_delta = float(yaw_delta)
        self.num_timesteps = 10
        self.calls = 0
        self.conditioned_inputs = []
        self.init_images = []
        self.skip_timesteps = []

    def p_sample_loop(self, model, shape, noise, clip_denoised, model_kwargs, init_image=None, skip_timesteps=0):
        del model, noise, clip_denoised
        self.calls += 1
        self.init_images.append(None if init_image is None else init_image.detach().cpu().clone())
        self.skip_timesteps.append(int(skip_timesteps))
        sample = model_kwargs["y"]["inpainted_motion"].clone()
        self.conditioned_inputs.append(sample.detach().cpu().numpy().copy())
        schema = get_schema_spec(model_kwargs["y"]["schema_name"])
        sample[:, schema.body_pose_slice(), REALTIME_POSE_TARGET_START] = torch.from_numpy(
            np.tile(IDENTITY_6D, 24)
        ).to(sample.device)
        sample[:, schema.root_yaw_delta_slice(), REALTIME_POSE_TARGET_START] = torch.tensor(
            [np.sin(self.yaw_delta), np.cos(self.yaw_delta)],
            dtype=sample.dtype,
            device=sample.device,
        )
        sample[:, schema.root_delta_xz_slice(), REALTIME_POSE_TARGET_START] = 0.0
        sample[:, schema.root_height_slice(), REALTIME_POSE_TARGET_START] = 0.0
        sample[:, schema.stationary_prob_slice(), REALTIME_POSE_TARGET_START] = 0.0
        return sample


def resolve_runtime_schema_for_test(*args, **kwargs):
    from utils.schema_resolution import resolve_runtime_schema

    return resolve_runtime_schema(*args, **kwargs)


def test_runtime_schema_resolution_prefers_checkpoint_exact_schema():
    schema = resolve_runtime_schema_for_test(
        cli_schema=None,
        checkpoint_args={"schema": REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME},
    )

    assert schema == REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME


def test_runtime_schema_resolution_reads_checkpoint_schema_name_fallback():
    schema = resolve_runtime_schema_for_test(
        cli_schema=None,
        checkpoint_args={"schema_name": REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME},
    )

    assert schema == REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME


def test_runtime_schema_resolution_rejects_explicit_cli_mismatch():
    with pytest.raises(ValueError, match="checkpoint schema"):
        resolve_runtime_schema_for_test(
            cli_schema=DEFAULT_REALTIME_POSE_SCHEMA_NAME,
            checkpoint_args={"schema": REALTIME_POSE_BODY_FBX_LOCAL_ROOT_Y0_SCHEMA_NAME},
            cli_schema_explicit=True,
        )


def test_runtime_schema_resolution_uses_cli_or_default_without_checkpoint_schema():
    assert (
        resolve_runtime_schema_for_test(
            cli_schema=REALTIME_POSE_SCHEMA_NAME,
            checkpoint_args={},
            cli_schema_explicit=True,
        )
        == REALTIME_POSE_SCHEMA_NAME
    )
    assert resolve_runtime_schema_for_test(cli_schema=None, checkpoint_args=None) == DEFAULT_REALTIME_POSE_SCHEMA_NAME


def test_unity_tracker_frame_matches_dataset_tracker_reference_v2():
    source = build_toy_realtime_source(frame_count=70)
    sensor_valid = np.ones((70, 6), dtype=bool)
    reference = encode_realtime_pose_features(
        {**source, "sensor_valid": sensor_valid},
        schema_name=REALTIME_POSE_SCHEMA_NAME,
    )
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    frame_index = 12

    encoded = encode_unity_tracker_frame(
        tracker_pos_world=source["tracker_pos_world"][frame_index],
        tracker_rot_world_6d=source["tracker_rot_world_6d"][frame_index],
        sensor_valid=sensor_valid[frame_index],
        reference_root_yaw=float(source["root_yaw"][frame_index - 1]),
        schema_name=REALTIME_POSE_SCHEMA_NAME,
        root_pos_world=source["root_pos_world"][frame_index],
    )

    np.testing.assert_allclose(encoded[schema.tracker_pos_slice()], reference[frame_index, schema.tracker_pos_slice()])
    np.testing.assert_allclose(encoded[schema.tracker_rot_slice()], reference[frame_index, schema.tracker_rot_slice()])
    np.testing.assert_array_equal(encoded[schema.sensor_valid_slice()], reference[frame_index, schema.sensor_valid_slice()])


def test_hip_tracker_root_estimate_removes_body_fbx_pelvis_offset():
    root_pos = np.asarray([1.2, 0.0, -0.4], dtype=np.float32)
    root_yaw = 0.55
    offsets = np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float32)
    offsets[0] = np.asarray([0.12, 0.97, 0.08], dtype=np.float32)
    yaw_rotation = make_yaw_rotation_np(np.asarray([root_yaw], dtype=np.float64))[0]

    tracker_pos = np.zeros((6, 3), dtype=np.float32)
    tracker_pos[HIP_TRACKER_INDEX] = root_pos + (yaw_rotation @ offsets[0].astype(np.float64)).astype(np.float32)

    estimated = estimate_root_pos_from_hip_tracker(
        tracker_pos,
        root_yaw=root_yaw,
        joint_offsets_parent=offsets,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
    )

    np.testing.assert_allclose(estimated, root_pos, atol=1e-6)


def test_simulate_unity_stream_corrects_root_state_from_hip_tracker():
    source = build_toy_realtime_source(frame_count=63)
    sensor_valid = np.ones((63, 6), dtype=bool)
    measured_yaw = source["root_yaw"].astype(np.float32)
    tracker_rot = source["tracker_rot_world_6d"].copy()
    tracker_rot[:, 3, 0] = np.sin(measured_yaw)
    tracker_rot[:, 3, 1] = 0.0
    tracker_rot[:, 3, 2] = np.cos(measured_yaw)
    tracker_rot[:, 3, 3:] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    diffusion = FixedV2Diffusion(yaw_delta=0.1)

    payload = simulate_unity_stream(
        model=object(),
        diffusion=diffusion,
        tracker_pos_world=source["tracker_pos_world"],
        tracker_rot_world_6d=tracker_rot,
        sensor_valid=sensor_valid,
        device=torch.device("cpu"),
        use_ddim=False,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
        normalizer=None,
        initial_root_yaw=0.0,
        joint_offsets_parent=source["joint_offsets_parent"],
        joint_rest_local_rotations_6d=source.get("joint_rest_local_rotations_6d"),
    )

    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    assert diffusion.calls == 3
    assert payload["predicted_features_raw"].shape == (1, 63, schema.feature_dim)
    assert not payload["eval_frame_mask"][0, :REALTIME_POSE_TARGET_START].any()
    assert payload["eval_frame_mask"][0, REALTIME_POSE_TARGET_START:].all()
    np.testing.assert_allclose(payload["root_yaw_predicted"][0, :REALTIME_POSE_TARGET_START], 0.0)
    np.testing.assert_allclose(
        payload["root_yaw_predicted"][0, REALTIME_POSE_TARGET_START:],
        measured_yaw[REALTIME_POSE_TARGET_START:],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        payload["root_pos_world_predicted"][0, REALTIME_POSE_TARGET_START:][:, [0, 2]],
        source["root_pos_world"][REALTIME_POSE_TARGET_START:][:, [0, 2]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        payload["root_pos_world_predicted"][0, REALTIME_POSE_TARGET_START:][:, 1],
        0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        payload["predicted_features_raw"][0, REALTIME_POSE_TARGET_START:, schema.root_height_slice()],
        source["tracker_pos_world"][REALTIME_POSE_TARGET_START:, 3, 1:2],
        atol=1e-6,
    )


def test_tracker_ik_pulls_known_hand_joint_toward_tracker():
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    offsets = np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float32)
    offsets[3] = np.asarray([0.0, 0.20, 0.0], dtype=np.float32)
    offsets[6] = np.asarray([0.0, 0.20, 0.0], dtype=np.float32)
    offsets[9] = np.asarray([0.0, 0.20, 0.0], dtype=np.float32)
    offsets[13] = np.asarray([-0.05, 0.05, 0.0], dtype=np.float32)
    offsets[16] = np.asarray([-0.18, 0.0, 0.0], dtype=np.float32)
    offsets[18] = np.asarray([-0.22, 0.0, 0.0], dtype=np.float32)
    offsets[20] = np.asarray([-0.22, 0.0, 0.0], dtype=np.float32)
    frame = initial_target_feature(schema.name, root_height=0.0)
    root_pos = np.zeros((3,), dtype=np.float32)
    tracker_pos = fk_tracker_positions_from_target(
        target_raw=frame,
        root_pos_world=root_pos,
        root_yaw=0.0,
        joint_offsets_parent=offsets,
        schema_name=schema.name,
    )
    tracker_pos[LEFT_HAND_TRACKER_INDEX] += np.asarray([0.12, -0.12, 0.22], dtype=np.float32)
    sensor_valid = np.zeros((6,), dtype=bool)
    sensor_valid[[HIP_TRACKER_INDEX, LEFT_HAND_TRACKER_INDEX]] = True

    before = fk_tracker_positions_from_target(
        target_raw=frame,
        root_pos_world=root_pos,
        root_yaw=0.0,
        joint_offsets_parent=offsets,
        schema_name=schema.name,
    )
    corrected = apply_tracker_position_ik(
        predicted_frame_raw=frame,
        root_pos_world=root_pos,
        root_yaw=0.0,
        tracker_pos_world=tracker_pos,
        sensor_valid=sensor_valid,
        joint_offsets_parent=offsets,
        schema_name=schema.name,
        iterations=80,
        lr=0.05,
        blend=1.0,
        delta_limit=0.0,
    )
    after = fk_tracker_positions_from_target(
        target_raw=corrected,
        root_pos_world=root_pos,
        root_yaw=0.0,
        joint_offsets_parent=offsets,
        schema_name=schema.name,
    )

    before_error = np.linalg.norm(before[LEFT_HAND_TRACKER_INDEX] - tracker_pos[LEFT_HAND_TRACKER_INDEX])
    after_error = np.linalg.norm(after[LEFT_HAND_TRACKER_INDEX] - tracker_pos[LEFT_HAND_TRACKER_INDEX])
    assert after_error < before_error * 0.55


def test_tracker_pose_init_refines_known_hand_tracker_with_rotation_loss_enabled():
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    offsets = np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float32)
    offsets[3] = np.asarray([0.0, 0.20, 0.0], dtype=np.float32)
    offsets[6] = np.asarray([0.0, 0.20, 0.0], dtype=np.float32)
    offsets[9] = np.asarray([0.0, 0.20, 0.0], dtype=np.float32)
    offsets[13] = np.asarray([-0.05, 0.05, 0.0], dtype=np.float32)
    offsets[16] = np.asarray([-0.18, 0.0, 0.0], dtype=np.float32)
    offsets[18] = np.asarray([-0.22, 0.0, 0.0], dtype=np.float32)
    offsets[20] = np.asarray([-0.22, 0.0, 0.0], dtype=np.float32)
    frame = initial_target_feature(schema.name, root_height=0.0)
    tracker_pos = fk_tracker_positions_from_target(
        target_raw=frame,
        root_pos_world=np.zeros((3,), dtype=np.float32),
        root_yaw=0.0,
        joint_offsets_parent=offsets,
        schema_name=schema.name,
    )
    tracker_pos[LEFT_HAND_TRACKER_INDEX] += np.asarray([0.12, -0.12, 0.22], dtype=np.float32)
    tracker_rot = np.tile(IDENTITY_6D, (6, 1)).astype(np.float32)
    sensor_valid = np.zeros((6,), dtype=bool)
    sensor_valid[[HIP_TRACKER_INDEX, LEFT_HAND_TRACKER_INDEX]] = True
    conditioned = torch.zeros((1, schema.feature_dim, REALTIME_POSE_SEQ_LEN), dtype=torch.float32)
    conditioned[0, schema.target_slice(), REALTIME_POSE_TARGET_START - 1] = torch.from_numpy(frame[schema.target_slice()])
    conditioned[0, schema.tracker_pos_slice(), REALTIME_POSE_TARGET_START] = torch.from_numpy(tracker_pos.reshape(-1))
    conditioned[0, schema.tracker_rot_slice(), REALTIME_POSE_TARGET_START] = torch.from_numpy(tracker_rot.reshape(-1))
    conditioned[0, schema.sensor_valid_slice(), REALTIME_POSE_TARGET_START] = torch.from_numpy(sensor_valid.astype(np.float32))

    init_image = build_tracker_pose_init_image(
        conditioned_x=conditioned,
        schema_name=schema.name,
        joint_offsets_parent=torch.from_numpy(offsets[None]),
        iterations=80,
        lr=0.05,
        pos_weight=1.0,
        rot_weight=0.2,
        reg_weight=0.01,
        delta_limit=0.0,
    )

    before = fk_tracker_positions_from_target(
        target_raw=frame,
        root_pos_world=np.zeros((3,), dtype=np.float32),
        root_yaw=0.0,
        joint_offsets_parent=offsets,
        schema_name=schema.name,
    )
    after_feature = init_image[0, :, REALTIME_POSE_TARGET_START].detach().numpy()
    after = fk_tracker_positions_from_target(
        target_raw=after_feature,
        root_pos_world=np.zeros((3,), dtype=np.float32),
        root_yaw=0.0,
        joint_offsets_parent=offsets,
        schema_name=schema.name,
    )
    before_error = np.linalg.norm(before[LEFT_HAND_TRACKER_INDEX] - tracker_pos[LEFT_HAND_TRACKER_INDEX])
    after_error = np.linalg.norm(after[LEFT_HAND_TRACKER_INDEX] - tracker_pos[LEFT_HAND_TRACKER_INDEX])
    assert np.isfinite(after_feature).all()
    assert after_error < before_error * 0.75


def test_tracker_ik_smoothing_and_delta_clamp_are_bounded():
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    measured = np.zeros((6, 3), dtype=np.float32)
    measured[LEFT_HAND_TRACKER_INDEX] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    previous = np.zeros((6, 3), dtype=np.float32)
    valid = np.zeros((6,), dtype=bool)
    valid[[HIP_TRACKER_INDEX, LEFT_HAND_TRACKER_INDEX]] = True

    smoothed, smoothed_valid = smooth_tracker_positions_for_ik(
        tracker_pos_world=measured,
        sensor_valid=valid,
        previous_smoothed_pos_world=previous,
        previous_valid=valid,
        smoothing=0.6,
    )

    np.testing.assert_array_equal(smoothed_valid, valid)
    np.testing.assert_allclose(smoothed[LEFT_HAND_TRACKER_INDEX], np.asarray([0.4, 0.0, 0.0], dtype=np.float32))

    base = initial_target_feature(schema.name)[schema.body_pose_slice()]
    moved = base.copy()
    moved[:6] += 1.0
    clamped = clamp_body_pose_delta(
        body_pose_parent_6d=moved,
        reference_body_pose_parent_6d=base,
        delta_limit=0.08,
    )
    per_joint_delta = np.linalg.norm(clamped.reshape(24, 6) - base.reshape(24, 6), axis=-1)
    assert float(per_joint_delta.max()) <= 0.081


def test_warmup_window_contains_tracker_history_and_tpose_targets():
    source = build_toy_realtime_source(frame_count=61)
    sensor_valid = np.ones((61, 6), dtype=bool)
    diffusion = FixedV2Diffusion(yaw_delta=0.0)

    simulate_unity_stream(
        model=object(),
        diffusion=diffusion,
        tracker_pos_world=source["tracker_pos_world"],
        tracker_rot_world_6d=source["tracker_rot_world_6d"],
        sensor_valid=sensor_valid,
        device=torch.device("cpu"),
        use_ddim=False,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
        normalizer=None,
        initial_root_yaw=0.0,
    )

    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    conditioned = diffusion.conditioned_inputs[0][0]
    history_frames = slice(0, REALTIME_POSE_TARGET_START)
    expected_history = np.stack(
        [
            encode_unity_tracker_frame(
                tracker_pos_world=source["tracker_pos_world"][frame_index],
                tracker_rot_world_6d=source["tracker_rot_world_6d"][frame_index],
                sensor_valid=sensor_valid[frame_index],
                reference_root_yaw=0.0,
                schema_name=REALTIME_POSE_SCHEMA_NAME,
            )
            for frame_index in range(REALTIME_POSE_TARGET_START)
        ],
        axis=0,
    ).T

    np.testing.assert_allclose(conditioned[schema.tracker_pos_slice(), history_frames], expected_history[schema.tracker_pos_slice()])
    np.testing.assert_allclose(conditioned[schema.tracker_rot_slice(), history_frames], expected_history[schema.tracker_rot_slice()])
    np.testing.assert_allclose(conditioned[schema.sensor_valid_slice(), history_frames], 1.0)
    np.testing.assert_allclose(
        conditioned[schema.body_pose_slice(), history_frames],
        np.tile(np.tile(IDENTITY_6D, 24)[:, None], (1, REALTIME_POSE_TARGET_START)),
    )
    np.testing.assert_allclose(
        conditioned[schema.root_yaw_delta_slice(), history_frames],
        np.tile(np.asarray([0.0, 1.0], dtype=np.float32)[:, None], (1, REALTIME_POSE_TARGET_START)),
    )
    np.testing.assert_allclose(conditioned[schema.target_slice(), REALTIME_POSE_TARGET_START], 0.0)


def test_unity_stream_records_tracker_pose_ik_init_metadata():
    source = build_toy_realtime_source(frame_count=61)
    sensor_valid = np.ones((61, 6), dtype=bool)
    diffusion = FixedV2Diffusion(yaw_delta=0.0)

    payload = simulate_unity_stream(
        model=object(),
        diffusion=diffusion,
        tracker_pos_world=source["tracker_pos_world"],
        tracker_rot_world_6d=source["tracker_rot_world_6d"],
        sensor_valid=sensor_valid,
        device=torch.device("cpu"),
        use_ddim=False,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
        normalizer=None,
        initial_root_yaw=0.0,
        joint_offsets_parent=source["joint_offsets_parent"],
        tracker_ik=False,
        ik_init_mode="tracker_pose",
        ik_init_timestep=3,
    )

    assert diffusion.init_images[0] is not None
    assert diffusion.skip_timesteps[0] == 6
    assert payload["ik_init_mode"].item() == "tracker_pose"
    assert int(payload["ik_init_timestep"]) == 3
    assert int(payload["ik_init_iterations"]) == 16
