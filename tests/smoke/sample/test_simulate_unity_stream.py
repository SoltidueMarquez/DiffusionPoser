from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import encode_realtime_pose_features
from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_START,
    REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    get_schema_spec,
)
from sample.simulate_unity_stream import (
    IDENTITY_6D,
    encode_unity_tracker_frame,
    simulate_unity_stream,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


class FixedV2Diffusion:
    def __init__(self, yaw_delta: float):
        self.yaw_delta = float(yaw_delta)
        self.calls = 0
        self.conditioned_inputs = []

    def p_sample_loop(self, model, shape, noise, clip_denoised, model_kwargs):
        del model, noise, clip_denoised
        self.calls += 1
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
        sample[:, schema.foot_contact_slice(), REALTIME_POSE_TARGET_START] = 0.0
        return sample


def test_unity_tracker_frame_matches_dataset_tracker_reference_v2():
    source = build_toy_realtime_source(frame_count=70)
    sensor_valid = np.ones((70, 6), dtype=bool)
    reference = encode_realtime_pose_features(
        {**source, "sensor_valid": sensor_valid},
        schema_name=REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    )
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    frame_index = 12

    encoded = encode_unity_tracker_frame(
        tracker_pos_world=source["tracker_pos_world"][frame_index],
        tracker_rot_world_6d=source["tracker_rot_world_6d"][frame_index],
        sensor_valid=sensor_valid[frame_index],
        reference_root_yaw=float(source["root_yaw"][frame_index - 1]),
        schema_name=REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
        root_pos_world=source["root_pos_world"][frame_index],
    )

    np.testing.assert_allclose(encoded[schema.tracker_pos_slice()], reference[frame_index, schema.tracker_pos_slice()])
    np.testing.assert_allclose(encoded[schema.tracker_rot_slice()], reference[frame_index, schema.tracker_rot_slice()])
    np.testing.assert_array_equal(encoded[schema.sensor_valid_slice()], reference[frame_index, schema.sensor_valid_slice()])


def test_simulate_unity_stream_predicts_after_warmup_and_updates_root_yaw():
    source = build_toy_realtime_source(frame_count=63)
    sensor_valid = np.ones((63, 6), dtype=bool)
    diffusion = FixedV2Diffusion(yaw_delta=0.1)

    payload = simulate_unity_stream(
        model=object(),
        diffusion=diffusion,
        tracker_pos_world=source["tracker_pos_world"],
        tracker_rot_world_6d=source["tracker_rot_world_6d"],
        sensor_valid=sensor_valid,
        device=torch.device("cpu"),
        use_ddim=False,
        schema_name=REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
        normalizer=None,
        initial_root_yaw=0.0,
    )

    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    assert diffusion.calls == 3
    assert payload["predicted_features_raw"].shape == (1, 63, schema.feature_dim)
    assert not payload["eval_frame_mask"][0, :REALTIME_POSE_TARGET_START].any()
    assert payload["eval_frame_mask"][0, REALTIME_POSE_TARGET_START:].all()
    np.testing.assert_allclose(payload["root_yaw_predicted"][0, :REALTIME_POSE_TARGET_START], 0.0)
    np.testing.assert_allclose(
        payload["root_yaw_predicted"][0, REALTIME_POSE_TARGET_START:],
        np.asarray([0.1, 0.2, 0.3]),
        atol=1e-6,
    )


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
        schema_name=REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
        normalizer=None,
        initial_root_yaw=0.0,
    )

    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    conditioned = diffusion.conditioned_inputs[0][0]
    history_frames = slice(0, REALTIME_POSE_TARGET_START)
    expected_history = np.stack(
        [
            encode_unity_tracker_frame(
                tracker_pos_world=source["tracker_pos_world"][frame_index],
                tracker_rot_world_6d=source["tracker_rot_world_6d"][frame_index],
                sensor_valid=sensor_valid[frame_index],
                reference_root_yaw=0.0,
                schema_name=REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
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
