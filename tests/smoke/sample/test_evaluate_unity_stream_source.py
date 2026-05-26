from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import encode_realtime_pose_features
from data_loaders.sensor_masking import REALTIME_POSE_TARGET_START, REALTIME_POSE_V2_CONTACT_SCHEMA_NAME, get_schema_spec
from sample.evaluate_unity_stream_source import build_long_sequence_payload, repeat_source_sequence
from sample.render_realtime_pose_comparison import render_realtime_pose_comparison
from sample.simulate_unity_stream import IDENTITY_6D
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


class FixedV2Diffusion:
    def __init__(self, yaw_delta: float = 0.0):
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


def build_source_with_sensor_valid(frame_count: int) -> dict[str, np.ndarray]:
    source = build_toy_realtime_source(frame_count=frame_count)
    source["sensor_valid"] = np.ones((frame_count, 6), dtype=bool)
    return source


def test_long_sequence_payload_aligns_gt_and_skips_warmup():
    source = build_source_with_sensor_valid(frame_count=62)
    diffusion = FixedV2Diffusion(yaw_delta=0.05)

    payload = build_long_sequence_payload(
        model=object(),
        diffusion=diffusion,
        source=source,
        device=torch.device("cpu"),
        use_ddim=False,
        normalizer=None,
        root_correction=False,
    )

    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    assert diffusion.calls == 2
    assert payload["reference_features_raw"].shape == (1, 62, schema.feature_dim)
    assert payload["predicted_features_raw"].shape == (1, 62, schema.feature_dim)
    assert payload["reference_joints_world"].shape == (1, 62, 24, 3)
    assert payload["predicted_joints_world"].shape == (1, 62, 24, 3)
    assert not payload["eval_frame_mask"][0, :REALTIME_POSE_TARGET_START].any()
    assert payload["eval_frame_mask"][0, REALTIME_POSE_TARGET_START:].all()
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    reference_features = encode_realtime_pose_features(source, schema_name=REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    np.testing.assert_allclose(
        payload["predicted_features_raw"][0, :REALTIME_POSE_TARGET_START, schema.target_slice()],
        reference_features[:REALTIME_POSE_TARGET_START, schema.target_slice()],
    )
    np.testing.assert_allclose(payload["root_yaw_predicted"][0, :REALTIME_POSE_TARGET_START], source["root_yaw"][:REALTIME_POSE_TARGET_START])
    conditioned = diffusion.conditioned_inputs[0][0]
    np.testing.assert_allclose(
        conditioned[schema.target_slice(), :REALTIME_POSE_TARGET_START].T,
        reference_features[:REALTIME_POSE_TARGET_START, schema.target_slice()],
    )
    np.testing.assert_allclose(conditioned[schema.target_slice(), REALTIME_POSE_TARGET_START], 0.0)
    next_conditioned = diffusion.conditioned_inputs[1][0]
    np.testing.assert_allclose(
        next_conditioned[schema.root_yaw_delta_slice(), REALTIME_POSE_TARGET_START - 1],
        np.asarray([np.sin(0.05), np.cos(0.05)], dtype=np.float32),
    )
    assert not np.allclose(
        next_conditioned[schema.root_yaw_delta_slice(), REALTIME_POSE_TARGET_START - 1],
        reference_features[REALTIME_POSE_TARGET_START, schema.root_yaw_delta_slice()],
    )
    assert payload["metadata"].item()["history_pose_source"] == "reference"
    assert payload["metadata"].item()["warmup_target_source"] == "first_frame"
    assert payload["metadata"].item()["reference_history_frames"] == REALTIME_POSE_TARGET_START
    assert payload["metadata"].item()["autoregressive_after_warmup"] is True


def test_predicted_history_identity_warmup_uses_identity_target():
    source = build_source_with_sensor_valid(frame_count=62)
    diffusion = FixedV2Diffusion(yaw_delta=0.0)

    payload = build_long_sequence_payload(
        model=object(),
        diffusion=diffusion,
        source=source,
        device=torch.device("cpu"),
        use_ddim=False,
        normalizer=None,
        history_pose_source="predicted",
        warmup_target_source="identity",
    )

    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    expected_pose = np.tile(IDENTITY_6D, 24)
    np.testing.assert_allclose(
        payload["predicted_features_raw"][0, :REALTIME_POSE_TARGET_START, schema.body_pose_slice()],
        np.tile(expected_pose[None], (REALTIME_POSE_TARGET_START, 1)),
    )
    np.testing.assert_allclose(
        payload["predicted_features_raw"][0, :REALTIME_POSE_TARGET_START, schema.root_yaw_delta_slice()],
        np.tile(np.asarray([0.0, 1.0], dtype=np.float32)[None], (REALTIME_POSE_TARGET_START, 1)),
    )
    assert payload["metadata"].item()["warmup_target_source"] == "identity"


def test_repeat_source_sequence_doubles_time_fields():
    source = build_source_with_sensor_valid(frame_count=61)
    repeated = repeat_source_sequence(source, loop_count=2)

    assert repeated["tracker_pos_world"].shape[0] == 122
    assert repeated["joints_world"].shape[0] == 122
    assert repeated["sensor_valid"].shape == (122, 6)
    assert repeated["joint_offsets_parent"].shape == (24, 3)


def test_render_realtime_pose_comparison_writes_mp4(tmp_path):
    source = build_source_with_sensor_valid(frame_count=5)
    output_path = tmp_path / "comparison.mp4"

    render_realtime_pose_comparison(
        output_path=output_path,
        reference_joints=source["joints_world"][None],
        predicted_joints=source["joints_world"][None].copy(),
        tracker_pos_world=source["tracker_pos_world"][None],
        sensor_valid=source["sensor_valid"][None],
        eval_frame_mask=np.ones((1, 5), dtype=bool),
        root_yaw_reference=source["root_yaw"][None],
        root_yaw_predicted=source["root_yaw"][None].copy(),
        fps=5,
        stride=1,
        camera_mode="follow",
        layout="overlay",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
