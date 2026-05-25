from __future__ import annotations

import numpy as np
import torch

from data_loaders.sensor_masking import REALTIME_POSE_TARGET_START, REALTIME_POSE_V2_CONTACT_SCHEMA_NAME, get_schema_spec
from sample.evaluate_unity_stream_source import build_long_sequence_payload, repeat_source_sequence
from sample.render_realtime_pose_comparison import render_realtime_pose_comparison
from sample.simulate_unity_stream import IDENTITY_6D
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


class FixedV2Diffusion:
    def __init__(self, yaw_delta: float = 0.0):
        self.yaw_delta = float(yaw_delta)
        self.calls = 0

    def p_sample_loop(self, model, shape, noise, clip_denoised, model_kwargs):
        del model, noise, clip_denoised
        self.calls += 1
        sample = model_kwargs["y"]["inpainted_motion"].clone()
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
    np.testing.assert_allclose(
        payload["predicted_features_raw"][0, 0, schema.target_slice()],
        payload["reference_features_raw"][0, 0, schema.target_slice()],
    )


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
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
