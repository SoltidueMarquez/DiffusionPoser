from __future__ import annotations

import torch

from data_loaders.sensor_masking import (
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_TARGET_START,
    REALTIME_POSE_V2_CONTACT_SCHEMA_NAME,
    get_schema_spec,
)
from model.causal_attention import build_frame_causal_mask, build_target_dit_causal_mask
from model.diffusionposer_dit import DiffusionPoserDiT
from model.realtime_pose_target_dit import RealtimePoseTargetDiT


def test_frame_causal_mask_blocks_future_frames():
    mask = build_frame_causal_mask(seq_len=5, device=torch.device("cpu"))
    assert tuple(mask.shape) == (5, 5)
    assert mask.dtype == torch.bool
    for query in range(5):
        for key in range(5):
            assert bool(mask[query, key]) is (key > query)


def test_target_dit_causal_mask_token_visibility():
    seq_len = 5
    tracker_count = 2
    target_frame = 3
    mask = build_target_dit_causal_mask(
        seq_len=seq_len,
        tracker_count=tracker_count,
        target_frame=target_frame,
        device=torch.device("cpu"),
    )
    frame_start = 1 + tracker_count
    assert tuple(mask.shape) == (frame_start + seq_len, frame_start + seq_len)

    # target 能读自身、当前 sensor 条件，以及目标帧之前/当前的 frame；不能读未来 frame。
    assert not bool(mask[0, 0])
    assert not mask[0, 1:frame_start].any()
    assert not mask[0, frame_start:frame_start + target_frame + 1].any()
    assert bool(mask[0, frame_start + target_frame + 1])

    # sensor token 不能读 target，也不能读目标帧之后的信息。
    assert bool(mask[1, 0])
    assert not mask[1, 1:frame_start].any()
    assert not mask[1, frame_start:frame_start + target_frame + 1].any()
    assert bool(mask[1, frame_start + target_frame + 1])

    # frame token 不读 target/sensor，只沿时间读过去和当前帧。
    for frame in range(seq_len):
        row = frame_start + frame
        assert mask[row, :frame_start].all()
        for key_frame in range(seq_len):
            assert bool(mask[row, frame_start + key_frame]) is (key_frame > frame)


def test_diffusionposer_dit_prefix_output_does_not_depend_on_future_frames():
    torch.manual_seed(7)
    seq_len = 12
    prefix_len = 6
    model = DiffusionPoserDiT(
        input_feats=REALTIME_POSE_INPUT_DIM,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_seq_len=seq_len,
    )
    model.eval()
    x_a = torch.randn(1, REALTIME_POSE_INPUT_DIM, seq_len)
    x_b = x_a.clone()
    x_b[:, :, prefix_len:] = torch.randn_like(x_b[:, :, prefix_len:])
    inpaint_cond = torch.ones_like(x_a, dtype=torch.bool)
    timestep = torch.zeros(1)

    with torch.no_grad():
        y_a = model(x_a, timestep, inpaint_cond=inpaint_cond)
        y_b = model(x_b, timestep, inpaint_cond=inpaint_cond)

    assert torch.allclose(y_a[:, :, :prefix_len], y_b[:, :, :prefix_len], atol=1e-6)


def test_target_dit_target_output_does_not_depend_on_future_frames():
    torch.manual_seed(11)
    schema = get_schema_spec(REALTIME_POSE_V2_CONTACT_SCHEMA_NAME)
    seq_len = REALTIME_POSE_TARGET_START + 5
    model = RealtimePoseTargetDiT(
        input_feats=schema.feature_dim,
        schema_name=schema.name,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_seq_len=seq_len,
    )
    model.eval()
    x_a = torch.randn(1, schema.feature_dim, seq_len)
    x_a[:, schema.sensor_valid_slice(), :] = 1.0
    x_b = x_a.clone()
    x_b[:, :, REALTIME_POSE_TARGET_START + 1:] = torch.randn_like(x_b[:, :, REALTIME_POSE_TARGET_START + 1:])
    inpaint_cond = torch.zeros_like(x_a, dtype=torch.bool)
    inpaint_cond[:, schema.target_slice(), REALTIME_POSE_TARGET_START] = True
    timestep = torch.zeros(1)

    with torch.no_grad():
        y_a = model(x_a, timestep, inpaint_cond=inpaint_cond)
        y_b = model(x_b, timestep, inpaint_cond=inpaint_cond)

    target = schema.target_slice()
    assert not torch.allclose(x_a[:, :, REALTIME_POSE_TARGET_START + 1:], x_b[:, :, REALTIME_POSE_TARGET_START + 1:])
    assert torch.allclose(
        y_a[:, target, REALTIME_POSE_TARGET_START],
        y_b[:, target, REALTIME_POSE_TARGET_START],
        atol=1e-6,
    )
