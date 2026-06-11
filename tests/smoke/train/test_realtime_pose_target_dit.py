from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    get_schema_spec,
)
from export.export_sentis_denoiser import SentisDenoiserWrapper, export_onnx
from utils.model_util import create_model_and_diffusion


def test_target_dit_predicts_only_target_slice_shape():
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    args = SimpleNamespace(
        model_arch="target_dit",
        schema=schema.name,
        input_feats=schema.feature_dim,
        latent_dim=32,
        layers=1,
        heads=4,
        dropout=0.0,
        zero_init=False,
        max_seq_len=REALTIME_POSE_SEQ_LEN,
        diffusion_steps=4,
        ts_respace="",
        noise_schedule="cosine",
        predict_xstart=1,
        sigma_small=True,
    )
    model, _diffusion = create_model_and_diffusion(args)
    x = torch.zeros(2, schema.feature_dim, REALTIME_POSE_SEQ_LEN)
    x[:, schema.sensor_valid_slice(), :] = 1.0
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, schema.target_slice(), REALTIME_POSE_TARGET_START] = True
    y = model(x, torch.zeros(2), inpaint_cond=mask)
    assert tuple(y.shape) == tuple(x.shape)
    non_target = torch.ones_like(x, dtype=torch.bool)
    non_target[:, schema.target_slice(), REALTIME_POSE_TARGET_START] = False
    assert torch.allclose(y[non_target], x[non_target])

    closed_mask = torch.zeros_like(x, dtype=torch.bool)
    closed = model(x, torch.zeros(2), inpaint_cond=closed_mask)
    assert torch.allclose(closed[:, schema.target_slice(), REALTIME_POSE_TARGET_START], x[:, schema.target_slice(), REALTIME_POSE_TARGET_START])


def test_target_dit_without_schema_uses_current_default_schema():
    schema = get_schema_spec(DEFAULT_REALTIME_POSE_SCHEMA_NAME)
    args = SimpleNamespace(
        model_arch="target_dit",
        input_feats=schema.feature_dim,
        latent_dim=32,
        layers=1,
        heads=4,
        dropout=0.0,
        zero_init=False,
        max_seq_len=REALTIME_POSE_SEQ_LEN,
        diffusion_steps=4,
        ts_respace="",
        noise_schedule="cosine",
        predict_xstart=1,
        sigma_small=True,
    )
    model, _diffusion = create_model_and_diffusion(args)
    assert model.schema.name == DEFAULT_REALTIME_POSE_SCHEMA_NAME


def test_target_dit_onnx_keeps_sentis_input_contract(tmp_path):
    onnx = pytest.importorskip("onnx")
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    args = SimpleNamespace(
        model_arch="target_dit",
        schema=schema.name,
        input_feats=schema.feature_dim,
        latent_dim=32,
        layers=1,
        heads=4,
        dropout=0.0,
        zero_init=False,
        max_seq_len=REALTIME_POSE_SEQ_LEN,
        diffusion_steps=4,
        ts_respace="",
        noise_schedule="cosine",
        predict_xstart=1,
        sigma_small=True,
    )
    model, _diffusion = create_model_and_diffusion(args)
    onnx_path = tmp_path / "target_dit.onnx"
    export_onnx(
        wrapper=SentisDenoiserWrapper(model),
        onnx_path=onnx_path,
        feature_dim=schema.feature_dim,
        sequence_length=REALTIME_POSE_SEQ_LEN,
        opset=17,
        device=torch.device("cpu"),
        schema_name=schema.name,
    )
    graph = onnx.load(str(onnx_path)).graph
    input_names = {value.name for value in graph.input}
    assert {"x_t", "timestep", "inpaint_mask", "valid_frame_mask"}.issubset(input_names)
