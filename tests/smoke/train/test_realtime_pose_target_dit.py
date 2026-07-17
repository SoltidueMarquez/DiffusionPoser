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
from model.diffusionposer_dit import DiffusionPoserDiT
from model.realtime_pose_target_dit import RealtimePoseTargetDiT
from utils.model_util import create_model_and_diffusion
from train.training_loop import validate_loaded_state_dict_keys


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


def test_target_dit_projects_predicted_stationary_logits_before_inpaint_blending():
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    model = RealtimePoseTargetDiT(
        input_feats=schema.feature_dim,
        schema_name=schema.name,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_seq_len=REALTIME_POSE_SEQ_LEN,
    )
    target_slice = schema.target_slice()
    stationary_slice = schema.stationary_prob_slice()
    stationary_start = stationary_slice.start - target_slice.start
    stationary_stop = stationary_slice.stop - target_slice.start
    stationary_logits = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    with torch.no_grad():
        model.output_proj.weight.zero_()
        model.output_proj.bias.zero_()
        model.output_proj.bias[0] = 2.0
        model.output_proj.bias[stationary_start:stationary_stop] = stationary_logits

    x = torch.zeros(1, schema.feature_dim, REALTIME_POSE_SEQ_LEN)
    known_stationary = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95])
    x[:, stationary_slice, REALTIME_POSE_TARGET_START] = known_stationary
    predicted_mask = torch.zeros_like(x, dtype=torch.bool)
    predicted_mask[:, target_slice, REALTIME_POSE_TARGET_START] = True

    predicted = model(x, torch.zeros(1), inpaint_cond=predicted_mask)
    predicted_stationary = predicted[:, stationary_slice, REALTIME_POSE_TARGET_START]
    assert torch.allclose(predicted_stationary, torch.sigmoid(stationary_logits)[None])
    assert torch.all((predicted_stationary >= 0.0) & (predicted_stationary <= 1.0))
    assert predicted[0, target_slice.start, REALTIME_POSE_TARGET_START].item() == pytest.approx(2.0)

    predicted_stationary.sum().backward()
    stationary_grad = model.output_proj.bias.grad[stationary_start:stationary_stop]
    assert torch.isfinite(stationary_grad).all()
    assert torch.all(stationary_grad > 0.0)

    known = model(x, torch.zeros(1), inpaint_cond=torch.zeros_like(x, dtype=torch.bool))
    assert torch.equal(
        known[:, stationary_slice, REALTIME_POSE_TARGET_START],
        known_stationary[None],
    )


def test_models_have_single_motion_output_and_no_stationary_head():
    schema = get_schema_spec(REALTIME_POSE_SCHEMA_NAME)
    model = DiffusionPoserDiT(
        input_feats=schema.feature_dim,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        max_seq_len=REALTIME_POSE_SEQ_LEN,
    )
    assert not hasattr(model, "stationary_head")
    x = torch.zeros(1, schema.feature_dim, REALTIME_POSE_SEQ_LEN)
    output = model(x, torch.zeros(1), inpaint_cond=torch.zeros_like(x, dtype=torch.bool))
    assert torch.is_tensor(output)
    assert output.shape == x.shape


def test_checkpoint_loading_rejects_all_missing_or_unexpected_keys():
    validate_loaded_state_dict_keys(missing_keys=[], unexpected_keys=[], source="checkpoint")
    with pytest.raises(RuntimeError, match="stationary_head"):
        validate_loaded_state_dict_keys(
            missing_keys=[],
            unexpected_keys=["stationary_head.net.0.weight"],
            source="checkpoint",
        )


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
    assert [value.name for value in graph.output] == ["pred_x0"]
    assert len(graph.output) == 1
    assert any(node.op_type == "Sigmoid" for node in graph.node)
