from __future__ import annotations

import numpy as np
import pytest
import torch

from data_loaders.sensor_masking import (
    ALL_SIX_AVAILABLE,
    CORE_TRACKER_INDICES,
    OPTIONAL_TRACKER_INDICES,
)
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from model.realtime_pose_current_dit import RealtimePoseCurrentDiT
from model.realtime_pose_predictor import RealtimePosePredictor
from sample import evaluate_progressive_tracker_dropout as progressive
from sample.realtime_pose_longseq_evaluator import (
    compute_sequence_metrics_by_stage,
    create_eval_noise_generator,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def _tiny_models():
    predictor = RealtimePosePredictor(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    ).eval()
    dit = RealtimePoseCurrentDiT(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    diffusion = GaussianDiffusion(
        betas=np.asarray([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    return predictor, dit, diffusion


def test_progressive_protocols_cover_all_six_optional_tracker_orders():
    protocols = progressive.build_progressive_dropout_protocols()

    assert len(protocols) == 6
    assert {
        tuple(protocol.metadata["drop_order"]) for protocol in protocols
    } == set(progressive.PROGRESSIVE_DROP_ORDERS)
    assert all(protocol.warmup_tracker_available == ALL_SIX_AVAILABLE for protocol in protocols)
    for protocol in protocols:
        assert [sum(stage.tracker_available) for stage in protocol.stages] == [
            6,
            5,
            4,
            3,
        ]
        assert all(
            all(stage.tracker_available[index] for index in CORE_TRACKER_INDICES)
            for stage in protocol.stages
        )
        dropped = [
            index
            for previous, current in zip(protocol.stages, protocol.stages[1:])
            for index in OPTIONAL_TRACKER_INDICES
            if previous.tracker_available[index] and not current.tracker_available[index]
        ]
        assert sorted(dropped) == sorted(OPTIONAL_TRACKER_INDICES)


def test_equal_quarter_schedule_has_monotonic_tracker_counts():
    schedule = progressive.build_equal_quarter_tracker_schedule(
        scored_frame_count=61,
        drop_order=OPTIONAL_TRACKER_INDICES,
    )

    stage_lengths = [
        int(np.count_nonzero(schedule.stage_indices == index)) for index in range(4)
    ]
    assert max(stage_lengths) - min(stage_lengths) <= 1
    assert [
        int(schedule.tracker_available[schedule.stage_indices == index][0].sum())
        for index in range(4)
    ] == [6, 5, 4, 3]
    assert schedule.tracker_available[:, list(CORE_TRACKER_INDICES)].all()


def test_progressive_schedule_requires_four_jitter_capable_stages():
    with pytest.raises(ValueError, match="至少需要 16 个计分帧"):
        progressive.build_equal_quarter_tracker_schedule(
            scored_frame_count=15,
            drop_order=OPTIONAL_TRACKER_INDICES,
        )


def test_progressive_orders_use_the_same_noise_sequence():
    first = create_eval_noise_generator(10, torch.device("cpu"))
    second = create_eval_noise_generator(10, torch.device("cpu"))
    for _ in range(3):
        torch.testing.assert_close(
            torch.randn((1, 144), generator=first),
            torch.randn((1, 144), generator=second),
        )


def test_whole_sequence_jitter_keeps_stage_boundary_jump():
    frame_count = 16
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float64), (frame_count, 24, 3, 3)
    ).copy()
    target_positions = np.zeros((frame_count, 24, 3), dtype=np.float64)
    predicted_positions = target_positions.copy()
    predicted_positions[..., 0] = np.repeat(
        np.asarray([0.0, 1.0, 2.0, 3.0]), 4
    )[:, None]
    stage_indices = np.repeat(np.arange(4, dtype=np.int64), 4)

    overall, stages = compute_sequence_metrics_by_stage(
        predicted_global_rotations=rotations,
        target_global_rotations=rotations,
        predicted_joint_positions=predicted_positions,
        target_joint_positions=target_positions,
        stage_indices=stage_indices,
        stage_count=4,
        fps=30.0,
    )

    assert overall["pred_jitter_m_per_s3"] > 0.0
    assert all(stage["pred_jitter_m_per_s3"] == pytest.approx(0.0) for stage in stages)


def test_progressive_main_writes_dynamic_only_json(tmp_path, monkeypatch):
    source_path = tmp_path / "toy_source.npz"
    np.savez(source_path, **build_toy_realtime_source(frame_count=46))
    predictor, dit, diffusion = _tiny_models()
    monkeypatch.setattr(
        progressive,
        "read_longseq_source_entries",
        lambda *args, **kwargs: [{"source_path": str(source_path)}],
    )
    monkeypatch.setattr(
        progressive,
        "create_model_and_diffusion",
        lambda args: (dit, diffusion),
    )
    monkeypatch.setattr(
        progressive,
        "load_checkpoint_model",
        lambda model, path, device, use_ema: (model, "raw"),
    )
    monkeypatch.setattr(
        progressive,
        "load_realtime_pose_predictor",
        lambda path, device: predictor,
    )
    monkeypatch.setattr(
        progressive,
        "RealtimePoseNormalizer",
        lambda path, disable: None,
    )
    output = tmp_path / "progressive.json"

    payload = progressive.main(
        [
            "--cuda",
            "false",
            "--normalizer_dir",
            "normalizer",
            "--predictor_model_path",
            "predictor.pt",
            "--dit_model_path",
            "dit.pt",
            "--diffusion_steps",
            "2",
            "--ts_respace",
            "2",
            "--fabrik_iterations",
            "1",
            "--ik_direction_only_quality",
            "0.8",
            "--ik_residual_scale",
            "0.5",
            "--ik_gap_low",
            "0.1",
            "--ik_gap_high",
            "0.5",
            "--output_json",
            str(output),
        ]
    )

    assert output.is_file()
    assert len(payload["progressive_orders"]) == 6
    assert set(payload["final_metrics"]["mean"]) == set(
        progressive.FINAL_METRIC_KEYS
    )
    assert "baselines" not in payload
    assert "predictor_only" not in payload
    assert [
        stage["tracker_count"] for stage in payload["progressive_stage_summary"]
    ] == [6, 5, 4, 3]
