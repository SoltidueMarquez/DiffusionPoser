from __future__ import annotations

import numpy as np
import pytest
import torch

from data_loaders.sensor_masking import (
    CORE_THREE_AVAILABLE,
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
from sample import evaluate_progressive_tracker_addition as addition
from sample import evaluate_progressive_tracker_dropout as dropout
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


def test_addition_protocols_cover_all_six_optional_tracker_orders():
    protocols = addition.build_progressive_addition_protocols()

    assert len(protocols) == 6
    assert {
        tuple(protocol.metadata["add_order"]) for protocol in protocols
    } == set(addition.PROGRESSIVE_ADD_ORDERS)
    assert all(
        protocol.warmup_tracker_available == CORE_THREE_AVAILABLE
        for protocol in protocols
    )
    for protocol in protocols:
        assert [sum(stage.tracker_available) for stage in protocol.stages] == [
            3,
            4,
            5,
            6,
        ]
        assert all(
            all(stage.tracker_available[index] for index in CORE_TRACKER_INDICES)
            for stage in protocol.stages
        )
        added = [
            index
            for previous, current in zip(protocol.stages, protocol.stages[1:])
            for index in OPTIONAL_TRACKER_INDICES
            if not previous.tracker_available[index] and current.tracker_available[index]
        ]
        assert sorted(added) == sorted(OPTIONAL_TRACKER_INDICES)
        assert tuple(protocol.metadata["paired_drop_order"]) == tuple(
            reversed(protocol.metadata["add_order"])
        )


def test_addition_schedule_is_reverse_path_of_paired_dropout():
    add_order_names = ("hip", "right_foot", "left_foot")
    add_indices = tuple(
        addition.OPTIONAL_TRACKER_NAME_TO_INDEX[name]
        for name in add_order_names
    )
    paired_drop_indices = tuple(reversed(add_indices))
    add_schedule = addition.build_equal_quarter_tracker_schedule(
        scored_frame_count=61,
        add_order=add_indices,
    )
    drop_schedule = dropout.build_equal_quarter_tracker_schedule(
        scored_frame_count=61,
        drop_order=paired_drop_indices,
    )

    stage_lengths = [
        int(np.count_nonzero(add_schedule.stage_indices == index))
        for index in range(4)
    ]
    assert max(stage_lengths) - min(stage_lengths) <= 1
    add_stage_masks = np.stack(
        [
            add_schedule.tracker_available[add_schedule.stage_indices == index][0]
            for index in range(4)
        ]
    )
    drop_stage_masks = np.stack(
        [
            drop_schedule.tracker_available[drop_schedule.stage_indices == index][0]
            for index in range(4)
        ]
    )
    assert add_stage_masks.sum(axis=1).tolist() == [3, 4, 5, 6]
    np.testing.assert_array_equal(add_stage_masks, drop_stage_masks[::-1])


def test_addition_schedule_requires_four_jitter_capable_stages():
    with pytest.raises(ValueError, match="至少需要 16 个计分帧"):
        addition.build_equal_quarter_tracker_schedule(
            scored_frame_count=15,
            add_order=OPTIONAL_TRACKER_INDICES,
        )


def test_addition_orders_use_the_same_noise_sequence():
    first = create_eval_noise_generator(10, torch.device("cpu"))
    second = create_eval_noise_generator(10, torch.device("cpu"))
    for _ in range(3):
        torch.testing.assert_close(
            torch.randn((1, 144), generator=first),
            torch.randn((1, 144), generator=second),
        )


def test_addition_whole_sequence_metrics_keep_stage_boundary_jump():
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
    assert all(
        stage["pred_jitter_m_per_s3"] == pytest.approx(0.0)
        for stage in stages
    )


def test_addition_main_writes_dynamic_only_json(tmp_path, monkeypatch):
    source_path = tmp_path / "toy_source.npz"
    np.savez(source_path, **build_toy_realtime_source(frame_count=46))
    predictor, dit, diffusion = _tiny_models()
    monkeypatch.setattr(
        addition,
        "read_longseq_source_entries",
        lambda *args, **kwargs: [{"source_path": str(source_path)}],
    )
    monkeypatch.setattr(
        addition,
        "create_model_and_diffusion",
        lambda args: (dit, diffusion),
    )
    monkeypatch.setattr(
        addition,
        "load_checkpoint_model",
        lambda model, path, device, use_ema: (model, "raw"),
    )
    monkeypatch.setattr(
        addition,
        "load_realtime_pose_predictor",
        lambda path, device: predictor,
    )
    monkeypatch.setattr(
        addition,
        "RealtimePoseNormalizer",
        lambda path, disable: None,
    )
    output = tmp_path / "progressive_addition.json"

    payload = addition.main(
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
    assert payload["experiment"] == "progressive_tracker_addition_3_to_6"
    assert payload["schedule"]["warmup_tracker_available"] == list(
        CORE_THREE_AVAILABLE
    )
    assert payload["schedule"]["tracker_counts"] == [3, 4, 5, 6]
    assert len(payload["progressive_orders"]) == 6
    assert all(
        report["paired_drop_order"] == list(reversed(report["add_order"]))
        for report in payload["progressive_orders"]
    )
    assert set(payload["final_metrics"]["mean"]) == set(
        addition.FINAL_METRIC_KEYS
    )
    assert "std_across_add_orders" in payload["final_metrics"]
    assert "baselines" not in payload
    assert "predictor_only" not in payload
    assert [
        stage["tracker_count"] for stage in payload["progressive_stage_summary"]
    ] == [3, 4, 5, 6]
