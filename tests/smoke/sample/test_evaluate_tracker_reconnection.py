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
from sample import evaluate_tracker_reconnection as reconnection
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


def test_reconnection_protocols_cover_all_three_optional_trackers() -> None:
    protocols = reconnection.build_tracker_reconnection_protocols(stage_frames=150)

    assert [protocol.name for protocol in protocols] == [
        "reconnect_hip",
        "reconnect_left_foot",
        "reconnect_right_foot",
    ]
    assert all(
        protocol.warmup_tracker_available == CORE_THREE_AVAILABLE
        for protocol in protocols
    )
    for protocol in protocols:
        assert [sum(stage.tracker_available) for stage in protocol.stages] == [3, 4]
        assert all(
            all(stage.tracker_available[index] for index in CORE_TRACKER_INDICES)
            for stage in protocol.stages
        )
        changed = np.flatnonzero(
            ~np.asarray(protocol.stages[0].tracker_available, dtype=bool)
            & np.asarray(protocol.stages[1].tracker_available, dtype=bool)
        )
        assert changed.shape == (1,)
        assert int(changed[0]) in OPTIONAL_TRACKER_INDICES


def test_reconnection_schedule_is_fixed_150_plus_150() -> None:
    tracker_index = reconnection.RECONNECT_TRACKER_NAME_TO_INDEX["hip"]
    schedule = reconnection.build_tracker_reconnection_schedule(
        scored_frame_count=300,
        reconnect_tracker_index=tracker_index,
        stage_frames=150,
    )

    assert schedule.stage_indices.tolist() == [0] * 150 + [1] * 150
    assert schedule.tracker_available[:150].sum(axis=1).tolist() == [3] * 150
    assert schedule.tracker_available[150:].sum(axis=1).tolist() == [4] * 150
    assert not schedule.tracker_available[:150, tracker_index].any()
    assert schedule.tracker_available[150:, tracker_index].all()


def test_reconnection_schedule_rejects_invalid_lengths() -> None:
    tracker_index = reconnection.RECONNECT_TRACKER_NAME_TO_INDEX["hip"]
    with pytest.raises(ValueError, match="至少为 4"):
        reconnection.build_tracker_reconnection_schedule(
            scored_frame_count=6,
            reconnect_tracker_index=tracker_index,
            stage_frames=3,
        )
    with pytest.raises(ValueError, match="恰好为"):
        reconnection.build_tracker_reconnection_schedule(
            scored_frame_count=299,
            reconnect_tracker_index=tracker_index,
            stage_frames=150,
        )


def test_reconnection_main_writes_three_protocol_json(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "toy_source.npz"
    np.savez(source_path, **build_toy_realtime_source(frame_count=46))
    predictor, dit, diffusion = _tiny_models()
    monkeypatch.setattr(
        reconnection,
        "read_longseq_source_entries",
        lambda *args, **kwargs: [{"source_path": str(source_path)}],
    )
    monkeypatch.setattr(
        reconnection,
        "create_model_and_diffusion",
        lambda args: (dit, diffusion),
    )
    monkeypatch.setattr(
        reconnection,
        "load_checkpoint_model",
        lambda model, path, device, use_ema: (model, "raw"),
    )
    monkeypatch.setattr(
        reconnection,
        "load_realtime_pose_predictor",
        lambda path, device: predictor,
    )
    monkeypatch.setattr(
        reconnection,
        "RealtimePoseNormalizer",
        lambda path, disable: None,
    )
    output = tmp_path / "tracker_reconnection.json"

    payload = reconnection.main(
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
            "--stage_frames",
            "8",
            "--output_json",
            str(output),
        ]
    )

    assert output.is_file()
    assert payload["experiment"] == "tracker_reconnection_3_to_4"
    assert payload["schedule"]["stage_frames"] == 8
    assert payload["schedule"]["tracker_counts"] == [3, 4]
    reports = payload["reconnection_protocols"]
    assert [report["reconnect_tracker"] for report in reports] == list(
        reconnection.RECONNECT_TRACKER_NAMES
    )
    assert all(
        [stage["tracker_count"] for stage in report["stages"]] == [3, 4]
        for report in reports
    )
    # 三种协议在重连前使用完全相同的 mask 和随机噪声，第一阶段应可直接比较。
    assert reports[0]["stages"][0]["dit_deployed"] == reports[1]["stages"][0][
        "dit_deployed"
    ]
    assert reports[0]["stages"][0]["dit_deployed"] == reports[2]["stages"][0][
        "dit_deployed"
    ]
