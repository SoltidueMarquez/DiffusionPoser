from __future__ import annotations

import pytest
import torch

from eval.evaluate_realtime_pose_predictor import (
    build_arg_parser,
    evaluation_last_frame_exclusive,
    evaluate_predictor_entries,
)
from model.realtime_pose_predictor import RealtimePosePredictor
from tests.smoke.realtime_pose_fixtures import write_toy_source_dataset
from utils.normalizer import RealtimePoseNormalizer


def test_predictor_eval_parser_does_not_require_dit(tmp_path):
    args = build_arg_parser().parse_args(
        [
            "--normalizer_dir", str(tmp_path / "normalizer"),
            "--predictor_model_path", str(tmp_path / "predictor.pt"),
            "--output_json", str(tmp_path / "report.json"),
        ]
    )
    assert args.predictor_model_path.endswith("predictor.pt")
    assert not hasattr(args, "dit_model_path")


def test_predictor_eval_rejects_empty_split():
    with pytest.raises(RuntimeError, match="评估集为空"):
        evaluate_predictor_entries(
            entries=[],
            predictor=torch.nn.Identity(),
            device=torch.device("cpu"),
            normalizer=None,
        )


def test_predictor_eval_runs_without_dit(tmp_path):
    source_path = write_toy_source_dataset(tmp_path / "source", frame_count=31)
    predictor = RealtimePosePredictor(
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    ).eval()
    report = evaluate_predictor_entries(
        entries=[{"source_path": str(source_path)}],
        predictor=predictor,
        device=torch.device("cpu"),
        normalizer=RealtimePoseNormalizer(tmp_path / "unused", disable=True),
        max_frames=1,
    )
    assert report["evaluated_sequences"] == 1
    assert report["generated_frames"] == 20
    assert report["evaluated_frames"] == 1
    assert report["rpm_p2_mc"]["mpjre_deg"] is not None
    assert report["rpm_p2_mc"]["mpjpe_cm"] is not None
    assert report["rpm_p2_mc"]["mpjve_cm_per_s"] is None
    assert report["rpm_p2_mc"]["pred_jitter_m_per_s3"] is None
    assert len(report["horizon_rotation_deg"]) == 11
    assert report["horizon_counts"][0] == 1
    assert report["rolling"]["first_30_generated_count"] == 20


def test_predictor_eval_max_frames_only_limits_scored_p2_frames():
    assert evaluation_last_frame_exclusive(100, 0) == 100
    assert evaluation_last_frame_exclusive(100, 1) == 31
    assert evaluation_last_frame_exclusive(31, 10) == 31
