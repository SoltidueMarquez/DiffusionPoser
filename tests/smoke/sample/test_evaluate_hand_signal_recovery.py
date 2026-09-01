from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from data_loaders.realtime_pose_predictor_features import (
    build_predictor_sparse_availability_mask_np,
)
from diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)
from model.realtime_pose_current_dit import RealtimePoseCurrentDiT
from model.realtime_pose_predictor import RealtimePosePredictor
from sample import evaluate_hand_signal_recovery as recovery
from sample.tracker_activation_blending import smoothstep_activation_alpha
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


def test_predictor_sparse_mask_closes_velocity_at_both_gap_boundaries():
    available = np.ones((12, 6), dtype=bool)
    available[5:8, 1] = False
    mask = build_predictor_sparse_availability_mask_np(available)

    # 输出 frame 4 对应输入 current=5，是左手掉线首帧：绝对量与速度量均关闭。
    assert not mask[4, 6:12].any()
    assert not mask[4, 24:30].any()
    assert not mask[4, 39:42].any()
    assert not mask[4, 48:51].any()
    # 输出 frame 7 是重连首帧：绝对测量恢复，但跨边界速度仍保持关闭。
    assert mask[7, 6:12].all()
    assert not mask[7, 24:30].any()
    assert mask[7, 39:42].all()
    assert not mask[7, 48:51].any()
    assert mask[8].all()


def test_gap_key_mapping_and_source_offset(tmp_path):
    split_file = tmp_path / "test.txt"
    split_file.write_text(
        "HumanEva/S1/Box_1_poses.npy\n"
        "Transitions_mocap/mazen_c3d/GUS_kick_poses.npy\n",
        encoding="utf-8",
    )
    mapping = recovery.build_gap_key_by_source(
        split_file,
        {"HumanEva-1", "Transitions_mocap-1"},
    )
    assert mapping["HumanEva/S1/Box_1_poses"] == "HumanEva-1"

    available = recovery.build_hand_tracker_availability(
        source_frame_count=12,
        hand_gap_intervals=(((2, 5),), ((7, 9),)),
        gap_frame_offset=1,
    )
    assert not available[3:6, 1].any()
    assert not available[8:10, 2].any()
    assert available[:, 0].all()
    assert not available[:, 3:].any()


def test_hand_reconnection_reuses_project_soft10f_policy():
    previous_available = np.array([True, False, True, False, False, False])
    current_available = np.array([True, True, True, False, False, False])
    measured_positions = np.zeros((6, 3), dtype=np.float32)
    measured_positions[1, 0] = 10.0
    measured_rotations_6d = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        (6, 1),
    )
    previous_joint_positions = np.zeros((24, 3), dtype=np.float32)
    previous_joint_rotations = np.tile(
        np.eye(3, dtype=np.float32)[None], (24, 1, 1)
    )
    ramps = {}

    positions, _, alpha, newly_added = recovery.apply_tracker_activation_blend(
        current_frame=31,
        blend_frames=10,
        previous_available=previous_available,
        current_available=current_available,
        measured_positions=measured_positions,
        measured_rotations_6d=measured_rotations_6d,
        previous_joint_positions=previous_joint_positions,
        previous_joint_rotations=previous_joint_rotations,
        activation_ramps=ramps,
    )

    expected_alpha = smoothstep_activation_alpha(0, 10)
    assert newly_added == (1,)
    assert alpha[1] == pytest.approx(expected_alpha)
    assert positions[1, 0] == pytest.approx(10.0 * expected_alpha)
    assert 1 in ramps


def test_hand_recovery_main_writes_ood_report(tmp_path, monkeypatch):
    source_path = tmp_path / "toy_source.npz"
    np.savez(source_path, **build_toy_realtime_source(frame_count=46))
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "test.txt").write_text(
        "HumanEva/S1/Box_1_poses.npy\n", encoding="utf-8"
    )
    gap_config = tmp_path / "hand_tracking.json"
    gap_config.write_text(
        json.dumps(
            {
                "metadata": {"masker": "seg_hands_idp", "min_f": 4},
                "gaps": {"HumanEva-1": [[[30, 34]], [[36, 40]]]},
            }
        ),
        encoding="utf-8",
    )
    predictor, dit, diffusion = _tiny_models()
    monkeypatch.setattr(
        recovery,
        "read_longseq_source_entries",
        lambda *args, **kwargs: [
            {
                "sequence_id": "toy",
                "source_path": str(source_path),
                "source_relative_path": "HumanEva/S1/Box_1_poses.npz",
            }
        ],
    )
    monkeypatch.setattr(
        recovery,
        "create_model_and_diffusion",
        lambda args: (dit, diffusion),
    )
    monkeypatch.setattr(
        recovery,
        "load_checkpoint_model",
        lambda model, path, device, use_ema: (model, "raw"),
    )
    monkeypatch.setattr(
        recovery,
        "load_realtime_pose_predictor",
        lambda path, device: predictor,
    )
    monkeypatch.setattr(
        recovery,
        "RealtimePoseNormalizer",
        lambda path, disable: None,
    )
    output = tmp_path / "hand_recovery.json"

    payload = recovery.main(
        [
            "--cuda",
            "false",
            "--split_dir",
            str(split_dir),
            "--normalizer_dir",
            "normalizer",
            "--gap_config",
            str(gap_config),
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
            "--transition_seconds",
            "0.1",
            "--output_json",
            str(output),
        ]
    )

    assert output.is_file()
    assert payload["experiment"] == "rpm_p2_hand_signal_loss_and_recovery"
    assert payload["source_sequence_count"] == 1
    assert "out_of_training_distribution" in payload["diagnostic_scope"]
    assert payload["gap_protocol"]["reconnection_policy"] == {
        "name": "project_soft_activation_blend",
        "activation_blend_frames": 10,
        "position": "LERP from previous deployed wrist to measurement",
        "rotation": "SLERP from previous deployed wrist to measurement",
        "weight": "smoothstep",
    }
    assert payload["report"]["gap_counts"] == {
        "left_hand": 1,
        "right_hand": 1,
    }
    per_sequence = payload["report"]["per_sequence"][0]
    assert per_sequence["gap_key"] == "HumanEva-1"
    assert per_sequence["lost_scored_frames"] == {
        "left_hand": 4,
        "right_hand": 4,
    }
    assert payload["report"]["smooth_reconnection"]["event_counts"] == {
        "left_hand": 1,
        "right_hand": 1,
    }
