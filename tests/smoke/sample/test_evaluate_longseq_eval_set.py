from __future__ import annotations

import numpy as np

from data_loaders.sensor_masking import STATIC_OPTIONAL_TRACKER_MASKS
from sample.evaluate_longseq_eval_set import (
    TRACKER_CONFIG_NAMES,
    build_arg_parser,
    pose_rotation_error_deg,
)
from utils.model_util import create_model_and_diffusion


def test_longseq_eval_declares_all_eight_static_configurations():
    assert len(TRACKER_CONFIG_NAMES) == len(STATIC_OPTIONAL_TRACKER_MASKS) == 8
    assert TRACKER_CONFIG_NAMES[0] == "core_only"
    assert TRACKER_CONFIG_NAMES[-1] == "all_six"
    assert all(all(mask[:3]) for mask in STATIC_OPTIONAL_TRACKER_MASKS)


def test_longseq_eval_can_select_only_training_seen_tracker_configurations():
    args = build_arg_parser().parse_args(
        [
            "--normalizer_dir",
            "normalizer",
            "--predictor_model_path",
            "predictor.pt",
            "--dit_model_path",
            "dit.pt",
            "--output_json",
            "report.json",
            "--tracker_configs",
            "core_only",
            "all_six",
        ]
    )
    assert args.tracker_configs == ["core_only", "all_six"]


def test_pose_rotation_error_is_zero_for_identical_pose():
    identity = np.tile(
        np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32), 24
    )
    assert pose_rotation_error_deg(identity, identity) < 1e-5


def test_longseq_parser_can_construct_current_dit_without_dataset_options():
    args = build_arg_parser().parse_args(
        [
            "--normalizer_dir",
            "normalizer",
            "--predictor_model_path",
            "predictor.pt",
            "--dit_model_path",
            "dit.pt",
            "--output_json",
            "report.json",
            "--latent_dim",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--diffusion_steps",
            "2",
        ]
    )
    model, diffusion = create_model_and_diffusion(args)
    assert model.input_feats == 144
    assert diffusion.num_timesteps == 2


def test_longseq_eval_can_respace_fifty_training_steps_to_five_ddim_steps():
    args = build_arg_parser().parse_args(
        [
            "--normalizer_dir",
            "normalizer",
            "--predictor_model_path",
            "predictor.pt",
            "--dit_model_path",
            "dit.pt",
            "--output_json",
            "report.json",
            "--latent_dim",
            "32",
            "--layers",
            "1",
            "--heads",
            "4",
            "--diffusion_steps",
            "50",
            "--ts_respace",
            "ddim5",
        ]
    )
    _, diffusion = create_model_and_diffusion(args)
    assert diffusion.original_num_steps == 50
    assert diffusion.num_timesteps == 5
