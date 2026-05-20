from __future__ import annotations

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import encode_realtime_pose_features
from data_loaders.sensor_masking import REALTIME_POSE_INPUT_DIM, REALTIME_POSE_TARGET_DIM, REALTIME_POSE_TARGET_START
from eval.evaluate_realtime_pose import evaluate_file
from sample.reconstruct_stream import build_realtime_inpaint_mask, save_reconstruction
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_realtime_pose_eval_reads_result_npz(tmp_path):
    source = build_toy_realtime_source(frame_count=61)
    source["sensor_valid"] = np.ones((61, 6), dtype=bool)
    features = encode_realtime_pose_features(source)
    reconstructed = features.copy()
    reconstructed[REALTIME_POSE_TARGET_START, 0] += 0.1
    mask = np.zeros((61, REALTIME_POSE_INPUT_DIM), dtype=bool)
    mask[REALTIME_POSE_TARGET_START, :REALTIME_POSE_TARGET_DIM] = True
    path = tmp_path / "result.npz"
    np.savez(path, reference_features=features, reconstructed_features=reconstructed, inpaint_mask=mask)
    metrics = evaluate_file(path)
    assert metrics["target_frames"] == 1
    assert metrics["pose_mse"] > 0.0


def test_realtime_pose_eval_reads_batched_reconstruction_npz(tmp_path):
    source = build_toy_realtime_source(frame_count=61)
    source["sensor_valid"] = np.ones((61, 6), dtype=bool)
    features = encode_realtime_pose_features(source)
    reference = torch.from_numpy(features.T).unsqueeze(0).float()
    reconstructed = reference.clone()
    reconstructed[:, 0, REALTIME_POSE_TARGET_START] += 0.1
    inpaint_mask = build_realtime_inpaint_mask(1, torch.device("cpu"))
    path = tmp_path / "batched_result.npz"
    save_reconstruction(path, reference, reference, reconstructed, inpaint_mask)

    metrics = evaluate_file(path)
    assert metrics["batch_size"] == 1
    assert metrics["target_frames"] == 1
    assert metrics["feature_space"] == "raw"
    assert metrics["pose_mse"] > 0.0


def test_realtime_pose_eval_prefers_raw_features_when_available(tmp_path):
    source = build_toy_realtime_source(frame_count=61)
    source["sensor_valid"] = np.ones((61, 6), dtype=bool)
    features = encode_realtime_pose_features(source)[None]
    raw_reconstructed = features.copy()
    raw_reconstructed[:, REALTIME_POSE_TARGET_START, 0] += 0.2
    normalized_reconstructed = features.copy()
    mask = np.zeros((1, 61, REALTIME_POSE_INPUT_DIM), dtype=bool)
    mask[:, REALTIME_POSE_TARGET_START, :REALTIME_POSE_TARGET_DIM] = True
    path = tmp_path / "raw_priority.npz"
    np.savez(
        path,
        reference_features_raw=features,
        reconstructed_features_raw=raw_reconstructed,
        reference_features_normalized=features,
        reconstructed_features_normalized=normalized_reconstructed,
        inpaint_mask=mask,
    )

    metrics = evaluate_file(path)
    assert metrics["feature_space"] == "raw"
    assert metrics["pose_mse"] > 0.0
