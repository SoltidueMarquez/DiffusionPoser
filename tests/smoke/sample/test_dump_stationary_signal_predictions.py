from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data_loaders.sensor_masking import (
    REALTIME_POSE_INPUT_DIM,
    REALTIME_POSE_SEQ_LEN,
    REALTIME_POSE_TARGET_START,
    get_schema_spec,
)
from sample import dump_stationary_signal_predictions as dump_module
from sample.dump_stationary_signal_predictions import save_stationary_prediction_payload


def test_save_stationary_prediction_payload_writes_main_feature_channel_only(tmp_path: Path):
    reference_features = np.zeros((1, 3, 214), dtype=np.float32)
    reconstructed_features = reference_features.copy()
    reference_stationary = np.ones((1, 3, 5), dtype=np.float32)
    feature_stationary = np.zeros((1, 3, 5), dtype=np.float32)
    path = tmp_path / "stationary_predictions.npz"

    save_stationary_prediction_payload(
        path=path,
        schema_name="realtime_pose_stationary5_v1",
        reference_features_raw=reference_features,
        reconstructed_features_raw=reconstructed_features,
        reference_stationary_prob_5=reference_stationary,
        feature_stationary_prob_5=feature_stationary,
    )

    with np.load(path, allow_pickle=False) as data:
        assert data["schema_name"].item() == "realtime_pose_stationary5_v1"
        assert data["reference_stationary_prob_5"].shape == (1, 3, 5)
        assert data["feature_stationary_prob_5"].shape == (1, 3, 5)
        assert "head_stationary_prob_5" not in data.files


def test_stationary_signal_dump_cli_writes_feature_channel_payload(tmp_path: Path, monkeypatch):
    schema = get_schema_spec("realtime_pose_stationary5_v1")
    model_path = tmp_path / "model000000000.pt"
    model_path.write_bytes(b"")
    reference = torch.zeros(REALTIME_POSE_INPUT_DIM, REALTIME_POSE_SEQ_LEN, dtype=torch.float32)
    reference[schema.stationary_prob_slice(), REALTIME_POSE_TARGET_START] = torch.tensor([1, 0, 1, 0, 1])

    class DummyDataset:
        normalizer = None

        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "x": reference,
                "conditioned_x": reference.clone(),
                "valid_frame_mask": torch.ones(REALTIME_POSE_SEQ_LEN, dtype=torch.bool),
            }

    class DummyModel(torch.nn.Module):
        def forward(self, x, timesteps, **kwargs):
            del timesteps, kwargs
            return x + 0.25

    def fake_reconstruct_batch(model, diffusion, batch, device, **kwargs):
        del diffusion, device, kwargs
        return model(batch["conditioned_x"], torch.zeros(batch["conditioned_x"].shape[0]))

    monkeypatch.setattr(dump_module.dist_util, "setup_dist", lambda *args, **kwargs: None)
    monkeypatch.setattr(dump_module.dist_util, "dev", lambda: torch.device("cpu"))
    monkeypatch.setattr(dump_module, "RealtimePoseTaskDataset", lambda *args, **kwargs: DummyDataset())
    monkeypatch.setattr(dump_module, "create_model_and_diffusion", lambda args: (DummyModel(), object()))
    monkeypatch.setattr(dump_module, "load_checkpoint_model", lambda model, *args, **kwargs: (model, "dummy"))
    monkeypatch.setattr(dump_module, "reconstruct_batch", fake_reconstruct_batch)
    monkeypatch.setattr(dump_module, "build_ik_init_image_for_batch", lambda *args, **kwargs: None)

    result = dump_module.main(
        [
            "--model_path",
            str(model_path),
            "--data_dir",
            str(tmp_path / "data"),
            "--output_dir",
            str(tmp_path / "out"),
            "--cuda",
            "false",
            "--max_batches",
            "1",
        ]
    )

    output_files = sorted(result["output_dir"].glob("*.npz"))
    assert len(output_files) == 1
    with np.load(output_files[0], allow_pickle=False) as data:
        assert "reference_stationary_prob_5" in data.files
        assert "feature_stationary_prob_5" in data.files
        assert "head_stationary_prob_5" not in data.files
