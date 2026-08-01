from __future__ import annotations

import pytest

from model.realtime_pose_target_dit import RealtimePoseTargetDiT


def test_target_dit_rejects_old_combined_feature_dimension():
    with pytest.raises(ValueError, match="140"):
        RealtimePoseTargetDiT(input_feats=214, latent_dim=32, num_layers=1, num_heads=4)


def test_target_dit_has_joint_root_and_tracker_history_modules():
    model = RealtimePoseTargetDiT(input_feats=140, latent_dim=32, num_layers=2, num_heads=4)
    assert len(model.blocks) == 2
    assert model.tracker_encoder.history_gru.input_size == 32
    assert model.joint_output.out_features == 6
    assert model.root_output.out_features == 2
