from __future__ import annotations

import random

import numpy as np
import torch

from data_loaders.realtime_pose_predictor_features import (
    build_predictor_sparse_availability_mask_np,
    build_predictor_sparse_availability_mask_torch,
)
from data_loaders.rpm_hand_dropout import (
    RPM_HAND_DROPOUT_MAX_FRAMES,
    build_rpm_dit_training_availability,
    build_rpm_predictor_training_availability,
    build_rpm_training_hand_availability,
)


def test_rpm_training_mask_matches_official_python_random_draw_order():
    # seed=127 的官方调用顺序会得到：左手 [0,40)，右手 [6,24)。
    available = build_rpm_training_hand_availability(
        frame_count=RPM_HAND_DROPOUT_MAX_FRAMES,
        seed=127,
    )

    assert available[:, 0].all()
    assert not available[0:40, 1].any()
    assert available[40, 1]
    assert not available[6:24, 2].any()
    assert available[:6, 2].all() and available[24:, 2].all()
    assert available[:, 3:].all()


def test_rpm_mask_uses_local_rng_and_is_repeatable():
    random.seed(2026)
    expected_next_global_draw = random.random()
    random.seed(2026)
    first = build_rpm_training_hand_availability(frame_count=41, seed=127)
    actual_next_global_draw = random.random()
    second = build_rpm_training_hand_availability(frame_count=41, seed=127)

    np.testing.assert_array_equal(first, second)
    assert actual_next_global_draw == expected_next_global_draw


def test_predictor_and_dit_windows_share_boundary_mask_semantics():
    predictor_available = build_rpm_predictor_training_availability(
        output_frame_count=52,
        seed=127,
    )
    dit_available = build_rpm_dit_training_availability(seed=127)
    np_mask = build_predictor_sparse_availability_mask_np(dit_available)
    torch_mask = build_predictor_sparse_availability_mask_torch(
        torch.from_numpy(dit_available)[None]
    )[0]

    assert predictor_available.shape == (52, 6)
    assert dit_available.shape == (12, 6)
    assert not dit_available[:, 1:3].any()
    np.testing.assert_array_equal(torch_mask.numpy(), np_mask)
