from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from data_loaders.realtime_pose_kinematics import rotation_6d_forward_up_np
from data_loaders.sensor_masking import (
    HEAD_TRACKER_INDEX,
    REALTIME_POSE_SEQ_LEN,
    STANDARD_THREE_TRACKER_INDICES,
    TRACKER_COUNT,
    make_window_patterns,
    make_dynamic_dropout_sensor_valid,
    make_tracker_pattern,
    validate_sensor_valid,
)
from data_loaders.tracker_codec import (
    decode_tracker_positions_np,
    decode_tracker_positions_torch,
    decode_tracker_rotations_np,
    encode_tracker_positions_np,
    encode_tracker_positions_torch,
    encode_tracker_rotations_np,
)


def test_all_validity_masks_follow_head_and_count_rule():
    for values in itertools.product((False, True), repeat=TRACKER_COUNT):
        valid = np.asarray(values, dtype=bool)[None]
        expected = bool(values[HEAD_TRACKER_INDEX] and sum(values) >= 3)
        if expected:
            np.testing.assert_array_equal(validate_sensor_valid(valid), valid)
        else:
            with pytest.raises(ValueError):
                validate_sensor_valid(valid)


def test_explicit_standard_three_and_dynamic_dropout_contract():
    rng = np.random.default_rng(20260713)
    pattern = make_tracker_pattern("standard_three", rng)
    expected = np.zeros(TRACKER_COUNT, dtype=bool)
    expected[list(STANDARD_THREE_TRACKER_INDICES)] = True
    np.testing.assert_array_equal(pattern.sensor_valid, expected)

    timeline = make_dynamic_dropout_sensor_valid(rng)
    assert timeline.shape == (REALTIME_POSE_SEQ_LEN, TRACKER_COUNT)
    assert timeline[:, HEAD_TRACKER_INDEX].all()
    assert (timeline.sum(axis=1) >= 3).all()
    assert (~timeline).any()


def test_final_mask_distribution_and_dynamic_run_lengths_are_reproducible():
    patterns = make_window_patterns(np.random.default_rng(10), patterns_per_window=10, ensure_pattern_categories=True)
    categories = [pattern.category for pattern in patterns]
    assert categories.count("full_six") == 3
    assert categories.count("standard_three") == 3
    assert categories.count("static_sparse") == 2
    assert categories.count("dynamic_dropout") == 2

    expected = make_dynamic_dropout_sensor_valid(np.random.default_rng(77), seq_len=240)
    actual = make_dynamic_dropout_sensor_valid(np.random.default_rng(77), seq_len=240)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0].all()
    assert actual[:, HEAD_TRACKER_INDEX].all()
    assert ((~actual).sum(axis=1) <= 3).all()
    for tracker_index in range(1, TRACKER_COUNT):
        missing_indices = np.flatnonzero(~actual[:, tracker_index])
        if missing_indices.size == 0:
            continue
        split_points = np.where(np.diff(missing_indices) > 1)[0] + 1
        for run in np.split(missing_indices, split_points):
            assert 2 <= run.size <= 30


def test_tracker_codec_numpy_and_torch_round_trip():
    rng = np.random.default_rng(7)
    batch = 9
    tracker_world = rng.normal(size=(batch, TRACKER_COUNT, 3)).astype(np.float32)
    root_world = rng.normal(size=(batch, 3)).astype(np.float32)
    yaw = rng.uniform(-np.pi, np.pi, size=(batch,)).astype(np.float32)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None, None], batch * TRACKER_COUNT, axis=0).reshape(
        batch, TRACKER_COUNT, 3, 3
    )
    rotations_6d = rotation_6d_forward_up_np(rotations).astype(np.float32)

    encoded = encode_tracker_positions_np(tracker_world, root_world, yaw)
    decoded = decode_tracker_positions_np(encoded, root_world, yaw)
    np.testing.assert_allclose(decoded, tracker_world, atol=1e-5)

    torch_encoded = encode_tracker_positions_torch(
        torch.from_numpy(tracker_world),
        torch.from_numpy(root_world),
        torch.from_numpy(yaw),
    )
    torch_decoded = decode_tracker_positions_torch(
        torch_encoded,
        torch.from_numpy(root_world),
        torch.from_numpy(yaw),
    )
    np.testing.assert_allclose(torch_encoded.numpy(), encoded, atol=1e-5)
    np.testing.assert_allclose(torch_decoded.numpy(), tracker_world, atol=1e-5)

    rotation_ref = encode_tracker_rotations_np(rotations_6d, yaw)
    rotation_world = decode_tracker_rotations_np(rotation_ref, yaw)
    np.testing.assert_allclose(rotation_world, rotations_6d, atol=1e-5)


def test_tracker_codec_matches_unity_contract_v2_golden_vector():
    positions = np.repeat(np.asarray([[3.0, 1.2, 4.0]], dtype=np.float32), TRACKER_COUNT, axis=0)
    rotations = np.repeat(
        np.asarray([[0.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32),
        TRACKER_COUNT,
        axis=0,
    )
    root = np.asarray([1.0, 0.2, -2.0], dtype=np.float32)
    yaw = np.float32(np.pi / 2.0)
    encoded_position = encode_tracker_positions_np(positions, root, yaw)
    encoded_rotation = encode_tracker_rotations_np(rotations, yaw)
    np.testing.assert_allclose(encoded_position[0], [-6.0, 1.0, 2.0], atol=1e-4)
    np.testing.assert_allclose(encoded_rotation[0], [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0], atol=1e-4)
