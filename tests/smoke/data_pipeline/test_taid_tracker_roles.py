from __future__ import annotations

import numpy as np
import torch

from data_loaders.tracker_roles import (
    TRACKER_ROLE_ANCHOR,
    TRACKER_ROLE_MISSING,
    TRACKER_ROLE_UNCONFIGURED,
    TRACKER_ROLE_UNCERTAIN,
    compute_tracker_roles_np,
    compute_tracker_roles_torch,
)


def _states(d_on: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = np.asarray(d_on, dtype=np.float32)
    configured = np.ones(duration.shape, dtype=bool)
    measured = configured.copy()
    return configured, measured, duration


def test_missing_measurement_has_strictly_zero_alpha_beta_and_distinct_identity():
    configured, measured, d_on = _states(np.full((2, 6), 20.0, dtype=np.float32))
    configured[0, 4] = False
    measured[0, 4] = False
    measured[1, 4] = False
    result = compute_tracker_roles_np(configured, measured, d_on)

    assert result.roles[0, 4] == TRACKER_ROLE_UNCONFIGURED
    assert result.roles[1, 4] == TRACKER_ROLE_MISSING
    assert result.alpha[0, 4] == result.beta[0, 4] == 0.0
    assert result.alpha[1, 4] == result.beta[1, 4] == 0.0


def test_role_boundaries_and_continuous_weights_follow_k0_k1_kr():
    durations = np.asarray([4.999, 5.0, 5.001, 14.999, 15.0, 15.001], dtype=np.float32)
    configured, measured, d_on = _states(np.tile(durations[:, None], (1, 6)))
    result = compute_tracker_roles_np(configured, measured, d_on)

    assert np.all(result.roles[:4, 3] == TRACKER_ROLE_UNCERTAIN)
    assert np.all(result.roles[4:, 3] == TRACKER_ROLE_ANCHOR)
    assert result.alpha[1, 3] == 0.0
    assert result.alpha[4, 3] == 1.0
    assert abs(float(result.alpha[0, 3] - result.alpha[2, 3])) < 3e-4
    assert abs(float(result.beta[3, 3] - result.beta[4, 3])) < 2e-4
    assert np.all(result.alpha[:, 3] + result.beta[:, 3] <= 1.0 + 1e-6)


def test_head_is_always_anchor_and_never_enters_innovation():
    configured, measured, d_on = _states(np.ones((1, 6), dtype=np.float32))
    result = compute_tracker_roles_np(configured, measured, d_on)
    assert result.roles[0, 0] == TRACKER_ROLE_ANCHOR
    assert result.alpha[0, 0] == 1.0
    assert result.beta[0, 0] == result.beta_hard[0, 0] == 0.0


def test_numpy_and_torch_role_managers_are_identical():
    configured, measured, d_on = _states(
        np.asarray([[1, 3, 7, 14, 15, 30], [2, 6, 9, 12, 18, 24]], dtype=np.float32)
    )
    measured[1, 2] = False
    numpy_result = compute_tracker_roles_np(configured, measured, d_on)
    torch_result = compute_tracker_roles_torch(
        torch.from_numpy(configured),
        torch.from_numpy(measured),
        torch.from_numpy(d_on),
    )

    np.testing.assert_array_equal(torch_result.roles.numpy(), numpy_result.roles)
    np.testing.assert_allclose(torch_result.alpha.numpy(), numpy_result.alpha, atol=1e-7)
    np.testing.assert_allclose(torch_result.beta.numpy(), numpy_result.beta, atol=1e-7)
    np.testing.assert_allclose(
        torch_result.region_coverage.numpy(),
        numpy_result.region_coverage,
        atol=1e-7,
    )
