from types import SimpleNamespace

import numpy as np
import pytest

from sample import render_fluid_teaser_left as teaser_left
from sample import render_fluid_teaser_right as teaser_right
from sample import render_fluid_teaser_right_reconnect as teaser_reconnect


def _build_left_results() -> dict[str, dict[str, np.ndarray]]:
    rotations = np.broadcast_to(
        np.eye(3, dtype=np.float32),
        (1, teaser_left.SMPL_JOINT_COUNT, 3, 3),
    ).copy()
    return {
        method_name: {
            "rotations": rotations.copy(),
            "positions": np.zeros(
                (1, teaser_left.SMPL_JOINT_COUNT, 3), dtype=np.float32
            ),
            "root_yaw": np.zeros((1,), dtype=np.float32),
            "ik_confidence": np.linspace(
                0.0,
                1.0,
                teaser_left.SMPL_JOINT_COUNT,
                dtype=np.float32,
            )[None],
        }
        for method_name in teaser_left.METHOD_ORDER
    }


def test_left_teaser_validates_confidence_contract() -> None:
    results = _build_left_results()

    teaser_left.validate_teaser_results(results)

    results[teaser_left.METHOD_ORDER[0]]["ik_confidence"][0, 0] = -0.01
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        teaser_left.validate_teaser_results(results)


def test_left_teaser_confidence_colors_are_opaque_and_bounded() -> None:
    weights = np.eye(teaser_left.SMPL_JOINT_COUNT, dtype=np.float32)
    low = teaser_left.confidence_to_vertex_colors(
        np.zeros((teaser_left.SMPL_JOINT_COUNT,), dtype=np.float32),
        weights,
    )
    high = teaser_left.confidence_to_vertex_colors(
        np.ones((teaser_left.SMPL_JOINT_COUNT,), dtype=np.float32),
        weights,
    )

    assert low.shape == high.shape == (teaser_left.SMPL_JOINT_COUNT, 4)
    assert low.dtype == high.dtype == np.uint8
    assert np.all(low[:, 3] == 255)
    assert np.all(high[:, 3] == 255)
    assert np.any(low[:, :3] != high[:, :3])


def test_progressive_teaser_alpha_schedule_emphasizes_boundaries() -> None:
    frames = np.asarray(teaser_right.SELECTED_SOURCE_FRAMES, dtype=np.int64)
    alphas = teaser_right.build_body_alphas(frames)

    assert alphas.shape == frames.shape
    assert np.isfinite(alphas).all()
    assert np.all((0.0 < alphas) & (alphas <= 1.0))
    for boundary in teaser_right.BOUNDARY_SOURCE_FRAMES:
        for slot, source_frame in enumerate(
            range(
                boundary - teaser_right.WINDOW_RADIUS,
                boundary + teaser_right.WINDOW_RADIUS,
            )
        ):
            selected = np.flatnonzero(frames == source_frame)
            assert selected.shape == (1,)
            assert alphas[selected[0]] >= teaser_right.BOUNDARY_ALPHAS[slot] - 1e-6


def test_reconnection_teaser_alpha_schedule_requires_all_fifteen_frames() -> None:
    inputs = SimpleNamespace(
        selected_indices=np.arange(
            teaser_reconnect.DISPLAY_FRAME_COUNT, dtype=np.int64
        )
    )

    alphas = teaser_reconnect.build_body_alphas(inputs)

    np.testing.assert_allclose(alphas, teaser_reconnect.BODY_RENDER_ALPHAS)
    inputs.selected_indices = inputs.selected_indices[:-1]
    with pytest.raises(ValueError, match="15 帧"):
        teaser_reconnect.build_body_alphas(inputs)
