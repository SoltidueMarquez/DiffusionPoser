from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_converter.amass_to_dance_bvh_json import (
    DANCE_BVH_FRAME_DIM,
    AMASS_ZUP_TO_SMPL_YUP_ROOT,
    build_dance_bvh_frames,
    main as amass_to_dance_bvh_json_main,
)


def write_toy_amass(path: Path, frame_count: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((frame_count, 72), dtype=np.float64)
    trans = np.zeros((frame_count, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(1.0, 1.3, frame_count)
    trans[:, 1] = 2.0
    trans[:, 2] = np.linspace(-0.5, 0.1, frame_count)
    np.savez(
        path,
        poses=poses,
        trans=trans,
        betas=np.zeros(10, dtype=np.float64),
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(60.0),
    )


def test_build_dance_bvh_frames_outputs_unity_json_frame_layout():
    class Source:
        pass

    source = Source()
    source.poses = np.zeros((3, 72), dtype=np.float64)
    source.trans = np.asarray(
        [
            [1.0, 2.0, -0.5],
            [1.25, 2.0, -0.25],
            [1.5, 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    frames = build_dance_bvh_frames(source, translation_scale=100.0, zero_origin_xz=True)

    assert frames.shape == (3, DANCE_BVH_FRAME_DIM)
    np.testing.assert_allclose(frames[0, :9], AMASS_ZUP_TO_SMPL_YUP_ROOT.astype(np.float32).reshape(-1))
    np.testing.assert_allclose(frames[0, 216:219], np.asarray([0.0, -50.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(frames[1, 216:219], np.asarray([-25.0, -25.0, 0.0], dtype=np.float32))


def test_amass_to_dance_bvh_json_cli_writes_playable_json(tmp_path):
    amass_dir = tmp_path / "AMASS"
    amass_path = amass_dir / "ACCAD" / "toy_motion.npz"
    output_json = tmp_path / "dance_output.json"
    write_toy_amass(amass_path)

    result = amass_to_dance_bvh_json_main(
        [
            "--amass_path",
            str(amass_path),
            "--amass_dir",
            str(amass_dir),
            "--output_json",
            str(output_json),
            "--target_fps",
            "60",
            "--translation_scale",
            "1",
            "--keep_world_translation",
            "--overwrite",
        ]
    )

    assert result == {"sequences": 1, "frames": 4}
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(payload["dance_array"]) == 1
    assert len(payload["dance_array"][0]) == 4
    assert len(payload["dance_array"][0][0]) == DANCE_BVH_FRAME_DIM
    assert payload["metadata"]["joint_order"][0] == "pelvis"
    assert payload["metadata"]["coordinate_basis"] == "amass_zup_to_smpl_yup"
    assert payload["metadata"]["sequences"][0]["source_relative_path"] == "ACCAD/toy_motion.npz"
    np.testing.assert_allclose(payload["dance_array"][0][0][:9], AMASS_ZUP_TO_SMPL_YUP_ROOT.reshape(-1))
    np.testing.assert_allclose(payload["dance_array"][0][0][216:219], np.asarray([-1.0, -0.5, 2.0]))
