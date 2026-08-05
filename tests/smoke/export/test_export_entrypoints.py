from __future__ import annotations

import numpy as np

from scripts import export_smpl_source_rest


def test_export_smpl_source_rest_imports_and_builds_parser():
    parser = export_smpl_source_rest.build_arg_parser()
    args = parser.parse_args(["--source_npz", "source.npz"])
    assert args.source_npz == "source.npz"


def test_source_rest_offsets_keep_positive_grounded_pelvis():
    joints = np.zeros((24, 3), dtype=np.float64)
    for joint_index in range(1, 24):
        joints[joint_index] = joints[joint_index - 1] + np.asarray([0.0, 0.1, 0.0])
    joints[:, 1] += 1.0
    vertices = np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)

    rest_offsets, source_fk_offsets = export_smpl_source_rest.build_rest_local_offsets(
        joints,
        vertices,
    )

    assert rest_offsets.shape == (24, 3)
    assert source_fk_offsets.shape == (24, 3)
    assert rest_offsets[0, 1] == 1.0
