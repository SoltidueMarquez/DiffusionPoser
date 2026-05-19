import unittest

import numpy as np

from data_converter.amass_to_x277 import (
    DEFAULT_SMPL_PARENTS,
    FEATURE_DIM,
    JOINT_INDEX,
    SMPL_JOINT_COUNT,
    SmplMotion,
    build_x277_features,
)


class Current277SchemaTest(unittest.TestCase):
    def test_build_x277_features_uses_current_frame_velocity_and_root_delta(self):
        frame_count = 3
        fps = 60.0
        joint_offsets = np.zeros((SMPL_JOINT_COUNT, 3), dtype=np.float64)
        joint_offsets[:, 1] = np.linspace(0.0, 1.0, SMPL_JOINT_COUNT)

        translations = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        joint_positions = joint_offsets[None] + translations[:, None, :]
        joint_rotations = np.repeat(np.eye(3, dtype=np.float64)[None, None], frame_count, axis=0)
        joint_rotations = np.repeat(joint_rotations, SMPL_JOINT_COUNT, axis=1)

        rest_joints = joint_offsets.copy()
        rest_joints[JOINT_INDEX["left_ankle"]] = np.asarray([-0.12, 0.08, -0.10])
        rest_joints[JOINT_INDEX["left_foot"]] = np.asarray([-0.12, 0.02, 0.18])
        rest_joints[JOINT_INDEX["right_ankle"]] = np.asarray([0.12, 0.08, -0.10])
        rest_joints[JOINT_INDEX["right_foot"]] = np.asarray([0.12, 0.02, 0.18])
        rest_vertices = build_rest_foot_vertices(rest_joints)
        vertices = rest_vertices[None] + translations[:, None, :]

        smpl_motion = SmplMotion(
            raw_joint_positions=joint_positions.copy(),
            joint_positions=joint_positions,
            joint_rotations=joint_rotations,
            vertices=vertices,
            rest_joints=rest_joints,
            rest_vertices=rest_vertices,
            parents=DEFAULT_SMPL_PARENTS,
        )

        x277 = build_x277_features(
            smpl_motion=smpl_motion,
            target_fps=fps,
            height_threshold=0.04,
            speed_threshold=0.20,
        )

        self.assertEqual(tuple(x277.shape), (frame_count - 1, FEATURE_DIM))
        velocity_row0 = x277[0, 144:216].reshape(SMPL_JOINT_COUNT, 3)
        np.testing.assert_allclose(velocity_row0[:, 0], np.full(SMPL_JOINT_COUNT, fps), atol=1e-6)
        np.testing.assert_allclose(velocity_row0[:, 1:], 0.0, atol=1e-6)
        np.testing.assert_allclose(x277[0, 270:272], np.asarray([1.0, 0.0], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(x277[1, 270:272], np.asarray([2.0, 0.0], dtype=np.float32), atol=1e-6)
        self.assertTrue(np.isin(x277[:, 273:277], [0.0, 1.0]).all())


def build_rest_foot_vertices(rest_joints: np.ndarray) -> np.ndarray:
    vertices = []
    for name in ("left_foot", "right_foot"):
        center = rest_joints[JOINT_INDEX[name]]
        for x_offset in np.linspace(-0.05, 0.05, 8):
            for z_offset in np.linspace(-0.18, 0.18, 8):
                vertices.append(center + np.asarray([x_offset, 0.0, z_offset], dtype=np.float64))
    return np.asarray(vertices, dtype=np.float64)


if __name__ == "__main__":
    unittest.main()
