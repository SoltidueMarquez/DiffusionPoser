import numpy as np

from data_loaders.body_fbx_kinematics import (
    SMPL_JOINT_COUNT,
    SOURCE_FK_TO_BODY_FBX_BASIS,
    source_global_rotations_to_body_fbx_local_delta_6d,
)
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np


def test_source_root_only_converts_local_basis() -> None:
    """pelvis 左侧是世界参考系，不能像普通局部关节一样再次转换。"""

    basis = SOURCE_FK_TO_BODY_FBX_BASIS
    global_rotations = np.repeat(
        basis[None, None],
        repeats=SMPL_JOINT_COUNT,
        axis=1,
    )

    pose_6d = source_global_rotations_to_body_fbx_local_delta_6d(
        global_rotations=global_rotations,
        root_heading=np.zeros((1,), dtype=np.float64),
    )
    body_delta = rotation_6d_to_matrix_np(
        pose_6d.reshape(1, SMPL_JOINT_COUNT, 6),
    )

    # source root 恰好等于基变换时，右乘 basis.T 后应恢复为 body.fbx 单位旋转。
    np.testing.assert_allclose(body_delta[0, 0], np.eye(3), atol=1e-6)
    # 其余关节的 source parent-local 均为单位旋转，也应继续保持单位旋转。
    np.testing.assert_allclose(
        body_delta[0, 1:],
        np.repeat(np.eye(3)[None], SMPL_JOINT_COUNT - 1, axis=0),
        atol=1e-6,
    )
