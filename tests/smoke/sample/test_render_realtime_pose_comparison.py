import numpy as np

from sample.render_realtime_pose_comparison import as_unbatched, mean_joint_speed


def test_as_unbatched_preserves_sequence_vector_and_single_frame_pose():
    eval_mask = np.ones(5, dtype=bool)
    single_frame_pose = np.zeros((1, 24, 3), dtype=np.float32)

    assert as_unbatched(eval_mask, "eval_frame_mask", 1).shape == (5,)
    assert as_unbatched(single_frame_pose, "reference_joints", 3).shape == (
        1,
        24,
        3,
    )


def test_as_unbatched_removes_only_explicit_single_batch_axis():
    batched_yaw = np.zeros((1, 5), dtype=np.float32)
    batched_pose = np.zeros((1, 5, 24, 3), dtype=np.float32)

    assert as_unbatched(batched_yaw, "root_yaw", 1).shape == (5,)
    assert as_unbatched(batched_pose, "reference_joints", 3).shape == (5, 24, 3)


def test_mean_joint_speed_preserves_stationary_frame_and_scales_by_fps():
    joints = np.zeros((3, 24, 3), dtype=np.float32)
    joints[1:, :, 0] = 0.1
    speed = mean_joint_speed(joints, fps=30.0)

    np.testing.assert_allclose(speed, [0.0, 3.0, 0.0], atol=1e-6)
