from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import sample.evaluate_longseq_eval_set as longseq
from data_loaders.generate_realtime_pose_tasks import compute_source_joint_rotations_world
from data_loaders.realtime_pose_geometry import build_pose_target_np, extract_forward_yaw_np
from data_loaders.realtime_pose_kinematics import rotation_6d_forward_up_np, rotation_6d_to_matrix_np
from data_loaders.tracker_timeline import TrackerTimeline, compute_missing_age
from eval.evaluate_realtime_pose_rollout import evaluate_rollout_file
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_longseq_140d_rollout_reinjects_prediction_and_keeps_missing_age(monkeypatch, tmp_path):
    source = build_toy_realtime_source(frame_count=63)
    rotations_world = compute_source_joint_rotations_world(source)
    head_rotations = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"][:, 0])
    head_yaws = extract_forward_yaw_np(head_rotations, initial_yaw=0.0)
    targets = [
        build_pose_target_np(
            rotations_world[frame_index : frame_index + 1],
            source["root_yaw"][frame_index : frame_index + 1],
            float(head_yaws[frame_index]),
        )[0]
        for frame_index in range(60, 63)
    ]
    # 第一个未知关节故意偏离 GT，用于确认下一帧读到的是预测历史。
    targets[0] = targets[0].copy()
    targets[0][:6] = rotation_6d_forward_up_np(
        Rotation.from_rotvec([0.35, 0.0, 0.0]).as_matrix()
    )

    configured = np.ones((63, 6), dtype=bool)
    measured = configured.copy()
    measured[61:, 3] = False
    missing_age = compute_missing_age(configured, measured)
    timeline = TrackerTimeline(
        configured=configured,
        measured_valid=measured,
        missing_age=missing_age,
        missing_age_norm=missing_age.astype(np.float32) / 60.0,
    )

    captured_histories = []
    original_build_conditioning = longseq.build_online_conditioning

    def capture_conditioning(*args, **kwargs):
        captured_histories.append(list(kwargs["pose_history_world"]))
        return original_build_conditioning(*args, **kwargs)

    sample_index = 0

    def fake_sample_online_target(**_kwargs):
        nonlocal sample_index
        result = targets[sample_index]
        sample_index += 1
        return result

    monkeypatch.setattr(longseq, "build_online_conditioning", capture_conditioning)
    monkeypatch.setattr(longseq, "sample_online_target", fake_sample_online_target)
    payload = longseq.rollout_long_sequence_source(
        model=None,
        diffusion=None,
        source=source,
        timeline=timeline,
        device=torch.device("cpu"),
        normalizer=None,
    )

    assert payload["reconstructed_target_raw"].shape == (1, 3, 140)
    assert payload["predicted_joints_world"].shape == (1, 3, 24, 3)
    assert payload["sampling_latency_ms"].shape == (1, 3)
    assert np.isnan(payload["sampling_latency_ms"]).all()
    assert payload["missing_age"][0, :, 3].tolist() == [0, 1, 2]
    assert payload["scenario"].tolist() == [["fixed_six", "dropout", "dropout"]]
    assert len(captured_histories) == 3
    assert not np.allclose(
        captured_histories[1][-1].joint_rotations_world[1],
        rotations_world[60, 1],
    )
    assert np.isfinite(payload["predicted_joints_world"]).all()
    assert float(payload["known_rotation_max_error"].max()) < 1e-5

    result_path = tmp_path / "rollout_result.npz"
    np.savez(result_path, **payload)
    result = evaluate_rollout_file(result_path)
    assert result["samples"] == 3
    assert result["velocity_pairs"] == 2
    assert result["acceleration_triplets"] == 1


def test_longseq_defaults_to_five_steps_and_latency_summary_excludes_warmup():
    args = longseq.build_arg_parser().parse_args(["--model_path", "model.pt"])
    assert args.inference_steps == 5
    summary = longseq.summarize_latency(np.asarray([100.0, 10.0, 20.0]), warmup_frames=1)
    assert summary["frames"] == 2
    assert summary["mean_ms"] == 15.0
    assert summary["p95_ms"] == 19.5
