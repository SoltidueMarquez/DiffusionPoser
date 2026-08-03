from __future__ import annotations

import json
import numpy as np
from torch.utils.data import DataLoader

import data_loaders.get_data as get_data
from data_loaders.generate_realtime_pose_tasks import (
    build_task_bundle_row,
    compute_source_joint_rotations_world,
    shard_fields,
)
from data_loaders.realtime_pose_dataset import (
    RealtimePoseBatchSampler,
    RealtimePoseTaskDataset,
    TaskRequest,
)
from data_loaders.realtime_pose_geometry import (
    decode_target_head_rotations_np,
    decode_target_root_yaw_world_np,
    extract_forward_yaw_np,
    extract_rotation_heading_np,
)
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np, wrap_radians
from data_loaders.tracker_timeline import build_task_config_plan
from data_loaders.realtime_pose_task_store import ShardWriter, read_store_metadata, write_json
from data_loaders.sensor_masking import (
    BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY,
    TRACKER_FEATURE_DIM,
    TRACKER_PATTERN_CATEGORIES,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def test_task_bundle_materializes_new_history_current_contract_without_writing_artifacts():
    source = build_toy_realtime_source(frame_count=70)
    pelvis_yaw_offset = 0.7
    source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY][:, :6] = np.asarray(
        [np.sin(pelvis_yaw_offset), 0.0, np.cos(pelvis_yaw_offset), 0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    joint_rotations = compute_source_joint_rotations_world(source)
    tracker_rotations = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"])
    head_yaws = extract_forward_yaw_np(tracker_rotations[:, 0])
    row = build_task_bundle_row(
        source=source,
        joint_rotations_world=joint_rotations,
        head_yaws=head_yaws,
        start_frame=0,
        source_index=0,
        config_plans=build_task_config_plan("toy", global_seed=10, max_rollout_steps=4),
        max_rollout_steps=4,
    )
    assert row["pose_history"].shape == (60, 144)
    assert row["current_target"].shape == (4, 144)
    assert row["tracker_history_continuous"].shape == (4, 60, 6, 9)
    assert row["current_tracker_continuous"].shape == (4, 6, 9)
    assert row["trajectory_history"].shape == (4, 60, 5)
    assert row["current_trajectory"].shape == (4, 1, 5)
    assert row["d_off"].shape == (5, 64, 6)
    assert row["d_on"].shape == (5, 64, 6)
    assert "position_quality" not in row and "rotation_quality" not in row
    assert row["hard_rotation_state"].shape == (5, 64, 6)
    assert row["future_leg_target"].shape == (4, 3, 8, 6)
    assert row["contact_target"].shape == (4, 2)
    assert np.isfinite(row["current_target"]).all()
    np.testing.assert_allclose(
        row["target_root_yaw_world"],
        extract_rotation_heading_np(joint_rotations[60:64, 0]),
        atol=1e-6,
    )
    _, pelvis_heading_head = decode_target_head_rotations_np(row["current_target"])
    np.testing.assert_allclose(
        wrap_radians(head_yaws[60:64] + pelvis_heading_head),
        row["target_root_yaw_world"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        decode_target_root_yaw_world_np(row["current_target"], head_yaws[60:64]),
        row["target_root_yaw_world"],
        atol=1e-6,
    )
    assert not np.allclose(row["target_root_yaw_world"], source["root_yaw"][60:64])


def test_target_root_yaw_stays_self_consistent_across_pi_boundary():
    source = build_toy_realtime_source(frame_count=70)
    pelvis_yaw_offset = np.pi - 0.02
    source[BODY_POSE_BODY_FBX_LOCAL_DELTA_KEY][:, :6] = np.asarray(
        [np.sin(pelvis_yaw_offset), 0.0, np.cos(pelvis_yaw_offset), 0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    joint_rotations = compute_source_joint_rotations_world(source)
    tracker_rotations = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"])
    head_yaws = extract_forward_yaw_np(tracker_rotations[:, 0])
    row = build_task_bundle_row(
        source,
        joint_rotations,
        head_yaws,
        start_frame=0,
        source_index=0,
        config_plans=build_task_config_plan("toy-pi", 10, max_rollout_steps=4),
        max_rollout_steps=4,
    )

    decoded = decode_target_root_yaw_world_np(row["current_target"], head_yaws[60:64])
    np.testing.assert_allclose(decoded, row["target_root_yaw_world"], atol=1e-6)
    source_difference = np.abs(
        wrap_radians(source["root_yaw"][60:64] - row["target_root_yaw_world"])
    )
    assert np.all(source_difference > 3.0)


def test_history_uses_previous_reference_while_current_head_is_local_origin():
    source = build_toy_realtime_source(frame_count=70)
    joint_rotations = compute_source_joint_rotations_world(source)
    tracker_rotations = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"])
    head_yaws = extract_forward_yaw_np(tracker_rotations[:, 0])
    row = build_task_bundle_row(
        source,
        joint_rotations,
        head_yaws,
        0,
        0,
        build_task_config_plan("coordinate", 10, 4),
        4,
    )
    current_head = row["current_tracker_continuous"][0, 0, :3]
    assert np.isclose(current_head[0], 0.0, atol=1e-6)
    assert np.isclose(current_head[2], 0.0, atol=1e-6)
    # 当前 Head 高度由 floor anchor 保留，不进入普通 position measurement token。
    assert current_head[1] > 0.0


def test_mmap_dataset_returns_new_batch_contract(tmp_path):
    source = build_toy_realtime_source(frame_count=70)
    joint_rotations = compute_source_joint_rotations_world(source)
    head_yaws = extract_forward_yaw_np(rotation_6d_to_matrix_np(source["tracker_rot_world_6d"])[:, 0])
    row = build_task_bundle_row(
        source,
        joint_rotations,
        head_yaws,
        0,
        0,
        build_task_config_plan("dataset", 10, 4),
        4,
    )
    split_dir = tmp_path / "tasks" / "train"
    writer = ShardWriter(split_dir / "shards" / "shard_00000", 1, shard_fields(4))
    writer.write_row(0, row)
    writer.finish()
    np.save(split_dir / "source_joint_offsets_parent.npy", source["joint_offsets_parent"][None])
    np.save(
        split_dir / "source_joint_rest_local_rotations_6d.npy",
        source["joint_rest_local_rotations_6d"][None],
    )
    (split_dir / "sources.jsonl").write_text(
        json.dumps(
            {
                "source_index": 0,
                "source_id": "toy",
                "source_path": "toy.npz",
                "source_relative_path": "toy.npz",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        split_dir / "task_store.json",
        {
            "generation_plan_hash": "temporary",
            "split": "train",
            "sample_count": 1,
            "source_count": 1,
            "max_rollout_steps": 4,
            "two_point_phase_counts": {"dropout": 1, "reconnect": 0},
            "config_names": list(TRACKER_PATTERN_CATEGORIES),
            "tracker_feature_dim": TRACKER_FEATURE_DIM,
            "schema_fields": sorted(shard_fields(4)),
            "shards": [{"index": 0, "row_count": 1, "path": "shards/shard_00000"}],
        },
    )
    dataset = RealtimePoseTaskDataset(tmp_path / "tasks", normalize_input=False)
    item = dataset[TaskRequest(0, 4, 4)]
    assert item["pose_history"].shape == (60, 144)
    assert item["tracker_history"].shape == (60, 6, 13)
    assert item["current_tracker"].shape == (6, 13)
    assert item["trajectory_history"].shape == (60, 5)
    assert item["current_trajectory"].shape == (1, 5)
    assert item["hard_rotation_state"].shape == (6,)
    assert len(item["rollout"]) == 3
    assert "known_mask" not in item and "inpaint_mask" not in item
    measured = item["measured_valid"]
    tracker = np.concatenate(
        [item["tracker_history"].numpy(), item["current_tracker"].numpy()[None]], axis=0
    )
    assert np.allclose(tracker[..., :9][~measured.numpy()], 0.0)

    cold = dataset[TaskRequest(0, 0, 4, history_length=0)]
    assert cold["history_length"].item() == 0
    assert not cold["valid_frame_mask"].any()
    assert np.count_nonzero(cold["pose_history"].numpy()) == 0
    assert np.count_nonzero(cold["tracker_history"].numpy()) == 0
    assert np.count_nonzero(cold["trajectory_history"].numpy()) == 0
    np.testing.assert_allclose(cold["current_trajectory"].numpy()[0, [0, 1, 3, 4]], [0, 0, 0, 1])
    np.testing.assert_array_equal(cold["d_on"].numpy()[-1], np.ones(6, dtype=np.int64))
    assert cold["hard_rotation_state"].tolist() == [True, False, False, False, False, False]
    assert [step["history_length"].item() for step in cold["rollout"]] == [1, 2, 3]

    warming = dataset[TaskRequest(0, 0, 1, history_length=13)]
    ready = dataset[TaskRequest(0, 0, 1, history_length=14)]
    np.testing.assert_array_equal(warming["d_on"].numpy()[-1], np.full(6, 14))
    assert warming["hard_rotation_state"].tolist() == [True, False, False, False, False, False]
    np.testing.assert_array_equal(ready["d_on"].numpy()[-1], np.full(6, 15))
    assert ready["hard_rotation_state"].all()

    almost_full = dataset[TaskRequest(0, 0, 4, history_length=59)]
    assert [
        almost_full["history_length"].item(),
        *[step["history_length"].item() for step in almost_full["rollout"]],
    ] == [59, 60, 60, 60]
    # step=2 的最早可见帧已经是会话第 2 帧，duration 必须延续为 2，不能重新从 1 计数。
    np.testing.assert_array_equal(almost_full["rollout"][1]["d_on"].numpy()[0], np.full(6, 2))
    worker_batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=2)))
    assert worker_batch["current_tracker"].shape == (1, 6, 13)
    dataset.close()


def test_eval_loader_honors_scenario_weights_instead_of_using_integer_indices(monkeypatch):
    class FakeDataset:
        max_rollout_steps = 4
        indices_by_shard = [list(range(8))]

        def __len__(self) -> int:
            return 8

        def task_id_at(self, task_index: int) -> str:
            return f"eval-{int(task_index)}"

        def __getitem__(self, request):
            if not isinstance(request, TaskRequest):
                return {"scenario_id": 0, "history_length": 60}
            return {
                "scenario_id": int(request.config_index),
                "history_length": int(request.history_length),
            }

    dataset = FakeDataset()
    monkeypatch.setattr(get_data, "RealtimePoseTaskDataset", lambda **_kwargs: dataset)
    loader = get_data.get_dataset_loader(
        data_dir="unused",
        batch_size=3,
        input_feats=144,
        seq_len=61,
        split="test",
        normalize_input=False,
        scenario_weights=(0.0, 0.0, 1.0, 0.0, 0.0),
    )
    scenario_ids = np.concatenate([batch["scenario_id"].numpy() for batch in loader])
    assert scenario_ids.tolist() == [2] * len(dataset)
    history_lengths = np.concatenate([batch["history_length"].numpy() for batch in loader])
    assert history_lengths.tolist() == [60] * len(dataset)


def test_cold_start_sampler_is_deterministic_and_only_samples_partial_history():
    class FakeDataset:
        max_rollout_steps = 4
        indices_by_shard = [list(range(8))]

        def __len__(self) -> int:
            return 8

        def task_id_at(self, task_index: int) -> str:
            return f"train-{int(task_index)}"

    sampler = RealtimePoseBatchSampler(
        dataset=FakeDataset(),
        batch_size=4,
        seed=10,
        scenario_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        rollout_steps=1,
        rollout_prob=0.0,
        cold_start_prob=1.0,
        shuffle=False,
        drop_last=False,
    )
    first = [request.history_length for batch in sampler for request in batch]
    sampler.set_epoch(0)
    second = [request.history_length for batch in sampler for request in batch]

    assert first == second
    assert all(0 <= history_length < 60 for history_length in first)


def test_old_task_metadata_is_rejected_explicitly(tmp_path):
    split_dir = tmp_path / "old" / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "task_store.json").write_text(
        json.dumps({"tracker_feature_dim": 12, "shards": []}) + "\n",
        encoding="utf-8",
    )
    try:
        read_store_metadata(split_dir)
    except ValueError as error:
        assert "旧 task 不可复用" in str(error)
    else:
        raise AssertionError("旧 task schema 必须明确拒绝。")
