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
from data_loaders.realtime_pose_geometry import extract_forward_yaw_np
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from data_loaders.realtime_pose_task_store import ShardWriter, read_store_metadata, write_json
from data_loaders.realtime_pose_validation import validate_realtime_task_arrays
from data_loaders.sensor_masking import TRACKER_FEATURE_DIM, TRACKER_PATTERN_CATEGORIES
from data_loaders.tracker_timeline import build_task_config_plan
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def _build_row():
    source = build_toy_realtime_source(frame_count=70)
    joint_rotations = compute_source_joint_rotations_world(source)
    tracker_rotations = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"])
    head_yaws = extract_forward_yaw_np(tracker_rotations[:, 0])
    row = build_task_bundle_row(
        source=source,
        joint_rotations_world=joint_rotations,
        head_yaws=head_yaws,
        start_frame=0,
        source_index=0,
        config_plans=build_task_config_plan("toy", global_seed=10, max_rollout_steps=1),
    )
    return source, row


def test_task_bundle_materializes_synchronized_spatiotemporal_window():
    _source, row = _build_row()
    assert row["pose_window_clean"].shape == (11, 144)
    assert row["tracker_window_continuous"].shape == (11, 6, 9)
    assert row["head_path_window"].shape == (11, 5)
    assert row["configured"].shape == (5, 61, 6)
    assert row["measured_valid"].shape == (5, 61, 6)
    assert row["future_leg_target"].shape == (3, 8, 6)
    assert row["previous_contact_target"].shape == (2,)
    assert row["contact_target"].shape == (2,)
    np.testing.assert_allclose(
        row["previous_contact_target"],
        _source["stationary_prob_5"][59, 1:3],
    )
    np.testing.assert_allclose(
        row["contact_target"],
        _source["stationary_prob_5"][60, 1:3],
    )
    np.testing.assert_allclose(row["head_path_window"][-1, :2], 0.0, atol=1e-7)
    np.testing.assert_allclose(row["head_path_window"][-1, 3:], [0.0, 1.0], atol=1e-7)
    # Head 路径和同一锚点的 Head Tracker 必须来自完全相同的参考系变换。
    np.testing.assert_allclose(
        row["tracker_window_continuous"][:, 0][:, [0, 2, 1]],
        row["head_path_window"][:, :3],
        atol=1e-6,
    )


def _write_store(tmp_path):
    source, row = _build_row()
    split_dir = tmp_path / "tasks" / "train"
    writer = ShardWriter(split_dir / "shards" / "shard_00000", 1, shard_fields())
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
            "two_point_phase_counts": {"dropout": 1, "reconnect": 0},
            "config_names": list(TRACKER_PATTERN_CATEGORIES),
            "tracker_feature_dim": TRACKER_FEATURE_DIM,
            "schema_fields": sorted(shard_fields()),
            "shards": [{"index": 0, "row_count": 1, "path": "shards/shard_00000"}],
        },
    )
    return tmp_path / "tasks"


def test_dataset_returns_window_contract_and_replays_cold_start(tmp_path):
    task_dir = _write_store(tmp_path)
    dataset = RealtimePoseTaskDataset(task_dir, normalize_input=False)
    item = dataset[TaskRequest(0, 0)]
    assert item["x"].shape == (11, 144)
    assert "pose_window_clean" not in item
    assert item["history_pose_observation"].shape == (10, 144)
    assert item["tracker_window"].shape == (11, 6, 13)
    assert item["tracker_window_raw"].shape == (11, 6, 13)
    assert item["hard_rotation_state_window"].shape == (11, 6)
    assert item["previous_contact_target"].shape == (2,)
    assert item["contact_target"].shape == (2,)
    assert not (
        item["hard_rotation_state_window"]
        & ~(item["configured"] & item["measured_valid"])
    ).any()
    assert item["head_path_window"].shape == (11, 5)
    assert item["history_region_confidence"].shape == (10, 5)
    assert item["window_valid_mask"].all()
    assert "rollout" not in item
    assert "prev_joints_head_ref" not in item
    assert "current_tracker_pos_head_ref" not in item
    assert "current_tracker_rot_head_ref_6d" not in item
    validate_realtime_task_arrays(item)

    cold = dataset[TaskRequest(0, 0, history_length=0)]
    assert cold["window_valid_mask"].tolist() == [False] * 10 + [True]
    assert np.count_nonzero(cold["x"].numpy()[:-1]) == 0
    assert np.count_nonzero(cold["tracker_window"].numpy()[:-1]) == 0
    assert np.count_nonzero(cold["head_path_window"].numpy()[:-1]) == 0
    np.testing.assert_array_equal(cold["d_on"].numpy()[-1], np.ones(6, dtype=np.int64))
    assert not cold["hard_rotation_state_window"][:-1].any()
    assert cold["hard_rotation_state_window"][-1].tolist() == [True, False, False, False, False, False]

    almost_full = dataset[TaskRequest(0, 0, history_length=59)]
    assert almost_full["window_valid_mask"].sum().item() == 10
    assert not almost_full["window_valid_mask"][0]
    worker_batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    assert worker_batch["tracker_window"].shape == (1, 11, 6, 13)
    assert worker_batch["tracker_window_raw"].shape == (1, 11, 6, 13)
    assert worker_batch["hard_rotation_state_window"].shape == (1, 11, 6)
    dataset.close()


def test_eval_loader_honors_scenario_weights(monkeypatch):
    class FakeDataset:
        indices_by_shard = [list(range(8))]

        def __len__(self):
            return 8

        def task_id_at(self, task_index: int):
            return f"eval-{task_index}"

        def __getitem__(self, request):
            return {"scenario_id": int(request.config_index)}

    monkeypatch.setattr(get_data, "RealtimePoseTaskDataset", lambda **_kwargs: FakeDataset())
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
    assert scenario_ids.tolist() == [2] * 8


def test_cold_start_sampler_is_deterministic():
    class FakeDataset:
        indices_by_shard = [list(range(8))]

        def __len__(self):
            return 8

        def task_id_at(self, task_index: int):
            return f"train-{task_index}"

    sampler = RealtimePoseBatchSampler(
        dataset=FakeDataset(),
        batch_size=4,
        seed=10,
        scenario_weights=(0.2,) * 5,
        cold_start_prob=1.0,
        shuffle=False,
        drop_last=False,
    )
    first = [request.history_length for batch in sampler for request in batch]
    sampler.set_epoch(0)
    second = [request.history_length for batch in sampler for request in batch]
    assert first == second
    assert all(0 <= value < 60 for value in first)


def test_old_task_metadata_is_rejected(tmp_path):
    split_dir = tmp_path / "old" / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "task_store.json").write_text(
        json.dumps({"tracker_feature_dim": 12, "shards": []}) + "\n",
        encoding="utf-8",
    )
    try:
        read_store_metadata(split_dir)
    except ValueError:
        pass
    else:
        raise AssertionError("旧 task schema 必须被拒绝。")
