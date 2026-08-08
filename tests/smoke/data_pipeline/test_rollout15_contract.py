from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from data_loaders.generate_realtime_pose_tasks import (
    TASK_OUTPUT_MARKER,
    build_argument_parser,
    build_task_bundle_row,
    compute_source_joint_rotations_world,
    select_limited_source_entries,
    shard_fields,
)
from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset, TaskRequest
from data_loaders.realtime_pose_geometry import extract_forward_yaw_np
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_np
from data_loaders.realtime_pose_task_store import ShardWriter, write_json
from data_loaders.realtime_pose_task_store import read_store_metadata
from data_loaders.reuse_realtime_pose_normalizer import (
    NORMALIZER_STAT_FILES,
    reuse_realtime_pose_normalizer,
    sha256_file,
)
from data_loaders.sensor_masking import (
    REALTIME_POSE_MAX_ROLLOUT_STEPS,
    TRACKER_FEATURE_DIM,
    TRACKER_PATTERN_CATEGORIES,
)
from data_loaders.tracker_timeline import build_task_config_plan
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def _make_row(steps: int, frame_count: int = 78) -> tuple[dict, dict[str, np.ndarray]]:
    source = build_toy_realtime_source(frame_count=frame_count)
    rotations = compute_source_joint_rotations_world(source)
    tracker_rotations = rotation_6d_to_matrix_np(source["tracker_rot_world_6d"])
    head_yaws = extract_forward_yaw_np(tracker_rotations[:, 0])
    row = build_task_bundle_row(
        source,
        rotations,
        head_yaws,
        0,
        0,
        build_task_config_plan("rollout15", 10, steps),
        steps,
    )
    return row, source


def _write_task_dir(
    task_dir: Path,
    *,
    steps: int,
    plan_hash: str,
    source_dir: Path,
    row: dict | None = None,
    source: dict[str, np.ndarray] | None = None,
) -> None:
    split_dir = task_dir / "train"
    shard_dir = split_dir / "shards" / "shard_00000"
    if row is None:
        shard_dir.mkdir(parents=True, exist_ok=True)
        np.save(shard_dir / "current_target.npy", np.zeros((1, steps, 144), dtype=np.float32))
        np.save(
            shard_dir / "tracker_history_continuous.npy",
            np.zeros((1, steps, 60, 6, 9), dtype=np.float32),
        )
        np.save(
            shard_dir / "configured.npy",
            np.zeros((1, 5, 60 + steps, 6), dtype=np.uint8),
        )
    else:
        writer = ShardWriter(shard_dir, 1, shard_fields(steps))
        writer.write_row(0, row)
        writer.finish()
        assert source is not None
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
            "generation_plan_hash": plan_hash,
            "split": "train",
            "sample_count": 1,
            "source_count": 1,
            "max_rollout_steps": steps,
            "two_point_phase_counts": {"dropout": 1, "reconnect": 0},
            "config_names": list(TRACKER_PATTERN_CATEGORIES),
            "tracker_feature_dim": TRACKER_FEATURE_DIM,
            "schema_fields": sorted(shard_fields(steps)),
            "shards": [{"index": 0, "row_count": 1, "path": "shards/shard_00000"}],
        },
    )
    write_json(task_dir / TASK_OUTPUT_MARKER, {"source_dir": str(source_dir.resolve())})
    (task_dir / "generation_plan.sha256").write_text(plan_hash + "\n", encoding="ascii")


def test_rollout_limit_defaults_to_four_accepts_15_and_rejects_16() -> None:
    assert build_argument_parser().parse_args([]).max_rollout_steps == 4
    assert len(build_task_config_plan("k15", 10, REALTIME_POSE_MAX_ROLLOUT_STEPS)) == 5
    with pytest.raises(ValueError, match=r"\[1,15\]"):
        build_task_config_plan("k16", 10, REALTIME_POSE_MAX_ROLLOUT_STEPS + 1)


def test_task_store_rejects_materialized_k16_metadata(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    task_dir = tmp_path / "tasks"
    _write_task_dir(task_dir, steps=16, plan_hash="k16-plan", source_dir=source_dir)
    with pytest.raises(ValueError, match="max_rollout_steps"):
        read_store_metadata(task_dir / "train")


def test_stratified_source_limit_is_deterministic_and_spans_sorted_split() -> None:
    entries = [{"index": index} for index in range(100)]
    first = select_limited_source_entries(
        entries, limit=10, selection="stratified", global_seed=10, split="train"
    )
    second = select_limited_source_entries(
        entries, limit=10, selection="stratified", global_seed=10, split="train"
    )
    assert first == second
    selected = [entry["index"] for entry in first]
    assert all(index * 10 <= value < (index + 1) * 10 for index, value in enumerate(selected))
    assert select_limited_source_entries(
        entries, limit=10, selection="prefix", global_seed=10, split="train"
    ) == entries[:10]


def test_k15_materialization_dataset_prefix_and_cold_start_contract(tmp_path: Path) -> None:
    row, source = _make_row(REALTIME_POSE_MAX_ROLLOUT_STEPS)
    assert row["current_target"].shape == (15, 144)
    assert row["configured"].shape == (5, 75, 6)
    assert row["future_leg_target"].shape == (15, 3, 8, 6)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    task_dir = tmp_path / "tasks"
    _write_task_dir(
        task_dir,
        steps=15,
        plan_hash="k15-plan",
        source_dir=source_dir,
        row=row,
        source=source,
    )

    dataset = RealtimePoseTaskDataset(task_dir, normalize_input=False)
    item15 = dataset[TaskRequest(0, 0, 15, history_length=60)]
    item4 = dataset[TaskRequest(0, 0, 4, history_length=60)]
    assert len(item15["rollout"]) == 14
    assert len(item4["rollout"]) == 3
    for step in range(4):
        left = item15 if step == 0 else item15["rollout"][step - 1]
        right = item4 if step == 0 else item4["rollout"][step - 1]
        assert set(left).difference({"rollout"}) == set(right).difference({"rollout"})
        for key in set(left).difference({"rollout"}):
            left_value = left[key]
            right_value = right[key]
            if hasattr(left_value, "numpy"):
                np.testing.assert_array_equal(left_value.numpy(), right_value.numpy())
            else:
                assert left_value == right_value

    cold = dataset[TaskRequest(0, 0, 15, history_length=0)]
    assert [cold["history_length"].item(), *[
        step["history_length"].item() for step in cold["rollout"]
    ]] == list(range(15))
    np.testing.assert_array_equal(cold["d_on"].numpy()[-1], np.ones(6, dtype=np.int64))
    np.testing.assert_array_equal(
        cold["rollout"][-1]["d_on"].numpy()[-1], np.full(6, 15, dtype=np.int64)
    )
    almost_full = dataset[TaskRequest(0, 0, 15, history_length=59)]
    assert [almost_full["history_length"].item(), *[
        step["history_length"].item() for step in almost_full["rollout"]
    ]] == [59] + [60] * 14
    dataset.close()


def test_k15_future_leg_requires_three_frames_after_last_target() -> None:
    _make_row(15, frame_count=78)
    with pytest.raises((ValueError, IndexError)):
        _make_row(15, frame_count=77)


def test_normalizer_reuse_copies_statistics_byte_for_byte_and_rebinds_plan(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    reference_task = tmp_path / "reference"
    target_task = tmp_path / "target"
    _write_task_dir(reference_task, steps=4, plan_hash="reference-plan", source_dir=source_dir)
    _write_task_dir(target_task, steps=15, plan_hash="target-plan", source_dir=source_dir)
    source_normalizer = tmp_path / "normalizer"
    source_normalizer.mkdir()
    for index, filename in enumerate(NORMALIZER_STAT_FILES):
        (source_normalizer / filename).write_bytes(bytes([index + 1]) * (index + 7))
    write_json(
        source_normalizer / "normalizer_meta.json",
        {"generation_plan_hash": "reference-plan", "sample_count": 123},
    )

    result = reuse_realtime_pose_normalizer(
        Namespace(
            source_normalizer_dir=str(source_normalizer),
            reference_task_dir=str(reference_task),
            target_task_dir=str(target_task),
            output_dir=str(tmp_path / "output"),
            split="train",
            run_name="rollout15",
        )
    )
    output_dir = Path(result["output_dir"])
    for filename in NORMALIZER_STAT_FILES:
        assert sha256_file(output_dir / filename) == sha256_file(source_normalizer / filename)
    metadata = json.loads((output_dir / "normalizer_meta.json").read_text(encoding="utf-8"))
    assert metadata["generation_plan_hash"] == "target-plan"
    assert metadata["statistics_reused_without_recomputation"] is True
    assert metadata["statistics_source_metadata_sha256"]


@pytest.mark.parametrize("mismatch", ["normalizer_plan", "source", "target_horizon"])
def test_normalizer_reuse_rejects_contract_mismatch(tmp_path: Path, mismatch: str) -> None:
    source_dir = tmp_path / "source"
    other_source = tmp_path / "other_source"
    source_dir.mkdir()
    other_source.mkdir()
    reference_task = tmp_path / "reference"
    target_task = tmp_path / "target"
    _write_task_dir(reference_task, steps=4, plan_hash="reference-plan", source_dir=source_dir)
    target_steps = 4 if mismatch == "target_horizon" else 15
    _write_task_dir(
        target_task,
        steps=target_steps,
        plan_hash="target-plan",
        source_dir=other_source if mismatch == "source" else source_dir,
    )
    source_normalizer = tmp_path / "normalizer"
    source_normalizer.mkdir()
    for filename in NORMALIZER_STAT_FILES:
        (source_normalizer / filename).write_bytes(b"stat")
    write_json(
        source_normalizer / "normalizer_meta.json",
        {
            "generation_plan_hash": (
                "wrong-plan" if mismatch == "normalizer_plan" else "reference-plan"
            )
        },
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        reuse_realtime_pose_normalizer(
            Namespace(
                source_normalizer_dir=str(source_normalizer),
                reference_task_dir=str(reference_task),
                target_task_dir=str(target_task),
                output_dir=str(tmp_path / "output"),
                split="train",
                run_name="rollout15",
            )
        )
