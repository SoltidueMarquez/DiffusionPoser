import json

import numpy as np

from data_converter.relabel_realtime_pose_stationary import main as relabel_main
from data_loaders.generate_realtime_pose_tasks import main as generate_tasks_main
from data_loaders.realtime_pose_contract import validate_realtime_source_contract
from data_loaders.sensor_masking import REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from data_loaders.stationary_label_config import stationary_label_metadata
from tests.smoke.realtime_pose_fixtures import write_toy_source_dataset


def test_relabel_realtime_pose_stationary_rewrites_only_label_and_metadata(tmp_path):
    source_dir = tmp_path / "old"
    output_dir = tmp_path / "new"
    write_toy_source_dataset(
        source_dir,
        frame_count=62,
        schema_name=REALTIME_POSE_SCHEMA_NAME,
    )
    source_path = next(source_dir.rglob("*.npz"))
    with np.load(source_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]).copy() for key in data.files if key != "metadata"}
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
    for key in stationary_label_metadata():
        metadata.pop(key, None)
    metadata.update(
        {
            "stationary_label_method": "joint_center_speed_only_v1",
            "stationary_speed_full_motion": 0.25,
            "stationary_median_window": 5,
        }
    )
    arrays["stationary_prob_5"][:] = 0.123
    np.savez(
        source_path,
        **arrays,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )

    manifest_path = relabel_main(
        [
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--source_set_name",
            "toy_causal_stationary",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--rebuild_manifest",
        ]
    )

    record = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    output_path = output_dir / source_path.relative_to(source_dir)
    assert record["status"] == "relabelled"
    assert record["source_set_name"] == "toy_causal_stationary"
    assert record["schema_name"] == REALTIME_POSE_SCHEMA_NAME
    assert record["pose_representation"] == "body_fbx_local_delta_6d"
    assert record["stablemotion_split_key"]
    with np.load(output_path, allow_pickle=False) as data:
        validate_realtime_source_contract(
            data,
            schema=get_schema_spec(REALTIME_POSE_SCHEMA_NAME),
            source=str(output_path),
        )
        next_metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        assert next_metadata["source_set_name"] == "toy_causal_stationary"
        assert next_metadata["stationary_label_method"] == "joint_center_speed_causal_fast_release_v2"
        assert not np.allclose(data["stationary_prob_5"], 0.123)
        np.testing.assert_allclose(data["joints_world"], arrays["joints_world"])

    task_root = tmp_path / "tasks"
    generate_tasks_main(
        [
            "--source_dir",
            str(output_dir),
            "--output_dir",
            str(task_root),
            "--source_set_name",
            "toy_causal_stationary",
            "--task_set_name",
            "toy_causal_stationary_rollout2",
            "--splits",
            "train",
            "--samples_per_source",
            "1",
            "--rollout_steps",
            "2",
            "--schema",
            REALTIME_POSE_SCHEMA_NAME,
            "--split_dir",
            "",
            "--overwrite",
        ]
    )
    task_manifest = next(task_root.rglob("train/manifest.jsonl"))
    task_record = json.loads(task_manifest.read_text(encoding="utf-8").splitlines()[0])
    for key, value in stationary_label_metadata().items():
        assert task_record[key] == value
