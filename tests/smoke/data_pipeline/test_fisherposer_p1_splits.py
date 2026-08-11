from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from data_loaders.build_fisherposer_p1_splits import (
    FISHERPOSER_P1_MIN_SOURCE_FRAMES,
    FISHERPOSER_P1_SUBSETS,
    build_fisherposer_p1_splits,
)
from data_loaders.generate_realtime_pose_tasks import normalize_split_key, read_source_entries


def test_local_fisherposer_p1_split_is_deterministic_and_task_compatible(tmp_path):
    records, eligible_keys = build_manifest_records()
    source_a = write_source_manifest(tmp_path / "source_a", records)
    source_b = write_source_manifest(tmp_path / "source_b", list(reversed(records)))
    output_a = tmp_path / "split_a"
    output_b = tmp_path / "split_b"

    build_fisherposer_p1_splits(source_dir=source_a, output_dir=output_a, seed=10)
    build_fisherposer_p1_splits(source_dir=source_b, output_dir=output_b, seed=10)

    for name in ("train.txt", "test.txt", "split_meta.json"):
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()
    assert not (output_a / "train_dev.txt").exists()
    assert not (output_a / "val.txt").exists()

    train_keys = read_generated_split(output_a / "train.txt")
    test_keys = read_generated_split(output_a / "test.txt")
    assert train_keys.isdisjoint(test_keys)
    assert train_keys | test_keys == eligible_keys

    usable_manifest_keys = {
        normalize_split_key(entry["stablemotion_split_key"])
        for entry in read_source_entries(source_a)
    }
    assert train_keys | test_keys <= usable_manifest_keys
    for subset in FISHERPOSER_P1_SUBSETS:
        subset_train = {key for key in train_keys if key.startswith(f"{subset}/")}
        subset_test = {key for key in test_keys if key.startswith(f"{subset}/")}
        assert len(subset_train) == 9
        assert len(subset_test) == math.ceil(10 * 0.1)

    assert all(not key.startswith("M/") for key in train_keys | test_keys)
    meta = json.loads((output_a / "split_meta.json").read_text(encoding="utf-8"))
    assert meta["split_origin"] == "deterministic_local_90_10"
    assert meta["counts"] == {
        "eligible": 30,
        "train": 27,
        "test": 3,
        "by_subset": {
            subset: {"eligible": 10, "train": 9, "test": 1}
            for subset in FISHERPOSER_P1_SUBSETS
        },
    }
    assert meta["eligibility_exclusion_counts"]["mirrored"] == 1
    assert meta["eligibility_exclusion_counts"]["unusable_status"] == 1
    assert meta["eligibility_exclusion_counts"]["too_short"] == 1
    assert meta["eligibility_exclusion_counts"]["target_fps_mismatch"] == 1
    assert meta["split_sha256"] == {
        "train.txt": sha256_file(output_a / "train.txt"),
        "test.txt": sha256_file(output_a / "test.txt"),
    }


def test_official_fisherposer_p1_split_projects_unusable_entries(tmp_path):
    records, eligible_keys = build_manifest_records()
    source_dir = write_source_manifest(tmp_path / "source", records)
    official_dir = tmp_path / "avatarposer" / "data_split"
    expected_train: set[str] = set()
    expected_test: set[str] = set()

    for subset in FISHERPOSER_P1_SUBSETS:
        subset_keys = sorted(key for key in eligible_keys if key.startswith(f"{subset}/"))
        test_keys = {subset_keys[0]}
        train_keys = set(subset_keys[1:])
        expected_train.update(train_keys)
        expected_test.update(test_keys)
        extra_train = []
        extra_test = []
        if subset == "CMU":
            extra_train = ["CMU/actor/shape.npz", "CMU/actor/failed_poses.npz"]
            extra_test = ["CMU/actor/short_poses.npz", "CMU/actor/missing_poses.npz"]
        write_official_file(official_dir / subset / "train_split.txt", train_keys, extra_train)
        write_official_file(official_dir / subset / "test_split.txt", test_keys, extra_test)

    output_dir = tmp_path / "official_split"
    metadata = build_fisherposer_p1_splits(
        source_dir=source_dir,
        output_dir=output_dir,
        official_split_dir=official_dir,
        seed=10,
    )

    assert read_generated_split(output_dir / "train.txt") == expected_train
    assert read_generated_split(output_dir / "test.txt") == expected_test
    assert metadata["split_origin"] == "avatarposer_official"
    exclusions = metadata["official_split"]["exclusions"]
    assert exclusions["non_motion_shape_file"] == ["CMU/actor/shape.npy"]
    assert exclusions["unusable_status"] == ["CMU/actor/failed_poses.npy"]
    assert exclusions["too_short"] == ["CMU/actor/short_poses.npy"]
    assert exclusions["not_in_usable_source_manifest"] == ["CMU/actor/missing_poses.npy"]
    assert metadata["counts"]["train"] == 27
    assert metadata["counts"]["test"] == 3


def build_manifest_records() -> tuple[list[dict], set[str]]:
    records: list[dict] = []
    eligible_keys: set[str] = set()
    for subset in FISHERPOSER_P1_SUBSETS:
        for index in range(10):
            key = f"{subset}/actor/motion_{index:02d}_poses"
            eligible_keys.add(key)
            records.append(source_record(key=key, frames=120))

    records.extend(
        [
            source_record(key="M/CMU/actor/mirrored_poses", frames=120, is_mirrored=True),
            source_record(key="CMU/actor/failed_poses", frames=120, status="failed"),
            source_record(key="CMU/actor/short_poses", frames=FISHERPOSER_P1_MIN_SOURCE_FRAMES - 1),
            source_record(key="CMU/actor/non60_poses", frames=120, target_fps=30.0),
            source_record(key="KIT/actor/outside_poses", frames=120),
        ]
    )
    return records, eligible_keys


def source_record(
    key: str,
    frames: int,
    target_fps: float = 60.0,
    status: str = "converted",
    is_mirrored: bool = False,
) -> dict:
    relative_path = f"{key}.npz"
    return {
        "status": status,
        "source_relative_path": relative_path,
        "stablemotion_split_key": f"{key}.npy",
        "output_path": relative_path,
        "frames": frames,
        "target_fps": target_fps,
        "is_mirrored": is_mirrored,
    }


def write_source_manifest(source_dir: Path, records: list[dict]) -> Path:
    source_dir.mkdir(parents=True)
    with (source_dir / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return source_dir


def write_official_file(path: Path, keys: set[str], extra_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}.npz" for key in sorted(keys)] + list(extra_keys)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_generated_split(path: Path) -> set[str]:
    return {
        normalize_split_key(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
