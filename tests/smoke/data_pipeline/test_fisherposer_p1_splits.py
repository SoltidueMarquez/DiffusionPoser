from __future__ import annotations

import math
from pathlib import Path

from data_loaders.build_fisherposer_p1_splits import (
    FISHERPOSER_P1_SUBSETS,
    build_fisherposer_p1_splits,
)
from data_loaders.generate_realtime_pose_tasks import normalize_split_key


def _write_source_tree(root: Path) -> set[str]:
    keys: set[str] = set()
    for subset in FISHERPOSER_P1_SUBSETS:
        for index in range(10):
            key = f"{subset}/actor/motion_{index:02d}"
            path = root / f"{key}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            keys.add(key)
    for key in ("M/CMU/actor/mirrored", "KIT/actor/outside"):
        path = root / f"{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return keys


def _read_keys(path: Path) -> set[str]:
    return {
        normalize_split_key(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_local_fisherposer_p1_split_is_deterministic_for_directory_sources(tmp_path):
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    eligible = _write_source_tree(source_a)
    _write_source_tree(source_b)
    output_a = tmp_path / "split_a"
    output_b = tmp_path / "split_b"

    first = build_fisherposer_p1_splits(source_a, output_a, seed=10)
    second = build_fisherposer_p1_splits(source_b, output_b, seed=10)

    assert (output_a / "train.txt").read_bytes() == (output_b / "train.txt").read_bytes()
    assert (output_a / "test.txt").read_bytes() == (output_b / "test.txt").read_bytes()
    train_keys = _read_keys(output_a / "train.txt")
    test_keys = _read_keys(output_a / "test.txt")
    assert train_keys.isdisjoint(test_keys)
    assert train_keys | test_keys == eligible
    for subset in FISHERPOSER_P1_SUBSETS:
        assert len({key for key in train_keys if key.startswith(f"{subset}/")}) == 9
        assert len({key for key in test_keys if key.startswith(f"{subset}/")}) == math.ceil(1.0)
    assert first["counts"] == second["counts"]
    assert first["eligibility_exclusion_counts"] == {
        "mirrored": 1,
        "outside_p1_subsets": 1,
    }


def test_official_fisherposer_p1_split_projects_missing_entries(tmp_path):
    source_dir = tmp_path / "source"
    eligible = _write_source_tree(source_dir)
    official_dir = tmp_path / "official"
    expected_train: set[str] = set()
    expected_test: set[str] = set()
    for subset in FISHERPOSER_P1_SUBSETS:
        subset_keys = sorted(key for key in eligible if key.startswith(f"{subset}/"))
        test_keys = {subset_keys[0]}
        train_keys = set(subset_keys[1:])
        expected_train.update(train_keys)
        expected_test.update(test_keys)
        for split, keys in (("train", train_keys), ("test", test_keys)):
            path = official_dir / subset / f"{split}_split.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            extra = ["CMU/actor/shape.npz", "CMU/actor/missing.npz"] if subset == "CMU" and split == "train" else []
            path.write_text(
                "\n".join([*(f"{key}.npz" for key in sorted(keys)), *extra]) + "\n",
                encoding="utf-8",
            )

    output_dir = tmp_path / "split"
    metadata = build_fisherposer_p1_splits(
        source_dir,
        output_dir,
        official_split_dir=official_dir,
        seed=10,
    )
    assert _read_keys(output_dir / "train.txt") == expected_train
    assert _read_keys(output_dir / "test.txt") == expected_test
    exclusions = metadata["official_split"]["exclusions"]
    assert exclusions["non_motion_shape_file"] == ["CMU/actor/shape.npy"]
    assert exclusions["not_in_source_directory"] == ["CMU/actor/missing.npy"]
