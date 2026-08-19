from __future__ import annotations

from pathlib import Path

import numpy as np

from data_loaders.build_realtime_longseq_eval_set import (
    build_replay_filename,
    build_sequence_output_dir_name,
    read_longseq_source_entries,
    resolve_source_entry_path,
)
from tests.smoke.realtime_pose_fixtures import build_toy_realtime_source


def _write_source(root: Path, relative: str, frame_count: int) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **build_toy_realtime_source(frame_count=frame_count))
    return path


def test_longseq_listing_selects_split_non_mirror_long_sources(tmp_path):
    source_dir = tmp_path / "source"
    _write_source(source_dir, "CMU/55/long_a.npz", 72)
    _write_source(source_dir, "KIT/442/long_b.npz", 73)
    _write_source(source_dir, "CMU/55/short.npz", 20)
    _write_source(source_dir, "M/CMU/55/mirror.npz", 80)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "test.txt").write_text(
        "CMU/55/long_a.npy\nKIT/442/long_b.npy\nCMU/55/short.npy\nM/CMU/55/mirror.npy\n",
        encoding="utf-8",
    )

    entries = read_longseq_source_entries(
        source_dir,
        split_dir,
        split="test",
        min_frames=30,
        include_mirror=False,
    )
    assert [entry["source_relative_path"] for entry in entries] == [
        "KIT/442/long_b.npz",
        "CMU/55/long_a.npz",
    ]
    assert [entry["num_frames"] for entry in entries] == [73, 72]
    for entry in entries:
        assert resolve_source_entry_path(entry).is_file()
        assert len(build_sequence_output_dir_name(entry)) <= 32
        assert build_replay_filename(entry).endswith("_replay.json")
