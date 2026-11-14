"""校验仓库输入、独立入口与短程闭环，避免实验重新依赖旧 output 目录。"""
import json
from pathlib import Path

import pytest
import torch

from eval.evaluate_scene_guidance import main, parse_args
from eval.scene_guidance_benchmark import load_snapshots, split_snapshot_indices
from eval.scene_guidance_sequence import world_reference
from eval.scene_guidance_setup import (
    REPOSITORY_ROOT, file_hash, guidance_configs, prepare_output_directory, source_paths, validate_inputs,
)
from sample import infer_unity_recording as bridge


def test_config_paths_and_cli_overrides_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = parse_args(["--config", "experiments/scene_guidance/sequence.json", "--seeds", "7", "--repeats", "2"])
    assert args.protocol == "sequence"
    assert args.seeds == [7] and args.repeats == 2
    assert args.measurement_scale == .0005 and args.ts_respace == "5"
    assert all(path.is_relative_to(REPOSITORY_ROOT) for path in source_paths(args))
    assert {config.method for config in guidance_configs(args)} == {"postedit", "lm", "pigdm"}
    assert validate_inputs(args)


def test_explicit_model_config_is_not_overwritten_by_checkpoint(tmp_path):
    config = json.loads((REPOSITORY_ROOT / "experiments/scene_guidance/snapshot.json").read_text())
    config["layers"] = 2
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert parse_args(["--config", str(path)]).layers == 2
    assert parse_args(["--config", str(path), "--layers", "3"]).layers == 3


def test_bundled_snapshots_and_held_out_split():
    args = parse_args([])
    validate_inputs(args)
    snapshots = load_snapshots(args.replay_inputs, "cpu")
    assert len(snapshots) == 65
    tuning, testing = split_snapshot_indices(65, 20, "uniform")
    assert len(tuning) == 20 and len(testing) == 45
    assert set(tuning).isdisjoint(testing)
    assert sorted(tuning + testing) == list(range(65))
    _, short = split_snapshot_indices(65, 20, "uniform", max_frames=3)
    assert short == [testing[0], testing[22], testing[-1]]


def test_bundled_recording_matches_snapshot_geometry():
    args = parse_args(["--config", "experiments/scene_guidance/sequence.json"])
    recording = bridge.resample_tracker_recording(bridge.apply_tracker_availability_overrides(
        bridge.load_unity_tracker_recording(args.recording), ignore_hip=True, ignore_feet=False))
    captures = json.loads(Path(args.replay_inputs).read_text(encoding="utf-8"))
    start = bridge.find_first_core_window(recording.available)
    boxes, checks = world_reference(recording, captures, start)
    assert len(recording.times) == 1108 and boxes.shape == (2, 15)
    assert checks["snapshot_tracker_position_max_error_m"] < 1e-5
    assert checks["static_world_box_variation"] < 1e-5


def test_missing_input_and_changed_pinned_input_fail_early(tmp_path, monkeypatch):
    args = parse_args(["--replay_inputs", str(tmp_path / "missing.json")])
    with pytest.raises(FileNotFoundError, match="missing.json"):
        validate_inputs(args)
    args = parse_args([])
    import eval.scene_guidance_setup as setup
    monkeypatch.setattr(setup, "file_hash", lambda path: "changed" if Path(path) == Path(args.replay_inputs) else file_hash(path))
    with pytest.raises(ValueError, match="校验失败"):
        validate_inputs(args)


def test_text_hash_survives_windows_checkout_and_output_is_preserved(tmp_path):
    path = tmp_path / "input.json"
    path.write_bytes(b'{\n  "value": 1\n}\n')
    expected = file_hash(path)
    path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
    assert file_hash(path) == expected
    args = parse_args(["--output_dir", str(tmp_path)])
    with pytest.raises(FileExistsError, match="非空"):
        prepare_output_directory(args)
    assert path.is_file()


@pytest.mark.parametrize("value", [0., float("nan"), float("inf")])
def test_invalid_guidance_parameter_is_rejected(value):
    args = parse_args([])
    args.guidance_configs = [{"method": "lm", "damping": value}]
    with pytest.raises(ValueError, match="有限正数"):
        guidance_configs(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="完整采样计时需要 CUDA")
@pytest.mark.parametrize("protocol,tune", [("snapshot", False), ("snapshot", True), ("sequence", False)])
def test_independent_entrypoint_short_run(tmp_path, protocol, tune):
    output = tmp_path / protocol
    argv = ["--config", f"experiments/scene_guidance/{protocol}.json", "--output_dir", str(output),
            "--methods", "lm", "pigdm", "--max_frames", "2", "--seeds", "10",
            "--warmup", "0", "--repeats", "2" if protocol == "sequence" else "1", "--rollout_warmup", "1"]
    main(argv + (["--tune", "--tune_frames", "2"] if tune else []))
    metadata = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert metadata["protocol"] == protocol and metadata["short_run"]
    assert (output / "report.html").is_file()
    if protocol == "snapshot":
        rows = json.loads((output / "test_summary.json").read_text(encoding="utf-8"))
        assert {row["method"] for row in rows} == {"none", "lm", "pigdm"}
        assert all(row["runs"] == 2 and row["failures"] == 0 for row in rows)
        assert metadata["test_frames"] == 2 and metadata["tuning_performed"] == tune
        if tune:
            selection = json.loads((output / "selection.json").read_text())
            assert selection["selection_split"] == "tuning only"
    else:
        rows = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert {row["method"] for row in rows} == {"lm", "pigdm"}
        assert all(row["evaluated_frames"] == 2 and row["timed_calls"] == 4 for row in rows)
        assert metadata["rollout_frames"] == 3
        import numpy as np
        for method in ("lm", "pigdm"):
            with np.load(output / f"{method}_seed10.npz") as poses:
                assert poses["poses"].shape == (3, 144)
