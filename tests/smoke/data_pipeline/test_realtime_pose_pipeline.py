from __future__ import annotations

from pathlib import Path

from scripts import run_realtime_pose_pipeline as pipeline


def _args(tmp_path: Path, *extra: str):
    return pipeline.build_arg_parser().parse_args(
        [
            "--source_dir", str(tmp_path / "source"),
            "--task_dir", str(tmp_path / "tasks"),
            "--normalizer_dir", str(tmp_path / "normalizer"),
            "--split_dir", str(tmp_path / "splits"),
            "--predictor_save_dir", str(tmp_path / "predictor"),
            "--save_dir", str(tmp_path / "dit"),
            *extra,
        ]
    )


def test_pipeline_stage_order_contains_single_predictor_stage():
    assert pipeline.selected_stages("tasks", "train") == (
        "tasks",
        "normalizer",
        "predictor",
        "calibrate",
        "train",
    )


def test_pipeline_passes_parallel_converter_args(tmp_path):
    rest_path = tmp_path / "body_fbx_rest.json"
    args = _args(
        tmp_path,
        "--convert_num_workers",
        "3",
        "--convert_worker_torch_threads",
        "2",
        "--body_fbx_rest_json",
        str(rest_path),
    )
    values = pipeline.build_convert_args(args)
    assert "--target_fps" not in values
    assert values[values.index("--num_workers") + 1] == "3"
    assert values[values.index("--worker_torch_threads") + 1] == "2"
    assert values[values.index("--body_fbx_rest_json") + 1] == str(rest_path)


def test_predictor_and_dit_use_single_predictor_checkpoint(tmp_path):
    args = _args(tmp_path)
    predictor = pipeline.build_predictor_args(args)
    dit = pipeline.build_train_args(args)
    assert dit[dit.index("--predictor_model_path") + 1].endswith(
        "predictor/model_latest.pt"
    )
    assert predictor[predictor.index("--checkpoint_max_keep") + 1] == "3"
    assert predictor[predictor.index("--num_steps") + 1] == "100000"
    assert dit[dit.index("--num_steps") + 1] == "1000000"
    assert dit[dit.index("--log_interval") + 1] == "10"
    assert "--stage" not in predictor
    assert "--stage1_model_path" not in predictor
    assert "--scenario_weights" not in dit
    assert "--tracker_confidence_warmup" not in dit
