from pathlib import Path

from sample.visualize_realtime_pose_sequences import (
    build_arg_parser,
    sequence_output_stem,
)


def test_visualization_parser_preserves_serial_source_order():
    args = build_arg_parser().parse_args(
        [
            "--predictor_model_path",
            "predictor.pt",
            "--dit_model_path",
            "dit.pt",
            "--normalizer_dir",
            "normalizer",
            "--output_dir",
            "output",
            "--source_npz",
            "run_stand.npz",
            "walking.npz",
        ]
    )

    assert args.source_npz == ["run_stand.npz", "walking.npz"]


def test_sequence_output_stem_keeps_dataset_context():
    stem = sequence_output_stem(
        Path("/source/Transitions_mocap/mazen_c3d/run_stand_poses.npz")
    )

    assert stem == "Transitions_mocap_mazen_c3d_run_stand_poses"
