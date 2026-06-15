from __future__ import annotations

import argparse
import json
import shutil
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any

from data_loaders.build_realtime_longseq_eval_set import (
    DEFAULT_LONGSEQ_EVAL_ROOT,
    build_replay_filename,
    read_longseq_manifest,
    resolve_longseq_eval_dir,
    resolve_manifest_source_path,
)
from data_loaders.generate_realtime_pose_tasks import load_realtime_source
from data_loaders.longseq_eval_dropout import (
    DROPOUT_PRESET_NONE,
    LongseqDropoutConfig,
    add_longseq_dropout_options,
    apply_longseq_dropout_to_source,
    build_longseq_dropout_config,
)
from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME
from export.write_unity_replay_stream import DEFAULT_REPLAY_FPS, load_sensor_valid, write_unity_replay_stream


DEFAULT_UNITY_REPLAY_DIR = (
    "../SIGGRAPH2024Unity/Assets/Projects/RealtimePose/Models/DiffusionPoserStationary5/Replays/"
    "root_y0_longseq_eval_stress_long"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Unity replay JSON files for a fixed longseq eval set.")
    paths = parser.add_argument_group("paths")
    paths.add_argument("--eval_root", default=DEFAULT_LONGSEQ_EVAL_ROOT, type=str)
    paths.add_argument("--eval_set", default="latest", type=str)
    paths.add_argument("--output_dir", default="", type=str)
    paths.add_argument("--also_write_unity", default=False, action=BooleanOptionalAction)
    paths.add_argument("--unity_output_dir", default=DEFAULT_UNITY_REPLAY_DIR, type=str)

    replay = parser.add_argument_group("replay")
    replay.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, type=str)
    replay.add_argument("--fps", default=DEFAULT_REPLAY_FPS, type=float)
    replay.add_argument("--frame_start", default=0, type=int)
    replay.add_argument("--frame_count", default=0, type=int)
    replay.add_argument("--identity_6d_rotations", default=False, action=BooleanOptionalAction)
    add_longseq_dropout_options(parser)
    return parser


def export_longseq_eval_unity_replays(
    eval_set_dir: Path,
    output_dir: Path | None = None,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    fps: float = DEFAULT_REPLAY_FPS,
    frame_start: int = 0,
    frame_count: int = 0,
    identity_6d_rotations: bool = False,
    dropout_config: LongseqDropoutConfig | None = None,
    also_write_unity: bool = False,
    unity_output_dir: Path | None = None,
) -> dict[str, Any]:
    eval_set_dir = Path(eval_set_dir).resolve()
    entries = read_longseq_manifest(eval_set_dir)
    dropout_config = dropout_config or LongseqDropoutConfig()
    replay_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else eval_set_dir / "unity_replays" / replay_subdir_name(dropout_config)
    )
    replay_dir.mkdir(parents=True, exist_ok=True)

    unity_dir = Path(unity_output_dir).resolve() if also_write_unity and unity_output_dir is not None else None
    if unity_dir is not None:
        unity_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for entry in entries:
        source_path = resolve_manifest_source_path(eval_set_dir=eval_set_dir, entry=entry)
        source = load_realtime_source(source_path, schema_name=schema_name)
        source["sensor_valid"] = load_sensor_valid(source_path, int(source["tracker_pos_world"].shape[0]))
        source, dropout_metadata = apply_longseq_dropout_to_source(
            source=source,
            sequence_id=str(entry["sequence_id"]),
            config=dropout_config,
        )
        replay_name = build_replay_filename(entry)
        replay_path = write_unity_replay_stream(
            source_npz=source_path,
            output_json=replay_dir / replay_name,
            schema_name=schema_name,
            fps=float(fps),
            frame_start=int(frame_start),
            frame_count=int(frame_count),
            identity_6d_rotations=bool(identity_6d_rotations),
            source_override=source,
            source_metadata_override={
                "sourcePath": str(source_path),
                "longseqEvalDropout": dropout_metadata,
            },
        )
        unity_path = None
        if unity_dir is not None:
            unity_path = unity_dir / replay_name
            shutil.copy2(replay_path, unity_path)
        files.append(
            {
                "sequence_id": entry["sequence_id"],
                "num_frames": int(entry["num_frames"]),
                "source_path": str(source_path),
                "output_json": str(replay_path),
                "unity_output_json": str(unity_path) if unity_path is not None else "",
                "valid_tracker_ratio": float(dropout_metadata["valid_tracker_ratio"]),
                "min_valid_trackers": int(dropout_metadata["min_valid_trackers"]),
            }
        )

    summary = {
        "kind": "longseq_eval_unity_replays",
        "eval_set_dir": str(eval_set_dir),
        "output_dir": str(replay_dir),
        "unity_output_dir": str(unity_dir) if unity_dir is not None else "",
        "schema_name": schema_name,
        "fps": float(fps),
        "frame_start": int(frame_start),
        "frame_count": int(frame_count),
        "identity_6d_rotations": bool(identity_6d_rotations),
        "dropout_config": dropout_config.to_dict(),
        "file_count": len(files),
        "files": files,
    }
    summary_path = replay_dir / "replay_export_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")
    summary["summary_path"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    eval_set_dir = resolve_longseq_eval_dir(eval_root=args.eval_root, eval_set=args.eval_set)
    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else None
    dropout_config = build_longseq_dropout_config(args)
    unity_output_dir = resolve_default_unity_output_dir(
        cli_value=str(args.unity_output_dir),
        dropout_config=dropout_config,
    )
    summary = export_longseq_eval_unity_replays(
        eval_set_dir=eval_set_dir,
        output_dir=output_dir,
        schema_name=str(args.schema),
        fps=float(args.fps),
        frame_start=int(args.frame_start),
        frame_count=int(args.frame_count),
        identity_6d_rotations=bool(args.identity_6d_rotations),
        dropout_config=dropout_config,
        also_write_unity=bool(args.also_write_unity),
        unity_output_dir=unity_output_dir,
    )
    print(
        "[write_longseq_eval_unity_replays] "
        f"files={summary['file_count']} output={summary['output_dir']} summary={summary['summary_path']}"
    )
    return summary


def replay_subdir_name(dropout_config: LongseqDropoutConfig) -> str:
    if dropout_config.preset != DROPOUT_PRESET_NONE:
        return dropout_config.preset
    if dropout_config.tracker_mask_policy != "task":
        return dropout_config.tracker_mask_policy
    return "full_trackers"


def resolve_default_unity_output_dir(cli_value: str, dropout_config: LongseqDropoutConfig) -> Path:
    if cli_value != DEFAULT_UNITY_REPLAY_DIR:
        return Path(cli_value)
    if dropout_config.preset == DROPOUT_PRESET_NONE and dropout_config.tracker_mask_policy == "task":
        return Path(cli_value)
    return Path(f"{DEFAULT_UNITY_REPLAY_DIR}_{replay_subdir_name(dropout_config)}")


if __name__ == "__main__":
    main()
