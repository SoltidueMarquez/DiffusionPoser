from __future__ import annotations

import argparse
import sys
from pathlib import Path


DIFFUSIONPOSER_ROOT = Path(__file__).resolve().parents[1]
if str(DIFFUSIONPOSER_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSIONPOSER_ROOT))


from scripts.realtime_pose_retarget_debug import (  # noqa: E402
    DEFAULT_BODY_FBX_META,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPLAY_JSON,
    IDENTITY_6D,
    build_debug_report,
    compute_fk_joints,
    parse_body_fbx_offsets_from_meta,
    write_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DEBUG_RETARGET_PROBE: diagnose realtime_pose_v2_contact SMPL feature to Unity body.fbx retarget errors."
    )
    parser.add_argument("--replay_json", default=str(DEFAULT_REPLAY_JSON), type=str)
    parser.add_argument("--body_fbx_meta", default=str(DEFAULT_BODY_FBX_META), type=str)
    parser.add_argument("--unity_dump_json", default="", type=str)
    parser.add_argument("--frame_start", default=0, type=int)
    parser.add_argument("--frame_count", default=0, type=int, help="0 means use all frames after frame_start.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), type=str)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Path]:
    args = build_arg_parser().parse_args(argv)
    report = build_debug_report(
        replay_json=Path(args.replay_json).resolve(),
        body_fbx_meta=Path(args.body_fbx_meta).resolve(),
        frame_start=int(args.frame_start),
        frame_count=int(args.frame_count),
        unity_dump_json=Path(args.unity_dump_json).resolve() if args.unity_dump_json else None,
    )
    paths = write_report(report, output_dir=Path(args.output_dir).resolve())

    print("[DEBUG_RETARGET_PROBE] wrote:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("[DEBUG_RETARGET_PROBE] classification:", report["classification"]["likelyCause"])
    return paths


__all__ = [
    "IDENTITY_6D",
    "build_arg_parser",
    "build_debug_report",
    "compute_fk_joints",
    "main",
    "parse_body_fbx_offsets_from_meta",
    "write_report",
]


if __name__ == "__main__":
    main()
