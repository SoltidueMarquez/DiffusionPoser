from __future__ import annotations

import argparse
import sys
from pathlib import Path


DIFFUSIONPOSER_ROOT = Path(__file__).resolve().parents[1]
if str(DIFFUSIONPOSER_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSIONPOSER_ROOT))


from scripts.realtime_pose_retarget_debug import DEFAULT_REPLAY_JSON, DEFAULT_SOURCE_REST_JSON  # noqa: E402
from scripts.realtime_pose_retarget_debug.source_rest import (  # noqa: E402
    DEFAULT_SMPL_MODEL_DIR,
    export_source_rest_pose_from_replay,
    export_source_rest_pose_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SMPL source rest pose JSON for Unity SmplWorldDeltaRetarget reference playback."
    )
    parser.add_argument("--replay_json", default=str(DEFAULT_REPLAY_JSON), type=str)
    parser.add_argument("--source_npz", default="", type=str, help="Optional explicit source npz; overrides replay metadata.")
    parser.add_argument("--output_json", default=str(DEFAULT_SOURCE_REST_JSON), type=str)
    parser.add_argument("--smpl_model_dir", default=str(DEFAULT_SMPL_MODEL_DIR), type=str)
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    output_json = Path(args.output_json).resolve()
    smpl_model_dir = Path(args.smpl_model_dir).resolve()
    if args.source_npz:
        path = export_source_rest_pose_json(
            source_npz=Path(args.source_npz).resolve(),
            output_json=output_json,
            smpl_model_dir=smpl_model_dir,
        )
    else:
        path = export_source_rest_pose_from_replay(
            replay_json=Path(args.replay_json).resolve(),
            output_json=output_json,
            smpl_model_dir=smpl_model_dir,
        )

    print(f"[DEBUG_RETARGET_SOURCE_REST] wrote: {path}")
    return path


if __name__ == "__main__":
    main()
