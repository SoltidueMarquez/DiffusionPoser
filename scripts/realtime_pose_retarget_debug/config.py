from __future__ import annotations

from pathlib import Path

import numpy as np


DIFFUSIONPOSER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY_JSON = (
    DIFFUSIONPOSER_ROOT.parent
    / "SIGGRAPH2024Unity"
    / "Assets"
    / "Projects"
    / "RealtimePose"
    / "TestData"
    / "Generated"
    / "realtime_pose_replay.json"
)
DEFAULT_BODY_FBX_META = DIFFUSIONPOSER_ROOT.parent / "SIGGRAPH2024Unity" / "Assets" / "Models" / "body.fbx.meta"
DEFAULT_SOURCE_REST_JSON = (
    DIFFUSIONPOSER_ROOT.parent
    / "SIGGRAPH2024Unity"
    / "Assets"
    / "Projects"
    / "RealtimePose"
    / "TestData"
    / "Generated"
    / "smpl_source_rest.json"
)
DEFAULT_OUTPUT_DIR = DIFFUSIONPOSER_ROOT / "output" / "realtime_pose_retarget_debug"
IDENTITY_6D = np.asarray([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
