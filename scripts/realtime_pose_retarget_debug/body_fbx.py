from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from data_loaders.realtime_pose_kinematics import SMPL_JOINT_NAMES


def canonical_bone_name(name: str) -> str:
    value = str(name).strip().lower()
    if value.startswith("m_avg_"):
        value = value[len("m_avg_") :]
    aliases = {
        "l_hip": "left_hip",
        "r_hip": "right_hip",
        "l_knee": "left_knee",
        "r_knee": "right_knee",
        "l_ankle": "left_ankle",
        "r_ankle": "right_ankle",
        "l_foot": "left_foot",
        "r_foot": "right_foot",
        "l_collar": "left_collar",
        "r_collar": "right_collar",
        "l_shoulder": "left_shoulder",
        "r_shoulder": "right_shoulder",
        "l_elbow": "left_elbow",
        "r_elbow": "right_elbow",
        "l_wrist": "left_wrist",
        "r_wrist": "right_wrist",
        "l_hand": "left_hand",
        "r_hand": "right_hand",
    }
    return aliases.get(value, value)


def parse_body_fbx_offsets_from_meta(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"body.fbx.meta not found: {path}")

    text = path.read_text(encoding="utf-8")
    float_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    entry_pattern = re.compile(
        rf"- name:\s*(?P<name>[^\n\r]+)\s+"
        rf"parentName:\s*(?P<parent>[^\n\r]*)\s+"
        rf"position:\s*\{{x:\s*(?P<x>{float_pattern}),\s*y:\s*(?P<y>{float_pattern}),\s*z:\s*(?P<z>{float_pattern})\}}",
        re.MULTILINE,
    )
    found: dict[str, np.ndarray] = {}
    for match in entry_pattern.finditer(text):
        canonical = canonical_bone_name(match.group("name"))
        if canonical not in SMPL_JOINT_NAMES or canonical in found:
            continue
        found[canonical] = np.asarray(
            [float(match.group("x")), float(match.group("y")), float(match.group("z"))],
            dtype=np.float32,
        )

    missing = [name for name in SMPL_JOINT_NAMES if name not in found]
    if missing:
        raise ValueError(f"body.fbx.meta missing SMPL24 local offsets for: {missing}")
    return np.stack([found[name] for name in SMPL_JOINT_NAMES], axis=0)
