from __future__ import annotations

from typing import Final

from schemas.base import SchemaSpec


SCHEMA_NAME: Final = "realtime_pose_stationary5_v1"
LEGACY_SCHEMA_NAME: Final = "realtime_pose_body_fbx_local_root_y0_v1"
RESOLVER_CONTRACT_VERSION: Final = "runtime_root_resolver_v2"

TASK_FORMAT: Final = "realtime_pose_source_reference"

POSE_REPRESENTATION: Final = "body_fbx_local_delta_6d"
BODY_POSE_KEY: Final = "body_pose_body_fbx_local_delta_6d"
ROOT_HEADING_DELTA_KEY: Final = "root_heading_delta_sincos"
PELVIS_HEIGHT_KEY: Final = "pelvis_height"
ROOT_Y_POLICY: Final = "fixed_zero"
PELVIS_HEIGHT_MODE: Final = "pelvis_local_offset_y"

SEQ_LEN: Final = 61
TARGET_START: Final = 60
TARGET_LENGTH: Final = 1
FEATURE_DIM: Final = 214
TARGET_DIM: Final = 154

BODY_POSE_DIM: Final = 144
ROOT_HEADING_DELTA_DIM: Final = 2
ROOT_DELTA_XZ_DIM: Final = 2
PELVIS_HEIGHT_DIM: Final = 1
STATIONARY_PROB_DIM: Final = 5
TRACKER_COUNT: Final = 6
TRACKER_POS_DIM: Final = TRACKER_COUNT * 3
TRACKER_ROT_DIM: Final = TRACKER_COUNT * 6
SENSOR_VALID_DIM: Final = TRACKER_COUNT

BODY_POSE_START: Final = 0
ROOT_HEADING_DELTA_START: Final = 144
ROOT_DELTA_XZ_START: Final = 146
PELVIS_HEIGHT_START: Final = 148
STATIONARY_PROB_START: Final = 149
TRACKER_POS_REF_START: Final = 154
TRACKER_ROT_REF_START: Final = 172
SENSOR_VALID_START: Final = 208

CHANNEL_RANGES: Final = {
    "body_pose_body_fbx_local_delta_6d": (BODY_POSE_START, ROOT_HEADING_DELTA_START),
    "root_heading_delta_sincos": (ROOT_HEADING_DELTA_START, ROOT_DELTA_XZ_START),
    "root_delta_xz_ref": (ROOT_DELTA_XZ_START, PELVIS_HEIGHT_START),
    "pelvis_height": (PELVIS_HEIGHT_START, STATIONARY_PROB_START),
    "stationary_prob_5": (STATIONARY_PROB_START, TRACKER_POS_REF_START),
    "tracker_pos_ref": (TRACKER_POS_REF_START, TRACKER_ROT_REF_START),
    "tracker_rot_ref_6d": (TRACKER_ROT_REF_START, SENSOR_VALID_START),
    "sensor_valid": (SENSOR_VALID_START, FEATURE_DIM),
}

SMPL_JOINT_NAMES: Final = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
TRACKER_NAMES: Final = ("head", "left_wrist", "right_wrist", "waist", "left_foot", "right_foot")
TRACKER_JOINT_INDICES: Final = (15, 20, 21, 0, 10, 11)
HIP_TRACKER_INDEX: Final = 3
MIN_VALID_TRACKERS: Final = 3
STATIONARY_JOINT_INDICES: Final = (0, 10, 11, 22, 23)
STATIONARY_JOINT_NAMES: Final = ("pelvis", "left_foot", "right_foot", "left_hand", "right_hand")


def build_stationary5_spec(
    name: str = SCHEMA_NAME,
    canonical_name: str = SCHEMA_NAME,
    task_format: str = TASK_FORMAT,
    one_line: str = "Realtime pose stationary5 canonical schema.",
) -> SchemaSpec:
    return SchemaSpec(
        name=name,
        canonical_name=canonical_name,
        one_line=one_line,
        task_format=task_format,
        feature_dim=FEATURE_DIM,
        target_dim=TARGET_DIM,
        body_pose_start=BODY_POSE_START,
        root_yaw_delta_start=ROOT_HEADING_DELTA_START,
        root_delta_xz_start=ROOT_DELTA_XZ_START,
        root_height_start=PELVIS_HEIGHT_START,
        stationary_prob_start=STATIONARY_PROB_START,
        tracker_pos_ref_start=TRACKER_POS_REF_START,
        tracker_rot_ref_start=TRACKER_ROT_REF_START,
        sensor_valid_start=SENSOR_VALID_START,
        seq_len=SEQ_LEN,
        target_start=TARGET_START,
        target_length=TARGET_LENGTH,
        pose_representation=POSE_REPRESENTATION,
        body_pose_key=BODY_POSE_KEY,
        root_heading_delta_key=ROOT_HEADING_DELTA_KEY,
        pelvis_height_key=PELVIS_HEIGHT_KEY,
        root_y_policy=ROOT_Y_POLICY,
        pelvis_height_mode=PELVIS_HEIGHT_MODE,
    )
