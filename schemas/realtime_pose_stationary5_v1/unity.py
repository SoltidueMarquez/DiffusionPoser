from __future__ import annotations

from typing import Any, Mapping

from schemas.base import SchemaSpec
from schemas.realtime_pose_stationary5_v1.contract import (
    BODY_POSE_DIM,
    HIP_TRACKER_INDEX,
    MIN_VALID_TRACKERS,
    PELVIS_HEIGHT_DIM,
    ROOT_DELTA_XZ_DIM,
    ROOT_HEADING_DELTA_DIM,
    SENSOR_VALID_DIM,
    SMPL_JOINT_NAMES,
    STATIONARY_JOINT_INDICES,
    STATIONARY_JOINT_NAMES,
    STATIONARY_PROB_DIM,
    TRACKER_COUNT,
    TRACKER_JOINT_INDICES,
    TRACKER_NAMES,
    TRACKER_POS_DIM,
    TRACKER_ROT_DIM,
)


def build_stationary5_unity_feature_schema(spec: SchemaSpec) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": 4,
        "schemaName": spec.name,
        "poseRepresentation": spec.pose_representation,
        "featureDim": spec.feature_dim,
        "sequenceLength": spec.seq_len,
        "targetStart": spec.target_start,
        "targetLength": spec.target_length,
        "targetFeatureLength": spec.target_dim,
        "boneCount": len(SMPL_JOINT_NAMES),
        "boneNames": list(SMPL_JOINT_NAMES),
        "trackerCount": TRACKER_COUNT,
        "trackerNames": list(TRACKER_NAMES),
        "trackerJointIndices": [int(value) for value in TRACKER_JOINT_INDICES],
        "hipTrackerIndex": HIP_TRACKER_INDEX,
        "minValidTrackers": MIN_VALID_TRACKERS,
        "bodyPoseBodyFbxLocalDelta6d": {
            "name": spec.body_pose_key,
            "start": spec.body_pose_start,
            "length": BODY_POSE_DIM,
        },
        "rootHeadingDeltaSinCos": {
            "name": spec.root_heading_delta_key,
            "start": spec.root_yaw_delta_start,
            "length": ROOT_HEADING_DELTA_DIM,
        },
        "rootDeltaXZReference": {
            "name": "root_delta_xz_ref",
            "start": spec.root_delta_xz_start,
            "length": ROOT_DELTA_XZ_DIM,
        },
        "pelvisHeight": {
            "name": spec.pelvis_height_key,
            "start": spec.root_height_start,
            "length": PELVIS_HEIGHT_DIM,
        },
        "stationaryProb5": {
            "name": "stationary_prob_5",
            "start": spec.stationary_prob_start,
            "length": STATIONARY_PROB_DIM,
            "jointIndices": [int(value) for value in STATIONARY_JOINT_INDICES],
            "jointNames": list(STATIONARY_JOINT_NAMES),
        },
        "trackerPositionReference": {
            "name": "tracker_pos_ref",
            "start": spec.tracker_pos_ref_start,
            "length": TRACKER_POS_DIM,
        },
        "trackerRotation6dReference": {
            "name": "tracker_rot_ref_6d",
            "start": spec.tracker_rot_ref_start,
            "length": TRACKER_ROT_DIM,
        },
        "sensorValid": {
            "name": "sensor_valid",
            "start": spec.sensor_valid_start,
            "length": SENSOR_VALID_DIM,
        },
        "runtimeRules": {
            "poseRepresentation": spec.pose_representation,
            "rootPositionY": spec.root_y_policy,
            "pelvisHeightApplication": spec.pelvis_height_mode,
            "requiresHeadTracker": True,
            "requiresHipTracker": False,
            "requiresTotalValidTrackersAtLeast": MIN_VALID_TRACKERS,
            "trackerReferenceYaw": "hip_current_else_previous_final",
            "trackerCodecVersion": "tracker_codec_v2",
            "referencePolicyVersion": "hip_current_else_previous_final_v1",
            "resolverContractVersion": "runtime_root_resolver_v1",
            "onnxDummyInputShape": [1, spec.feature_dim, spec.seq_len],
            "failSafe": "hold_previous_frame_when_tracker_validity_fails",
        },
    }
    return payload
