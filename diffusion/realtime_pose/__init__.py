"""Realtime pose loss, resolver, and configuration interfaces."""

from diffusion.realtime_pose.config import (
    REALTIME_POSE_LOSS_DEFAULTS,
    REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS,
    REALTIME_POSE_LOSS_TERM_TO_WEIGHT,
    RealtimePoseLossConfig,
)
from diffusion.realtime_pose.loss_terms import RealtimePoseAuxiliaryLoss
from diffusion.realtime_pose.resolver import (
    DifferentiableResolverResult,
    resolve_realtime_pose_frame_torch,
    rotation_matrix_log_vector,
    wrap_angle_torch,
)

__all__ = [
    "DifferentiableResolverResult",
    "REALTIME_POSE_LOSS_DEFAULTS",
    "REALTIME_POSE_LOSS_GRADIENT_TARGET_RATIOS",
    "REALTIME_POSE_LOSS_TERM_TO_WEIGHT",
    "RealtimePoseAuxiliaryLoss",
    "RealtimePoseLossConfig",
    "resolve_realtime_pose_frame_torch",
    "rotation_matrix_log_vector",
    "wrap_angle_torch",
]
