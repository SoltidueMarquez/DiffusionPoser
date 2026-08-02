from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from data_loaders.realtime_pose_kinematics import JOINT_INDEX
from data_loaders.sensor_masking import REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM


@dataclass
class MotionEncoding:
    temporal_tokens: torch.Tensor
    latents: torch.Tensor
    valid_frame_mask: torch.Tensor


class RegionalMotionEncoder(nn.Module):
    """严格只读取过去 60 帧，输出 global/pelvis/左右腿四类 motion prior。"""

    def __init__(self, latent_dim: int = 512, num_layers: int = 4, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.latent_dim = int(latent_dim)
        region_joint_names = (
            tuple(JOINT_INDEX),
            ("pelvis", "spine1", "spine2", "spine3", "neck", "head", "left_collar", "right_collar"),
            ("left_hip", "left_knee", "left_ankle", "left_foot"),
            ("right_hip", "right_knee", "right_ankle", "right_foot"),
        )
        self.region_joint_indices = tuple(
            tuple(JOINT_INDEX[name] for name in names) for names in region_joint_names
        )
        self.pose_projections = nn.ModuleList(
            [nn.Linear(len(indices) * 6, self.latent_dim) for indices in self.region_joint_indices]
        )
        self.trajectory_projection = nn.Linear(5, self.latent_dim)
        self.history_context_projections = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(4)]
        )
        self.temporal_embedding = nn.Parameter(
            torch.zeros(1, 1, REALTIME_POSE_HISTORY_LENGTH, self.latent_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=int(num_heads),
            dim_feedforward=self.latent_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))

    def forward(
        self,
        pose_history: torch.Tensor,
        tracker_history_summary: torch.Tensor,
        trajectory_history: torch.Tensor,
        valid_frame_mask: torch.Tensor,
    ) -> MotionEncoding:
        batch_size = pose_history.shape[0]
        if tuple(pose_history.shape) != (batch_size, REALTIME_POSE_HISTORY_LENGTH, REALTIME_POSE_TARGET_DIM):
            raise ValueError("pose_history 必须为 [B,60,144]。")
        if tuple(tracker_history_summary.shape[:2]) != (batch_size, 6):
            raise ValueError("tracker_history_summary 必须为 [B,6,D]。")
        if tuple(trajectory_history.shape) != (batch_size, REALTIME_POSE_HISTORY_LENGTH, 5):
            raise ValueError("trajectory_history 必须为 [B,60,5]。")
        if tuple(valid_frame_mask.shape) != (batch_size, REALTIME_POSE_HISTORY_LENGTH):
            raise ValueError("valid_frame_mask 必须为 [B,60]。")

        pose = pose_history.reshape(batch_size, REALTIME_POSE_HISTORY_LENGTH, 24, 6)
        tracker_routes = ((0, 1, 2, 3, 4, 5), (0, 3), (0, 3, 4), (0, 3, 5))
        trajectory_token = self.trajectory_projection(trajectory_history)
        region_tokens: list[torch.Tensor] = []
        for region_index, joint_indices in enumerate(self.region_joint_indices):
            joint_index = torch.as_tensor(joint_indices, device=pose.device, dtype=torch.long)
            pose_token = self.pose_projections[region_index](
                pose.index_select(2, joint_index).flatten(2)
            )
            tracker_context = tracker_history_summary[:, tracker_routes[region_index]].mean(dim=1)
            tracker_context = self.history_context_projections[region_index](tracker_context)[:, None]
            region_tokens.append(pose_token + trajectory_token + tracker_context)
        tokens = torch.stack(region_tokens, dim=1) + self.temporal_embedding

        flat_tokens = tokens.reshape(batch_size * 4, REALTIME_POSE_HISTORY_LENGTH, self.latent_dim)
        flat_valid = valid_frame_mask[:, None].expand(-1, 4, -1).reshape(
            batch_size * 4, REALTIME_POSE_HISTORY_LENGTH
        )
        safe_valid = flat_valid.clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[empty, 0] = True
        causal_mask = torch.triu(
            torch.ones(
                REALTIME_POSE_HISTORY_LENGTH,
                REALTIME_POSE_HISTORY_LENGTH,
                dtype=torch.bool,
                device=pose.device,
            ),
            diagonal=1,
        )
        encoded = self.temporal_encoder(
            flat_tokens,
            mask=causal_mask,
            src_key_padding_mask=~safe_valid,
        )
        encoded = encoded * flat_valid[..., None].to(encoded.dtype)
        lengths = flat_valid.long().sum(dim=1)
        frame_indices = torch.arange(REALTIME_POSE_HISTORY_LENGTH, device=pose.device)
        last = torch.where(flat_valid, frame_indices[None], -1).max(dim=1).values.clamp_min(0)
        latents = encoded[torch.arange(encoded.shape[0], device=encoded.device), last]
        latents = latents * (lengths > 0)[:, None].to(latents.dtype)
        return MotionEncoding(
            temporal_tokens=encoded.reshape(batch_size, 4, REALTIME_POSE_HISTORY_LENGTH, self.latent_dim),
            latents=latents.reshape(batch_size, 4, self.latent_dim),
            valid_frame_mask=valid_frame_mask.bool(),
        )
