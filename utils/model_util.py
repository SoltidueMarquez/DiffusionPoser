from __future__ import annotations

import json
from pathlib import Path

import torch

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.realtime_pose_current_dit import RealtimePoseCurrentDiT
from model.realtime_pose_predictor import RealtimePosePredictor


def create_model_and_diffusion(args):
    model_arch = getattr(args, "model_arch", "current_dit")
    if model_arch != "current_dit":
        raise ValueError("当前主链路只支持 current_dit。")
    model = RealtimePoseCurrentDiT(
        input_feats=getattr(args, "input_feats", 144),
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    )
    return model, create_gaussian_diffusion(args)


def create_gaussian_diffusion(args):
    """创建只恢复当前 144D Predictor residual 的扩散过程。"""

    steps = int(args.diffusion_steps)
    if not bool(args.predict_xstart):
        raise ValueError("Predictor residual diffusion 固定要求 --predict_xstart 1。")
    timestep_respacing = args.ts_respace if getattr(args, "ts_respace", "") else [steps]
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_betas=1.0)
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X if args.predict_xstart else gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_SMALL if args.sigma_small else gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        aux_loss_weight=getattr(args, "aux_loss_weight", 1.0),
        rotation_loss_weight=getattr(args, "rotation_loss_weight", 1.0),
        fk_loss_weight=getattr(args, "fk_loss_weight", 2.0),
        local_rot_loss_weight=getattr(args, "local_rot_loss_weight", 1.0),
        tracker_pos_loss_weight=getattr(args, "tracker_pos_loss_weight", 10.0),
        tracker_pos_huber_beta=getattr(args, "tracker_pos_huber_beta", 0.05),
        tracker_rot_loss_weight=getattr(args, "tracker_rot_loss_weight", 1.0),
        root_loss_weight=getattr(args, "root_loss_weight", 1.0),
        head_ref_joint_distance_loss_weight=getattr(
            args, "head_ref_joint_distance_loss_weight", 1.0
        ),
        head_to_root_xz_loss_weight=getattr(args, "head_to_root_xz_loss_weight", 1.0),
        hip_height_loss_weight=getattr(args, "hip_height_loss_weight", 1.0),
        rotation_velocity_loss_weight=getattr(
            args, "rotation_velocity_loss_weight", 1.0
        ),
    )


def load_realtime_pose_predictor(
    checkpoint_path: str | Path,
    device: torch.device,
) -> RealtimePosePredictor:
    """从 checkpoint 邻接的训练参数恢复冻结 Predictor。"""

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Predictor checkpoint 不存在：{checkpoint}")
    args_path = checkpoint.with_name("args.json")
    values = json.loads(args_path.read_text(encoding="utf-8")) if args_path.is_file() else {}
    model = RealtimePosePredictor(
        latent_dim=int(values.get("latent_dim", 512)),
        num_layers=int(values.get("layers", 4)),
        num_heads=int(values.get("heads", 4)),
        feedforward_dim=int(values.get("feedforward_dim", 1024)),
        dropout=float(values.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return model.eval().requires_grad_(False)
