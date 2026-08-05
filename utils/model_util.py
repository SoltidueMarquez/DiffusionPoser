from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.realtime_pose_spatiotemporal_dit import RealtimePoseSpatioTemporalDiT
from data_loaders.realtime_pose_config import TrackerReliabilityConfig


def create_model_and_diffusion(args):
    model_arch = getattr(args, "model_arch", "spatiotemporal_dit")
    if model_arch != "spatiotemporal_dit":
        raise ValueError("当前主链路只支持 spatiotemporal_dit；旧模型架构不兼容。")
    model = RealtimePoseSpatioTemporalDiT(
        input_feats=args.input_feats,
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        zero_init=args.zero_init,
        max_seq_len=args.max_seq_len,
        # Task 与 Runtime 当前都固定使用默认可靠度配置；在 metadata 契约完成前，
        # 模型工厂也禁止从 CLI 覆盖，避免同一持续时间被按不同 duration_cap 解释。
        reliability_config=TrackerReliabilityConfig().validate(),
    )
    return model, create_gaussian_diffusion(args)


def create_gaussian_diffusion(args):
    """创建直接作用于 11 帧、每帧 144 维 Pose 窗口的扩散过程。"""

    steps = int(args.diffusion_steps)
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
        future_leg_loss_weight=getattr(args, "future_leg_loss_weight", 0.5),
        contact_loss_weight=getattr(args, "contact_loss_weight", 0.1),
    )
