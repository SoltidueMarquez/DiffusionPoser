from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.realtime_pose_spatiotemporal_dit import RealtimePoseSpatioTemporalDiT


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
    )
    return model, create_gaussian_diffusion(args)


def create_gaussian_diffusion(args):
    """创建联合恢复当前帧和未来 10 帧、以 10 帧历史为条件的扩散过程。"""

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
        hip_height_loss_weight=getattr(args, "hip_height_loss_weight", 1.0),
        rotation_velocity_loss_weight=getattr(
            args, "rotation_velocity_loss_weight", 1.0
        ),
        contact_loss_weight=getattr(args, "contact_loss_weight", 0.1),
        contact_slide_loss_weight=getattr(args, "contact_slide_loss_weight", 0.1),
    )
