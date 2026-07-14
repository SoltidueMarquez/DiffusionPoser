from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME
from model.diffusionposer_dit import DiffusionPoserDiT
from model.realtime_pose_target_dit import RealtimePoseTargetDiT


def create_model_and_diffusion(args):
    model_arch = getattr(args, "model_arch", "full_feature_dit")
    if model_arch == "target_dit":
        model = RealtimePoseTargetDiT(
            input_feats=args.input_feats,
            schema_name=getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME),
            latent_dim=args.latent_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            dropout=args.dropout,
            zero_init=args.zero_init,
            max_seq_len=args.max_seq_len,
        )
    elif model_arch == "full_feature_dit":
        model = DiffusionPoserDiT(
            input_feats=args.input_feats,
            latent_dim=args.latent_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            dropout=args.dropout,
            zero_init=args.zero_init,
            max_seq_len=args.max_seq_len,
        )
    else:
        raise ValueError(f"未知 model_arch={model_arch}")
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def create_gaussian_diffusion(args):
    """创建与 StableMotion 兼容的扩散调度器，默认预测干净样本 x0。"""

    steps = args.diffusion_steps
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
        yaw_loss_weight=getattr(args, "yaw_loss_weight", 10.0),
        fk_loss_weight=getattr(args, "fk_loss_weight", 2.0),
        joint_vel_loss_weight=getattr(args, "joint_vel_loss_weight", 0.5),
        foot_lock_loss_weight=getattr(args, "foot_lock_loss_weight", 0.5),
        root_delta_loss_weight=getattr(args, "root_delta_loss_weight", 1.0),
        root_height_loss_weight=getattr(args, "root_height_loss_weight", 1.0),
        contact_loss_weight=getattr(args, "contact_loss_weight", 0.5),
        tracker_pos_loss_weight=getattr(args, "tracker_pos_loss_weight", 10.0),
        tracker_pos_huber_beta=getattr(args, "tracker_pos_huber_beta", 0.05),
        tracker_pos_timestep_min_weight=getattr(args, "tracker_pos_timestep_min_weight", 0.1),
        tracker_pos_timestep_gamma=getattr(args, "tracker_pos_timestep_gamma", 2.0),
        tracker_rot_loss_weight=getattr(args, "tracker_rot_loss_weight", 2.0),
        head_anchor_loss_weight=getattr(args, "head_anchor_loss_weight", 1.0),
        hip_root_position_loss_weight=getattr(args, "hip_root_position_loss_weight", 1.0),
        hip_root_yaw_loss_weight=getattr(args, "hip_root_yaw_loss_weight", 1.0),
        hip_root_height_loss_weight=getattr(args, "hip_root_height_loss_weight", 1.0),
    )
