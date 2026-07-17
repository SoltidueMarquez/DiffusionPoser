from diffusion import gaussian_diffusion as gd
from diffusion.realtime_pose import REALTIME_POSE_LOSS_DEFAULTS
from diffusion.respace import SpacedDiffusion, space_timesteps
from data_loaders.sensor_masking import DEFAULT_REALTIME_POSE_SCHEMA_NAME
from model.diffusionposer_dit import DiffusionPoserDiT
from model.realtime_pose_target_dit import RealtimePoseTargetDiT


def create_model_and_diffusion(args):
    model_arch = getattr(args, "model_arch", "target_dit")
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

    loss_options = {
        name: getattr(args, name, default)
        for name, default in REALTIME_POSE_LOSS_DEFAULTS.items()
    }
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X if args.predict_xstart else gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_SMALL if args.sigma_small else gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        **loss_options,
    )
