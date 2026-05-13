from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from model.diffusionposer_dit import DiffusionPoserDiT


def create_model_and_diffusion(args):
    model = DiffusionPoserDiT(
        input_feats=args.input_feats,
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        zero_init=args.zero_init,
    )
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
    )
