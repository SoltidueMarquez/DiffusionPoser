# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch
import torch as th
from copy import deepcopy
from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
)
from diffusion.nn import mean_flat, sum_flat
from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood, compute_snr
from diffusion.realtime_pose_losses import compute_raw_deployed_losses
from diffusion.realtime_pose_inpainting import (
    apply_realtime_pose_inpainting,
    validate_realtime_pose_inpainting_condition,
)
from diffusion.realtime_pose_projection import project_realtime_pose_xstart


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: 一个一维 numpy 数组，表示每个扩散时间步的 beta，从 T 到 1。
    :param model_mean_type: 一个 ModelMeanType，决定模型输出的含义。
    :param model_var_type: 一个 ModelVarType，决定方差的输出方式。
    :param loss_type: 一个 LossType，决定要使用的损失函数。
    :param rescale_timesteps: 若为 True，将传入浮点时间步，使其始终按原论文的尺度（0 到 1000）处理。
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        aux_loss_weight=1.0,
        rotation_loss_weight=1.0,
        fk_loss_weight=2.0,
        local_rot_loss_weight=1.0,
        tracker_pos_loss_weight=10.0,
        tracker_pos_huber_beta=0.05,
        diffusion_loss_weight=1.0,
        tracker_rot_loss_weight=1.0,
        root_loss_weight=1.0,
        head_ref_joint_distance_loss_weight=1.0,
        head_to_root_xz_loss_weight=1.0,
        hip_height_loss_weight=1.0,
        rotation_velocity_loss_weight=1.0,
        contact_loss_weight=0.0,
        contact_slide_loss_weight=0.0,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.aux_loss_weight = float(aux_loss_weight)
        self.rotation_loss_weight = float(rotation_loss_weight)
        self.fk_loss_weight = float(fk_loss_weight)
        self.local_rot_loss_weight = float(local_rot_loss_weight)
        self.tracker_pos_loss_weight = float(tracker_pos_loss_weight)
        self.tracker_pos_huber_beta = float(tracker_pos_huber_beta)
        if not np.isfinite(self.tracker_pos_huber_beta) or self.tracker_pos_huber_beta <= 0.0:
            raise ValueError("tracker_pos_huber_beta 必须是有限正数。")
        self.diffusion_loss_weight = float(diffusion_loss_weight)
        self.tracker_rot_loss_weight = float(tracker_rot_loss_weight)
        self.root_loss_weight = float(root_loss_weight)
        self.head_ref_joint_distance_loss_weight = float(
            head_ref_joint_distance_loss_weight
        )
        self.head_to_root_xz_loss_weight = float(head_to_root_xz_loss_weight)
        self.hip_height_loss_weight = float(hip_height_loss_weight)
        self.rotation_velocity_loss_weight = float(rotation_velocity_loss_weight)
        self.contact_loss_weight = float(contact_loss_weight)
        self.contact_slide_loss_weight = float(contact_slide_loss_weight)

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.l2_loss = th.nn.MSELoss(reduction='none')
        self.l1_loss = th.nn.L1Loss(reduction='none')

    def masked_l2(self, a, b, mask, feature_w=None, use_l1=False):
        # assuming a.shape == b.shape == bs, Jdim, seqlen
        # assuming mask.shape == bs, 1, seqlen
        # print(mask.shape, a.shape)
        eps = 1e-6
        if mask.dim() != a.dim():
            mask = mask.squeeze()
            mask = mask.unsqueeze(1)
        # print(mask.shape == a.shape)
        mask = torch.broadcast_to(mask, a.shape)
        if use_l1:
            loss = self.l1_loss(a, b)
        else:
            loss = self.l2_loss(a, b)
        if feature_w is not None:
            loss = loss * feature_w
        loss = sum_flat(loss * mask.float())  # gives \sigma_euclidean over unmasked elements

        non_zero_elements = sum_flat(mask) + eps
        mse_loss_val = loss / non_zero_elements

        return mse_loss_val


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:  # 未指定噪声时，按输入形状采样标准正态噪声
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape  # 噪声必须与 x_start 形状一致
        res =  (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start  # 缩放 x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise  # 加上按时间步缩放后的噪声
        )
        return res  # 返回对应时间步的带噪样本

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        由模型预测 p(x_{t-1} | x_t) 的分布，包括均值、方差以及 x_0 的估计。

        :param model: 接受带噪输入与时间步并输出预测结果的模型。
        :param x: 当前时间步的样本张量（形状 [N, C, ...]）。
        :param t: 每个样本对应的时间步（1D 张量，长度等于 batch）。
        :param clip_denoised: 若为 True，则将模型预测的 x_start 限制在 [-1, 1] 。
        :param denoised_fn: 若传入函数，在裁剪之前对 x_start 做一次自定义处理。
        :param model_kwargs: 额外的条件输入（如 inpainting 掩码、长度等）。
        :return: 字典，包含 mean/variance/log_variance/pred_xstart 四个键。
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        # 模型的时间步输入可能需要 rescale（_scale_timesteps），因此封装它
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        auxiliary_outputs = None
        if isinstance(model_output, tuple):
            if len(model_output) != 2 or not isinstance(model_output[1], dict):
                raise TypeError("模型 tuple 输出必须为 (prediction, auxiliary_outputs)。")
            model_output, auxiliary_outputs = model_output

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                # 模型直接输出 log variance
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # model_var_values 被限制在 [-1,1]，用于在最小/最大 log var 之间插值
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # 对于 FIXED_LARGE，初始 log variance 取 posterior_variance[1] 拼接 beta
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        result = {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
        if auxiliary_outputs is not None:
            result["auxiliary_outputs"] = auxiliary_outputs
        return result

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_mean_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, p_mean_var, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def condition_score_with_grad(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, t, p_mean_var, **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        在给定时间步下，根据模型预测从 x_t 跳到 x_{t-1}。

        :param model: 用于去噪预测的模型模块。
        :param x: 当前的带噪样本（即 x_t）。
        :param t: 本次采样的时间步，0 表示第一步的扩散末端。
        :param clip_denoised: 若为 True，则将模型预测的 x_start 裁剪到 [-1, 1]。
        :param denoised_fn: 若传入函数，则在使用模型预测前先处理一次 x_start。
        :param cond_fn: 若不为 None，则该函数计算一个与模型等效的梯度，作为额外条件。
        :param model_kwargs: 扩散模型的额外条件。
        :return: 包含以下键的字典：
                 - 'sample': 本步采样得到的 x_{t-1}。
                 - 'pred_xstart': 当前模型对 x_0 的预测值。
        """

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)

        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        # nonzero_mask 控制只有 t!=0 时才叠加随机噪声
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        if cond_fn is not None:
            # 可选的 classifier guidance：通过 cond_fn 修改均值
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        与 p_sample 相似，但在内部打开梯度计算，使得 cond_fn_with_grad 也能使用。

        :param model: 去噪模型。
        :param x: 当前时间步的样本 x_t。
        :param t: 当前时间步索引（0 表示扩散末端）。
        :param clip_denoised: 若为 True，则对模型预测的 x_start 做 [-1, 1] 裁剪。
        :param denoised_fn: 若传入函数，在使用模型预测前先处理一次。
        :param cond_fn: 若不为空，则表示需要在梯度流中应用额外的条件引导。
        :param model_kwargs: 模型额外输入。
        :return: dict，包含
                 - 'sample': 当前时间步的去噪输出 x_{t-1}。
                 - 'pred_xstart': 模型对 x_0 的预测（detach 后输出）。
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            noise = th.randn_like(x)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # 只有 t != 0 时才加入噪声
            if cond_fn is not None:
                out["mean"] = self.condition_mean_with_grad(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        # pred_xstart 直接 detach，避免后续 backprop 影响原始图
        return {"sample": sample, "pred_xstart": out["pred_xstart"].detach()}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None  # 用于存储最后一个时间步的采样结果
        if dump_steps is not None:  # 如果需要保存中间步骤
            dump = []  # 初始化中间结果列表

        # 遍历渐进式采样生成器，从 T 到 0 逐步去噪
        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
        )):
            # 如果当前步数在 dump_steps 列表中，则保存当前采样结果的副本
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            final = sample  # 更新最后一个时间步的结果
        if dump_steps is not None:  # 如果是 dump 模式，返回保存的所有步骤列表
            return dump
        return final["sample"]  # 默认返回最终生成的采样结果 (x_0)

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        cond_fn_with_grad=False,
        const_noise=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)

            with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    const_noise=const_noise,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out_orig = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
        else:
            out = out_orig

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"]}

    def ddim_sample_with_grad(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        with th.enable_grad():
            x = x.detach().requires_grad_()
            out_orig = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if cond_fn is not None:
                out = self.condition_score_with_grad(cond_fn, out_orig, x, t,
                                                     model_kwargs=model_kwargs)
            else:
                out = out_orig

        out["pred_xstart"] = out["pred_xstart"].detach()

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"].detach()}

    def projected_ddim_sample(
        self,
        model,
        x,
        t,
        projection_fn,
        clip_denoised=False,
        model_kwargs=None,
        eta=0.0,
        inpaint_condition=None,
    ):
        """在模型前注入 IK 条件，并用最终 deployed x0 重算 epsilon。"""

        if model_kwargs is None:
            model_kwargs = {}
        x_model = x
        if inpaint_condition is not None:
            x_model, _ = apply_realtime_pose_inpainting(
                x_t=x,
                t=t,
                condition=inpaint_condition,
                alphas_cumprod=self.alphas_cumprod,
            )
        step_model_kwargs = dict(model_kwargs)
        if bool(torch.all(t == 0)):
            step_model_kwargs["return_aux_outputs"] = True
        out = self.p_mean_variance(
            model,
            x_model,
            t,
            clip_denoised=clip_denoised,
            model_kwargs=step_model_kwargs,
        )
        raw_pred_xstart = out["pred_xstart"]
        # 置信度只通过 inpainting 生效；最终步额外执行一次 Tracker hard projection。
        should_project = t == 0
        if bool(should_project.any()):
            projected = projection_fn(raw_pred_xstart)
            selector = should_project.view(-1, *([1] * (x.ndim - 1)))
            deployed_pred_xstart = torch.where(selector, projected, raw_pred_xstart)
        else:
            # 早期步骤保持 raw x0，禁止提前覆盖 hard Tracker。
            deployed_pred_xstart = raw_pred_xstart
        eps = self._predict_eps_from_xstart(x_model, t, deployed_pred_xstart)
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x_model.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x_model.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        mean_pred = (
            deployed_pred_xstart * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        sample = mean_pred + nonzero_mask * sigma * th.randn_like(x)
        # 最终步无条件返回 deployed pose，保证 late/final ablation 也遵守部署契约。
        sample = torch.where(nonzero_mask.bool(), sample, deployed_pred_xstart)
        result = {
            "sample": sample,
            "pred_xstart": deployed_pred_xstart,
            "raw_pred_xstart": raw_pred_xstart,
            "deployed_pred_xstart": deployed_pred_xstart,
        }
        if "auxiliary_outputs" in out:
            result["auxiliary_outputs"] = out["auxiliary_outputs"]
        return result

    def projected_ddim_sample_loop(
        self,
        model,
        shape,
        projection_fn,
        noise=None,
        clip_denoised=False,
        model_kwargs=None,
        device=None,
        eta=0.0,
        progress=False,
        inpaint_condition=None,
    ):
        """执行完整 IK-Inpainting DDIM，并保留最终 raw/deployed x0。"""

        if device is None:
            device = next(model.parameters()).device
        image = noise if noise is not None else th.randn(*shape, device=device)
        if inpaint_condition is not None:
            validate_realtime_pose_inpainting_condition(inpaint_condition)
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm

            indices = tqdm(indices)
        final = None
        for index in indices:
            timestep = th.full((shape[0],), index, device=device, dtype=th.long)
            with th.no_grad():
                final = self.projected_ddim_sample(
                    model,
                    image,
                    timestep,
                    projection_fn=projection_fn,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    inpaint_condition=inpaint_condition,
                )
            image = final["sample"]
        if final is None:
            raise RuntimeError("Projected DDIM 没有执行任何时间步。")
        return final

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        A bit different from ddim inverse scheduler in diffuser.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
        self,
        model,
        img,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        dump_steps=None,
    ):
        """
        Inverse noise latent from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_reverse_sample_loop_progressive(
            model,
            img,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
        ):
            final = sample
        return final["sample"]

    def ddim_reverse_sample_loop_progressive(
        self,
        model,
        img,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
    ):
        """
        Use DDIM inversion to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device

        indices = list(range(self.num_timesteps - skip_timesteps))

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * img.shape[0], device=device)
            with th.no_grad():
                sample_fn = self.ddim_reverse_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        if dump_steps is not None:
            raise NotImplementedError()
        if const_noise == True:
            raise NotImplementedError()

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def plms_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cond_fn_with_grad=False,
        order=2,
        old_out=None,
    ):
        """
        Sample x_{t-1} from the model using Pseudo Linear Multistep.

        Same usage as p_sample().
        """
        if not int(order) or not 1 <= order <= 4:
            raise ValueError('order is invalid (should be int from 1-4).')

        def get_model_output(x, t):
            with th.set_grad_enabled(cond_fn_with_grad and cond_fn is not None):
                x = x.detach().requires_grad_() if cond_fn_with_grad else x
                out_orig = self.p_mean_variance(
                    model,
                    x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                if cond_fn is not None:
                    if cond_fn_with_grad:
                        out = self.condition_score_with_grad(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                        x = x.detach()
                    else:
                        out = self.condition_score(cond_fn, out_orig, x, t, model_kwargs=model_kwargs)
                else:
                    out = out_orig

            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            return eps, out, out_orig

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        eps, out, out_orig = get_model_output(x, t)

        if order > 1 and old_out is None:
            # Pseudo Improved Euler
            old_eps = [eps]
            mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps
            eps_2, _, _ = get_model_output(mean_pred, t - 1)
            eps_prime = (eps + eps_2) / 2
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime
        else:
            # Pseudo Linear Multistep (Adams-Bashforth)
            old_eps = old_out["old_eps"]
            old_eps.append(eps)
            cur_order = min(order, len(old_eps))
            if cur_order == 1:
                eps_prime = old_eps[-1]
            elif cur_order == 2:
                eps_prime = (3 * old_eps[-1] - old_eps[-2]) / 2
            elif cur_order == 3:
                eps_prime = (23 * old_eps[-1] - 16 * old_eps[-2] + 5 * old_eps[-3]) / 12
            elif cur_order == 4:
                eps_prime = (55 * old_eps[-1] - 59 * old_eps[-2] + 37 * old_eps[-3] - 9 * old_eps[-4]) / 24
            else:
                raise RuntimeError('cur_order is invalid.')
            pred_prime = self._predict_xstart_from_eps(x, t, eps_prime)
            mean_pred = pred_prime * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev) * eps_prime

        if len(old_eps) >= order:
            old_eps.pop(0)

        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred * nonzero_mask + out["pred_xstart"] * (1 - nonzero_mask)

        return {"sample": sample, "pred_xstart": out_orig["pred_xstart"], "old_eps": old_eps}

    def plms_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Generate samples from the model using Pseudo Linear Multistep.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.plms_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
            order=order,
        ):
            final = sample
        return final["sample"]

    def plms_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        randomize_class=False,
        cond_fn_with_grad=False,
        order=2,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from each
        timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_out = None

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if randomize_class and 'y' in model_kwargs:
                model_kwargs['y'] = th.randint(low=0, high=model.num_classes,
                                               size=model_kwargs['y'].shape,
                                               device=model_kwargs['y'].device)
            with th.no_grad():
                out = self.plms_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cond_fn_with_grad=cond_fn_with_grad,
                    order=order,
                    old_out=old_out,
                )
                yield out
                old_out = out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def my_compd_q_sample(self, x_start, t, noise=None):
        x_t = self.q_sample(x_start, t, noise=noise)
        negnoise_level_mask = (t < 0).float()
        while len(negnoise_level_mask.shape) < len(x_start.shape):
            negnoise_level_mask = negnoise_level_mask[:, None]
        x_t = x_start * negnoise_level_mask + x_t * (1 - negnoise_level_mask)
        return x_t

    def training_losses(
        self,
        model,
        x_start,
        t,
        model_kwargs=None,
        noise=None,
        inpaint_condition=None,
        feature_w=None,
        snr_gamma=0.0,
        use_l1=False,
        return_pred_xstart=False,
    ):
        """计算当前 144D RealtimePose 路径的单时间步训练损失。"""

        if model_kwargs is None:
            raise ValueError("RealtimePose 训练必须提供历史、路径与监督字段。")
        return self._realtime_pose_training_losses(
            model=model,
            x_start=x_start,
            t=t,
            model_kwargs=model_kwargs,
            noise=noise,
            inpaint_condition=inpaint_condition,
            feature_w=feature_w,
            snr_gamma=snr_gamma,
            use_l1=use_l1,
            return_pred_xstart=return_pred_xstart,
        )

    def _realtime_pose_training_losses(
        self,
        model,
        x_start,
        t,
        model_kwargs,
        noise=None,
        inpaint_condition=None,
        feature_w=None,
        snr_gamma=0.0,
        use_l1=False,
        return_pred_xstart=False,
    ):
        """只对当前 144D Pose 加噪，并显式分离 raw/deployed x0 路径。"""

        if x_start.ndim != 2 or x_start.shape[1] != REALTIME_POSE_TARGET_DIM:
            raise ValueError("RealtimePose diffusion target 必须为 [B,144]。")
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        if inpaint_condition is None:
            x_model = x_t
        else:
            validate_realtime_pose_inpainting_condition(inpaint_condition)
            x_model, _ = apply_realtime_pose_inpainting(
                x_t=x_t,
                t=t,
                condition=inpaint_condition,
                alphas_cumprod=self.alphas_cumprod,
            )
        batch = model_kwargs.get("y", model_kwargs)
        call_kwargs = dict(model_kwargs)
        call_kwargs["return_aux_outputs"] = True
        model_result = model(x_model, self._scale_timesteps(t), **call_kwargs)
        if not isinstance(model_result, tuple) or len(model_result) != 2:
            raise TypeError("RealtimePose 模型训练时必须返回 (raw_output, auxiliary_outputs)。")
        model_output, auxiliary_outputs = model_result
        target = (
            x_start
            if self.model_mean_type == ModelMeanType.START_X
            else self._predict_eps_from_xstart(x_model, t, x_start)
        )
        if model_output.shape != target.shape:
            raise ValueError("模型输出、diffusion target 和 x_start 必须同形。")
        raw_pred_xstart = (
            model_output
            if self.model_mean_type == ModelMeanType.START_X
            else self._predict_xstart_from_eps(x_t=x_model, t=t, eps=model_output)
        )
        deployed_pred_xstart = project_realtime_pose_xstart(
            pred_xstart=raw_pred_xstart,
            current_tracker_raw=batch["current_tracker_raw"],
            pose_mean=batch.get("pose_mean"),
            pose_scale=batch.get("pose_scale"),
        )
        # 单帧分支遵守公共训练 CLI：L1/MSE 只切换 diffusion
        # reconstruction term，feature_w 按 [B,144] 对特征维加权。
        elementwise_loss = (
            (target - model_output).abs()
            if bool(use_l1)
            else (target - model_output).square()
        )
        if feature_w is not None:
            feature_weight = feature_w.to(
                device=elementwise_loss.device,
                dtype=elementwise_loss.dtype,
            )
            if feature_weight.ndim == 1:
                feature_weight = feature_weight[None]
            if (
                feature_weight.shape[-1] != elementwise_loss.shape[-1]
                or feature_weight.shape[0] not in (1, elementwise_loss.shape[0])
            ):
                raise ValueError(
                    "动态 RealtimePose feature_w 必须为 [144] 或 [B,144]，"
                    f"实际为 {tuple(feature_weight.shape)}"
                )
            elementwise_loss = elementwise_loss * feature_weight
        simple_loss = elementwise_loss.flatten(1).mean(dim=1)
        if snr_gamma:
            snr = compute_snr(self, t)
            weight = torch.minimum(snr, torch.full_like(snr, float(snr_gamma)))
            if self.model_mean_type == ModelMeanType.EPSILON:
                weight = weight / snr.clamp_min(1e-8)
            simple_loss = simple_loss * weight
        terms = {"simple_loss": simple_loss}
        terms.update(
            compute_raw_deployed_losses(
                raw_pred_xstart,
                deployed_pred_xstart,
                x_start,
                batch,
                auxiliary_outputs,
                tracker_pos_huber_beta=self.tracker_pos_huber_beta,
            )
        )
        auxiliary_loss = (
            self.rotation_loss_weight * terms["global_rotation_loss"]
            + self.local_rot_loss_weight * terms["local_rotation_loss"]
            + self.tracker_rot_loss_weight * terms["tracker_rotation_loss"]
            + self.fk_loss_weight * terms["fk_loss"]
            + self.tracker_pos_loss_weight * terms["tracker_position_loss"]
            + self.root_loss_weight * terms["root_loss"]
            + self.head_ref_joint_distance_loss_weight
            * terms["head_ref_joint_distance_loss"]
            + self.head_to_root_xz_loss_weight * terms["head_to_root_xz_loss"]
            + self.hip_height_loss_weight * terms["hip_height_loss"]
            + self.rotation_velocity_loss_weight
            * terms["rotation_velocity_loss"]
            + self.contact_loss_weight * terms["contact_loss"]
            + self.contact_slide_loss_weight * terms["contact_slide_loss"]
        )
        terms["aux_loss"] = auxiliary_loss
        terms["loss"] = self.diffusion_loss_weight * simple_loss + self.aux_loss_weight * auxiliary_loss
        if return_pred_xstart:
            terms["raw_pred_xstart"] = raw_pred_xstart
            terms["deployed_pred_xstart"] = deployed_pred_xstart
            terms["pred_xstart"] = deployed_pred_xstart
        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        # res = res[..., None]
        res = res[:, None]
    return res.expand(broadcast_shape)
