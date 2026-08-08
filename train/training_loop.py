from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from data_loaders.realtime_pose_geometry import advance_rollout_pose_history_torch
from data_loaders.realtime_pose_config import TAID_ABLATIONS

from data_loaders.sensor_masking import (
    REALTIME_POSE_HISTORY_LENGTH,
    REALTIME_POSE_MAX_ROLLOUT_STEPS,
    REALTIME_POSE_TARGET_DIM,
    TASK_MODE_REALTIME_POSE,
)
from diffusion import logger
from diffusion.realtime_pose_temporal_losses import compute_rollout_temporal_losses
from utils import dist_util


ROLLOUT_PRIOR_DIAGNOSTIC_NAMES = (
    "prior_fk_loss",
    "prior_internal_fk_loss",
    "prior_root_pose_gap_m",
    "prior_root_pose_gap_xz_m",
    "prior_joint_resolver_gap_m",
)
ROLLOUT_FRAME_WEIGHTINGS = ("uniform", "linear_late")


def build_rollout_frame_weights(
    rollout_steps: int,
    weighting: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """构造第 2～K 帧的归一化权重，第一帧主损失不包含在这里。"""

    future_steps = int(rollout_steps) - 1
    if future_steps < 1:
        raise ValueError("rollout frame weighting 至少需要两个预测帧。")
    if weighting not in ROLLOUT_FRAME_WEIGHTINGS:
        raise ValueError(
            f"rollout_frame_weighting 必须属于 {ROLLOUT_FRAME_WEIGHTINGS}，实际为 {weighting!r}。"
        )
    if weighting == "uniform":
        weights = torch.ones(future_steps, device=device, dtype=dtype)
    else:
        weights = torch.arange(1, future_steps + 1, device=device, dtype=dtype)
    return weights / weights.sum()


def aggregate_rollout_frame_losses(
    rollout_terms: list[dict[str, torch.Tensor]],
    weighting: str,
) -> dict[str, torch.Tensor]:
    """只聚合逐帧主 loss；时序 loss 和其他诊断仍走原来的独立路径。"""

    if not rollout_terms:
        raise ValueError("rollout frame loss 聚合至少需要一个后续预测。")
    step_losses = [terms["loss"] for terms in rollout_terms]
    first_shape = step_losses[0].shape
    if any(loss.shape != first_shape for loss in step_losses):
        raise ValueError("rollout 各后续帧 loss 形状必须一致。")
    stacked = torch.stack(step_losses, dim=0)
    weights = build_rollout_frame_weights(
        len(rollout_terms) + 1,
        weighting,
        device=stacked.device,
        dtype=stacked.dtype,
    )
    broadcast_shape = (len(weights),) + (1,) * (stacked.ndim - 1)
    uniform_frame_loss = stacked.mean(dim=0)
    weighted_frame_loss = (
        uniform_frame_loss
        if weighting == "uniform"
        else (stacked * weights.reshape(broadcast_shape)).sum(dim=0)
    )
    result = {
        "loss": weighted_frame_loss,
        "uniform_frame_loss": uniform_frame_loss,
    }
    for step_index, (step_loss, weight) in enumerate(zip(step_losses, weights), start=1):
        result[f"step_{step_index}_loss"] = step_loss
        result[f"step_{step_index}_weight"] = torch.ones_like(step_loss) * weight
        result[f"step_{step_index}_weighted_loss"] = step_loss * weight
    return result


def add_rollout_step_diagnostics(
    result: dict[str, torch.Tensor],
    terms: dict[str, torch.Tensor],
    step_index: int,
) -> None:
    """把第 2～15 帧的 Prior/FK 诊断写入稳定的 rollout 日志键。"""

    for diagnostic_name in ROLLOUT_PRIOR_DIAGNOSTIC_NAMES:
        if diagnostic_name in terms:
            result[f"step_{step_index}_{diagnostic_name}"] = terms[diagnostic_name]


def validate_taid_checkpoint_stage(
    model,
    state_dict: dict[str, torch.Tensor],
    *,
    allow_stage_transition: bool,
) -> tuple[str, str]:
    """在加载参数前核对 TAID 阶段，防止同构 B1～B6 静默串用配置。"""

    model_impl = getattr(model, "module", model)
    target = getattr(getattr(model_impl, "taid_config", None), "ablation", "B0")
    code = state_dict.get("taid_conditioner.ablation_code")
    if code is None:
        source = "B0"
    else:
        source_index = int(code.item())
        if not 0 <= source_index < len(TAID_ABLATIONS):
            raise RuntimeError(f"checkpoint TAID ablation code 非法：{source_index}。")
        source = TAID_ABLATIONS[source_index]
    if source != "B0" and target != "B0":
        target_state = model_impl.state_dict()
        contract_keys = (
            "taid_conditioner.config_contract",
            "taid_conditioner.position_scales",
            "taid_conditioner.rotation_scales",
        )
        mismatched = []
        for key in contract_keys:
            source_value = state_dict.get(key)
            target_value = target_state.get(key)
            if (
                source_value is None
                or target_value is None
                or source_value.shape != target_value.shape
                or not torch.allclose(
                    source_value.detach().cpu().to(torch.float64),
                    target_value.detach().cpu().to(torch.float64),
                    atol=1e-12,
                    rtol=0.0,
                )
            ):
                mismatched.append(key)
        if mismatched:
            raise RuntimeError(
                "TAID checkpoint 配置与当前模型不兼容："
                f" mismatched={mismatched}"
            )
    if source == target:
        return source, target
    allowed = allow_stage_transition and (
        (source == "B0" and target == "B1")
        or (source == "B1" and target in {"B2", "B3", "B4", "B5", "B6"})
    )
    if not allowed:
        mode = "init" if allow_stage_transition else "resume"
        raise RuntimeError(
            f"TAID {mode} 阶段不兼容：checkpoint={source}, current={target}。"
        )
    if code is not None:
        # B1→B2+ 复用同构参数，但 checkpoint 中的阶段标记必须改写为当前配置，
        # 否则下一次保存会把新实验错误标成 B1。
        state_dict["taid_conditioner.ablation_code"] = code.new_tensor(
            TAID_ABLATIONS.index(target)
        )
    return source, target


def zero_invalid_pose_history(
    pose_history: torch.Tensor,
    valid_frame_mask: torch.Tensor,
) -> torch.Tensor:
    """把 `[B,60,144]` 中无效历史恢复为模型输入空间的字面量零。"""

    if pose_history.ndim != 3 or pose_history.shape[1:] != (
        REALTIME_POSE_HISTORY_LENGTH,
        REALTIME_POSE_TARGET_DIM,
    ):
        raise ValueError("pose_history 必须为 [B,60,144]。")
    if tuple(valid_frame_mask.shape) != tuple(pose_history.shape[:2]):
        raise ValueError("valid_frame_mask 必须与 pose_history 的前两维一致。")
    return pose_history.masked_fill(~valid_frame_mask.bool()[..., None], 0.0)


class TrainLoop:
    """root-y0 realtime_pose 扩散训练循环。"""

    def __init__(self, args, train_platform, model, diffusion, data, eval_data=None):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data

        self.batch_size = args.batch_size
        self.microbatch = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.init_checkpoint = getattr(args, "init_checkpoint", "")
        if self.resume_checkpoint and self.init_checkpoint:
            raise ValueError("resume_checkpoint 与 init_checkpoint 不能同时设置。")
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.gradient_clip = args.gradient_clip
        self.snr_gamma = args.snr_gamma
        self.use_l1 = args.l1_loss
        self.task_mode = getattr(args, "task_mode", TASK_MODE_REALTIME_POSE)
        self.checkpoint_max_keep = max(0, int(args.checkpoint_max_keep))
        self.rollout_steps = int(getattr(args, "rollout_steps", 1))
        self.rollout_loss_weight = float(getattr(args, "rollout_loss_weight", 0.0))
        self.rollout_frame_weighting = str(
            getattr(args, "rollout_frame_weighting", "uniform")
        )
        self.rollout_prob = float(getattr(args, "rollout_prob", 0.0))
        self.detach_rollout_history = bool(getattr(args, "detach_rollout_history", True))
        self.rollout_joint_vel_loss_weight = float(
            getattr(args, "rollout_joint_vel_loss_weight", 0.05)
        )
        self.rollout_rot_vel_loss_weight = float(
            getattr(args, "rollout_rot_vel_loss_weight", 0.02)
        )
        if self.rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {self.rollout_steps}")
        if self.rollout_steps > REALTIME_POSE_MAX_ROLLOUT_STEPS:
            raise ValueError(
                "rollout_steps 超过离线 task 可物化上限："
                f"{REALTIME_POSE_MAX_ROLLOUT_STEPS}。"
            )
        if self.rollout_loss_weight < 0.0:
            raise ValueError(f"rollout_loss_weight must be >= 0, got {self.rollout_loss_weight}")
        if self.rollout_frame_weighting not in ROLLOUT_FRAME_WEIGHTINGS:
            raise ValueError(
                "rollout_frame_weighting 必须属于 "
                f"{ROLLOUT_FRAME_WEIGHTINGS}，实际为 {self.rollout_frame_weighting!r}。"
            )
        if self.rollout_joint_vel_loss_weight < 0.0:
            raise ValueError("rollout_joint_vel_loss_weight 必须大于等于 0。")
        if self.rollout_rot_vel_loss_weight < 0.0:
            raise ValueError("rollout_rot_vel_loss_weight 必须大于等于 0。")
        if not 0.0 <= self.rollout_prob <= 1.0:
            raise ValueError(f"rollout_prob must be in [0, 1], got {self.rollout_prob}")

        self.save_dir = Path(args.save_dir)
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.data_num_batches = len(self.data)
        self.data_wait_times: list[float] = []
        if self.data_num_batches <= 0:
            raise RuntimeError(
                "训练 DataLoader 没有可用 batch；请降低 --batch_size 或增加训练样本，"
                "当前 train split 会 drop_last。"
            )
        self.eval_num_batches = max(0, int(getattr(args, "eval_num_batches", 0)))
        if getattr(args, "eval_during_training", False):
            if self.eval_data is None:
                raise RuntimeError("开启 --eval_during_training 时必须提供 eval DataLoader。")
            if len(self.eval_data) <= 0:
                raise RuntimeError("eval DataLoader 没有可用 batch；请检查 --eval_split 或降低 --batch_size。")
        self.num_epochs = self.num_steps // self.data_num_batches + 1
        self.device = dist_util.dev()

        logger.log(f"training device: {self.device}")
        logger.log(f"task mode: {self.task_mode}")
        logger.log(
            f"rollout: steps={self.rollout_steps}, prob={self.rollout_prob}, "
            f"weight={self.rollout_loss_weight}, frame_weighting={self.rollout_frame_weighting}, "
            f"detach_history={self.detach_rollout_history}, "
            f"joint_vel_weight={self.rollout_joint_vel_loss_weight}, "
            f"rot_vel_weight={self.rollout_rot_vel_loss_weight}"
        )
        if self.task_mode != TASK_MODE_REALTIME_POSE:
            raise ValueError(f"当前训练链路只支持 {TASK_MODE_REALTIME_POSE}，实际为 {self.task_mode}")
        if self.device.type == "cuda":
            logger.log(f"cuda device name: {torch.cuda.get_device_name(self.device)}")

        self.feature_w = self._load_feature_weights(args)
        (
            self.normalizer_mean,
            self.normalizer_std,
            self.tracker_normalizer_mean,
            self.tracker_normalizer_std,
        ) = self._read_dataset_normalizer_stats(data)
        self._load_and_sync_parameters()

        self.amp_dtype = self._select_amp_dtype()
        self.scaler = GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and self.amp_dtype == torch.float16,
            init_scale=1024.0,
        )
        if self.device.type == "cuda":
            logger.log(f"amp dtype: {self.amp_dtype}, grad scaler enabled: {self.scaler.is_enabled()}")
        self.opt = AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.resume_step:
            self._load_optimizer_state()

        self.ema_model = self._create_ema_model(args)
        if self.resume_step and self.ema_model is not None:
            self._load_ema_state()

    # region 初始化与恢复
    def _select_amp_dtype(self) -> torch.dtype:
        """CUDA 上优先使用 bf16，避免 realtime_pose 几何辅助 loss 在 fp16 下溢出。"""

        if self.device.type != "cuda":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _load_feature_weights(self, args):
        if not args.weighted_loss:
            return None

        feature_w_path = Path(args.normalizer_dir) / args.feature_w_file
        if not feature_w_path.exists():
            raise FileNotFoundError(f"开启 --weighted_loss 后找不到特征权重文件：{feature_w_path}")

        feature_w = torch.load(feature_w_path, map_location="cpu", weights_only=True).float().flatten()
        if feature_w.numel() != REALTIME_POSE_TARGET_DIM:
            raise ValueError(f"feature_w 应为 {REALTIME_POSE_TARGET_DIM} 维，实际为 {feature_w.numel()} 维")
        feature_w.requires_grad_(False)
        return feature_w

    @staticmethod
    def _read_dataset_normalizer_stats(data):
        dataset = getattr(data, "dataset", None)
        normalizer = getattr(dataset, "normalizer", None)
        if normalizer is None or getattr(normalizer, "disable", False):
            return None, None, None, None
        mean = getattr(normalizer, "mean", None)
        std = getattr(normalizer, "std", None)
        if mean is None or std is None:
            return None, None, None, None
        tracker_mean = getattr(normalizer, "tracker_mean", None)
        tracker_std = getattr(normalizer, "tracker_std", None)
        return (
            mean.detach().float().clone(),
            std.detach().float().clone(),
            None if tracker_mean is None else tracker_mean.detach().float().clone(),
            None if tracker_std is None else tracker_std.detach().float().clone(),
        )

    def _load_and_sync_parameters(self):
        if self.init_checkpoint:
            logger.log(f"initializing model weights from checkpoint: {self.init_checkpoint}...")
            state_dict = dist_util.load_state_dict(self.init_checkpoint, map_location=self.device)
            source_stage, target_stage = validate_taid_checkpoint_stage(
                self.model, state_dict, allow_stage_transition=True
            )
            incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
            missing_keys = list(incompatible_keys.missing_keys)
            unexpected_keys = list(incompatible_keys.unexpected_keys)
            allowed_missing = source_stage == "B0" and target_stage == "B1" and all(
                key.startswith("taid_conditioner.") for key in missing_keys
            )
            if unexpected_keys or (missing_keys and not allowed_missing):
                raise RuntimeError(
                    "init checkpoint 与当前模型不兼容："
                    f" missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
                )
            if allowed_missing:
                logger.log("B0 checkpoint 未含 TAID 参数；保留新 TAID 模块初始化值。")
            return
        resume_checkpoint = find_resume_checkpoint(
            save_dir=self.save_dir,
            requested_checkpoint=self.resume_checkpoint,
        )
        if not resume_checkpoint:
            return

        self.resume_checkpoint = resume_checkpoint
        self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
        logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
        state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=self.device)
        validate_taid_checkpoint_stage(self.model, state_dict, allow_stage_transition=False)
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        missing_keys = list(incompatible_keys.missing_keys)
        unexpected_keys = list(incompatible_keys.unexpected_keys)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "checkpoint 与当前 root-y0 realtime_pose 模型结构不匹配，已停止恢复。"
                f" missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
            )

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = Path(main_checkpoint).with_name(f"opt{self.resume_step:09d}.pt")
        if not opt_checkpoint.exists():
            logger.log(f"optimizer checkpoint not found, skip: {opt_checkpoint}")
            return

        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=self.device)
        self.opt.load_state_dict(state_dict)

    def _create_ema_model(self, args):
        if not args.model_ema:
            return None

        from ema_pytorch import EMA

        return EMA(
            self.model,
            beta=args.model_ema_decay,
            update_every=args.model_ema_steps,
            update_after_step=args.model_ema_update_after,
            include_online_model=False,
        )

    def _load_ema_state(self):
        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = Path(main_checkpoint).with_name(f"ema{self.resume_step:09d}.pt")
        if not ema_checkpoint.exists():
            logger.log(f"ema checkpoint not found, skip: {ema_checkpoint}")
            return

        logger.log(f"loading ema state from checkpoint: {ema_checkpoint}")
        state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=self.device)
        self.ema_model.load_state_dict(state_dict)

    # endregion

    # region 训练循环
    def run_loop(self):
        self.model.train()
        if not hasattr(self, "data_wait_times"):
            self.data_wait_times = []
        if self._should_stop():
            logger.log(f"resume step {self.resume_step} has already reached num_steps={self.num_steps}; skip training.")
            return

        last_step_end = time.perf_counter()
        for epoch in range(self.num_epochs):
            batch_sampler = getattr(self.data, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            for batch in self.data:
                batch_ready_time = time.perf_counter()
                data_wait = batch_ready_time - last_step_end
                if len(self.data_wait_times) < 100:
                    self.data_wait_times.append(data_wait)
                    if len(self.data_wait_times) == 100:
                        logger.log(
                            "first 100 batch data wait: "
                            f"mean={sum(self.data_wait_times) / 100.0:.4f}s, "
                            f"max={max(self.data_wait_times):.4f}s"
                        )
                batch = move_batch_to_device(batch, self.device)
                train_start_time = time.perf_counter()
                self.run_step(batch)
                self.step += 1
                train_end_time = time.perf_counter()
                global_step = self.step + self.resume_step
                print(
                    f"step[{global_step}] "
                    f"data={data_wait:.3f}s "
                    f"train={train_end_time - train_start_time:.3f}s",
                    flush=True,
                )
                last_step_end = train_end_time

                self.log_step()
                if self.log_interval > 0 and global_step % self.log_interval == 0:
                    self.report_metrics()

                if self.save_interval > 0 and global_step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    if os.environ.get("DIFFUSION_TRAINING_TEST", ""):
                        return

                if self._should_stop():
                    break

            if self._should_stop():
                break

        global_step = self.step + self.resume_step
        if self.step > 0 and (self.save_interval <= 0 or global_step % self.save_interval != 0):
            self.save()
            self.evaluate()

    def _should_stop(self) -> bool:
        global_step = self.step + self.resume_step
        return global_step >= self.num_steps

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        if self.eval_data is None:
            raise RuntimeError("开启 --eval_during_training 时必须提供 eval DataLoader。")

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        totals: dict[str, torch.Tensor] = {}
        count = 0
        max_batches = self.eval_num_batches if self.eval_num_batches > 0 else len(self.eval_data)

        with torch.no_grad():
            for batch_index, batch in enumerate(self.eval_data):
                if batch_index >= max_batches:
                    break
                batch = move_batch_to_device(batch, self.device)
                sample = batch["x"]
                batch_size = sample.shape[0]
                # eval 使用固定时间步轮转，避免验证 loss 随随机 timestep 抖动太大。
                timesteps = (torch.arange(batch_size, device=self.device) + batch_index) % self.diffusion.num_timesteps
                with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
                    losses = self.compute_losses(batch=batch, timesteps=timesteps)
                    loss = losses["loss"].mean()
                validate_finite_losses(losses=losses, loss=loss, batch=batch)
                for key, values in losses.items():
                    if torch.is_tensor(values):
                        totals[key] = totals.get(key, torch.zeros((), device=self.device)) + values.detach().float().mean()
                count += 1

        if count <= 0:
            raise RuntimeError("eval DataLoader 没有产出 batch，无法计算训练期评估指标。")

        global_step = self.step + self.resume_step
        for key, total in totals.items():
            value = float((total / count).item())
            metric_name = f"eval/{key}"
            logger.logkv_mean(metric_name, value)
            self.train_platform.report_scalar(
                name=metric_name,
                value=value,
                iteration=global_step,
                group_name="Eval",
            )
            if key == "loss":
                print(f"step[{global_step}]: eval/loss[{value:0.5f}]")

        if was_training:
            self.model.train()

    def run_step(self, batch):
        self.forward_backward(batch)

        # 先恢复真实梯度，再统一检查 NaN/Inf；关闭裁剪时仍需在 optimizer step 前失败。
        self.scaler.unscale_(self.opt)
        max_grad_norm = 1.0 if self.gradient_clip else float("inf")
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        )

        self._anneal_lr()
        self.scaler.step(self.opt)
        self.scaler.update()

        if self.ema_model is not None:
            self.ema_model.update()

    def forward_backward(self, batch):
        self.opt.zero_grad(set_to_none=True)

        with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            for i in range(0, batch["x"].shape[0], self.microbatch):
                # 当前保持整 batch 训练，后续如果 batch 很大再引入 microbatch 梯度累积。
                assert i == 0
                assert self.microbatch == self.batch_size

                timesteps = torch.randint(
                    low=0,
                    high=self.diffusion.num_timesteps,
                    size=(batch["x"].shape[0],),
                    device=self.device,
                )
                losses = self.compute_losses(batch=batch, timesteps=timesteps)
                loss = losses["loss"].mean()
                validate_finite_losses(losses=losses, loss=loss, batch=batch)
                log_loss_dict(self.diffusion, timesteps, losses)
                self.scaler.scale(loss).backward()

    def compute_losses(self, batch: dict, timesteps: torch.Tensor) -> dict:
        sample = batch["x"]  # [B,144]
        if sample.ndim != 2 or sample.shape[1] != REALTIME_POSE_TARGET_DIM:
            raise ValueError(f"训练输入应为 [B,{REALTIME_POSE_TARGET_DIM}]，实际为 {tuple(sample.shape)}")
        batch_size = sample.shape[0]
        feature_w = self._feature_weights_for_batch(batch_size)
        do_rollout = self.should_compute_rollout_loss(batch)
        model_kwargs = self.mask_manager(batch, sample)
        losses = self.diffusion.training_losses(
            self.model,
            sample,
            timesteps,
            model_kwargs=model_kwargs,
            feature_w=feature_w,
            snr_gamma=self.snr_gamma,
            use_l1=self.use_l1,
            return_pred_xstart=do_rollout,
        )
        pred_xstart = losses.pop("pred_xstart", None)
        losses.pop("raw_pred_xstart", None)
        losses.pop("deployed_pred_xstart", None)
        if not do_rollout:
            return losses

        rollout_losses = self.compute_rollout_losses(
            batch=batch,
            pred_xstart=pred_xstart,
            timesteps=timesteps,
        )
        base_loss = losses["loss"]
        rollout_frame_loss = rollout_losses.pop("loss")
        joint_vel_loss = rollout_losses.pop("joint_vel_loss")
        rotation_vel_loss = rollout_losses.pop("rotation_vel_loss")
        foot_slide_loss = rollout_losses.pop("foot_slide_loss")
        rollout_loss_weighted = rollout_frame_loss * self.rollout_loss_weight
        joint_vel_loss_weighted = joint_vel_loss * self.rollout_joint_vel_loss_weight
        rotation_vel_loss_weighted = rotation_vel_loss * self.rollout_rot_vel_loss_weight
        foot_slide_loss_weighted = foot_slide_loss * float(self.diffusion.foot_slide_loss_weight)
        losses["base_loss"] = base_loss
        losses["rollout_loss"] = rollout_frame_loss
        losses["rollout_loss_weighted"] = rollout_loss_weighted
        losses["rollout_joint_vel_loss"] = joint_vel_loss
        losses["rollout_joint_vel_loss_weighted"] = joint_vel_loss_weighted
        losses["rollout_rotation_vel_loss"] = rotation_vel_loss
        losses["rollout_rotation_vel_loss_weighted"] = rotation_vel_loss_weighted
        losses["rollout_foot_slide_loss"] = foot_slide_loss
        losses["rollout_foot_slide_loss_weighted"] = foot_slide_loss_weighted
        losses["loss"] = (
            base_loss
            + rollout_loss_weighted
            + joint_vel_loss_weighted
            + rotation_vel_loss_weighted
            + foot_slide_loss_weighted
        )
        for key, value in rollout_losses.items():
            losses[f"rollout_{key}"] = value
        return losses

    def should_compute_rollout_loss(self, batch: dict) -> bool:
        rollout_weight_enabled = (
            self.rollout_loss_weight > 0.0
            or self.rollout_joint_vel_loss_weight > 0.0
            or self.rollout_rot_vel_loss_weight > 0.0
            or float(self.diffusion.foot_slide_loss_weight) > 0.0
        )
        if self.rollout_steps <= 1 or not rollout_weight_enabled:
            return False
        if not self.model.training or not torch.is_grad_enabled():
            return False
        rollout = batch.get("rollout")
        if rollout is None:
            return False
        if not isinstance(rollout, (list, tuple)) or len(rollout) < self.rollout_steps - 1:
            raise ValueError(f"batch rollout 不足以提供 rollout_steps={self.rollout_steps}。")
        return True

    def compute_rollout_losses(
        self,
        batch: dict,
        pred_xstart: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict:
        if pred_xstart is None:
            raise ValueError("rollout loss 需要第一步 training_losses 返回 pred_xstart。")

        predictions = [pred_xstart]
        step_batches = [batch]
        rollout_terms: list[dict] = []
        pose_history = batch["pose_history"]
        current_batch = batch

        for rollout_index, materialized_batch in enumerate(
            batch["rollout"][: self.rollout_steps - 1],
            start=1,
        ):
            next_batch = dict(materialized_batch)
            next_sample = next_batch["x"]
            if next_sample.shape != batch["x"].shape:
                raise ValueError(
                    f"rollout[{rollout_index - 1}]['x'] 应为 {tuple(batch['x'].shape)}，"
                    f"实际为 {tuple(next_sample.shape)}"
                )

            # 持续滚动同一份历史，防止第三步以后较早的模型预测被离线 GT 历史重新覆盖。
            pose_history = self.build_next_rollout_pose_history(
                pose_history=pose_history,
                source_head_yaw_world=current_batch["history_head_yaw_world"],
                destination_head_yaw_world=next_batch["history_head_yaw_world"],
                pred_xstart=predictions[-1],
            )
            # reference 变换会改变归一化 padding 的数值；下一步送入模型前必须再次严格清零。
            pose_history = zero_invalid_pose_history(
                pose_history,
                next_batch["valid_frame_mask"],
            )
            next_batch["pose_history"] = pose_history

            feature_w = self._feature_weights_for_batch(next_sample.shape[0])
            next_terms = self.diffusion.training_losses(
                self.model,
                next_sample,
                timesteps,
                model_kwargs=self.mask_manager(next_batch, next_sample),
                feature_w=feature_w,
                snr_gamma=self.snr_gamma,
                use_l1=self.use_l1,
                return_pred_xstart=True,
            )
            next_prediction = next_terms.pop("pred_xstart", None)
            next_terms.pop("raw_pred_xstart", None)
            next_terms.pop("deployed_pred_xstart", None)
            if next_prediction is None:
                raise ValueError(f"rollout step {rollout_index} 没有返回 pred_xstart。")
            predictions.append(next_prediction)
            step_batches.append(next_batch)
            rollout_terms.append(next_terms)
            current_batch = next_batch

        if not rollout_terms:
            raise ValueError("rollout_steps>1 时至少应产生一个后续预测。")

        result = aggregate_rollout_frame_losses(
            rollout_terms,
            getattr(self, "rollout_frame_weighting", "uniform"),
        )
        for step_index, terms in enumerate(rollout_terms, start=1):
            add_rollout_step_diagnostics(result, terms, step_index)

        common_keys = set.intersection(*(set(terms.keys()) for terms in rollout_terms))
        for key in sorted(common_keys - {"loss"}):
            values = [terms[key] for terms in rollout_terms]
            if all(torch.is_tensor(value) and value.shape == values[0].shape for value in values):
                result[key] = torch.stack(values, dim=0).mean(dim=0)

        result.update(
            compute_rollout_temporal_losses(
                predictions=predictions,
                step_batches=step_batches,
                normalizer_mean=self.normalizer_mean,
                normalizer_std=self.normalizer_std,
            )
        )
        return result

    def build_next_rollout_pose_history(
        self,
        pose_history: torch.Tensor,
        source_head_yaw_world: torch.Tensor,
        destination_head_yaw_world: torch.Tensor,
        pred_xstart: torch.Tensor,
    ) -> torch.Tensor:
        """把当前历史整体换到下一帧参考系，并追加当前模型预测。"""

        return advance_rollout_pose_history_torch(
            pose_history=pose_history,
            prediction=pred_xstart,
            source_head_yaw_world=source_head_yaw_world,
            destination_head_yaw_world=destination_head_yaw_world,
            normalizer_mean=self.normalizer_mean,
            normalizer_std=self.normalizer_std,
            detach_prediction=self.detach_rollout_history,
        )

    def _anneal_lr(self):
        if self.lr_anneal_steps <= 0:
            return
        global_step = self.step + self.resume_step
        frac_done = min(float(global_step) / float(self.lr_anneal_steps), 1.0)
        lr = self.lr * (1.0 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def _feature_weights_for_batch(self, batch_size: int):
        if self.feature_w is None:
            return None
        return self.feature_w.to(self.device)[None].repeat(batch_size, 1)

    def mask_manager(self, batch, sample):
        """构造新动态观测条件；扩散状态始终对完整 144D 加噪。"""

        batch_size = sample.shape[0]
        valid_frame_mask = batch.get("valid_frame_mask")
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch_size, 60, dtype=torch.bool, device=sample.device)
        valid_frame_mask = valid_frame_mask.bool()
        if valid_frame_mask.shape != (batch_size, 60):
            raise ValueError(f"valid_frame_mask 应为 [B,60]，实际为 {tuple(valid_frame_mask.shape)}")

        y = {
            "pose_history": batch["pose_history"],
            "tracker_history": batch["tracker_history"],
            "current_tracker": batch["current_tracker"],
            "current_tracker_raw": batch["current_tracker_raw"],
            "trajectory_history": batch["trajectory_history"],
            "current_trajectory": batch["current_trajectory"],
            "valid_frame_mask": valid_frame_mask,
            "hard_rotation_state": batch["hard_rotation_state"],
            "target_joints_head_ref": batch["target_joints_head_ref"],
            "prev_joints_head_ref": batch["prev_joints_head_ref"],
            "current_tracker_pos_head_ref": batch["current_tracker_pos_head_ref"],
            "joint_offsets_parent": batch["joint_offsets_parent"],
            "joint_rest_local_rotations_6d": batch["joint_rest_local_rotations_6d"],
            "configured": batch["configured"],
            "measured_valid": batch["measured_valid"],
            "d_off": batch["d_off"],
            "d_on": batch["d_on"],
            "target_root_position_head_ref": batch["target_root_position_head_ref"],
            "target_root_yaw_world": batch["target_root_yaw_world"],
            "current_head_yaw_world": batch["current_head_yaw_world"],
            "future_leg_target": batch["future_leg_target"],
            "contact_target": batch["contact_target"],
        }
        if self.normalizer_mean is not None and self.normalizer_std is not None:
            y["normalizer_mean"] = self.normalizer_mean.to(device=sample.device, dtype=sample.dtype)
            y["normalizer_std"] = self.normalizer_std.to(device=sample.device, dtype=sample.dtype)
        if self.tracker_normalizer_mean is not None and self.tracker_normalizer_std is not None:
            y["tracker_normalizer_mean"] = self.tracker_normalizer_mean.to(
                device=sample.device, dtype=sample.dtype
            )
            y["tracker_normalizer_std"] = self.tracker_normalizer_std.to(
                device=sample.device, dtype=sample.dtype
            )

        return {
            "pose_history": batch["pose_history"],
            "tracker_history": batch["tracker_history"],
            "current_tracker": batch["current_tracker"],
            "trajectory_history": batch["trajectory_history"],
            "current_trajectory": batch["current_trajectory"],
            "valid_frame_mask": valid_frame_mask,
            "y": y,
        }

    # endregion

    # region 日志与 checkpoint
    def log_step(self):
        global_step = self.step + self.resume_step
        logger.logkv("step", global_step)
        logger.logkv("samples", global_step * self.global_batch)
        if hasattr(self, "opt"):
            logger.logkv("lr", self.opt.param_groups[0]["lr"])

    def report_metrics(self):
        current = logger.get_current().name2val
        for key, value in current.items():
            if key in {"step", "samples"} or "_q" in key:
                continue
            self.train_platform.report_scalar(
                name=key,
                value=value,
                iteration=self.step + self.resume_step,
                group_name="Loss",
            )
            if key == "loss":
                print(f"step[{self.step + self.resume_step}]: loss[{value:0.5f}]")
        logger.dumpkvs()

    def ckpt_file_name(self):
        return f"model{self.step + self.resume_step:09d}.pt"

    def save(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        step = self.step + self.resume_step
        logger.log(f"saving checkpoint step {step}...")

        torch.save(self.model.state_dict(), self.save_dir / self.ckpt_file_name())
        torch.save(self.opt.state_dict(), self.save_dir / f"opt{step:09d}.pt")
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), self.save_dir / f"ema{step:09d}.pt")

        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        if self.checkpoint_max_keep <= 0 or not self.save_dir.exists():
            return

        pattern = re.compile(r"^(?:model|opt|ema)(\d{9})\.pt$")
        step_to_files: dict[int, list[Path]] = {}
        for path in self.save_dir.iterdir():
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if match is None:
                continue
            step_to_files.setdefault(int(match.group(1)), []).append(path)

        if len(step_to_files) <= self.checkpoint_max_keep:
            return

        sorted_steps = sorted(step_to_files.keys(), reverse=True)
        expired_steps = sorted_steps[self.checkpoint_max_keep :]
        removed_files = 0
        for step in expired_steps:
            for path in step_to_files[step]:
                try:
                    path.unlink()
                    removed_files += 1
                except OSError as exc:
                    logger.log(f"[ckpt-prune] remove failed: {path}, err={exc}")

        if removed_files:
            logger.log(
                f"[ckpt-prune] keep={self.checkpoint_max_keep}, "
                f"removed_steps={len(expired_steps)}, removed_files={removed_files}"
            )

    # endregion


def move_batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def parse_resume_step_from_filename(filename):
    match = re.search(r"model(\d+)\.pt$", str(filename))
    if match is None:
        return 0
    return int(match.group(1))


def find_resume_checkpoint(save_dir: str | Path, requested_checkpoint: str | Path | None = "") -> str:
    """解析恢复训练使用的主模型 checkpoint。"""

    requested_text = "" if requested_checkpoint is None else str(requested_checkpoint).strip()
    if not requested_text:
        return ""

    if requested_text.lower() in {"latest", "auto"}:
        latest_checkpoint = find_latest_model_checkpoint(save_dir)
        if latest_checkpoint is None:
            latest_run_dir = find_latest_run_dir(save_dir)
            if latest_run_dir is not None:
                latest_checkpoint = find_latest_model_checkpoint(latest_run_dir)
        if latest_checkpoint is None:
            raise FileNotFoundError(f"save_dir 中没有可恢复的 model*.pt checkpoint：{Path(save_dir)}")
        logger.log(f"auto resume from latest checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    requested_path = Path(requested_text).expanduser()
    if requested_path.exists():
        return str(requested_path)

    raise FileNotFoundError(f"--resume_checkpoint 指向的文件不存在：{requested_path}")


def find_latest_model_checkpoint(save_dir: str | Path) -> Path | None:
    """在实验目录中查找 step 最大的 `model*.pt`。"""

    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for path in save_dir.iterdir():
        if not path.is_file():
            continue
        match = re.fullmatch(r"model(\d+)\.pt", path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def find_latest_run_dir(save_dir: str | Path) -> Path | None:
    """读取 run 根目录里的 latest 指针，定位最近一次自动创建的 run 子目录。"""

    save_dir = Path(save_dir)
    json_path = save_dir / "latest_run.json"
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            latest_dir = Path(str(payload.get("save_dir", ""))).expanduser()
            if latest_dir.exists():
                return latest_dir
        except (OSError, json.JSONDecodeError):
            return None

    text_path = save_dir / "latest_run.txt"
    if text_path.exists():
        try:
            latest_dir = Path(text_path.read_text(encoding="utf-8").strip()).expanduser()
        except OSError:
            return None
        if latest_dir.exists():
            return latest_dir
    return None


def log_loss_dict(diffusion, timesteps, losses):
    for key, values in losses.items():
        if not torch.is_tensor(values):
            continue
        logger.logkv_mean(key, values.mean().item())
        for timestep, loss in zip(timesteps.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * timestep / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", float(loss))


def validate_finite_losses(losses: dict, loss: torch.Tensor, batch: dict) -> None:
    """训练中一旦出现 NaN/Inf 立即停止，避免继续写出不可用日志和 checkpoint。"""

    bad_terms = []
    for key, values in losses.items():
        if torch.is_tensor(values) and not torch.isfinite(values).all():
            bad_terms.append(key)
    if torch.isfinite(loss).all() and not bad_terms:
        return

    keyids = batch.get("keyid", [])
    if isinstance(keyids, (list, tuple)):
        keyid_preview = [str(value) for value in keyids[:3]]
    else:
        keyid_preview = [str(keyids)]
    raise FloatingPointError(
        "训练 loss 出现 NaN/Inf；"
        f"bad_terms={bad_terms or ['loss']}; "
        f"keyid_preview={keyid_preview}"
    )
