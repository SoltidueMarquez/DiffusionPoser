from __future__ import annotations

import json
import os
import re
import time
import copy
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    HIP_TRACKER_INDEX,
    POSE_REPRESENTATION_KEY,
    REALTIME_POSE_TARGET_START,
    TASK_MODE_REALTIME_POSE,
    get_schema_spec,
)
from data_loaders.realtime_pose_kinematics import fk_body_fbx_local_torch
from data_loaders.tracker_codec import (
    build_tracker_reference_np,
    decode_tracker_positions_np,
    decode_tracker_rotations_np,
    encode_tracker_positions_np,
    encode_tracker_rotations_np,
)
from diffusion import logger
from sample.runtime_root_resolver import (
    RuntimeRootResolver,
    RuntimeRootResolverState,
)
from train.realtime_rollout import (
    REALTIME_LR_DEFAULTS,
    REALTIME_ROLLOUT_DEFAULTS,
    rollout_curriculum_state_from_args,
    sampling_epoch_for_global_step,
    scheduled_learning_rate,
)
from utils import dist_util


CHECKPOINT_CONTRACT_KEYS = ("task_mode", "schema", "input_feats", "seq_len", "max_seq_len", "model_arch")
CHECKPOINT_INT_CONTRACT_KEYS = {"input_feats", "seq_len", "max_seq_len"}


def validate_realtime_pose_training_args(args):
    """校验 realtime_pose 训练入口和所选 schema 的固定契约。"""

    schema = get_schema_spec(getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME))

    input_feats = int(getattr(args, "input_feats", schema.feature_dim))
    if input_feats != schema.feature_dim:
        raise ValueError(f"{schema.name} 训练输入维度必须为 {schema.feature_dim}，实际为 {input_feats}。")

    seq_len = int(getattr(args, "seq_len", schema.seq_len))
    if seq_len != schema.seq_len:
        raise ValueError(f"{schema.name} 固定使用 seq_len={schema.seq_len}，实际为 {seq_len}。")

    max_seq_len = int(getattr(args, "max_seq_len", schema.seq_len))
    if max_seq_len != schema.seq_len:
        raise ValueError(f"{schema.name} 固定使用 max_seq_len={schema.seq_len}，实际为 {max_seq_len}。")

    return schema


def validate_root_y0_training_args(args):
    """旧导入名保留为别名，实际校验由通用 realtime_pose schema 入口负责。"""

    return validate_realtime_pose_training_args(args)


def validate_resume_checkpoint_contract(
    resume_checkpoint: str | Path,
    args,
    schema_name: str | None = None,
    *,
    require_schedule_signature: bool = True,
) -> None:
    """恢复训练前校验 checkpoint 的 exact schema 契约，防止权重被误加载。"""

    checkpoint_path = Path(resume_checkpoint)
    args_path = checkpoint_path.with_name("args.json")
    if not args_path.exists():
        raise FileNotFoundError(f"恢复训练需要 checkpoint 同目录存在 args.json，用于校验 schema 契约：{args_path}")

    with args_path.open("r", encoding="utf-8") as file:
        checkpoint_args = json.load(file)
    if not isinstance(checkpoint_args, dict):
        raise ValueError(f"{args_path} 必须是 JSON object。")

    schema = get_schema_spec(schema_name or getattr(args, "schema", DEFAULT_REALTIME_POSE_SCHEMA_NAME))
    expected_contract = {
        "task_mode": TASK_MODE_REALTIME_POSE,
        "schema": schema.name,
        "input_feats": schema.feature_dim,
        "seq_len": schema.seq_len,
        "max_seq_len": schema.seq_len,
        "model_arch": str(getattr(args, "model_arch", "target_dit")),
    }

    missing = [key for key in CHECKPOINT_CONTRACT_KEYS if key not in checkpoint_args]
    if missing:
        raise ValueError(f"{args_path} 缺少 checkpoint 训练契约字段：{missing}，不能恢复到 {schema.name} 训练。")

    mismatches: list[str] = []
    for key, expected in expected_contract.items():
        actual = checkpoint_args[key]
        if key in CHECKPOINT_INT_CONTRACT_KEYS:
            try:
                actual = int(actual)
            except (TypeError, ValueError):
                mismatches.append(f"{key}={checkpoint_args[key]!r}, expected {expected!r}")
                continue
        else:
            actual = str(actual)
        if actual != expected:
            mismatches.append(f"{key}={actual!r}, expected {expected!r}")

    optional_schema_metadata = {
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        POSE_REPRESENTATION_KEY: schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
    }
    for key, expected in optional_schema_metadata.items():
        if key in checkpoint_args and str(checkpoint_args[key]) != expected:
            mismatches.append(f"{key}={checkpoint_args[key]!r}, expected {expected!r}")

    if require_schedule_signature:
        expected_schedule_signature = str(getattr(args, "training_schedule_signature", ""))
        checkpoint_schedule_signature = str(checkpoint_args.get("training_schedule_signature", ""))
        if not expected_schedule_signature:
            mismatches.append("current training_schedule_signature is missing")
        elif checkpoint_schedule_signature != expected_schedule_signature:
            mismatches.append(
                "training_schedule_signature="
                f"{checkpoint_schedule_signature!r}, expected {expected_schedule_signature!r}"
            )

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"{args_path} 与当前 {schema.name} 训练配置不兼容：{joined}")


def validate_loaded_state_dict_keys(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    source: str,
) -> None:
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"{source} does not match current model. "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )


class TrainLoop:
    """realtime_pose 扩散训练循环。"""

    def __init__(self, args, train_platform, model, diffusion, data, eval_data=None, data_factory=None):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.data_factory = data_factory
        self.eval_data = eval_data

        self.batch_size = args.batch_size
        self.microbatch = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.init_checkpoint = getattr(args, "init_checkpoint", "")
        self.weight_decay = args.weight_decay
        self.gradient_clip = args.gradient_clip
        self.snr_gamma = args.snr_gamma
        self.use_l1 = args.l1_loss
        self.task_mode = getattr(args, "task_mode", TASK_MODE_REALTIME_POSE)
        self.schema = validate_realtime_pose_training_args(args)
        self.checkpoint_max_keep = max(0, int(args.checkpoint_max_keep))
        self.rollout_steps = int(getattr(args, "rollout_steps", 1))
        for name, default in REALTIME_ROLLOUT_DEFAULTS.items():
            setattr(self, name, type(default)(getattr(args, name, default)))
        for name, default in REALTIME_LR_DEFAULTS.items():
            setattr(self, name, type(default)(getattr(args, name, default)))
        if self.rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {self.rollout_steps}")
        if self.rollout_steps > 9:
            raise ValueError("RPM 风格训练最多支持 base + 8 个相邻窗口，即 rollout_steps<=9。")
        for name in (
            "short_rollout_prob",
            "long_rollout_prob",
            "rollout_max_horizon_prob",
            "long_rollout_transition_prob",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name in ("short_rollout_loss_weight", "long_rollout_loss_weight"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.long_rollout_smooth_l1_beta <= 0.0:
            raise ValueError("long_rollout_smooth_l1_beta 必须大于 0")
        rollout_curriculum_state_from_args(args, 0)
        scheduled_learning_rate(
            global_step=0,
            num_steps=int(args.num_steps),
            lr=float(args.lr),
            lr_warmup_start=float(self.lr_warmup_start),
            lr_warmup_steps=int(self.lr_warmup_steps),
            lr_min=float(self.lr_min),
        )

        self.save_dir = Path(args.save_dir)
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.data_num_batches = len(self.data)
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
        self._data_iterator = None
        self._sampling_epoch = 0
        self._active_rollout_steps = int(getattr(getattr(self.data, "dataset", None), "rollout_steps", 1))
        self._data_initialized_for_training = False
        self._last_logged_curriculum_phase: str | None = None
        self.device = dist_util.dev()
        self.rollout_diffusion = self._create_rollout_diffusion(args)

        logger.log(f"training device: {self.device}")
        logger.log(f"task mode: {self.task_mode}")
        logger.log(
            "rollout curriculum: "
            f"task_steps={self.rollout_steps}, short_prob={self.short_rollout_prob}, "
            f"short_weight={self.short_rollout_loss_weight}, "
            f"long_prob={self.long_rollout_prob}, long_weight={self.long_rollout_loss_weight}, "
            f"max_horizon_prob={self.rollout_max_horizon_prob}, "
            f"transition_prob={self.long_rollout_transition_prob}, prefix=no_grad"
        )
        if self.task_mode != TASK_MODE_REALTIME_POSE:
            raise ValueError(f"当前训练链路只支持 {TASK_MODE_REALTIME_POSE}，实际为 {self.task_mode}")
        if self.device.type == "cuda":
            logger.log(f"cuda device name: {torch.cuda.get_device_name(self.device)}")

        self.feature_w = self._load_feature_weights(args)
        self.normalizer_mean, self.normalizer_std = self._read_dataset_normalizer_stats(data)
        self._load_and_sync_parameters()
        self._configure_trainable_parameters()

        self.amp_dtype = self._select_amp_dtype()
        self.scaler = GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and self.amp_dtype == torch.float16,
            init_scale=1024.0,
        )
        if self.device.type == "cuda":
            logger.log(f"amp dtype: {self.amp_dtype}, grad scaler enabled: {self.scaler.is_enabled()}")
        self.opt = AdamW(self._trainable_parameters(), lr=self.lr, weight_decay=self.weight_decay)

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

    def _create_rollout_diffusion(self, args):
        rollout_enabled = (
            self.rollout_steps > 1
            and (
                (self.short_rollout_prob > 0.0 and self.short_rollout_loss_weight > 0.0)
                or (self.long_rollout_prob > 0.0 and self.long_rollout_loss_weight > 0.0)
            )
        )
        if not rollout_enabled:
            return None
        if self.rollout_ddim_steps != 10:
            raise ValueError("当前运行时契约固定使用 DDIM10 rollout。")
        from utils.model_util import create_gaussian_diffusion

        rollout_args = copy.copy(args)
        rollout_args.diffusion_steps = int(
            getattr(args, "diffusion_steps", getattr(self.diffusion, "original_num_steps", self.diffusion.num_timesteps))
        )
        rollout_args.noise_schedule = str(getattr(args, "noise_schedule", "cosine"))
        rollout_args.predict_xstart = bool(
            getattr(args, "predict_xstart", self.diffusion.model_mean_type.name == "START_X")
        )
        rollout_args.sigma_small = bool(getattr(args, "sigma_small", True))
        rollout_args.ts_respace = f"ddim{self.rollout_ddim_steps}"
        return create_gaussian_diffusion(rollout_args)

    def _load_feature_weights(self, args):
        if not args.weighted_loss:
            return None

        feature_w_path = Path(args.normalizer_dir) / args.feature_w_file
        if not feature_w_path.exists():
            raise FileNotFoundError(f"开启 --weighted_loss 后找不到特征权重文件：{feature_w_path}")

        feature_w = torch.load(feature_w_path, map_location="cpu", weights_only=True).float().flatten()
        if feature_w.numel() != self.schema.feature_dim:
            raise ValueError(f"feature_w 应为 {self.schema.feature_dim} 维，实际为 {feature_w.numel()} 维")
        feature_w.requires_grad_(False)
        return feature_w

    def _configure_trainable_parameters(self) -> None:
        """正式训练从起始 step 起始终训练完整 pose backbone。"""

        for param in self.model.parameters():
            param.requires_grad_(True)

        trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        total = sum(param.numel() for param in self.model.parameters())
        if trainable <= 0:
            raise RuntimeError("当前训练配置没有任何可训练参数。")
        logger.log(f"trainable parameters: {trainable}/{total}")

    def _trainable_parameters(self):
        return [param for param in self.model.parameters() if param.requires_grad]

    @staticmethod
    def _read_dataset_normalizer_stats(data):
        dataset = getattr(data, "dataset", None)
        normalizer = getattr(dataset, "normalizer", None)
        if normalizer is None or getattr(normalizer, "disable", False):
            return None, None
        mean = getattr(normalizer, "mean", None)
        std = getattr(normalizer, "std", None)
        if mean is None or std is None:
            return None, None
        return mean.detach().float().clone(), std.detach().float().clone()

    def _load_and_sync_parameters(self):
        if self.resume_checkpoint:
            self._load_resume_checkpoint()
            return
        if self.init_checkpoint:
            self._load_init_checkpoint()

    def _load_resume_checkpoint(self):
        resume_checkpoint = find_resume_checkpoint(save_dir=self.save_dir, requested_checkpoint=self.resume_checkpoint)
        if not resume_checkpoint:
            return

        self.resume_checkpoint = resume_checkpoint
        self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
        validate_resume_checkpoint_contract(resume_checkpoint=resume_checkpoint, args=self.args, schema_name=self.schema.name)
        logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
        state_dict = dist_util.load_state_dict(resume_checkpoint, map_location=self.device)
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        validate_loaded_state_dict_keys(
            missing_keys=list(incompatible_keys.missing_keys),
            unexpected_keys=list(incompatible_keys.unexpected_keys),
            source="resume checkpoint",
        )

    def _load_init_checkpoint(self):
        init_checkpoint = Path(str(self.init_checkpoint)).expanduser()
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"--init_checkpoint file does not exist: {init_checkpoint}")

        validate_resume_checkpoint_contract(
            resume_checkpoint=init_checkpoint,
            args=self.args,
            schema_name=self.schema.name,
            require_schedule_signature=False,
        )
        logger.log(f"warm-start model from checkpoint: {init_checkpoint}")
        state_dict = dist_util.load_state_dict(init_checkpoint, map_location=self.device)
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        validate_loaded_state_dict_keys(
            missing_keys=list(incompatible_keys.missing_keys),
            unexpected_keys=list(incompatible_keys.unexpected_keys),
            source="init checkpoint",
        )
        logger.log("warm-start keeps optimizer/EMA/global step fresh; use --resume_checkpoint for full training resume.")

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = Path(main_checkpoint).with_name(f"opt{self.resume_step:09d}.pt")
        if not opt_checkpoint.exists():
            logger.log(f"optimizer checkpoint not found, skip: {opt_checkpoint}")
            return

        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=self.device)
        self.opt.load_state_dict(state_dict)
        # 保留 Adam moments，但恢复点学习率只由 global step 的连续曲线决定。
        current_lr = self.learning_rate_at(self.resume_step)
        for param_group in self.opt.param_groups:
            param_group["lr"] = current_lr
            param_group["initial_lr"] = self.lr
        logger.log(f"resume optimizer lr restored from schedule at step {self.resume_step}: {current_lr}")

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
        if self._should_stop():
            logger.log(f"resume step {self.resume_step} has already reached num_steps={self.num_steps}; skip training.")
            return

        last_step_end = time.perf_counter()
        while not self._should_stop():
            current_global_step = self.step + self.resume_step
            self._ensure_training_data_for_step(current_global_step)
            try:
                batch = next(self._data_iterator)
            except StopIteration:
                self._sampling_epoch += 1
                self._set_dataset_sampling_epoch(self._sampling_epoch)
                self._data_iterator = iter(self.data)
                batch = next(self._data_iterator)

            batch_ready_time = time.perf_counter()
            batch = move_batch_to_device(batch, self.device)
            train_start_time = time.perf_counter()
            self.run_step(batch)
            self.step += 1
            train_end_time = time.perf_counter()
            global_step = self.step + self.resume_step
            print(
                f"step[{global_step}] "
                f"data={batch_ready_time - last_step_end:.3f}s "
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
                    self._shutdown_data_iterator()
                    return

        global_step = self.step + self.resume_step
        if self.step > 0 and (self.save_interval <= 0 or global_step % self.save_interval != 0):
            self.save()
            self.evaluate()
        self._shutdown_data_iterator()

    def _ensure_training_data_for_step(self, global_step: int) -> None:
        state = rollout_curriculum_state_from_args(self.args, global_step)
        desired_steps = int(state.active_rollout_steps)
        needs_rebuild = desired_steps != int(self._active_rollout_steps)
        if needs_rebuild:
            if self.data_factory is None:
                raise RuntimeError(
                    f"课程进入 {state.phase} 需要 rollout_steps={desired_steps}，但没有提供 DataLoader 工厂。"
                )
            self._shutdown_data_iterator()
            replacement = self.data_factory(desired_steps)
            replacement_batches = len(replacement)
            if replacement_batches != self.data_num_batches:
                raise RuntimeError(
                    "课程切换前后 DataLoader batch 数必须一致："
                    f"expected={self.data_num_batches}, actual={replacement_batches}, phase={state.phase}"
                )
            self.data = replacement
            self._active_rollout_steps = desired_steps
            logger.log(
                f"rollout DataLoader rebuilt: phase={state.phase}, "
                f"active_rollout_steps={desired_steps}, batches={replacement_batches}"
            )

        if needs_rebuild or not self._data_initialized_for_training:
            self._sampling_epoch = sampling_epoch_for_global_step(
                global_step=global_step,
                batches_per_epoch=self.data_num_batches,
                phase_start_steps=(
                    self.rollout_h1_start_step,
                    self.rollout_h2_start_step,
                    self.rollout_h4_start_step,
                    self.rollout_h8_start_step,
                ),
                resume_mid_epoch=bool(
                    not self._data_initialized_for_training and self.resume_step > 0
                ),
            )
            self._set_dataset_sampling_epoch(self._sampling_epoch)
            self._data_iterator = iter(self.data)
            self._data_initialized_for_training = True

        self.current_curriculum_state = state
        if state.phase != getattr(self, "_last_logged_curriculum_phase", None):
            logger.log(
                "rollout curriculum phase: "
                f"phase={state.phase}, active_rollout_steps={state.active_rollout_steps}, "
                f"max_horizon={state.max_horizon}, short_prob={state.short_prob:.6f}, "
                f"long_prob={state.long_prob:.6f}, max_horizon_prob={state.max_horizon_prob:.6f}"
            )
            self._last_logged_curriculum_phase = state.phase

    def _set_dataset_sampling_epoch(self, sampling_epoch: int) -> None:
        dataset = getattr(self.data, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(int(sampling_epoch))

    def _shutdown_data_iterator(self) -> None:
        iterator = self._data_iterator
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        self._data_iterator = None

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
        self._update_learning_rate()
        self.forward_backward(batch)

        self.scaler.unscale_(self.opt)
        max_norm = 1.0 if self.gradient_clip else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm,
            error_if_nonfinite=True,
        )
        logger.logkv_mean("grad_norm_pre_clip", float(grad_norm.detach().float().item()))
        logger.logkv_mean(
            "grad_clipped_fraction",
            float(self.gradient_clip and grad_norm.detach().float().item() > max_norm),
        )

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
                prepared_batch = self.prepare_teacher_forced_temporal_state(batch)
                base_losses = self.compute_losses(batch=prepared_batch, timesteps=timesteps)
                base_loss = base_losses["loss"].mean()
                validate_finite_losses(losses=base_losses, loss=base_loss, batch=prepared_batch)
                base_log = {key: value for key, value in base_losses.items() if key != "loss"}
                base_log["base_loss"] = base_losses["loss"]
                log_loss_dict(self.diffusion, timesteps, base_log)
                self.scaler.scale(base_loss).backward()
                total_loss_for_log = base_losses["loss"].detach().float().clone()

                curriculum = rollout_curriculum_state_from_args(
                    self.args, self.step + self.resume_step
                )
                self.current_curriculum_state = curriculum
                logger.logkv_mean("rollout_phase_max_horizon", float(curriculum.max_horizon))
                logger.logkv_mean("active_rollout_steps", float(curriculum.active_rollout_steps))
                logger.logkv_mean("rollout_short_probability", float(curriculum.short_prob))
                logger.logkv_mean("rollout_long_probability", float(curriculum.long_prob))
                logger.logkv_mean("rollout_max_horizon_probability", float(curriculum.max_horizon_prob))
                short_event, long_horizon, event_stats = self.sample_rollout_events(prepared_batch)
                logger.logkv_mean("short_rollout_event_fraction", float(short_event))
                logger.logkv_mean("long_rollout_event_fraction", float(long_horizon > 0))
                for name, value in event_stats.items():
                    logger.logkv_mean(name, float(value))
                if not short_event and long_horizon <= 0:
                    log_loss_dict(self.diffusion, timesteps, {"loss": total_loss_for_log})
                    continue

                current_batch = prepared_batch
                predicted_history: list[torch.Tensor] = []
                max_horizon = max(1 if short_event else 0, long_horizon)
                for prefix_index in range(max_horizon):
                    pred_xstart = self.sample_rollout_history(current_batch)
                    rollout_batch = prepared_batch["rollout"][prefix_index]
                    current_batch, next_sample, predicted_history = self.prepare_rollout_batch(
                        batch=current_batch,
                        rollout_batch=rollout_batch,
                        pred_xstart=pred_xstart,
                        predicted_history=predicted_history,
                    )
                    horizon = prefix_index + 1

                    if short_event and horizon == 1:
                        short_losses, short_timesteps = self.compute_rollout_endpoint_losses(
                            batch=current_batch,
                            sample=next_sample,
                            simple_loss_mode="mse",
                        )
                        short_weighted = short_losses["loss"] * self.short_rollout_loss_weight
                        short_loss = short_weighted.mean()
                        validate_finite_losses(
                            losses=short_losses,
                            loss=short_loss,
                            batch=current_batch,
                        )
                        short_log = {
                            f"short_rollout_{key}": value
                            for key, value in short_losses.items()
                            if key != "loss"
                        }
                        short_log["short_rollout_loss"] = short_losses["loss"]
                        short_log["short_rollout_loss_weighted"] = short_weighted
                        short_log["short_rollout_horizon"] = torch.ones_like(short_losses["loss"])
                        log_loss_dict(self.diffusion, short_timesteps, short_log)
                        self.scaler.scale(short_loss).backward()
                        total_loss_for_log = total_loss_for_log + short_weighted.detach().float()

                    if long_horizon > 0 and horizon == long_horizon:
                        long_losses, long_timesteps = self.compute_rollout_endpoint_losses(
                            batch=current_batch,
                            sample=next_sample,
                            simple_loss_mode="smooth_l1",
                        )
                        long_weighted = long_losses["loss"] * self.long_rollout_loss_weight
                        long_loss = long_weighted.mean()
                        validate_finite_losses(
                            losses=long_losses,
                            loss=long_loss,
                            batch=current_batch,
                        )
                        long_log = {
                            f"long_rollout_{key}": value
                            for key, value in long_losses.items()
                            if key != "loss"
                        }
                        long_log["long_rollout_loss"] = long_losses["loss"]
                        long_log["long_rollout_loss_weighted"] = long_weighted
                        long_log["long_rollout_horizon"] = torch.full_like(
                            long_losses["loss"],
                            float(long_horizon),
                        )
                        log_loss_dict(self.diffusion, long_timesteps, long_log)
                        self.scaler.scale(long_loss).backward()
                        total_loss_for_log = total_loss_for_log + long_weighted.detach().float()

                log_loss_dict(self.diffusion, timesteps, {"loss": total_loss_for_log})

    def compute_losses(self, batch: dict, timesteps: torch.Tensor) -> dict:
        batch = self.prepare_teacher_forced_temporal_state(batch)
        sample = batch["x"]  # [B, C, 61]
        batch_size, channels, seq_len = sample.shape
        if channels != self.schema.feature_dim:
            raise ValueError(f"训练输入应为 [B, {self.schema.feature_dim}, T]，实际为 {tuple(sample.shape)}")

        return self.diffusion.training_losses(
            self.model,
            sample,
            timesteps,
            model_kwargs=self.mask_manager(batch, sample),
            feature_w=self._feature_weights_for_batch(batch_size, seq_len),
            snr_gamma=self.snr_gamma,
            use_l1=self.use_l1,
            return_pred_xstart=False,
        )

    @staticmethod
    def prepare_teacher_forced_temporal_state(batch: dict) -> dict:
        """为基础样本显式建立 teacher-forced previous state，不保留旧字段别名。"""

        if "prev_joints_world" in batch:
            raise KeyError("prev_joints_world 已移除，请改用 gt_prev_joints_world/pred_prev_joints_world。")
        required = (
            "gt_prev_joints_world",
            "gt_prev_local_pose_6d",
            "gt_prev_root_yaw",
            "target_frame_dt_seconds",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"训练 batch 缺少 temporal 字段：{missing}")
        has_pred = "pred_prev_joints_world" in batch
        has_pred_pose = "pred_prev_local_pose_6d" in batch
        has_flag = "previous_state_is_predicted" in batch
        if len({has_pred, has_pred_pose, has_flag}) != 1:
            raise KeyError(
                "pred_prev_joints_world、pred_prev_local_pose_6d 与 previous_state_is_predicted 必须同时存在。"
            )
        if has_pred:
            return batch

        result = dict(batch)
        result["pred_prev_joints_world"] = batch["gt_prev_joints_world"]
        result["pred_prev_local_pose_6d"] = batch["gt_prev_local_pose_6d"]
        result["previous_state_is_predicted"] = torch.zeros(
            batch["x"].shape[0],
            dtype=torch.bool,
            device=batch["x"].device,
        )
        return result

    def sample_rollout_events(self, batch: dict) -> tuple[bool, int, dict[str, float]]:
        stats = {
            "rollout_active_max_horizon": 0.0,
            "long_rollout_selected_max_horizon_fraction": 0.0,
            "long_rollout_transition_aware_fraction": 0.0,
            "long_rollout_selected_reconnect_samples": 0.0,
            "long_rollout_selected_dropout_samples": 0.0,
            "short_rollout_reconnect_samples": 0.0,
            "short_rollout_dropout_samples": 0.0,
        }
        for horizon in range(1, 9):
            stats[f"rollout_h{horizon}_event_fraction"] = 0.0
        state = rollout_curriculum_state_from_args(self.args, self.step + self.resume_step)
        self.current_curriculum_state = state
        active_rollout_steps = int(state.active_rollout_steps)
        if active_rollout_steps <= 1 or not self.model.training or not torch.is_grad_enabled():
            return False, 0, stats
        short_enabled = state.short_prob > 0.0 and self.short_rollout_loss_weight > 0.0
        long_enabled = (
            active_rollout_steps > 2
            and state.long_prob > 0.0
            and self.long_rollout_loss_weight > 0.0
        )
        if not short_enabled and not long_enabled:
            return False, 0, stats
        rollout = batch.get("rollout")
        if not isinstance(rollout, (list, tuple)) or len(rollout) < active_rollout_steps - 1:
            raise ValueError(
                f"active_rollout_steps={active_rollout_steps} 需要 Dataset 返回 rollout 子窗口，"
                "请先生成 rollout task 并启用 enable_rollout。"
            )

        short_event = short_enabled and (
            torch.rand((), device=self.device).item() < state.short_prob
        )
        if short_event:
            stats["rollout_h1_event_fraction"] = 1.0
            reconnect, dropout = self.rollout_transition_counts(batch=batch, horizon=1)
            stats["short_rollout_reconnect_samples"] = float(reconnect)
            stats["short_rollout_dropout_samples"] = float(dropout)

        max_horizon = int(state.max_horizon)
        stats["rollout_active_max_horizon"] = float(max_horizon)
        long_event = long_enabled and max_horizon >= 2 and (
            torch.rand((), device=self.device).item() < state.long_prob
        )
        if not long_event:
            return short_event, 0, stats

        horizon, transition_aware, selected_max = self.sample_long_rollout_horizon(
            batch=batch,
            max_horizon=max_horizon,
            max_horizon_prob=float(state.max_horizon_prob),
        )
        stats[f"rollout_h{horizon}_event_fraction"] = 1.0
        stats["long_rollout_selected_max_horizon_fraction"] = float(selected_max)
        reconnect, dropout = self.rollout_transition_counts(batch=batch, horizon=horizon)
        stats["long_rollout_transition_aware_fraction"] = float(transition_aware)
        stats["long_rollout_selected_reconnect_samples"] = float(reconnect)
        stats["long_rollout_selected_dropout_samples"] = float(dropout)
        return short_event, horizon, stats

    def sample_long_rollout_horizon(
        self,
        batch: dict,
        max_horizon: int,
        max_horizon_prob: float,
    ) -> tuple[int, bool, bool]:
        max_horizon = int(max_horizon)
        if max_horizon < 2:
            return 0, False, False
        if max_horizon == 2 or torch.rand((), device=self.device).item() < float(max_horizon_prob):
            return max_horizon, False, True

        horizons = torch.arange(2, max_horizon, device=self.device, dtype=torch.long)
        if horizons.numel() <= 0:
            return max_horizon, False, True
        use_transition = (
            torch.rand((), device=self.device).item() < self.long_rollout_transition_prob
        )
        if use_transition:
            scores = []
            for horizon in horizons.tolist():
                reconnect, dropout = self.rollout_transition_counts(
                    batch=batch,
                    horizon=int(horizon),
                )
                scores.append(float(reconnect * 2 + dropout))
            score_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
            if torch.any(score_tensor > 0.0):
                selected = int(torch.multinomial(score_tensor, 1).item())
                return int(horizons[selected].item()), True, False
        selected = int(torch.randint(0, horizons.numel(), (), device=self.device).item())
        return int(horizons[selected].item()), False, False

    def rollout_transition_counts(self, batch: dict, horizon: int) -> tuple[int, int]:
        rollout_items = batch.get("rollout")
        if not isinstance(rollout_items, (list, tuple)) or not 1 <= int(horizon) <= len(rollout_items):
            raise ValueError(f"无法读取 horizon={horizon} 的 rollout transition")
        previous_batch = batch if int(horizon) == 1 else rollout_items[int(horizon) - 2]
        current_batch = rollout_items[int(horizon) - 1]
        previous_hip = previous_batch["target_sensor_valid"][:, HIP_TRACKER_INDEX].bool()
        current_hip = current_batch["target_sensor_valid"][:, HIP_TRACKER_INDEX].bool()
        reconnect = int(((~previous_hip) & current_hip).sum().item())
        dropout = int((previous_hip & (~current_hip)).sum().item())
        return reconnect, dropout

    def compute_rollout_endpoint_losses(
        self,
        *,
        batch: dict,
        sample: torch.Tensor,
        simple_loss_mode: str,
    ) -> tuple[dict, torch.Tensor]:
        batch_size, _, seq_len = sample.shape
        timesteps = torch.randint(
            low=0,
            high=self.diffusion.num_timesteps,
            size=(batch_size,),
            device=self.device,
        )
        losses = self.diffusion.training_losses(
            self.model,
            sample,
            timesteps,
            model_kwargs=self.mask_manager(batch, sample),
            noise=torch.randn_like(sample),
            feature_w=self._feature_weights_for_batch(batch_size, seq_len),
            snr_gamma=self.snr_gamma,
            use_l1=self.use_l1,
            simple_loss_mode=simple_loss_mode,
            simple_loss_huber_beta=self.long_rollout_smooth_l1_beta,
        )
        return losses, timesteps

    def compute_rollout_terminal_losses(
        self,
        batch: dict,
        horizon: int,
        simple_loss_mode: str = "mse",
    ) -> dict:
        batch = self.prepare_teacher_forced_temporal_state(batch)
        if not 1 <= int(horizon) < self.rollout_steps:
            raise ValueError(f"rollout horizon 必须在 [1,{self.rollout_steps - 1}]，实际为 {horizon}")
        rollout_items = batch.get("rollout")
        if not isinstance(rollout_items, (list, tuple)) or len(rollout_items) < horizon:
            raise ValueError(f"horizon={horizon} 需要至少 {horizon} 个 rollout 子窗口。")

        current_batch = batch
        predicted_history: list[torch.Tensor] = []
        for prefix_index in range(horizon):
            # RPM 风格前缀固定 no_grad；每一步都传播 Resolver 最终状态。
            pred_xstart = self.sample_rollout_history(current_batch)
            rollout_batch = rollout_items[prefix_index]
            if rollout_batch["x"].shape != batch["x"].shape:
                raise ValueError(
                    f"rollout[{prefix_index}]['x'] 应为 {tuple(batch['x'].shape)}，"
                    f"实际为 {tuple(rollout_batch['x'].shape)}"
                )
            current_batch, next_sample, predicted_history = self.prepare_rollout_batch(
                batch=current_batch,
                rollout_batch=rollout_batch,
                pred_xstart=pred_xstart,
                predicted_history=predicted_history,
            )

        losses, _ = self.compute_rollout_endpoint_losses(
            batch=current_batch,
            sample=next_sample,
            simple_loss_mode=simple_loss_mode,
        )
        return losses

    def compute_one_step_rollout_losses(
        self,
        batch: dict,
        pred_xstart: torch.Tensor | None = None,
    ) -> dict:
        """保留测试入口；正式路径统一走 horizon=1 的 terminal rollout。"""

        if pred_xstart is None:
            return self.compute_rollout_terminal_losses(batch=batch, horizon=1)
        next_batch, next_sample, _ = self.prepare_rollout_batch(
            batch=batch,
            rollout_batch=batch["rollout"][0],
            pred_xstart=pred_xstart,
            predicted_history=[],
        )
        losses, _ = self.compute_rollout_endpoint_losses(
            batch=next_batch,
            sample=next_sample,
            simple_loss_mode="mse",
        )
        return losses

    def sample_rollout_history(self, batch: dict) -> torch.Tensor:
        if self.rollout_diffusion is None:
            raise RuntimeError("rollout diffusion 尚未初始化。")
        sample = batch["conditioned_x"]
        model_kwargs = self.mask_manager(batch, batch["x"])
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return self.rollout_diffusion.ddim_sample_loop(
                    self.model,
                    shape=tuple(sample.shape),
                    noise=torch.randn_like(sample),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=self.device,
                    progress=False,
                    eta=0.0,
                ).detach()
        finally:
            self.model.train(was_training)

    def prepare_rollout_batch(
        self,
        *,
        batch: dict,
        rollout_batch: dict,
        pred_xstart: torch.Tensor,
        predicted_history: list[torch.Tensor],
    ) -> tuple[dict, torch.Tensor, list[torch.Tensor]]:
        """把当前 DDIM 输出经 Resolver 校正后写入下一窗口全部重叠预测历史。"""

        final_target, final_root, final_yaw, final_height, final_joints, final_state = self.resolve_rollout_target(
            batch=batch,
            pred_xstart=pred_xstart,
        )
        finalized_prediction = pred_xstart.detach().clone()
        finalized_prediction[:, self.schema.target_slice(), REALTIME_POSE_TARGET_START] = final_target
        # 保存完整 214 维帧：154 维预测结果之外，还要保留该帧实际使用的 tracker reference。
        # 否则 H>1 再次滑窗时，no-Hip 历史帧会悄悄退回 Dataset 中按 GT state 编码的条件。
        finalized_history_frame = batch["conditioned_x"][
            :, :, REALTIME_POSE_TARGET_START
        ].detach().clone()
        finalized_history_frame[:, self.schema.target_slice()] = final_target.detach()
        updated_history = [*predicted_history, finalized_history_frame]
        next_conditioned = self.build_rollout_conditioned_x(
            rollout_batch=rollout_batch,
            predicted_history=updated_history,
        )
        next_batch = dict(rollout_batch)
        next_sample = rollout_batch["x"].clone()
        final_raw_target = self._target_features_to_raw(finalized_prediction)

        next_batch["prev_root_pos_world"] = final_root
        next_batch["prev_root_yaw"] = final_yaw
        next_batch["pred_prev_joints_world"] = final_joints
        next_batch["pred_prev_local_pose_6d"] = final_raw_target[:, self.schema.body_pose_slice()]
        next_batch["previous_state_is_predicted"] = torch.ones(
            final_joints.shape[0],
            dtype=torch.bool,
            device=final_joints.device,
        )
        next_batch["resolver_before_target_root_pos_world"] = final_root
        next_batch["resolver_before_target_root_yaw"] = final_yaw
        next_batch["resolver_before_target_pelvis_height"] = final_height
        next_batch["resolver_before_target_joints_world"] = final_joints
        next_batch.update(final_state)

        # Rollout 的 x0 target 继续表达 GT 帧间运动量。预测上一状态只用于条件和
        # Resolver state，不能把累积漂移压缩成当前帧的 root/yaw “瞬时拉回”目标。
        next_batch["x"] = next_sample

        self._reencode_rollout_target_trackers(
            next_batch=next_batch,
            next_conditioned=next_conditioned,
            previous_final_root=final_root,
            previous_final_yaw=final_yaw,
        )
        next_batch["conditioned_x"] = next_conditioned
        return next_batch, next_sample, updated_history

    def prepare_one_step_rollout_batch(
        self,
        *,
        batch: dict,
        rollout_batch: dict,
        pred_xstart: torch.Tensor,
    ) -> tuple[dict, torch.Tensor]:
        """兼容原有 H=1 测试入口。"""

        next_batch, next_sample, _ = self.prepare_rollout_batch(
            batch=batch,
            rollout_batch=rollout_batch,
            pred_xstart=pred_xstart,
            predicted_history=[],
        )
        return next_batch, next_sample

    def resolve_rollout_target(
        self,
        *,
        batch: dict,
        pred_xstart: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        raw_target = self._target_features_to_raw(pred_xstart)
        raw_target_np = raw_target.detach().cpu().numpy().astype(np.float32)
        batch_size = raw_target_np.shape[0]
        ref_pos = batch["tracker_ref_root_pos_world"].detach().cpu().numpy()
        ref_yaw = batch["tracker_ref_root_yaw"].detach().cpu().numpy()
        tracker_pos_world = decode_tracker_positions_np(
            batch["target_tracker_pos_ref"].detach().cpu().numpy(), ref_pos, ref_yaw
        )
        tracker_rot_world = decode_tracker_rotations_np(
            batch["target_tracker_rot_ref_6d"].detach().cpu().numpy(), ref_yaw
        )
        valid = batch["target_sensor_valid"].detach().cpu().numpy().astype(bool)
        offsets = batch["joint_offsets_parent"].detach().cpu().numpy().astype(np.float32)
        rest_rotations = batch.get("joint_rest_local_rotations_6d")
        rest_rotations_np = None if rest_rotations is None else rest_rotations.detach().cpu().numpy().astype(np.float32)

        roots: list[np.ndarray] = []
        yaws: list[float] = []
        heights: list[float] = []
        joints: list[np.ndarray] = []
        states: list[RuntimeRootResolverState] = []
        finalized = raw_target_np.copy()
        for index in range(batch_size):
            state = RuntimeRootResolverState(
                initialized=True,
                final_root_pos_world=batch["resolver_before_target_root_pos_world"][index].detach().cpu().numpy(),
                final_root_yaw=float(batch["resolver_before_target_root_yaw"][index].item()),
                final_pelvis_height=float(batch["resolver_before_target_pelvis_height"][index].item()),
                final_joints_world=batch["resolver_before_target_joints_world"][index].detach().cpu().numpy(),
                hip_was_valid=bool(batch["resolver_before_target_hip_valid"][index].item()),
                reconnect_active=(
                    0.0 < float(batch["resolver_before_target_reconnect_elapsed_seconds"][index].item()) < 0.1
                ),
                reconnect_elapsed_seconds=float(
                    batch["resolver_before_target_reconnect_elapsed_seconds"][index].item()
                ),
                reconnect_start_root_pos_world=batch[
                    "resolver_before_target_reconnect_start_root_pos_world"
                ][index].detach().cpu().numpy(),
                reconnect_start_root_yaw=float(
                    batch["resolver_before_target_reconnect_start_root_yaw"][index].item()
                ),
                reconnect_start_pelvis_height=float(
                    batch["resolver_before_target_reconnect_start_pelvis_height"][index].item()
                ),
                last_timestamp=float(batch["resolver_before_target_last_timestamp_seconds"][index].item()),
                floor_y=float(batch["resolver_before_target_floor_y"][index].item()),
                tracking_origin_revision=int(
                    batch["resolver_before_target_tracking_origin_revision"][index].item()
                ),
            )
            pose = raw_target_np[index, self.schema.body_pose_slice()].copy()
            local_offsets = offsets[index].copy()
            local_rest = None if rest_rotations_np is None else rest_rotations_np[index].copy()

            def fk_callback(root: np.ndarray, yaw: float, pelvis_height: float) -> np.ndarray:
                fk_offsets = local_offsets.copy()
                fk_offsets[0, 1] = float(pelvis_height)
                with torch.no_grad():
                    output = fk_body_fbx_local_torch(
                        body_pose_local_delta_6d=torch.from_numpy(pose[None]).float(),
                        actor_root_pos_world=torch.from_numpy(root[None]).float(),
                        root_heading=torch.tensor([yaw], dtype=torch.float32),
                        rest_local_positions=torch.from_numpy(fk_offsets[None]).float(),
                        rest_local_rotations_6d=(
                            None if local_rest is None else torch.from_numpy(local_rest[None]).float()
                        ),
                    )
                return output[0].cpu().numpy()

            resolver = RuntimeRootResolver(
                pelvis_offset_parent=local_offsets[0],
                state=state,
            )
            result = resolver.resolve(
                tracker_pos_world=tracker_pos_world[index],
                tracker_rot_world_6d=tracker_rot_world[index],
                sensor_valid=valid[index],
                timestamp=float(batch["target_timestamp_seconds"][index].item()),
                floor_y=float(batch["target_floor_y"][index].item()),
                tracking_origin_revision=int(batch["target_tracking_origin_revision"][index].item()),
                model_root_delta_xz_ref=raw_target_np[index, self.schema.root_delta_xz_slice()],
                model_yaw_delta_sincos=raw_target_np[index, self.schema.root_yaw_delta_slice()],
                model_pelvis_height=float(raw_target_np[index, self.schema.root_height_slice()][0]),
                fk_callback=fk_callback,
            )
            finalized[index, self.schema.root_delta_xz_slice()] = result.final_root_delta_xz_ref
            finalized[index, self.schema.root_yaw_delta_slice()] = result.final_yaw_delta_sincos
            finalized[index, self.schema.root_height_slice()] = result.final_pelvis_height
            roots.append(result.final_root_pos_world)
            yaws.append(result.final_root_yaw)
            heights.append(result.final_pelvis_height)
            joints.append(result.final_joints_world)
            states.append(result.state)

        device = pred_xstart.device
        dtype = pred_xstart.dtype
        final_target = self._raw_target_to_normalized(torch.from_numpy(finalized).to(device=device, dtype=dtype))
        state_tensors = {
            "resolver_before_target_hip_valid": torch.tensor(
                [state.hip_was_valid for state in states], device=device, dtype=torch.bool
            ),
            "resolver_before_target_reconnect_start_root_pos_world": torch.from_numpy(
                np.stack([state.reconnect_start_root_pos_world for state in states])
            ).to(device=device, dtype=dtype),
            "resolver_before_target_reconnect_start_root_yaw": torch.tensor(
                [state.reconnect_start_root_yaw for state in states], device=device, dtype=dtype
            ),
            "resolver_before_target_reconnect_start_pelvis_height": torch.tensor(
                [state.reconnect_start_pelvis_height for state in states], device=device, dtype=dtype
            ),
            "resolver_before_target_reconnect_elapsed_seconds": torch.tensor(
                [state.reconnect_elapsed_seconds for state in states], device=device, dtype=dtype
            ),
            "resolver_before_target_last_timestamp_seconds": torch.tensor(
                [float(state.last_timestamp) for state in states], device=device, dtype=torch.float64
            ),
            "resolver_before_target_floor_y": torch.tensor(
                [state.floor_y for state in states], device=device, dtype=dtype
            ),
            "resolver_before_target_tracking_origin_revision": torch.tensor(
                [state.tracking_origin_revision for state in states], device=device, dtype=torch.int64
            ),
        }
        return (
            final_target,
            torch.from_numpy(np.stack(roots)).to(device=device, dtype=dtype),
            torch.tensor(yaws, device=device, dtype=dtype),
            torch.tensor(heights, device=device, dtype=dtype),
            torch.from_numpy(np.stack(joints)).to(device=device, dtype=dtype),
            state_tensors,
        )

    def _reencode_rollout_target_trackers(
        self,
        *,
        next_batch: dict,
        next_conditioned: torch.Tensor,
        previous_final_root: torch.Tensor,
        previous_final_yaw: torch.Tensor,
    ) -> None:
        old_ref_pos = next_batch["tracker_ref_root_pos_world"].detach().cpu().numpy()
        old_ref_yaw = next_batch["tracker_ref_root_yaw"].detach().cpu().numpy()
        tracker_world = decode_tracker_positions_np(
            next_batch["target_tracker_pos_ref"].detach().cpu().numpy(), old_ref_pos, old_ref_yaw
        )
        tracker_rot_world = decode_tracker_rotations_np(
            next_batch["target_tracker_rot_ref_6d"].detach().cpu().numpy(), old_ref_yaw
        )
        valid = next_batch["target_sensor_valid"].detach().cpu().numpy().astype(bool)
        offsets = next_batch["joint_offsets_parent"].detach().cpu().numpy()
        floor = next_batch["target_floor_y"].detach().cpu().numpy()
        previous_root_np = previous_final_root.detach().cpu().numpy()
        previous_yaw_np = previous_final_yaw.detach().cpu().numpy()
        ref_positions = []
        ref_yaws = []
        ref_sources = []
        encoded_positions = []
        encoded_rotations = []
        for index in range(valid.shape[0]):
            ref_pos, ref_yaw, ref_source = build_tracker_reference_np(
                tracker_pos_world=tracker_world[index:index + 1],
                tracker_rot_world_6d=tracker_rot_world[index:index + 1],
                sensor_valid=valid[index:index + 1],
                previous_final_root_pos_world=previous_root_np[index:index + 1],
                previous_final_root_yaw=previous_yaw_np[index:index + 1],
                pelvis_offset_parent=offsets[index, 0],
                floor_y=floor[index:index + 1],
            )
            ref_positions.append(ref_pos[0])
            ref_yaws.append(ref_yaw[0])
            ref_sources.append(ref_source[0])
            encoded_positions.append(encode_tracker_positions_np(tracker_world[index:index + 1], ref_pos, ref_yaw)[0])
            encoded_rotations.append(encode_tracker_rotations_np(tracker_rot_world[index:index + 1], ref_yaw)[0])

        device = next_conditioned.device
        dtype = next_conditioned.dtype
        encoded_pos = torch.from_numpy(np.stack(encoded_positions)).to(device=device, dtype=dtype)
        encoded_rot = torch.from_numpy(np.stack(encoded_rotations)).to(device=device, dtype=dtype)
        valid_tensor = next_batch["target_sensor_valid"].to(device=device)
        next_batch["tracker_ref_root_pos_world"] = torch.from_numpy(np.stack(ref_positions)).to(device=device, dtype=dtype)
        next_batch["tracker_ref_root_yaw"] = torch.tensor(ref_yaws, device=device, dtype=dtype)
        next_batch["tracker_ref_source"] = torch.tensor(ref_sources, device=device, dtype=torch.int64)
        next_batch["target_tracker_pos_ref"] = encoded_pos
        next_batch["target_tracker_rot_ref_6d"] = encoded_rot

        pos_slice = self.schema.tracker_pos_slice()
        rot_slice = self.schema.tracker_rot_slice()
        pos_values = encoded_pos.reshape(encoded_pos.shape[0], -1)
        rot_values = encoded_rot.reshape(encoded_rot.shape[0], -1)
        if self.normalizer_mean is not None and self.normalizer_std is not None:
            mean = self.normalizer_mean.to(device=device, dtype=dtype)
            std = self.normalizer_std.to(device=device, dtype=dtype)
            pos_values = (pos_values - mean[pos_slice]) / std[pos_slice]
            rot_values = (rot_values - mean[rot_slice]) / std[rot_slice]
        for tracker_index in range(valid_tensor.shape[1]):
            missing = ~valid_tensor[:, tracker_index]
            pos_values[missing, tracker_index * 3 : tracker_index * 3 + 3] = 0.0
            rot_values[missing, tracker_index * 6 : tracker_index * 6 + 6] = 0.0
        next_conditioned[:, pos_slice, REALTIME_POSE_TARGET_START] = pos_values
        next_conditioned[:, rot_slice, REALTIME_POSE_TARGET_START] = rot_values
        next_conditioned[:, self.schema.sensor_valid_slice(), REALTIME_POSE_TARGET_START] = valid_tensor.to(dtype=dtype)

    def _target_features_to_raw(self, features: torch.Tensor) -> torch.Tensor:
        values = features[:, self.schema.target_slice(), REALTIME_POSE_TARGET_START]
        if self.normalizer_mean is None or self.normalizer_std is None:
            return values.clone()
        mean = self.normalizer_mean.to(device=features.device, dtype=features.dtype)[self.schema.target_slice()]
        std = self.normalizer_std.to(device=features.device, dtype=features.dtype)[self.schema.target_slice()]
        return values * std + mean

    def _raw_target_to_normalized(self, values: torch.Tensor) -> torch.Tensor:
        if self.normalizer_mean is None or self.normalizer_std is None:
            return values
        mean = self.normalizer_mean.to(device=values.device, dtype=values.dtype)[self.schema.target_slice()]
        std = self.normalizer_std.to(device=values.device, dtype=values.dtype)[self.schema.target_slice()]
        return (values - mean) / std

    def build_rollout_conditioned_x(
        self,
        rollout_batch: dict,
        predicted_history: list[torch.Tensor],
    ) -> torch.Tensor:
        next_conditioned = rollout_batch["conditioned_x"].clone()
        if not predicted_history:
            return next_conditioned
        if len(predicted_history) > REALTIME_POSE_TARGET_START:
            raise ValueError(f"预测历史最多回填 {REALTIME_POSE_TARGET_START} 帧，实际为 {len(predicted_history)}")
        history = torch.stack([value.detach() for value in predicted_history], dim=-1)
        history_start = REALTIME_POSE_TARGET_START - len(predicted_history)
        full_expected = (next_conditioned.shape[0], self.schema.feature_dim, len(predicted_history))
        target_expected = (next_conditioned.shape[0], self.schema.target_dim, len(predicted_history))
        if history.shape == full_expected:
            next_conditioned[:, :, history_start:REALTIME_POSE_TARGET_START] = history
        elif history.shape == target_expected:
            # 只保留给旧 H=1 测试入口；正式 rollout 路径始终传播完整 feature frame。
            next_conditioned[
                :, self.schema.target_slice(), history_start:REALTIME_POSE_TARGET_START
            ] = history
        else:
            raise ValueError(
                f"predicted history 应为 {full_expected}（兼容 {target_expected}），"
                f"实际为 {tuple(history.shape)}"
            )
        return next_conditioned

    def build_one_step_rollout_conditioned_x(self, rollout_batch: dict, pred_xstart: torch.Tensor) -> torch.Tensor:
        """兼容 H=1 调用；历史始终 detach。"""

        pred_target = pred_xstart[:, self.schema.target_slice(), REALTIME_POSE_TARGET_START]
        return self.build_rollout_conditioned_x(
            rollout_batch=rollout_batch,
            predicted_history=[pred_target],
        )

    def learning_rate_at(self, global_step: int) -> float:
        return scheduled_learning_rate(
            global_step=int(global_step),
            num_steps=int(self.num_steps),
            lr=float(self.lr),
            lr_warmup_start=float(self.lr_warmup_start),
            lr_warmup_steps=int(self.lr_warmup_steps),
            lr_min=float(self.lr_min),
        )

    def _update_learning_rate(self) -> float:
        lr = self.learning_rate_at(self.step + self.resume_step)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr
        return lr

    def _feature_weights_for_batch(self, batch_size: int, seq_len: int):
        if self.feature_w is None:
            return None
        return self.feature_w.to(self.device)[None, :, None].repeat(batch_size, 1, seq_len)

    def mask_manager(self, batch, sample):
        """
        `inpaint_mask=True` 表示该位置需要加噪、预测并参与 denoise loss。
        tracker 条件和 sensor_valid 永远作为观测条件，不参与 diffusion loss。
        """

        batch_size, channels, seq_len = sample.shape
        valid_frame_mask = batch.get("valid_frame_mask", batch.get("attention_mask"))
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=sample.device)
        valid_frame_mask = valid_frame_mask.bool()
        if valid_frame_mask.shape != (batch_size, seq_len):
            raise ValueError(f"valid_frame_mask 应为 [B, T]，实际为 {tuple(valid_frame_mask.shape)}")

        inpaint_mask = batch.get("inpaint_mask")
        if inpaint_mask is None:
            raise ValueError("训练 batch 缺少 inpaint_mask，请先生成 realtime_pose task。")
        inpaint_mask = inpaint_mask.bool()
        if inpaint_mask.shape != sample.shape:
            raise ValueError(f"inpaint_mask 应为 {tuple(sample.shape)}，实际为 {tuple(inpaint_mask.shape)}")

        inpaint_mask = inpaint_mask & valid_frame_mask.unsqueeze(1)
        inpaint_mask[:, self.schema.target_dim:self.schema.feature_dim, :] = False
        sensor_slice = self.schema.sensor_valid_slice()
        if inpaint_mask[:, sensor_slice, :].any():
            raise ValueError("sensor_valid 不能参与 diffusion loss，请检查 task 的 inpaint_mask。")
        if not inpaint_mask.any():
            raise ValueError("当前 batch 的 inpaint_mask 没有待补全部分，请检查离线任务生成结果。")

        conditioned_sample = batch.get("conditioned_x", sample)
        if conditioned_sample.shape != sample.shape:
            raise ValueError(f"conditioned_x 应为 {tuple(sample.shape)}，实际为 {tuple(conditioned_sample.shape)}")

        y = {
            "mask": inpaint_mask,
            "inpainted_motion": conditioned_sample,
            "schema_name": self.schema.name,
            "target_joints_world": batch["target_joints_world"],
            "gt_prev_joints_world": batch["gt_prev_joints_world"],
            "pred_prev_joints_world": batch["pred_prev_joints_world"],
            "gt_prev_local_pose_6d": batch["gt_prev_local_pose_6d"],
            "pred_prev_local_pose_6d": batch["pred_prev_local_pose_6d"],
            "previous_state_is_predicted": batch["previous_state_is_predicted"],
            "target_frame_dt_seconds": batch["target_frame_dt_seconds"],
            "target_root_pos_world": batch["target_root_pos_world"],
            "prev_root_yaw": batch["prev_root_yaw"],
            "gt_prev_root_yaw": batch["gt_prev_root_yaw"],
            "target_root_yaw": batch["target_root_yaw"],
            "tracker_ref_root_pos_world": batch["tracker_ref_root_pos_world"],
            "tracker_ref_root_yaw": batch["tracker_ref_root_yaw"],
            "tracker_ref_source": batch["tracker_ref_source"],
            "target_floor_y": batch["target_floor_y"],
            "target_tracker_pos_ref": batch["target_tracker_pos_ref"],
            "target_tracker_rot_ref_6d": batch["target_tracker_rot_ref_6d"],
            "target_sensor_valid": batch["target_sensor_valid"],
            "joint_offsets_parent": batch["joint_offsets_parent"],
            "sensor_valid": batch["sensor_valid"],
            "target_timestamp_seconds": batch["target_timestamp_seconds"],
            "target_tracking_origin_revision": batch["target_tracking_origin_revision"],
            "resolver_before_target_root_pos_world": batch["resolver_before_target_root_pos_world"],
            "resolver_before_target_root_yaw": batch["resolver_before_target_root_yaw"],
            "resolver_before_target_pelvis_height": batch["resolver_before_target_pelvis_height"],
            "resolver_before_target_hip_valid": batch["resolver_before_target_hip_valid"],
            "resolver_before_target_reconnect_start_root_pos_world": batch[
                "resolver_before_target_reconnect_start_root_pos_world"
            ],
            "resolver_before_target_reconnect_start_root_yaw": batch[
                "resolver_before_target_reconnect_start_root_yaw"
            ],
            "resolver_before_target_reconnect_start_pelvis_height": batch[
                "resolver_before_target_reconnect_start_pelvis_height"
            ],
            "resolver_before_target_reconnect_elapsed_seconds": batch[
                "resolver_before_target_reconnect_elapsed_seconds"
            ],
            "resolver_before_target_last_timestamp_seconds": batch[
                "resolver_before_target_last_timestamp_seconds"
            ],
            "resolver_before_target_tracking_origin_revision": batch[
                "resolver_before_target_tracking_origin_revision"
            ],
        }
        if "joint_rest_local_rotations_6d" in batch:
            y["joint_rest_local_rotations_6d"] = batch["joint_rest_local_rotations_6d"]
        if self.schema.supports_root_motion:
            y["prev_root_pos_world"] = batch["prev_root_pos_world"]
            y["target_root_delta_xz_ref"] = batch["target_root_delta_xz_ref"]
            y["target_root_height"] = batch["target_root_height"]
        if self.schema.supports_stationary_prob:
            y["target_stationary_prob_5"] = batch["target_stationary_prob_5"]
        if self.normalizer_mean is not None and self.normalizer_std is not None:
            y["normalizer_mean"] = self.normalizer_mean.to(device=sample.device, dtype=sample.dtype)
            y["normalizer_std"] = self.normalizer_std.to(device=sample.device, dtype=sample.dtype)

        return {
            "inpaint_cond": inpaint_mask,
            "valid_frame_mask": valid_frame_mask,
            "attention_mask": valid_frame_mask,
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
        log_values = values.detach().float()
        logger.logkv_mean(key, log_values.mean().item())
        for timestep, loss in zip(timesteps.detach().cpu().numpy(), log_values.cpu().numpy()):
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
