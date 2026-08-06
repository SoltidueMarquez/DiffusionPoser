from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from data_loaders.realtime_pose_history_noise import (
    HistoryPoseNoiseConfig,
    corrupt_history_pose_observation,
)

from data_loaders.sensor_masking import (
    REALTIME_POSE_TARGET_DIM,
    REALTIME_POSE_WINDOW_LENGTH,
    TASK_MODE_REALTIME_POSE,
)
from diffusion import logger
from utils import dist_util


TRAIN_DEVICE_FIELDS = frozenset(
    {
        "x",
        "history_pose_observation",
        "tracker_window",
        "head_path_window",
        "history_region_confidence",
        "window_valid_mask",
        "frame_offsets",
        "tracker_window_raw",
        "hard_rotation_state_window",
        "target_joints_head_ref",
        "joint_offsets_parent",
        "target_root_position_head_ref",
        "target_root_yaw_world",
        "current_head_yaw_world",
        "future_leg_target",
        "contact_target",
    }
)


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
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.gradient_clip = args.gradient_clip
        self.snr_gamma = args.snr_gamma
        self.use_l1 = args.l1_loss
        self.task_mode = getattr(args, "task_mode", TASK_MODE_REALTIME_POSE)
        self.checkpoint_max_keep = max(0, int(args.checkpoint_max_keep))
        self.history_noise_config = HistoryPoseNoiseConfig(
            probability=float(getattr(args, "history_noise_prob", 0.8)),
            min_degrees=float(getattr(args, "history_noise_min_deg", 2.0)),
            max_degrees=float(getattr(args, "history_noise_max_deg", 10.0)),
            temporal_rho=float(getattr(args, "history_noise_temporal_rho", 0.95)),
            region_ratio=float(getattr(args, "history_noise_region_ratio", 0.75)),
            joint_ratio=float(getattr(args, "history_noise_joint_ratio", 0.25)),
        ).validate()
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
            "history pose noise: "
            f"prob={self.history_noise_config.probability}, "
            f"degrees=[{self.history_noise_config.min_degrees},"
            f"{self.history_noise_config.max_degrees}], "
            f"temporal_rho={self.history_noise_config.temporal_rho}"
        )
        if self.task_mode != TASK_MODE_REALTIME_POSE:
            raise ValueError(f"当前训练链路只支持 {TASK_MODE_REALTIME_POSE}，实际为 {self.task_mode}")
        if self.device.type == "cuda":
            logger.log(f"cuda device name: {torch.cuda.get_device_name(self.device)}")

        self.feature_w = self._load_feature_weights(args)
        self.pose_mean, self.pose_scale = self._read_dataset_normalizer_stats(data)
        self._load_and_sync_parameters()

        self.amp_dtype = self._select_amp_dtype()
        self.scaler = GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and self.amp_dtype == torch.float16,
            init_scale=1024.0,
        )
        if self.device.type == "cuda":
            logger.log(f"amp dtype: {self.amp_dtype}, grad scaler enabled: {self.scaler.is_enabled()}")
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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
            return None, None
        mean = getattr(normalizer, "pose_mean", None)
        scale = getattr(normalizer, "pose_scale", None)
        if mean is None or scale is None:
            return None, None
        return mean.detach().float().clone(), scale.detach().float().clone()

    def _load_and_sync_parameters(self):
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
                batch = move_training_batch_to_device(batch, self.device)
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
                batch = move_training_batch_to_device(batch, self.device)
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
        sample = batch["x"]  # [B,11,144]
        if sample.ndim != 3 or tuple(sample.shape[1:]) != (
            REALTIME_POSE_WINDOW_LENGTH,
            REALTIME_POSE_TARGET_DIM,
        ):
            raise ValueError(
                f"训练输入应为 [B,11,{REALTIME_POSE_TARGET_DIM}]，"
                f"实际为 {tuple(sample.shape)}"
            )
        feature_w = self._feature_weights_for_batch(sample.shape[0])
        return self.diffusion.training_losses(
            self.model,
            sample,
            timesteps,
            model_kwargs=self.mask_manager(batch, sample),
            feature_w=feature_w,
            snr_gamma=self.snr_gamma,
            use_l1=self.use_l1,
            return_pred_xstart=False,
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
        window_valid_mask = batch["window_valid_mask"].bool()
        if tuple(window_valid_mask.shape) != tuple(sample.shape[:2]):
            raise ValueError("window_valid_mask 必须为 [B,11]。")
        history_observation = batch["history_pose_observation"]
        if self.model.training and torch.is_grad_enabled():
            # SO(3) 扰动始终使用 float32，避免 AMP 降低小角度指数映射精度。
            with autocast(device_type=self.device.type, enabled=False):
                history_observation = corrupt_history_pose_observation(
                    history_pose_clean=history_observation.float(),
                    history_region_confidence=batch["history_region_confidence"].float(),
                    history_valid_mask=window_valid_mask[:, :-1],
                    pose_mean=self.pose_mean,
                    pose_scale=self.pose_scale,
                    config=self.history_noise_config,
                )
        conditions = {
            "history_pose_observation": history_observation,
            "tracker_window": batch["tracker_window"],
            "head_path_window": batch["head_path_window"],
            "history_region_confidence": batch["history_region_confidence"],
            "window_valid_mask": window_valid_mask,
            "frame_offsets": batch["frame_offsets"],
        }
        y = {
            **conditions,
            "tracker_window_raw": batch["tracker_window_raw"],
            "hard_rotation_state_window": batch["hard_rotation_state_window"],
            "target_joints_head_ref": batch["target_joints_head_ref"],
            "joint_offsets_parent": batch["joint_offsets_parent"],
            "target_root_position_head_ref": batch["target_root_position_head_ref"],
            "target_root_yaw_world": batch["target_root_yaw_world"],
            "current_head_yaw_world": batch["current_head_yaw_world"],
            "future_leg_target": batch["future_leg_target"],
            "contact_target": batch["contact_target"],
        }
        if self.pose_mean is not None and self.pose_scale is not None:
            y["pose_mean"] = self.pose_mean.to(
                device=sample.device, dtype=sample.dtype
            )
            y["pose_scale"] = self.pose_scale.to(
                device=sample.device, dtype=sample.dtype
            )
        # 模型、Loss 和 hard projection 共用同一个 y，避免条件在两层字典中静默分叉。
        return {"y": y}


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


def move_training_batch_to_device(batch: dict, device: torch.device) -> dict:
    """只搬运训练和 loss 实际读取的张量，重建专用字段继续留在 CPU。"""

    return {
        key: move_batch_to_device(value, device) if key in TRAIN_DEVICE_FIELDS else value
        for key, value in batch.items()
    }


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
