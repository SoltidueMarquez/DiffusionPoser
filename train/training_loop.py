import os
import re
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from diffusion import logger
from data_loaders.sensor_masking import MODEL_INPUT_DIM, X277_FEATURE_DIM
from utils import dist_util


class TrainLoop:
    """
    DiffusionPoser 的 fix-only 扩散训练循环。

    本文件沿用 StableMotion 的训练骨架：run_loop -> run_step ->
    forward_backward -> mask_manager -> save/log_step。与 StableMotion 不同的是，
    DiffusionPoser 不再训练 detection 模式，训练目标完全由离线生成的
    `inpaint_mask: [B, 283, T]` 决定。
    """

    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.data = data

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
        self.checkpoint_max_keep = max(0, int(args.checkpoint_max_keep))

        self.save_dir = Path(args.save_dir)
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // max(1, len(self.data)) + 1
        self.device = dist_util.dev()
        logger.log(f"training device: {self.device}")
        if self.device.type == "cuda":
            logger.log(f"cuda device name: {torch.cuda.get_device_name(self.device)}")

        self.feature_w = self._load_feature_weights(args)
        self._load_and_sync_parameters()

        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.resume_step:
            self._load_optimizer_state()

        self.ema_model = self._create_ema_model(args)
        if self.resume_step and self.ema_model is not None:
            self._load_ema_state()

    # region 初始化与恢复
    def _load_feature_weights(self, args):
        if not args.weighted_loss:
            return None

        feature_w_path = Path(args.normalizer_dir) / args.feature_w_file
        if not feature_w_path.exists():
            raise FileNotFoundError(f"开启 --weighted_loss 后找不到特征权重文件：{feature_w_path}")

        feature_w = torch.load(feature_w_path, map_location="cpu", weights_only=True).float()
        feature_w = feature_w.flatten()
        if feature_w.numel() == X277_FEATURE_DIM:
            # X277 权重只覆盖真实动作特征；6 维缺失标签只作为条件，补 1 只是为了形状对齐。
            feature_w = torch.cat([feature_w, torch.ones(MODEL_INPUT_DIM - X277_FEATURE_DIM)])
        if feature_w.numel() != MODEL_INPUT_DIM:
            raise ValueError(
                f"feature_w 应为 {X277_FEATURE_DIM} 或 {MODEL_INPUT_DIM} 维，实际为 {feature_w.numel()} 维"
            )
        feature_w.requires_grad_(False)
        return feature_w

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if not resume_checkpoint:
            return

        self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
        logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
        self.model.load_state_dict(
            dist_util.load_state_dict(resume_checkpoint, map_location=self.device),
            strict=False,
        )

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
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
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
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
        last_step_end = time.perf_counter()
        for _epoch in range(self.num_epochs):
            for batch in self.data:
                batch_ready_time = time.perf_counter()
                batch = move_batch_to_device(batch, self.device)
                train_start_time = time.perf_counter()
                self.run_step(batch)
                train_end_time = time.perf_counter()
                print(
                    f"step[{self.step + self.resume_step}] "
                    f"data={batch_ready_time - last_step_end:.3f}s "
                    f"train={train_end_time - train_start_time:.3f}s",
                    flush=True,
                )
                last_step_end = train_end_time

                if self.step % self.log_interval == 0:
                    self.report_metrics()

                if self.step % self.save_interval == 0:
                    self.save()
                    self.model.eval()
                    self.evaluate()
                    self.model.train()

                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                self.step += 1
                if self._should_stop():
                    break

            if self._should_stop():
                break

        if (self.step - 1) % self.save_interval != 0:
            self.save()
            self.evaluate()

    def _should_stop(self) -> bool:
        global_step = self.step + self.resume_step
        if self.lr_anneal_steps and global_step >= self.lr_anneal_steps:
            return True
        return global_step >= self.num_steps

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        raise NotImplementedError("训练期评估尚未实现；请先关闭 --eval_during_training。")

    def run_step(self, batch):
        self.forward_backward(batch)

        if self.gradient_clip:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.scaler.step(self.opt)
        self.scaler.update()

        if self.ema_model is not None:
            self.ema_model.update()

        self.log_step()

    def forward_backward(self, batch):
        self.opt.zero_grad(set_to_none=True)

        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            for i in range(0, batch["x"].shape[0], self.microbatch):
                # 目前保持 StableMotion 的整 batch 训练语义，先不引入 microbatch 梯度累积。
                assert i == 0
                assert self.microbatch == self.batch_size

                sample = batch["x"]  # [B, 283, T]
                batch_size, channels, seq_len = sample.shape
                if channels != MODEL_INPUT_DIM:
                    raise ValueError(f"训练输入应为 [B, {MODEL_INPUT_DIM}, T]，实际为 {tuple(sample.shape)}")

                feature_w = self._feature_weights_for_batch(batch_size, seq_len)
                timesteps = torch.randint(
                    low=0,
                    high=self.diffusion.num_timesteps,
                    size=(batch_size,),
                    device=self.device,
                )
                model_kwargs = self.mask_manager(batch, sample)

                losses = self.diffusion.training_losses(
                    self.model,
                    sample,
                    timesteps,
                    model_kwargs=model_kwargs,
                    feature_w=feature_w,
                    snr_gamma=self.snr_gamma,
                    use_l1=self.use_l1,
                )
                loss = losses["loss"].mean()
                log_loss_dict(self.diffusion, timesteps, losses)
                self.scaler.scale(loss).backward()

    def _feature_weights_for_batch(self, batch_size: int, seq_len: int):
        if self.feature_w is None:
            return None
        return self.feature_w.to(self.device)[None, :, None].repeat(batch_size, 1, seq_len)

    def mask_manager(self, batch, sample):
        """
        构建 fix-only 的扩散条件。

        `inpaint_mask=True` 表示该位置需要加噪、预测并参与 loss；False 表示该位置是观测条件。
        其中 `[277:283)` 的 6 维传感器缺失标签必须一直是条件，不允许被模型预测。
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
            raise ValueError("训练 batch 缺少 inpaint_mask，请先生成 X277 传感器缺失任务。")
        inpaint_mask = inpaint_mask.bool()
        if inpaint_mask.shape != sample.shape:
            raise ValueError(f"inpaint_mask 应为 {tuple(sample.shape)}，实际为 {tuple(inpaint_mask.shape)}")

        inpaint_mask = inpaint_mask & valid_frame_mask.unsqueeze(1)
        inpaint_mask[:, X277_FEATURE_DIM:MODEL_INPUT_DIM, :] = False
        if not inpaint_mask.any():
            raise ValueError("当前 batch 的 inpaint_mask 没有待补全位置，请检查离线任务生成结果。")

        return {
            "inpaint_cond": inpaint_mask,
            "valid_frame_mask": valid_frame_mask,
            "attention_mask": valid_frame_mask,
            "y": {
                "mask": inpaint_mask,
                "inpainted_motion": sample,
            },
        }

    # endregion

    # region 日志与 checkpoint
    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

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
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def parse_resume_step_from_filename(filename):
    match = re.search(r"model(\d+)\.pt$", str(filename))
    if match is None:
        return 0
    return int(match.group(1))


def find_resume_checkpoint():
    return None


def log_loss_dict(diffusion, timesteps, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for timestep, loss in zip(timesteps.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * timestep / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", loss)
