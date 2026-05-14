import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from diffusion import logger
from diffusion.resample import create_named_schedule_sampler
from utils import dist_util


class TrainLoop:
    """DiffusionPoser 的最小训练循环，保留 StableMotion 的 diffusion loss 调用方式。"""

    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.device = dist_util.dev()
        self.step = 0
        self.resume_step = 0
        self.save_dir = Path(args.save_dir)
        self.num_steps = args.num_steps
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.gradient_clip = args.gradient_clip

        self.opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = GradScaler("cuda", enabled=self.device.type == "cuda")
        self.schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
        self.ema_model = self._create_ema_model(args)

        self._load_checkpoint_if_needed(args.resume_checkpoint)

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

    def _load_checkpoint_if_needed(self, checkpoint_path: str):
        if not checkpoint_path:
            return
        state_dict = dist_util.load_state_dict(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.resume_step = parse_resume_step_from_filename(checkpoint_path)

    def run_loop(self):
        self.model.train()
        while self.step + self.resume_step < self.num_steps:
            for batch in self.data:
                batch = move_batch_to_device(batch, self.device)
                self.run_step(batch)

                if self.step % self.log_interval == 0:
                    self.report_metrics()
                if self.step % self.save_interval == 0:
                    self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

                self.step += 1
                if self.step + self.resume_step >= self.num_steps:
                    break

        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        self.opt.zero_grad(set_to_none=True)
        sample = batch["x"]
        model_kwargs = self.mask_manager(batch, sample)
        timesteps, weights = self.schedule_sampler.sample(sample.shape[0], self.device)

        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            losses = self.diffusion.training_losses(
                self.model,
                sample,
                timesteps,
                model_kwargs=model_kwargs,
                snr_gamma=self.args.snr_gamma,
                use_l1=self.args.l1_loss,
            )
            loss = (losses["loss"] * weights).mean()

        self.scaler.scale(loss).backward()
        if self.gradient_clip:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        if self.ema_model is not None:
            self.ema_model.update()

        log_loss_dict(self.diffusion, timesteps, losses)
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * sample.shape[0])

    def mask_manager(self, batch, sample):
        batch_size, channels, seq_len = sample.shape
        valid_frame_mask = batch.get("valid_frame_mask", batch.get("attention_mask"))
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=sample.device)
        valid_frame_mask = valid_frame_mask.bool()

        inpaint_mask = batch.get("inpaint_mask")
        if inpaint_mask is None:
            raise ValueError("训练 batch 缺少 inpaint_mask，请先用 generate_x277_missing_tasks 生成离线缺失任务。")

        # 真实 X277 数据已经离线生成了精确 mask；这里只叠加有效帧，避免 padding 进入 loss。
        inpaint_mask = inpaint_mask.bool() & valid_frame_mask.unsqueeze(1)
        if inpaint_mask.shape != sample.shape:
            raise ValueError(f"inpaint_mask 应为 {tuple(sample.shape)}，实际为 {tuple(inpaint_mask.shape)}")
        if not inpaint_mask.any():
            raise ValueError("batch 中的 inpaint_mask 没有任何待补全位置，请检查离线任务生成。")

        return {
            "inpaint_cond": inpaint_mask,
            "valid_frame_mask": valid_frame_mask,
            "attention_mask": valid_frame_mask,
            "y": {
                "mask": inpaint_mask,
                "inpainted_motion": sample,
            },
        }

    def report_metrics(self):
        current = logger.get_current().name2val
        for key, value in current.items():
            if key not in {"step", "samples"} and "_q" not in key:
                self.train_platform.report_scalar(key, value, self.step + self.resume_step, group_name="Loss")
        logger.dumpkvs()

    def save(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        step = self.step + self.resume_step
        torch.save(self.model.state_dict(), self.save_dir / f"model{step:09d}.pt")
        torch.save(self.opt.state_dict(), self.save_dir / f"opt{step:09d}.pt")
        if self.ema_model is not None:
            torch.save(self.ema_model.state_dict(), self.save_dir / f"ema{step:09d}.pt")


def move_batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def parse_resume_step_from_filename(filename):
    stem = Path(filename).stem
    if not stem.startswith("model"):
        return 0
    try:
        return int(stem.replace("model", ""))
    except ValueError:
        return 0


def log_loss_dict(diffusion, timesteps, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for timestep, loss in zip(timesteps.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * timestep / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", loss)
