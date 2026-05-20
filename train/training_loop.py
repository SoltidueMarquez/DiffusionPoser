import os
import re
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from diffusion import logger
from data_loaders.sensor_masking import MODEL_INPUT_DIM, TASK_MODE_FULL_RECONSTRUCTION_CURRENT, X277_FEATURE_DIM
from utils import dist_util


class TrainLoop:
    """
    DiffusionPoser 的 current277 扩散训练循环。

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
        self.task_mode = getattr(args, "task_mode", TASK_MODE_FULL_RECONSTRUCTION_CURRENT)
        self.checkpoint_max_keep = max(0, int(args.checkpoint_max_keep))

        self.save_dir = Path(args.save_dir)
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // max(1, len(self.data)) + 1
        self.device = dist_util.dev()
        self._eval_skip_logged = False
        logger.log(f"training device: {self.device}")
        logger.log(f"task mode: {self.task_mode}")
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
                "恢复训练 checkpoint 与当前模型结构不匹配，已停止以避免继续训练错误实验。"
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
        if self._should_stop():
            logger.log(
                f"resume step {self.resume_step} has already reached num_steps={self.num_steps}; skip training."
            )
            return

        last_step_end = time.perf_counter()
        for _epoch in range(self.num_epochs):
            for batch in self.data:
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
        if self.lr_anneal_steps and global_step >= self.lr_anneal_steps:
            return True
        return global_step >= self.num_steps

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        if not self._eval_skip_logged:
            logger.log("训练期评估入口尚未接入采样评估流程，本次训练将跳过 --eval_during_training。")
            self._eval_skip_logged = True

    def run_step(self, batch):
        self.forward_backward(batch)

        if self.gradient_clip:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.scaler.step(self.opt)
        self.scaler.update()

        if self.ema_model is not None:
            self.ema_model.update()

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
        根据 task_mode 生成扩散条件与 loss mask。

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

        conditioned_sample = batch.get("conditioned_x", sample)
        if conditioned_sample.shape != sample.shape:
            raise ValueError(
                f"conditioned_x 应为 {tuple(sample.shape)}，实际为 {tuple(conditioned_sample.shape)}"
            )

        return {
            "inpaint_cond": inpaint_mask,
            "valid_frame_mask": valid_frame_mask,
            "attention_mask": valid_frame_mask,
            "y": {
                "mask": inpaint_mask,
                "inpainted_motion": conditioned_sample,
            },
        }

    # endregion

    # region 日志与 checkpoint
    def log_step(self):
        global_step = self.step + self.resume_step
        logger.logkv("step", global_step)
        logger.logkv("samples", global_step * self.global_batch)

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


def find_resume_checkpoint(save_dir: str | Path, requested_checkpoint: str | Path | None = "") -> str:
    """
    解析恢复训练用的主模型 checkpoint。

    - `requested_checkpoint` 为空：表示用户明确要从头训练，不自动恢复。
    - `requested_checkpoint` 为 `latest`/`auto`：从 `save_dir` 扫描最新的 `model*.pt`。
    - `requested_checkpoint` 是具体路径但文件不存在：回退到 `save_dir` 中最新的 checkpoint，
      这样 VSCode 配置里写死的旧 checkpoint 被清理后仍然能继续训练。
    """

    requested_text = "" if requested_checkpoint is None else str(requested_checkpoint).strip()
    if not requested_text:
        return ""

    latest_checkpoint = find_latest_model_checkpoint(save_dir)
    if requested_text.lower() in {"latest", "auto"}:
        if latest_checkpoint is None:
            raise FileNotFoundError(f"save_dir 中没有可恢复的 model*.pt checkpoint：{Path(save_dir)}")
        logger.log(f"auto resume from latest checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    requested_path = Path(requested_text).expanduser()
    if requested_path.exists():
        return str(requested_path)

    if latest_checkpoint is not None:
        logger.log(
            f"requested resume checkpoint not found: {requested_path}; "
            f"fallback to latest checkpoint: {latest_checkpoint}"
        )
        return str(latest_checkpoint)

    raise FileNotFoundError(
        f"--resume_checkpoint 指向的文件不存在：{requested_path}；"
        f"并且 save_dir 中也没有可恢复的 model*.pt checkpoint：{Path(save_dir)}"
    )


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


def log_loss_dict(diffusion, timesteps, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for timestep, loss in zip(timesteps.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * timestep / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", loss)
