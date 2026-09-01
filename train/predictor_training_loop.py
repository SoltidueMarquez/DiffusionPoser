from __future__ import annotations

import copy
import re
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from data_loaders.realtime_pose_predictor_features import (
    build_predictor_sparse_availability_mask_torch,
    build_predictor_step_features_torch,
    pose_head_to_world_rotations_torch,
)
from data_loaders.realtime_pose_kinematics import rotation_6d_to_matrix_torch
from data_loaders.sensor_masking import PREDICTOR_FREE_RUNNING_MAX_STEPS
from train.predictor_losses import compute_predictor_losses
from utils.normalizer import RealtimePoseNormalizer
from utils.training_precision import TrainingPrecision


PREDICTOR_DEVICE_FIELDS = frozenset(
    {
        "joint_rotations_world_6d",
        "tracker_positions_world",
        "tracker_rotations_world_6d",
        "floor_y",
        "joint_offsets_parent",
        "tracker_available",
        "current_frame",
    }
)


class PredictorTrainLoop:
    """单阶段 Predictor 训练；每批均匀采样 0～30 步闭环历史。"""

    def __init__(self, args, model: torch.nn.Module, train_data, device):
        self.args = args
        self.model = model
        self.train_data = train_data
        self.device = torch.device(device)
        self.precision = TrainingPrecision(
            getattr(args, "precision", "fp32"), self.device
        )
        self.save_dir = Path(args.save_dir)
        self.normalizer = RealtimePoseNormalizer(args.normalizer_dir)
        self.pose_mean = self.normalizer.pose_mean.to(self.device)
        self.pose_scale = self.normalizer.pose_scale.to(self.device)
        self.sparse_mean = self.normalizer.predictor_sparse_mean.to(self.device)
        self.sparse_std = self.normalizer.predictor_sparse_std.to(self.device)
        self.free_running_max_steps = PREDICTOR_FREE_RUNNING_MAX_STEPS
        self.base_lr = float(args.lr)
        self.lr_drop_step = int(args.lr_drop_step)
        self.lr_drop_factor = float(args.lr_drop_factor)
        if self.lr_drop_step < 0:
            raise ValueError("lr_drop_step 必须大于等于 0。")
        if self.lr_drop_factor < 1.0:
            raise ValueError("lr_drop_factor 必须大于等于 1。")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=float(args.weight_decay),
        )
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema_decay = float(args.ema_decay)
        self.checkpoint_max_keep = max(
            0, int(getattr(args, "checkpoint_max_keep", 3))
        )
        self.step = 0
        self._load_initial_state()

    def _load_initial_state(self) -> None:
        requested = str(getattr(self.args, "resume_checkpoint", "") or "").strip()
        if requested:
            checkpoint = resolve_predictor_resume_checkpoint(self.save_dir, requested)
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
            match = re.search(r"model(\d+)\.pt$", checkpoint.name)
            self.step = int(match.group(1)) if match else 0
            optimizer_path = checkpoint.with_name(f"opt{self.step:09d}.pt")
            ema_path = checkpoint.with_name(f"ema{self.step:09d}.pt")
            if optimizer_path.is_file():
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device, weights_only=True))
            if ema_path.is_file():
                self.ema_model.load_state_dict(torch.load(ema_path, map_location=self.device, weights_only=True))
            else:
                self.ema_model.load_state_dict(self.model.state_dict())
        self._update_learning_rate()

    def run(self) -> None:
        iterator = iter(self.train_data)
        while self.step < int(self.args.num_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.train_data)
                batch = next(iterator)
            batch = move_predictor_batch_to_device(batch, self.device)
            rollout_steps = self.sample_rollout_steps()
            self._update_learning_rate()
            losses = self.train_step(batch, rollout_steps)
            self.step += 1
            if int(self.args.log_interval) > 0 and self.step % int(self.args.log_interval) == 0:
                values = ", ".join(
                    f"{name}={float(value.mean().detach()):.6f}"
                    for name, value in losses.items()
                )
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"predictor step[{self.step}] fr={rollout_steps} lr={lr:.8g} {values}",
                    flush=True,
                )
            if int(self.args.save_interval) > 0 and self.step % int(self.args.save_interval) == 0:
                self.save()
        self.save()

    def sample_rollout_steps(self) -> int:
        """每个 batch 独立均匀采样闭环步数，包含 0 与 30 两端。"""

        return int(
            torch.randint(
                0,
                self.free_running_max_steps + 1,
                (),
                device=self.device,
            ).item()
        )

    def _update_learning_rate(self) -> None:
        lr = self.base_lr
        if self.lr_drop_step > 0 and self.step >= self.lr_drop_step:
            lr /= self.lr_drop_factor
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def train_step(self, batch: dict, rollout_steps: int) -> dict[str, torch.Tensor]:
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        losses = self._forward_loss(batch, int(rollout_steps), gradient=True)
        losses["loss"].mean().backward()
        if float(self.args.gradient_clip_norm) > 0.0:
            clip_grad_norm_(self.model.parameters(), float(self.args.gradient_clip_norm))
        self.optimizer.step()
        self._update_ema()
        return losses

    def _forward_loss(
        self,
        batch: dict,
        rollout_steps: int,
        *,
        gradient: bool,
        model_override: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        if not 0 <= int(rollout_steps) <= PREDICTOR_FREE_RUNNING_MAX_STEPS:
            raise ValueError("rollout_steps 必须位于 [0,30]。")
        model = self.model if model_override is None else model_override
        # Dataset 常驻 6D 以节省内存；batch 到设备后再重建合法 SO(3) 矩阵。
        rotations = rotation_6d_to_matrix_torch(
            batch["joint_rotations_world_6d"]
        )
        tracker_positions = batch["tracker_positions_world"]
        tracker_rotations = batch["tracker_rotations_world_6d"]
        tracker_available = batch["tracker_available"]
        floor_y = batch["floor_y"]
        motion_world = rotations[:, 1:11].clone()

        if rollout_steps:
            model.eval()
            with torch.no_grad():
                for step_index in range(int(rollout_steps)):
                    motion, sparse, _, head_yaw = build_predictor_step_features_torch(
                        motion_world,
                        tracker_positions[:, step_index : step_index + 12],
                        tracker_rotations[:, step_index : step_index + 12],
                        floor_y[:, step_index + 11],
                    )
                    prediction = self.precision.forward(
                        model,
                        self._normalize_pose(motion),
                        self._normalize_sparse(
                            sparse,
                            tracker_available[
                                :, step_index : step_index + 12
                            ],
                        ),
                    )
                    motion_world = append_predictor_current_prediction(
                        motion_world=motion_world,
                        prediction_normalized=prediction,
                        head_yaw_world=head_yaw,
                        pose_mean=self.pose_mean,
                        pose_scale=self.pose_scale,
                    )

        step_index = int(rollout_steps)
        model.train(bool(gradient))
        context = torch.enable_grad() if gradient else torch.no_grad()
        with context:
            motion, sparse, target, _ = build_predictor_step_features_torch(
                motion_world,
                tracker_positions[:, step_index : step_index + 12],
                tracker_rotations[:, step_index : step_index + 12],
                floor_y[:, step_index + 11],
                rotations[:, step_index + 11 : step_index + 22],
            )
            if target is None:
                raise RuntimeError("Predictor 监督 target 不应为空。")
            motion_normalized = self._normalize_pose(motion)
            prediction = self.precision.forward(
                model,
                motion_normalized,
                self._normalize_sparse(
                    sparse,
                    tracker_available[:, step_index : step_index + 12],
                ),
            )
            return compute_predictor_losses(
                prediction_normalized=prediction,
                target_normalized=self._normalize_pose(target),
                motion_context_normalized=motion_normalized,
                joint_offsets_parent=batch["joint_offsets_parent"],
                pose_mean=self.pose_mean,
                pose_scale=self.pose_scale,
            )

    def _normalize_pose(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.pose_mean) / self.pose_scale

    def _inverse_pose(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.pose_scale + self.pose_mean

    def _normalize_sparse(
        self,
        value: torch.Tensor,
        tracker_available_with_previous: torch.Tensor,
    ) -> torch.Tensor:
        normalized = (value - self.sparse_mean) / (
            self.sparse_std + self.normalizer.eps
        )
        available = build_predictor_sparse_availability_mask_torch(
            tracker_available_with_previous
        )
        # 与 runtime 一致，缺失观测使用归一化域零向量；速度通道在掉线和
        # 重连首帧都关闭，避免模型把跨 gap 差分误认为真实高速运动。
        return torch.where(available, normalized, torch.zeros_like(normalized))

    @torch.no_grad()
    def _update_ema(self) -> None:
        online = self.model.state_dict()
        for name, value in self.ema_model.state_dict().items():
            source = online[name].detach()
            if value.is_floating_point():
                value.lerp_(source, 1.0 - self.ema_decay)
            else:
                value.copy_(source)

    def save(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_dir / f"model{self.step:09d}.pt")
        torch.save(self.ema_model.state_dict(), self.save_dir / f"ema{self.step:09d}.pt")
        torch.save(self.optimizer.state_dict(), self.save_dir / f"opt{self.step:09d}.pt")
        # 下游只需要一份稳定路径；latest 使用更平滑的 EMA 权重作为推理模型。
        torch.save(self.ema_model.state_dict(), self.save_dir / "model_latest.pt")
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self) -> None:
        """按训练 step 成组保留最近 N 份 model/ema/opt checkpoint。"""

        if self.checkpoint_max_keep <= 0 or not self.save_dir.exists():
            return
        pattern = re.compile(r"^(?:model|ema|opt)(\d{9})\.pt$")
        step_to_files: dict[int, list[Path]] = {}
        for path in self.save_dir.iterdir():
            match = pattern.match(path.name)
            if match is not None:
                step_to_files.setdefault(int(match.group(1)), []).append(path)
        expired_steps = sorted(step_to_files, reverse=True)[self.checkpoint_max_keep :]
        for step in expired_steps:
            for path in step_to_files[step]:
                path.unlink()


def move_predictor_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if key in PREDICTOR_DEVICE_FIELDS and torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def append_predictor_current_prediction(
    *,
    motion_world: torch.Tensor,
    prediction_normalized: torch.Tensor,
    head_yaw_world: torch.Tensor,
    pose_mean: torch.Tensor,
    pose_scale: torch.Tensor,
) -> torch.Tensor:
    """只把 Predictor horizon 0 转回 world 并追加到下一步历史。"""

    if prediction_normalized.ndim != 3 or tuple(prediction_normalized.shape[1:]) != (
        11,
        144,
    ):
        raise ValueError("prediction_normalized 必须为 [B,11,144]。")
    current_raw = prediction_normalized[:, 0] * pose_scale + pose_mean
    current_world = pose_head_to_world_rotations_torch(current_raw, head_yaw_world)
    return torch.cat([motion_world[:, 1:], current_world[:, None]], dim=1)


def resolve_predictor_resume_checkpoint(
    save_dir: str | Path,
    requested_checkpoint: str | Path,
) -> Path:
    """解析 Predictor 恢复点；`latest` 只选择带 step 的训练权重。"""

    requested = str(requested_checkpoint).strip()
    if requested.lower() not in {"latest", "auto"}:
        checkpoint = Path(requested).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Predictor checkpoint 不存在：{checkpoint}")
        if re.fullmatch(r"model\d{9}\.pt", checkpoint.name) is None:
            raise ValueError(
                "恢复训练必须指定带 9 位 step 的 modelXXXXXXXXX.pt；"
                "model_latest.pt 仅用于推理。"
            )
        return checkpoint
    pattern = re.compile(r"model\d{9}\.pt")
    candidates = sorted(
        path
        for path in Path(save_dir).resolve().glob("model*.pt")
        if pattern.fullmatch(path.name) is not None
    )
    if not candidates:
        raise FileNotFoundError(f"没有可恢复的 Predictor model*.pt：{save_dir}")
    return candidates[-1]
