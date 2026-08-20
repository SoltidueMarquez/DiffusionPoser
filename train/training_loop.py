from __future__ import annotations

import copy
import re
from itertools import chain
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from data_loaders.realtime_pose_config import IKInpaintingConfig
from diffusion.realtime_pose_inpainting import build_current_realtime_pose_conditions
from utils.model_util import load_realtime_pose_predictor
from utils.normalizer import RealtimePoseNormalizer
from utils.training_precision import TrainingPrecision


class TrainLoop:
    """训练单帧 DiT；联合阶段额外以小学习率更新 Predictor。"""

    def __init__(self, args, train_platform, model, diffusion, data, eval_data=None):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.eval_data = eval_data
        self.device = next(model.parameters()).device
        self.precision = TrainingPrecision(
            getattr(args, "precision", "fp32"), self.device
        )
        self.save_dir = Path(args.save_dir)
        self.step = 0
        self.joint_finetune = bool(getattr(args, "joint_finetune", False))
        self.normalizer = RealtimePoseNormalizer(
            args.normalizer_dir, disable=not bool(args.normalize_input)
        )
        self.pose_mean = (
            None if self.normalizer.disable else self.normalizer.pose_mean.to(self.device)
        )
        self.pose_scale = (
            None if self.normalizer.disable else self.normalizer.pose_scale.to(self.device)
        )
        self.tracker_mean = (
            None
            if self.normalizer.disable
            else self.normalizer.tracker_mean.to(self.device)
        )
        self.tracker_scale = (
            None
            if self.normalizer.disable
            else (self.normalizer.tracker_std + self.normalizer.eps).to(self.device)
        )
        self.ik_config = IKInpaintingConfig(
            fabrik_iterations=int(args.fabrik_iterations),
            direction_only_quality=args.ik_direction_only_quality,
            residual_scale=args.ik_residual_scale,
            position_solved_quality=args.ik_position_solved_quality,
            gap_low=args.ik_gap_low,
            gap_high=args.ik_gap_high,
            direction_support=args.ik_direction_support,
            untracked_strength=args.ik_untracked_strength,
        ).validate()
        self.predictor = load_realtime_pose_predictor(
            args.predictor_model_path, self.device
        )
        if self.joint_finetune:
            self.predictor.train().requires_grad_(True)
        parameter_groups = [
            {
                "params": self.model.parameters(),
                "lr": float(args.lr),
            }
        ]
        if self.joint_finetune:
            parameter_groups.append(
                {
                    "params": self.predictor.parameters(),
                    "lr": float(args.predictor_lr),
                }
            )
        self.optimizer = AdamW(
            parameter_groups,
            weight_decay=float(args.weight_decay),
        )
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.ema_predictor = (
            copy.deepcopy(self.predictor).eval().requires_grad_(False)
            if self.joint_finetune
            else None
        )
        self.ema_decay = float(args.model_ema_decay)
        self.feature_w = self._load_feature_weights()
        self._load_resume()

    def _load_feature_weights(self) -> torch.Tensor | None:
        if not bool(self.args.weighted_loss):
            return None
        value = torch.load(
            Path(self.args.normalizer_dir) / self.args.feature_w_file,
            map_location=self.device,
            weights_only=True,
        ).float().flatten()
        if value.numel() != 144:
            raise ValueError("feature_w 必须为 [144]。")
        return value

    def _load_resume(self) -> None:
        requested = str(getattr(self.args, "resume_checkpoint", "") or "").strip()
        if not requested:
            return
        checkpoint = Path(find_resume_checkpoint(self.save_dir, requested))
        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=True)
        )
        self.step = parse_resume_step_from_filename(checkpoint)
        optimizer_path = checkpoint.with_name(f"opt{self.step:09d}.pt")
        ema_path = checkpoint.with_name(f"ema{self.step:09d}.pt")
        if optimizer_path.is_file():
            self.optimizer.load_state_dict(
                torch.load(optimizer_path, map_location=self.device, weights_only=True)
            )
        if ema_path.is_file():
            self.ema_model.load_state_dict(
                torch.load(ema_path, map_location=self.device, weights_only=True)
            )
        else:
            self.ema_model.load_state_dict(self.model.state_dict())
        if self.joint_finetune:
            predictor_path = checkpoint.with_name(f"predictor{self.step:09d}.pt")
            predictor_ema_path = checkpoint.with_name(
                f"predictor_ema{self.step:09d}.pt"
            )
            if not predictor_path.is_file():
                raise FileNotFoundError(f"联合 checkpoint 缺少 Predictor：{predictor_path}")
            self.predictor.load_state_dict(
                torch.load(predictor_path, map_location=self.device, weights_only=True)
            )
            if predictor_ema_path.is_file():
                self.ema_predictor.load_state_dict(
                    torch.load(
                        predictor_ema_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                )
            else:
                self.ema_predictor.load_state_dict(self.predictor.state_dict())

    def run_loop(self) -> None:
        iterator = iter(self.data)
        while self.step < int(self.args.num_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                batch_sampler = getattr(self.data, "batch_sampler", None)
                if hasattr(batch_sampler, "set_epoch"):
                    batch_sampler.set_epoch(self.step // max(len(self.data), 1) + 1)
                iterator = iter(self.data)
                batch = next(iterator)
            batch = move_batch_to_device(batch, self.device)
            losses = self.run_step(batch)
            self.step += 1
            if int(self.args.log_interval) > 0 and self.step % int(self.args.log_interval) == 0:
                self._report(losses, "train")
            if (
                bool(self.args.eval_during_training)
                and int(self.args.save_interval) > 0
                and self.step % int(self.args.save_interval) == 0
            ):
                self.evaluate()
            if int(self.args.save_interval) > 0 and self.step % int(self.args.save_interval) == 0:
                self.save()
        self.save()

    def run_step(self, batch: dict) -> dict[str, torch.Tensor]:
        self.model.train()
        if self.joint_finetune:
            self.predictor.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self._forward_losses(batch)
        losses["loss"].mean().backward()
        if bool(self.args.gradient_clip):
            parameters = (
                chain(self.model.parameters(), self.predictor.parameters())
                if self.joint_finetune
                else self.model.parameters()
            )
            clip_grad_norm_(parameters, 1.0)
        self.optimizer.step()
        self._update_ema()
        if not self.joint_finetune and any(
            parameter.grad is not None for parameter in self.predictor.parameters()
        ):
            raise RuntimeError("冻结 Predictor 不应产生梯度。")
        return losses

    def _forward_losses(
        self,
        batch: dict,
        *,
        model_override: torch.nn.Module | None = None,
        predictor_override: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        motion_context = batch["motion_context"]
        # Predictor 与 DiT 共享 Task Store 中的干净完整历史；Predictor 训练时
        # 已用 0～30 步闭环回填覆盖部署历史分布，不再额外构造人工历史噪声。
        predictor_model = (
            self.predictor if predictor_override is None else predictor_override
        )
        if self.joint_finetune and predictor_override is None:
            predictor_pose_horizon = self.precision.forward(
                predictor_model,
                motion_context, batch["core_tracker_context"]
            )
        else:
            with torch.no_grad():
                predictor_pose_horizon = self.precision.forward(
                    predictor_model,
                    motion_context,
                    batch["core_tracker_context"],
                )
        # IK 与扩散监督描述的是当前 Predictor 给出的固定基线。联合阶段不让
        # Predictor 通过移动 target 或 IK 条件来降低 diffusion loss。
        conditioning_horizon = (
            predictor_pose_horizon.detach()
            if self.joint_finetune
            else predictor_pose_horizon
        )
        initial_pose_raw = self._inverse_pose(conditioning_horizon[:, 0])
        _, ik_condition, tracker_geometry = (
            build_current_realtime_pose_conditions(
                initial_pose_raw=initial_pose_raw,
                current_tracker_raw=batch["current_tracker_raw"],
                joint_offsets_parent=batch["joint_offsets_parent"],
                pose_mean=self.pose_mean,
                pose_scale=self.pose_scale,
                tracker_mean=self.tracker_mean,
                tracker_scale=self.tracker_scale,
                config=self.ik_config,
            )
        )
        loss_batch = dict(batch)
        if self.pose_mean is not None:
            loss_batch["pose_mean"] = self.pose_mean
            loss_batch["pose_scale"] = self.pose_scale
        timestep = torch.randint(
            0,
            self.diffusion.num_timesteps,
            (batch["x"].shape[0],),
            device=self.device,
        )
        model_kwargs = {
            "motion_context": motion_context,
            "predictor_pose_horizon": conditioning_horizon,
            "tracker_geometry": tracker_geometry,
            "tracker_available": batch["tracker_available"].bool(),
            "ik_residual": ik_condition.ik_residual,
            "ik_gap": ik_condition.ik_gap,
            "ik_confidence": ik_condition.ik_confidence,
            "denoise_strength": ik_condition.denoise_strength,
            "constraint_type": ik_condition.constraint_type,
            "y": loss_batch,
        }
        predictor_current = predictor_pose_horizon[:, 0]
        diffusion_target = batch["x"] - predictor_current.detach()
        model = self.model if model_override is None else model_override
        # diffusion 只看到 FP32 输出，BF16 的数值边界严格限制在 DiT forward。
        def model_forward(*model_args, **model_kwargs):
            return self.precision.forward(
                model, *model_args, **model_kwargs
            )

        terms = self.diffusion.training_losses(
            model_forward,
            diffusion_target,
            timestep,
            model_kwargs=model_kwargs,
            predictor_current=predictor_current,
            feature_w=self.feature_w,
            snr_gamma=float(self.args.snr_gamma),
            use_l1=bool(self.args.l1_loss),
        )
        if self.joint_finetune:
            predictor_pose_loss = (
                predictor_current - batch["x"]
            ).square().mean(dim=1)
            terms["predictor_pose_loss"] = predictor_pose_loss
            terms["loss"] = terms["loss"] + float(
                self.args.predictor_loss_weight
            ) * predictor_pose_loss
        return terms

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        if self.eval_data is None:
            return {}
        totals: dict[str, float] = {}
        count = 0
        self.ema_model.eval()
        if self.ema_predictor is not None:
            self.ema_predictor.eval()
        for batch in self.eval_data:
            batch = move_batch_to_device(batch, self.device)
            losses = self._forward_losses(
                batch,
                model_override=self.ema_model,
                predictor_override=self.ema_predictor,
            )
            for name, value in losses.items():
                if torch.is_tensor(value) and value.ndim <= 1:
                    totals[name] = totals.get(name, 0.0) + float(value.mean())
            count += 1
            if count >= int(self.args.eval_num_batches):
                break
        metrics = {name: value / max(count, 1) for name, value in totals.items()}
        self._report(metrics, "validation")
        return metrics

    def _inverse_pose(self, value: torch.Tensor) -> torch.Tensor:
        if self.pose_mean is None:
            return value
        return value * self.pose_scale + self.pose_mean

    @torch.no_grad()
    def _update_ema(self) -> None:
        _update_model_ema(self.ema_model, self.model, self.ema_decay)
        if self.ema_predictor is not None:
            _update_model_ema(
                self.ema_predictor,
                self.predictor,
                self.ema_decay,
            )

    def _report(self, losses: dict, group: str) -> None:
        values = {}
        for name, value in losses.items():
            if torch.is_tensor(value):
                if value.ndim > 1:
                    continue
                scalar = float(value.mean().detach())
            else:
                scalar = float(value)
            values[name] = scalar
            self.train_platform.report_scalar(name, scalar, self.step, group)
        stage = "joint" if self.joint_finetune else "dit"
        print(f"{stage} {group} step[{self.step}]: {values}", flush=True)

    def save(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_dir / f"model{self.step:09d}.pt")
        torch.save(self.ema_model.state_dict(), self.save_dir / f"ema{self.step:09d}.pt")
        torch.save(self.optimizer.state_dict(), self.save_dir / f"opt{self.step:09d}.pt")
        if self.joint_finetune:
            torch.save(
                self.predictor.state_dict(),
                self.save_dir / f"predictor{self.step:09d}.pt",
            )
            torch.save(
                self.ema_predictor.state_dict(),
                self.save_dir / f"predictor_ema{self.step:09d}.pt",
            )
            # latest 始终保存同一步的两份 EMA，运行时必须成对使用。
            torch.save(self.ema_model.state_dict(), self.save_dir / "model_latest.pt")
            torch.save(
                self.ema_predictor.state_dict(),
                self.save_dir / "predictor_latest.pt",
            )


@torch.no_grad()
def _update_model_ema(
    target: torch.nn.Module,
    online: torch.nn.Module,
    decay: float,
) -> None:
    online_state = online.state_dict()
    for name, value in target.state_dict().items():
        source = online_state[name].detach()
        if value.is_floating_point():
            value.lerp_(source, 1.0 - decay)
        else:
            value.copy_(source)


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


def parse_resume_step_from_filename(filename: str | Path) -> int:
    match = re.search(r"model(\d+)\.pt$", str(filename))
    return int(match.group(1)) if match else 0


def find_latest_model_checkpoint(save_dir: str | Path) -> Path | None:
    candidates = sorted(Path(save_dir).glob("model[0-9]*.pt"))
    return candidates[-1] if candidates else None


def find_latest_run_dir(save_dir: str | Path) -> Path | None:
    root = Path(save_dir)
    candidates = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    return candidates[-1] if candidates else None


def find_resume_checkpoint(
    save_dir: str | Path,
    requested_checkpoint: str | Path | None = "",
) -> str:
    requested = str(requested_checkpoint or "").strip()
    if not requested:
        return ""
    if requested.lower() not in {"latest", "auto"}:
        path = Path(requested).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在：{path}")
        return str(path)
    checkpoint = find_latest_model_checkpoint(save_dir)
    if checkpoint is None:
        latest_run = find_latest_run_dir(save_dir)
        checkpoint = (
            find_latest_model_checkpoint(latest_run) if latest_run is not None else None
        )
    if checkpoint is None:
        raise FileNotFoundError(f"没有可恢复的 model*.pt：{save_dir}")
    return str(checkpoint)
