import json
import os
import re
from datetime import datetime
from pathlib import Path

import torch

from data_loaders.get_data import get_dataset_loader
from data_loaders.sensor_masking import POSE_REPRESENTATION_KEY, REALTIME_POSE_SCHEMA_NAME, get_schema_spec
from diffusion import logger
from train.train_platforms import NoPlatform, TensorboardPlatform
from train.training_loop import TrainLoop, find_resume_checkpoint
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.parser_util import train_args
from utils.run_dirs import resolve_latest_or_self


TRAIN_PLATFORMS = {
    "NoPlatform": NoPlatform,
    "TensorboardPlatform": TensorboardPlatform,
}


def main():
    args = train_args()
    fixseed(args.seed)
    resolve_save_dir(args)
    resolve_input_artifact_dirs(args)
    prepare_save_dir(args)
    dist_util.setup_dist(args.device if args.cuda else -1)
    logger.configure(dir=args.save_dir)
    torch.backends.cudnn.benchmark = True

    train_platform = TRAIN_PLATFORMS[args.train_platform_type](args.save_dir)
    try:
        train_platform.report_args(args, name="Args")
        save_args(args)
        enable_rollout_training = (
            args.rollout_steps > 1
            and args.rollout_loss_weight > 0.0
            and args.rollout_prob > 0.0
        )

        print("creating data loader...")
        data = get_dataset_loader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            input_feats=args.input_feats,
            seq_len=args.seq_len,
            split=args.data_split,
            normalizer_dir=args.normalizer_dir,
            normalize_input=args.normalize_input,
            preload_data=args.preload_data,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
            schema_name=args.schema,
            tracker_pos_noise_std=args.tracker_pos_noise_std,
            tracker_rot_noise_std=args.tracker_rot_noise_std,
            non_hip_tracker_dropout_prob=args.non_hip_tracker_dropout_prob,
            history_pose_noise_std=args.history_pose_noise_std,
            history_yaw_noise_std=args.history_yaw_noise_std,
            root_yaw_ref_noise_std=args.root_yaw_ref_noise_std,
            history_pose_dropout_prob=args.history_pose_dropout_prob,
            history_pose_replace_prob=args.history_pose_replace_prob,
            history_yaw_replace_prob=args.history_yaw_replace_prob,
            history_root_yaw_drift_std=args.history_root_yaw_drift_std,
            tracker_latency_max_frames=args.tracker_latency_max_frames,
            tracker_burst_dropout_prob=args.tracker_burst_dropout_prob,
            tracker_outlier_prob=args.tracker_outlier_prob,
            predicted_history_cache_dir=args.predicted_history_cache_dir or None,
            predicted_history_prob=args.predicted_history_prob,
            enable_rollout=enable_rollout_training,
            rollout_steps=args.rollout_steps,
            tracker_mask_policy=args.tracker_mask_policy,
            tracker_mask_seed=args.tracker_mask_seed,
            tracker_mask_fill=args.tracker_mask_fill,
            tracker_mask_categories=args.tracker_mask_categories,
        )
        eval_data = None
        if args.eval_during_training:
            print("creating eval data loader...")
            eval_data = get_dataset_loader(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                input_feats=args.input_feats,
                seq_len=args.seq_len,
                split=args.eval_split,
                normalizer_dir=args.normalizer_dir,
                normalize_input=args.normalize_input,
                preload_data=args.preload_data,
                num_workers=args.num_workers,
                pin_memory=args.cuda,
                schema_name=args.schema,
                tracker_pos_noise_std=args.tracker_pos_noise_std,
                tracker_rot_noise_std=args.tracker_rot_noise_std,
                non_hip_tracker_dropout_prob=args.non_hip_tracker_dropout_prob,
                history_pose_noise_std=args.history_pose_noise_std,
                history_yaw_noise_std=args.history_yaw_noise_std,
                root_yaw_ref_noise_std=args.root_yaw_ref_noise_std,
                history_pose_dropout_prob=args.history_pose_dropout_prob,
                history_pose_replace_prob=args.history_pose_replace_prob,
                history_yaw_replace_prob=args.history_yaw_replace_prob,
                history_root_yaw_drift_std=args.history_root_yaw_drift_std,
                tracker_latency_max_frames=args.tracker_latency_max_frames,
                tracker_burst_dropout_prob=args.tracker_burst_dropout_prob,
                tracker_outlier_prob=args.tracker_outlier_prob,
                predicted_history_cache_dir=args.predicted_history_cache_dir or None,
                predicted_history_prob=args.predicted_history_prob,
                enable_rollout=False,
                rollout_steps=1,
                tracker_mask_policy=args.tracker_mask_policy,
                tracker_mask_seed=args.tracker_mask_seed,
                tracker_mask_fill=args.tracker_mask_fill,
                tracker_mask_categories=args.tracker_mask_categories,
            )

        print("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args)
        model.to(dist_util.dev())
        print(f"Total params: {model.num_parameters() / 1_000_000.0:.2f}M")

        print(f"training DiffusionPoser model, task_mode={args.task_mode}...")
        TrainLoop(args, train_platform, model, diffusion, data, eval_data=eval_data).run_loop()
    finally:
        train_platform.close()


def prepare_save_dir(args):
    save_dir = Path(args.save_dir)
    if save_dir.exists() and not args.overwrite and not args.resume_checkpoint:
        raise FileExistsError(
            f"save_dir [{save_dir}] already exists. "
            "For a fresh run, choose a new --save_dir or pass --overwrite to reuse it. "
            "To continue training, pass --resume_checkpoint latest."
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume_checkpoint:
        write_latest_run_pointer(args)


def resolve_save_dir(args):
    """把用户给的 save_dir 解析成本次训练实际写入的 run 目录。"""

    if args.resume_checkpoint:
        args.resume_checkpoint = find_resume_checkpoint(
            save_dir=args.save_dir,
            requested_checkpoint=args.resume_checkpoint,
        )
        args.save_dir = str(Path(args.resume_checkpoint).resolve().parent)
        return

    run_root = Path(args.save_dir).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = resolve_run_label(args)
    candidate = run_root / f"{run_id}_{run_label}"
    suffix = 2
    while candidate.exists():
        candidate = run_root / f"{run_id}_{run_label}_{suffix:02d}"
        suffix += 1
    args.run_root = str(run_root)
    args.run_id = run_id
    args.save_dir = str(candidate)


def resolve_input_artifact_dirs(args):
    """把 data_dir/normalizer_dir 根目录解析到 latest 指向的实际产物目录。"""

    args.data_dir = str(resolve_latest_or_self(args.data_dir, kind="tasks"))
    if getattr(args, "normalizer_dir", ""):
        args.normalizer_dir = str(resolve_latest_or_self(args.normalizer_dir, kind="normalizer"))


def resolve_run_label(args) -> str:
    run_name = str(getattr(args, "run_name", "auto") or "auto").strip()
    if run_name.lower() in {"auto", ""}:
        run_name = f"{getattr(args, 'schema', 'schema')}_{getattr(args, 'model_arch', 'model')}_seed{getattr(args, 'seed', 0)}"
    run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("._-")
    return run_name or "run"


def write_latest_run_pointer(args):
    """在 run 根目录留下稳定指针，方便脚本和 AI 快速定位最近一次训练。"""

    run_root = Path(getattr(args, "run_root", Path(args.save_dir).parent)).resolve()
    save_dir = Path(args.save_dir).resolve()
    schema = get_schema_spec(getattr(args, "schema", REALTIME_POSE_SCHEMA_NAME))
    run_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "save_dir": str(save_dir),
        "run_root": str(run_root),
        "run_id": getattr(args, "run_id", ""),
        "run_name": getattr(args, "run_name", "auto"),
        "schema": schema.name,
        "schema_name": schema.name,
        "schema_canonical_name": str(schema.canonical_name),
        POSE_REPRESENTATION_KEY: schema.pose_representation,
        "root_y_policy": schema.root_y_policy,
        "pelvis_height_mode": schema.pelvis_height_mode,
        "model_arch": getattr(args, "model_arch", ""),
        "seed": getattr(args, "seed", None),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_root / "latest_run.txt").write_text(str(save_dir), encoding="utf-8")
    with (run_root / "latest_run.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)


def save_args(args):
    args_file = "resume_args.json" if args.resume_checkpoint else "args.json"
    args_path = os.path.join(args.save_dir, args_file)
    payload = vars(args).copy()
    schema = get_schema_spec(payload.get("schema", REALTIME_POSE_SCHEMA_NAME))
    payload["schema"] = schema.name
    payload["schema_name"] = schema.name
    payload["schema_canonical_name"] = str(schema.canonical_name)
    payload[POSE_REPRESENTATION_KEY] = schema.pose_representation
    payload["root_y_policy"] = schema.root_y_policy
    payload["pelvis_height_mode"] = schema.pelvis_height_mode
    with open(args_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    main()
