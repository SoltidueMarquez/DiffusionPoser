from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from data_loaders.realtime_pose_dataset import RealtimePoseTaskDataset
from data_loaders.realtime_pose_kinematics import fk_parent_local_torch, integrate_root_delta_xz_ref
from data_loaders.sensor_masking import (
    REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_TARGET_START,
    SchemaSpec,
    TRACKER_COUNT,
)
from sample.reconstruct_stream import build_realtime_inpaint_mask, reconstruct_batch, tensor_bct_to_numpy_btc
from sample.utils import load_checkpoint_model
from utils import dist_util
from utils.model_util import create_model_and_diffusion
from utils.parser_util import add_base_options, add_data_options, add_diffusion_options, add_model_options, add_sampling_options, parse_and_load_from_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run explicit realtime_pose_v1 rollout baseline.")
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_sampling_options(parser)
    parser.add_argument("--rollout_limit", default=0, type=int, help="最多处理多少个 materialized windows；0 表示全量。")
    return parser


def predicted_target_to_joints(
    predicted_target_raw: np.ndarray,
    task: dict[str, np.ndarray],
    schema: SchemaSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """把第 61 帧预测 target raw feature 转成 root yaw 和 FK joints。"""

    target_feature = np.asarray(predicted_target_raw, dtype=np.float32)
    prev_root_yaw = task["root_yaw"][REALTIME_POSE_TARGET_START - 1:REALTIME_POSE_TARGET_START].astype(np.float32)
    yaw_delta = target_feature[schema.root_yaw_delta_slice()]
    pred_root_yaw = prev_root_yaw + np.asarray([np.arctan2(yaw_delta[0], yaw_delta[1])], dtype=np.float32)

    root_pos = task["root_pos_world"][REALTIME_POSE_TARGET_START:REALTIME_POSE_TARGET_START + 1].astype(np.float32).copy()
    offsets = task["joint_offsets_parent"][None].astype(np.float32).copy()
    if schema.supports_root_motion:
        root_delta = target_feature[schema.root_delta_xz_slice()][None].astype(np.float32)
        prev_root_pos = task["root_pos_world"][REALTIME_POSE_TARGET_START - 1:REALTIME_POSE_TARGET_START].astype(np.float32)
        root_pos = integrate_root_delta_xz_ref(
            prev_root_pos_world=prev_root_pos,
            prev_root_yaw=prev_root_yaw,
            root_delta_xz_ref=root_delta,
        )
        root_pos[:, 1] = 0.0
        offsets[:, 0, 1] = float(target_feature[schema.root_height_slice()][0])

    with torch.no_grad():
        joints = fk_parent_local_torch(
            body_pose_parent_6d=torch.from_numpy(target_feature[schema.body_pose_slice()][None].astype(np.float32)),
            root_pos_world=torch.from_numpy(root_pos),
            root_yaw=torch.from_numpy(pred_root_yaw),
            parent_offsets=torch.from_numpy(offsets),
        )
    return pred_root_yaw.astype(np.float32), joints.numpy()[0].astype(np.float32)


def sorted_rollout_indices(dataset: RealtimePoseTaskDataset, limit: int = 0) -> list[int]:
    """按 source/start_frame 排序，让预测帧能写回后续重叠窗口 history。"""

    indices = sorted(
        range(len(dataset)),
        key=lambda index: (
            str(dataset.entries[index].get("source_path", "")),
            int(dataset.entries[index].get("start_frame", index)),
            index,
        ),
    )
    return indices if int(limit) <= 0 else indices[: int(limit)]


def source_key(entry: dict) -> str:
    return str(entry.get("source_path") or entry.get("source_relative_path") or "")


def inject_predicted_history(
    batch: dict,
    entry: dict,
    schema: SchemaSpec,
    predicted_history: dict[str, dict[int, np.ndarray]],
) -> None:
    """把已预测的绝对帧 target slice 写回当前窗口 history 条件。"""

    source = source_key(entry)
    if not source:
        return
    source_cache = predicted_history.get(source)
    if not source_cache:
        return
    start_frame = int(entry.get("start_frame", 0))
    conditioned = batch["conditioned_x"]
    for history_frame in range(REALTIME_POSE_TARGET_START):
        cached_target = source_cache.get(start_frame + history_frame)
        if cached_target is None:
            continue
        conditioned[:, schema.target_slice(), history_frame] = torch.as_tensor(
            cached_target,
            dtype=conditioned.dtype,
            device=conditioned.device,
        )


def rollout_dataset(
    model,
    diffusion,
    dataset: RealtimePoseTaskDataset,
    device: torch.device,
    use_ddim: bool,
    limit: int = 0,
) -> dict[str, np.ndarray]:
    """
    对 materialized windows 执行显式 rollout baseline。

    输出按“每个窗口的 target 帧”组织为 `[1, N, ...]`，N 是处理过的窗口数。
    当后续窗口 history 覆盖到已经预测过的绝对帧时，会用预测 target slice 替换 GT history。
    """

    schema = dataset.schema
    predicted_history: dict[str, dict[int, np.ndarray]] = {}
    reference_features = []
    predicted_features = []
    reference_joints = []
    predicted_joints = []
    root_yaw_reference = []
    root_yaw_predicted = []
    tracker_pos_ref = []
    sensor_valid = []

    for index in sorted_rollout_indices(dataset, limit=limit):
        entry = dataset.entries[index]
        item = dataset[index]
        batch = {key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value for key, value in item.items()}
        inject_predicted_history(
            batch=batch,
            entry=entry,
            schema=schema,
            predicted_history=predicted_history,
        )
        reconstructed = reconstruct_batch(
            model=model,
            diffusion=diffusion,
            batch=batch,
            device=device,
            use_ddim=use_ddim,
            schema_name=schema.name,
        )
        inpaint_mask = build_realtime_inpaint_mask(1, device, schema_name=schema.name)

        reference_normalized = tensor_bct_to_numpy_btc(batch["x"])
        predicted_normalized = tensor_bct_to_numpy_btc(torch.where(inpaint_mask, reconstructed, batch["conditioned_x"]))
        if dataset.normalizer is not None:
            reference_raw = dataset.normalizer.inverse(reference_normalized)
            predicted_raw = dataset.normalizer.inverse(predicted_normalized)
        else:
            reference_raw = reference_normalized
            predicted_raw = predicted_normalized

        task = dataset.load_task(index, entry)
        target_raw = predicted_raw[0, REALTIME_POSE_TARGET_START]
        pred_yaw, pred_joints = predicted_target_to_joints(target_raw, task=task, schema=schema)
        reference_features.append(reference_raw[0, REALTIME_POSE_TARGET_START])
        predicted_features.append(target_raw)
        reference_joints.append(task["joints_world"][REALTIME_POSE_TARGET_START].astype(np.float32))
        predicted_joints.append(pred_joints)
        root_yaw_reference.append(float(task["root_yaw"][REALTIME_POSE_TARGET_START]))
        root_yaw_predicted.append(float(pred_yaw[0]))
        tracker_pos_ref.append(
            reference_raw[0, REALTIME_POSE_TARGET_START, schema.tracker_pos_slice()].reshape(TRACKER_COUNT, 3)
        )
        sensor_valid.append(reference_raw[0, REALTIME_POSE_TARGET_START, schema.sensor_valid_slice()])

        source = source_key(entry)
        if source:
            target_abs_frame = int(entry.get("start_frame", 0)) + REALTIME_POSE_TARGET_START
            predicted_history.setdefault(source, {})[target_abs_frame] = (
                reconstructed[0, schema.target_slice(), REALTIME_POSE_TARGET_START]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )

    if not reference_features:
        raise RuntimeError("没有可 rollout 的 dataset window。")

    return {
        "reference_features_raw": np.asarray(reference_features, dtype=np.float32)[None],
        "predicted_features_raw": np.asarray(predicted_features, dtype=np.float32)[None],
        "reference_joints_world": np.asarray(reference_joints, dtype=np.float32)[None],
        "predicted_joints_world": np.asarray(predicted_joints, dtype=np.float32)[None],
        "root_yaw_reference": np.asarray(root_yaw_reference, dtype=np.float32)[None],
        "root_yaw_predicted": np.asarray(root_yaw_predicted, dtype=np.float32)[None],
        "tracker_pos_ref": np.asarray(tracker_pos_ref, dtype=np.float32)[None],
        "sensor_valid": np.asarray(sensor_valid, dtype=np.float32)[None],
        "metadata": np.asarray(
            {"schema_name": schema.name, "rollout_frames": int(len(reference_features))},
            dtype=object,
        ),
    }


def rollout_dataset_item(
    model,
    diffusion,
    dataset: RealtimePoseTaskDataset,
    device: torch.device,
    use_ddim: bool,
) -> dict[str, np.ndarray]:
    return rollout_dataset(
        model=model,
        diffusion=diffusion,
        dataset=dataset,
        device=device,
        use_ddim=use_ddim,
        limit=1,
    )


def save_rollout(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = build_arg_parser()
    args = parse_and_load_from_model(parser, argv=argv)
    if args.schema != REALTIME_POSE_SCHEMA_NAME:
        raise ValueError("reconstruct_rollout 当前只作为 v1 explicit baseline，请传 --schema realtime_pose_v1。")
    dist_util.setup_dist(args.device if args.cuda else -1)
    device = dist_util.dev()
    dataset = RealtimePoseTaskDataset(
        data_dir=args.data_dir,
        split=args.data_split,
        seq_len=args.seq_len,
        normalizer_dir=args.normalizer_dir,
        normalize_input=args.normalize_input,
        folder_path=getattr(args, "folder_path", "") or None,
        tracker_mask_policy="task",
        schema_name=args.schema,
    )
    model, diffusion = create_model_and_diffusion(args)
    model, source = load_checkpoint_model(model, args.model_path, device=device, use_ema=args.use_ema)
    payload = rollout_dataset(
        model=model,
        diffusion=diffusion,
        dataset=dataset,
        device=device,
        use_ddim=str(args.ts_respace).startswith("ddim"),
        limit=int(args.rollout_limit),
    )
    output_dir = Path(args.output_dir or "output/realtime_pose_v1_rollout").resolve()
    output_path = output_dir / "rollout_result.npz"
    save_rollout(output_path, payload)
    print(f"[reconstruct_rollout] weights={source} output={output_path}")
    return {"output_path": output_path}


if __name__ == "__main__":
    main()
