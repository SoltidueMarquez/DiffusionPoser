from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _configuration_map() -> dict[str, dict]:
    payload = json.loads((ROOT / ".vscode" / "launch.json").read_text(encoding="utf-8"))
    return {value["name"]: value for value in payload["configurations"]}


def _option(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _normalize_paired_training_args(args: list[str]) -> list[str]:
    values = list(args)
    for option in ("--save_dir", "--run_name", "--rollout_steps"):
        values[values.index(option) + 1] = f"<{option}>"
    return values


def _normalize_p100_training_args(args: list[str]) -> list[str]:
    values = list(args)
    for option in ("--save_dir", "--run_name", "--rollout_prob"):
        values[values.index(option) + 1] = f"<{option}>"
    return values


def _normalize_linear_late_training_args(args: list[str]) -> list[str]:
    values = list(args)
    for option in ("--save_dir", "--run_name", "--rollout_frame_weighting"):
        values[values.index(option) + 1] = f"<{option}>"
    return values


def _without_option(args: list[str], name: str) -> list[str]:
    values = list(args)
    index = values.index(name)
    del values[index : index + 2]
    return values


def test_rollout15_launch_chain_and_paired_training_contract() -> None:
    configs = _configuration_map()
    expected_modules = {
        "TAID 29F | 生成15帧小型task": "data_loaders.generate_realtime_pose_tasks",
        "TAID 29G | 复用正式normalizer统计": "data_loaders.reuse_realtime_pose_normalizer",
        "TAID 29H | 15帧真实batch显存与梯度预检": "train.validate_realtime_pose_rollout",
        "TAID 29I | 训练匹配R4-Control 5k": "train.train_diffusionposer",
        "TAID 29J | 训练R15 pilot 5k": "train.train_diffusionposer",
        "TAID 29K | 诊断R4-Control的29E曲线": "sample.diagnose_taid_history_horizon",
        "TAID 29L | 诊断R15的29E曲线": "sample.diagnose_taid_history_horizon",
        "TAID 29M | 禁止运行（R15曲线未通过）fixed-three": "sample.evaluate_longseq_eval_set",
        "TAID 29N | 禁止运行（R15曲线未通过）fixed-six": "sample.evaluate_longseq_eval_set",
        "TAID 29O | 训练R15-P100 pilot 5k": "train.train_diffusionposer",
        "TAID 29P | 诊断R15-P100的29E曲线": "sample.diagnose_taid_history_horizon",
        "TAID 29Q | 禁止运行（P100曲线未通过）fixed-three": "sample.evaluate_longseq_eval_set",
        "TAID 29R | 禁止运行（P100曲线未通过）fixed-six": "sample.evaluate_longseq_eval_set",
        "TAID 29S | 线性后段权重真实batch预检": "train.validate_realtime_pose_rollout",
        "TAID 29T | 训练R15-LW pilot 5k": "train.train_diffusionposer",
        "TAID 29U | 诊断R15-LW的29E曲线": "sample.diagnose_taid_history_horizon",
        "TAID 29V | 禁止运行（R15-LW曲线未通过）fixed-three": "sample.evaluate_longseq_eval_set",
        "TAID 29W | 禁止运行（R15-LW曲线未通过）fixed-six": "sample.evaluate_longseq_eval_set",
        "TAID 29X | 审计当前B1 Prior输入与监督能力": "train.audit_taid_prior_capacity",
        "TAID 29Y | Tracker-history Prior真实K15 batch预检": "train.validate_realtime_pose_rollout",
        "TAID 29Z | 训练Tracker-history R15-LW B1 pilot 5k": "train.train_diffusionposer",
        "TAID 29AA | 诊断Tracker-history B1的29E曲线": "sample.diagnose_taid_history_horizon",
        "TAID 29AB | 禁止运行（Tracker-history曲线未通过）fixed-three": "sample.evaluate_longseq_eval_set",
        "TAID 29AC | 禁止运行（Tracker-history曲线未通过）fixed-six": "sample.evaluate_longseq_eval_set",
        "TAID 29AD | 固定六槽 Prior真实K15 batch预检": "train.validate_realtime_pose_rollout",
        "TAID 29AE | 训练固定六槽 R15-LW B1 pilot 5k": "train.train_diffusionposer",
        "TAID 29AF | 诊断固定六槽 B1的29E曲线": "sample.diagnose_taid_history_horizon",
        "TAID 29AG | 评估固定六槽 B1 fixed-three前三条": "sample.evaluate_longseq_eval_set",
        "TAID 29AH | 评估固定六槽 B1 fixed-six前三条": "sample.evaluate_longseq_eval_set",
    }
    assert all(configs[name]["module"] == module for name, module in expected_modules.items())

    generator = configs["TAID 29F | 生成15帧小型task"]["args"]
    assert _option(generator, "--max_rollout_steps") == "15"
    assert _option(generator, "--limit") == "1024"
    assert _option(generator, "--limit_selection") == "stratified"

    r4 = configs["TAID 29I | 训练匹配R4-Control 5k"]["args"]
    r15 = configs["TAID 29J | 训练R15 pilot 5k"]["args"]
    assert _option(r4, "--rollout_steps") == "4"
    assert _option(r15, "--rollout_steps") == "15"
    assert _option(r4, "--init_checkpoint") == _option(r15, "--init_checkpoint")
    assert _option(r4, "--batch_size") == _option(r15, "--batch_size")
    assert _option(r4, "--rollout_frame_weighting") == "uniform"
    assert _option(r15, "--rollout_frame_weighting") == "uniform"
    assert _normalize_paired_training_args(r4) == _normalize_paired_training_args(r15)

    # P100 只提高闭环 batch 的出现概率；模型、数据、loss、seed和训练预算都必须与R15一致。
    r15_p100 = configs["TAID 29O | 训练R15-P100 pilot 5k"]["args"]
    assert _option(r15_p100, "--rollout_steps") == "15"
    assert _option(r15, "--rollout_prob") == "0.5"
    assert _option(r15_p100, "--rollout_prob") == "1.0"
    assert _option(r15_p100, "--batch_size") == "${input:taidRollout15TrainBatchSize}"
    assert _normalize_p100_training_args(r15) == _normalize_p100_training_args(r15_p100)

    checkpoint_input = "${input:taidR15P100Checkpoint}"
    diagnostic = configs["TAID 29P | 诊断R15-P100的29E曲线"]["args"]
    fixed_three = configs["TAID 29Q | 禁止运行（P100曲线未通过）fixed-three"]["args"]
    fixed_six = configs["TAID 29R | 禁止运行（P100曲线未通过）fixed-six"]["args"]
    for args in (diagnostic, fixed_three, fixed_six):
        assert _option(args, "--model_path") == checkpoint_input
        assert _option(args, "--use_ema") == "true"
        assert _option(args, "--limit") == "3"
        assert _option(args, "--inference_steps") == "5"
        assert _option(args, "--projected_ddim_mode") == "all_steps"
    condition_index = diagnostic.index("--conditions")
    assert diagnostic[condition_index + 1 : condition_index + 3] == ["fixed_three", "fixed_six"]
    assert _option(fixed_three, "--conditions") == "fixed_three"
    assert _option(fixed_six, "--conditions") == "fixed_six"
    assert "TAID 29M | 评估R15 fixed-three前三条" not in configs
    assert "TAID 29N | 评估R15 fixed-six前三条" not in configs

    preflight = configs["TAID 29S | 线性后段权重真实batch预检"]["args"]
    assert _option(preflight, "--rollout_steps") == "15"
    assert _option(preflight, "--batch_candidates") == "4"
    assert _option(preflight, "--max_reserved_gib") == "14"
    assert _option(preflight, "--rollout_frame_weighting") == "linear_late"

    # R15-LW相对原R15只能改变run信息和后续帧时间权重。
    r15_linear_late = configs["TAID 29T | 训练R15-LW pilot 5k"]["args"]
    assert _option(r15_linear_late, "--rollout_steps") == "15"
    assert _option(r15_linear_late, "--rollout_prob") == "0.5"
    assert _option(r15_linear_late, "--rollout_frame_weighting") == "linear_late"
    assert _normalize_linear_late_training_args(r15) == _normalize_linear_late_training_args(
        r15_linear_late
    )

    linear_late_checkpoint = "${input:taidR15LinearLateCheckpoint}"
    linear_diagnostic = configs["TAID 29U | 诊断R15-LW的29E曲线"]["args"]
    linear_fixed_three = configs[
        "TAID 29V | 禁止运行（R15-LW曲线未通过）fixed-three"
    ]["args"]
    linear_fixed_six = configs[
        "TAID 29W | 禁止运行（R15-LW曲线未通过）fixed-six"
    ]["args"]
    for args in (linear_diagnostic, linear_fixed_three, linear_fixed_six):
        assert _option(args, "--model_path") == linear_late_checkpoint
        assert _option(args, "--use_ema") == "true"
        assert _option(args, "--limit") == "3"
        assert _option(args, "--inference_steps") == "5"
        assert _option(args, "--projected_ddim_mode") == "all_steps"
    condition_index = linear_diagnostic.index("--conditions")
    assert linear_diagnostic[condition_index + 1 : condition_index + 3] == [
        "fixed_three",
        "fixed_six",
    ]
    assert _option(linear_fixed_three, "--conditions") == "fixed_three"
    assert _option(linear_fixed_six, "--conditions") == "fixed_six"
    assert "TAID 29Q | 评估R15-P100 fixed-three前三条" not in configs
    assert "TAID 29R | 评估R15-P100 fixed-six前三条" not in configs
    assert "TAID 29V | 评估R15-LW fixed-three前三条" not in configs
    assert "TAID 29W | 评估R15-LW fixed-six前三条" not in configs

    audit = configs["TAID 29X | 审计当前B1 Prior输入与监督能力"]["args"]
    assert _option(audit, "--model_path") == linear_late_checkpoint
    assert _option(audit, "--batch_size") == "2"
    assert _option(audit, "--rollout_steps") == "15"
    assert _option(audit, "--use_ema") == "true"
    assert "5000e-h60-b2-837c4b0327" in _option(
        audit, "--history_diagnostic_dir"
    )

    tracker_preflight = configs[
        "TAID 29Y | Tracker-history Prior真实K15 batch预检"
    ]["args"]
    assert _option(tracker_preflight, "--init_checkpoint") == "${input:taidB0Checkpoint}"
    assert _option(tracker_preflight, "--batch_candidates") == "4"
    assert _option(tracker_preflight, "--rollout_steps") == "15"
    assert _option(tracker_preflight, "--rollout_frame_weighting") == "linear_late"
    assert "--require_tracker_history_prior" in tracker_preflight

    tracker_train = configs[
        "TAID 29Z | 训练Tracker-history R15-LW B1 pilot 5k"
    ]["args"]
    assert _normalize_linear_late_training_args(tracker_train) == (
        _normalize_linear_late_training_args(r15_linear_late)
    )
    assert _option(tracker_train, "--init_checkpoint") == "${input:taidB0Checkpoint}"
    assert _option(tracker_train, "--rollout_steps") == "15"
    assert _option(tracker_train, "--rollout_prob") == "0.5"
    assert _option(tracker_train, "--rollout_frame_weighting") == "linear_late"
    assert _option(tracker_train, "--batch_size") == "${input:taidRollout15TrainBatchSize}"

    tracker_checkpoint = "${input:taidTrackerHistoryCheckpoint}"
    tracker_diagnostic = configs[
        "TAID 29AA | 诊断Tracker-history B1的29E曲线"
    ]["args"]
    tracker_fixed_three = configs[
        "TAID 29AB | 禁止运行（Tracker-history曲线未通过）fixed-three"
    ]["args"]
    tracker_fixed_six = configs[
        "TAID 29AC | 禁止运行（Tracker-history曲线未通过）fixed-six"
    ]["args"]
    for args in (tracker_diagnostic, tracker_fixed_three, tracker_fixed_six):
        assert _option(args, "--model_path") == tracker_checkpoint
        assert _option(args, "--use_ema") == "true"
        assert _option(args, "--limit") == "3"
        assert _option(args, "--inference_steps") == "5"
        assert _option(args, "--projected_ddim_mode") == "all_steps"
    condition_index = tracker_diagnostic.index("--conditions")
    assert tracker_diagnostic[condition_index + 1 : condition_index + 3] == [
        "fixed_three",
        "fixed_six",
    ]
    assert _option(tracker_fixed_three, "--conditions") == "fixed_three"
    assert _option(tracker_fixed_six, "--conditions") == "fixed_six"

    fixed_slot_preflight = configs[
        "TAID 29AD | 固定六槽 Prior真实K15 batch预检"
    ]["args"]
    assert _option(fixed_slot_preflight, "--init_checkpoint") == "${input:taidB0Checkpoint}"
    assert _option(fixed_slot_preflight, "--batch_candidates") == "4"
    assert _option(fixed_slot_preflight, "--rollout_steps") == "15"
    assert _option(fixed_slot_preflight, "--rollout_frame_weighting") == "linear_late"
    assert _option(fixed_slot_preflight, "--taid_prior_tracker_aggregation") == "fixed_slots"
    assert "--require_tracker_history_prior" in fixed_slot_preflight
    assert "--require_fixed_slot_prior" in fixed_slot_preflight

    fixed_slot_train = configs[
        "TAID 29AE | 训练固定六槽 R15-LW B1 pilot 5k"
    ]["args"]
    assert _option(fixed_slot_train, "--taid_prior_tracker_aggregation") == "fixed_slots"
    assert _option(fixed_slot_train, "--init_checkpoint") == "${input:taidB0Checkpoint}"
    assert _option(fixed_slot_train, "--rollout_steps") == "15"
    assert _option(fixed_slot_train, "--rollout_prob") == "0.5"
    assert _option(fixed_slot_train, "--rollout_frame_weighting") == "linear_late"
    assert _normalize_linear_late_training_args(
        _without_option(fixed_slot_train, "--taid_prior_tracker_aggregation")
    ) == _normalize_linear_late_training_args(tracker_train)

    fixed_slot_checkpoint = "${input:taidFixedSlotsCheckpoint}"
    fixed_slot_diagnostic = configs[
        "TAID 29AF | 诊断固定六槽 B1的29E曲线"
    ]["args"]
    fixed_slot_three = configs[
        "TAID 29AG | 评估固定六槽 B1 fixed-three前三条"
    ]["args"]
    fixed_slot_six = configs[
        "TAID 29AH | 评估固定六槽 B1 fixed-six前三条"
    ]["args"]
    for args in (fixed_slot_diagnostic, fixed_slot_three, fixed_slot_six):
        assert _option(args, "--model_path") == fixed_slot_checkpoint
        assert _option(args, "--use_ema") == "true"
        assert _option(args, "--limit") == "3"
        assert _option(args, "--inference_steps") == "5"
        assert _option(args, "--projected_ddim_mode") == "all_steps"
    condition_index = fixed_slot_diagnostic.index("--conditions")
    assert fixed_slot_diagnostic[condition_index + 1 : condition_index + 3] == [
        "fixed_three",
        "fixed_six",
    ]
    assert _option(fixed_slot_three, "--conditions") == "fixed_three"
    assert _option(fixed_slot_six, "--conditions") == "fixed_six"
    assert any(name.startswith("TAID 30 | 禁止运行") for name in configs)
