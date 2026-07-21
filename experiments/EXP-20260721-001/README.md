---
schema_version: 1
experiment_id: "EXP-20260721-001"
summary: "无小模型直接扩散强基线：单进程 H1 到 H8 自回归课程训练"
experiment_type: "code_change"
status: "running"
created_at: "2026-07-21T01:15:30.6254351+08:00"
script: "experiments/EXP-20260721-001/run.ps1"
repositories:
  diffusionposer:
    root: "."
    remote: "https://github.com/SoltidueMarquez/DiffusionPoser.git"
    branch_at_snapshot: "main"
    participation: "primary"
    changed: true
    baseline_commit: "9e338bd264d6f62f14862894985f991af76a1485"
    baseline_subject: "refactor: fold stationary regression into simple loss"
    experiment_commit: "02c00b1af5a069be1d626af81c554c7fa68ba6aa"
    experiment_subject: "experiment: unify EXP-20260721-001 rollout curriculum"
  unity:
    root: "../SIGGRAPH2024Unity"
    remote: "https://github.com/2333qbyqby/SIGGRAPH2024Unity.git"
    branch_at_snapshot: "TestSMPLTracking"
    participation: "reference_only"
    changed: false
    baseline_commit: "784574418852264f7d206bbf2343fb8e76b5237c"
    baseline_subject: "feat: load versioned realtime pose model profiles"
    experiment_commit: "784574418852264f7d206bbf2343fb8e76b5237c"
    experiment_subject: "feat: load versioned realtime pose model profiles"
paths:
  dataset:
    path: "../artifactStore/DiffusionPoser/active/generated/experiments/EXP-20260721-001/source"
    note: "复用本实验第一次运行已完成的 exact-schema source，不重复转换 AMASS。"
  task:
    path: "../artifactStore/DiffusionPoser/active/generated/experiments/EXP-20260721-001/tasks"
    note: "train 与 eval 直接写入 tasks/train 和 tasks/eval；训练集每 source 两个在线窗口，Dataset 在内存中构造 base/H1～H8。"
  normalizer:
    path: "../artifactStore/DiffusionPoser/active/generated/experiments/EXP-20260721-001/normalizer"
    note: "仅从 train source-reference task 统计；K4 生成正式统计，K8 执行收敛门禁。"
  input_checkpoint:
    path: null
    note: "从零训练直接扩散基线，不使用旧 200k 或其他初始化 checkpoint。"
  run_dir:
    path: "runs/EXP-20260721-001"
    note: "正式实验协调 manifest、日志与评估索引目录；Canary 使用其独立子目录，均不提交 Git。"
  log_dir:
    path: "runs/EXP-20260721-001/logs"
    note: "所有转换、标定、训练和评估命令的控制台日志，不提交 Git。"
  output_checkpoint:
    path: "../artifactStore/DiffusionPoser/active/generated/experiments/EXP-20260721-001/runs/direct-baseline"
    note: "正式单进程 130k 训练目录；Canary 使用同级 canary-b16-h8-s200 独立目录，二者均保留。"
  sample_output:
    path: null
    note: "本实验不单独生成展示采样；长序列推理产物写入 eval_output。"
  eval_output:
    path: "../artifactStore/DiffusionPoser/active/generated/experiments/EXP-20260721-001/output"
    note: "包含各阶段 limit-4 评估，以及 90k、115k、130k 三个 checkpoint 的完整评估矩阵。"
  export_output:
    path: null
    note: "训练基线阶段不导出 ONNX/Sentis；选定 checkpoint 后另行执行导出实验。"
  unity_assets:
    path: null
    note: "Unity 为 reference_only，本实验不生成或修改 Unity asset。"
runtime:
  manifest: "runs/EXP-20260721-001/experiment_runtime.json"
  diffusionposer_commit: "02c00b1af5a069be1d626af81c554c7fa68ba6aa"
  unity_commit: "784574418852264f7d206bbf2343fb8e76b5237c"
  command: "powershell.exe -NoProfile -ExecutionPolicy Bypass -File experiments/EXP-20260721-001/run.ps1 -SkipConversion -ReusePreparedData"
tests:
  - command: >-
      conda run --no-capture-output -n diffusionposer5070 pytest
      tests/smoke/train/test_rollout_curriculum_temporary.py
      tests/smoke/data_pipeline/test_prepared_data_temporary.py
      tests/smoke/experiments/test_exp_20260721_001_temporary.py
      tests/smoke/data_pipeline/test_realtime_pose_prepared_data.py
    result: "开发期临时测试累计 28 项通过；记录结果后四个临时测试文件均已删除。"
  - command: >-
      conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke -q
      --basetemp t -p no:cacheprovider
    result: "删除临时测试后的最终状态为 370 passed, 27 warnings；basetemp 与 pytest cache 随后删除。"
  - command: >-
      conda run --no-capture-output -n diffusionposer5070 python -m
      data_loaders.validate_realtime_pose_prepared_data
    result: "真实 prepared-data 严格校验通过：17,526 个 train entry，seed/mask/K4/K8 与 task/source SHA 全部匹配。"
  - command: >-
      powershell.exe -NoProfile -ExecutionPolicy Bypass -File
      experiments/EXP-20260721-001/run.ps1 -DryRun -SkipConversion -ReusePreparedData
    result: "Formal DryRun 通过；10k calibration 独立执行，正式 130k 训练仅调用一次。"
  - command: >-
      powershell.exe -NoProfile -ExecutionPolicy Bypass -File
      experiments/EXP-20260721-001/run.ps1 -DryRun -CanarySteps 2 -BatchSize 16
      -NumWorkers 2 -SkipConversion -ReusePreparedData
    result: "Canary DryRun 通过；独立目录、H8 每 step 强制执行，且不修改正式 checkpoint/latest。"
  - command: >-
      powershell.exe -NoProfile -ExecutionPolicy Bypass -File
      experiments/EXP-20260721-001/run.ps1 -CanarySteps 200 -BatchSize 16
      -NumWorkers 2 -SkipConversion -ReusePreparedData
    result: "Canary completed，exit 0；200/200 step 全部执行 H8，checkpoint、runtime manifest 与日志已保留。"
result:
  metrics:
    canary_steps: 200
    canary_h8_event_fraction: 1.0
    canary_step_200_loss: 0.66911
    canary_step_200_long_rollout_loss: 0.288
    canary_step_200_grad_norm_pre_clip: 4.56
    canary_status: "completed"
  conclusion: null
  failure_reason: null
---

# EXP-20260721-001｜无小模型直接扩散强基线：单进程 H1 到 H8 自回归课程训练

## 实验目的

在加入 PriorNet 或下半身初值小模型前，得到一版可作为论文消融基准的直接扩散模型。正式训练在同一个进程中从 base 平滑扩展到 H1/H2/H4/H8，避免阶段重启模型、Optimizer 和 EMA，同时保留 10k loss calibration、完整 Tracker mask 分布和既定长序列评估协议。

## 改动说明

- DiffusionPoser：正式 130k 训练只启动一次；global step 在 30k/60k/70k/90k 依次启用 H1/H2/H4/H8，DataLoader 只重建当前所需的 2/3/5/9 个窗口，模型、Optimizer、Scaler 与 EMA 连续保留。
- 学习率使用 2k warmup 后从 `5e-5` cosine decay 到 `1e-5`；rollout 概率独立 ramp。课程参数生成稳定 `training_schedule_signature`，任意 step 恢复都会校验 signature、恢复 Adam moments 并覆盖为当前 global-step LR。
- source-reference task 和 K4/K8 Normalizer 保持不变；`-ReusePreparedData` 严格检查 seed、mask policy、patterns、schema、split、samples、rollout steps、task/source SHA 和 Normalizer 收敛状态，任何不匹配都直接失败。
- 10k loss calibration 保持独立；正式训练全程使用 `tracker_mask_categories=all`。训练后评估 30k、60k、70k、90k、105k、115k、125k、130k，并对 90k、115k、130k 执行完整动态、full-six、standard-three 评估矩阵。
- 独立 `-CanarySteps` 模式把所有 horizon 起点和 ramp 设为 0、`long_rollout_prob=1`、`max_horizon_prob=1`，每个 step 强制执行 H8。Canary 使用独立 training/runtime/log 目录，checkpoint、manifest 与日志全部保留，不执行删除或 cleanup。
- 开发期间新增的课程、prepared-data 和实验脚本临时 smoke 测试累计 28 项通过；按计划在最终提交前全部删除，只保留对既有 smoke 的必要断言更新。
- 本实验固定 `stationary_margin_loss_weight=0`；保留 `stationary_prob_5` 的主扩散 MSE、sigmoid 概率投影和 `stationary_simple_loss_channel_weight=1.6232687317836745`，并忽略 calibration 报告对 margin 项给出的权重。
- Unity：不修改、不运行，仅冻结版本供后续导出和部署对照。

## 版本定位

| 仓库 | 实验前版本 | 实验版本 | 参与方式 |
|---|---|---|---|
| DiffusionPoser | `9e338bd264d6f62f14862894985f991af76a1485` | `02c00b1af5a069be1d626af81c554c7fa68ba6aa` | `primary` |
| Unity | `784574418852264f7d206bbf2343fb8e76b5237c` | `784574418852264f7d206bbf2343fb8e76b5237c` | `reference_only` |

## 执行与测试

- 本编号实验采用统一目录布局：当前 `README.md`、`experiment.json` 与 `run.ps1` 位于同一目录；大型运行产物仍写入仓库外的 artifactStore。
- 实验脚本：`experiments/EXP-20260721-001/run.ps1`
- Canary 命令：`powershell.exe -NoProfile -ExecutionPolicy Bypass -File experiments/EXP-20260721-001/run.ps1 -CanarySteps 200 -BatchSize 16 -NumWorkers 2 -SkipConversion -ReusePreparedData`
- 正式命令：`powershell.exe -NoProfile -ExecutionPolicy Bypass -File experiments/EXP-20260721-001/run.ps1 -SkipConversion -ReusePreparedData`
- 测试结果：临时专项测试累计 28 项通过后删除；删除后的完整 `tests/smoke` 为 370 项通过、27 条 warning。PowerShell 语法、Formal DryRun、Canary DryRun、真实 prepared-data 校验和 `git diff --check` 均通过。
- Canary 结果：2026-07-21 20:21–20:24 完成 200/200 step，exit code 0；`rollout_h8_event_fraction=1`，step 200 总 loss `0.66911`、H8 loss `0.288`，checkpoint 位于 `.../runs/canary-b16-h8-s200/20260721_202116_EXP-20260721-001-canary-b16-h8-s200/model000000200.pt`。Canary runtime manifest 和日志均保留，正式训练目录/latest 未修改。
- 正式实验：2026-07-21 20:27 启动，runtime manifest 已记录 `status=running` 和实现 commit `02c00b1af5a069be1d626af81c554c7fa68ba6aa`；record/preflight、prepared-data、test manifest 和 longseq 准备均已完成，当前执行独立 10k calibration warm-up。旧实现 commit `08c3a3b…` 的失败 manifest/log 已原样归档到 `runs/EXP-20260721-001/archive/formal-failed-08c3a3b-20260721T105529/`，未删除任何历史 Formal 或 Canary 产物。

## 数据与产物路径

路径以实验配置和运行后生成的 `runs/EXP-20260721-001/experiment_runtime.json` 为准。实验主体固定在 `generated/experiments/EXP-20260721-001/` 下，目录为 `source/`、`tasks/`、`normalizer/`、`longseq/`、`runs/`、`output/`。task 产物只有 source-reference marker/manifest，训练 task NPZ 数固定为 0；保留约 14.95 GiB source，脚本要求 artifact 盘启动前至少剩余 40 GiB，用于 checkpoint、Normalizer 与评估输出。

数据加载 benchmark 使用 128 个真实 train source、batch size 16、每 worker 512 MiB source LRU，预热 2 batch 后统计 8 batch。base 的平均等待为 72/45/22 ms（worker 0/2/4），H8 为 456/311/324 ms；4 worker 的 H8 P95 升至 1.56 s，因此实验固定 `num_workers=2`。

## 指标与结论

200-step H8 Canary 已通过并保留全部产物；正式实验已启动，当前状态为 `running`。训练完成后使用 `runs/EXP-20260721-001/evaluation_index.json` 比较课程 checkpoint，并重点分析 90k、115k、130k 的全量 predicted-history 指标，不能默认选择最后一个 checkpoint。
