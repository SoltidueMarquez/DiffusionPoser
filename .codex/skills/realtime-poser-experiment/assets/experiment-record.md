---
schema_version: 1
experiment_id: "{{EXPERIMENT_ID}}"
summary: "{{SUMMARY}}"
experiment_type: "{{EXPERIMENT_TYPE}}"
status: "ready"
created_at: "{{CREATED_AT_ISO8601}}"
script: "experiments/{{EXPERIMENT_ID}}/run.ps1"
repositories:
  diffusionposer:
    root: "."
    remote: "{{DIFFUSIONPOSER_REMOTE}}"
    branch_at_snapshot: "{{DIFFUSIONPOSER_BRANCH}}"
    participation: "primary"
    changed: true
    baseline_commit: "{{DIFFUSIONPOSER_BASELINE_SHA}}"
    baseline_subject: "{{DIFFUSIONPOSER_BASELINE_SUBJECT}}"
    experiment_commit: "{{DIFFUSIONPOSER_EXPERIMENT_SHA}}"
    experiment_subject: "{{DIFFUSIONPOSER_EXPERIMENT_SUBJECT}}"
  unity:
    root: "../SIGGRAPH2024Unity"
    remote: "{{UNITY_REMOTE}}"
    branch_at_snapshot: "{{UNITY_BRANCH}}"
    participation: "{{UNITY_PARTICIPATION}}"
    changed: false
    baseline_commit: "{{UNITY_BASELINE_SHA}}"
    baseline_subject: "{{UNITY_BASELINE_SUBJECT}}"
    experiment_commit: "{{UNITY_EXPERIMENT_SHA}}"
    experiment_subject: "{{UNITY_EXPERIMENT_SUBJECT}}"
paths:
  dataset:
    path: null
    note: "{{DATASET_PATH_OR_UNUSED_REASON}}"
  task:
    path: null
    note: "{{TASK_PATH_OR_UNUSED_REASON}}"
  normalizer:
    path: null
    note: "{{NORMALIZER_PATH_OR_UNUSED_REASON}}"
  input_checkpoint:
    path: null
    note: "{{INPUT_CHECKPOINT_PATH_OR_UNUSED_REASON}}"
  run_dir:
    path: "runs/{{EXPERIMENT_ID}}"
    note: "训练或执行产物根目录，不提交 Git。"
  log_dir:
    path: "runs/{{EXPERIMENT_ID}}/logs"
    note: "控制台日志目录，不提交 Git。"
  output_checkpoint:
    path: null
    note: "{{OUTPUT_CHECKPOINT_PATH_OR_UNUSED_REASON}}"
  sample_output:
    path: null
    note: "{{SAMPLE_OUTPUT_PATH_OR_UNUSED_REASON}}"
  eval_output:
    path: null
    note: "{{EVAL_OUTPUT_PATH_OR_UNUSED_REASON}}"
  export_output:
    path: null
    note: "{{EXPORT_OUTPUT_PATH_OR_UNUSED_REASON}}"
  unity_assets:
    path: null
    note: "{{UNITY_ASSET_PATH_OR_UNUSED_REASON}}"
runtime:
  manifest: "runs/{{EXPERIMENT_ID}}/experiment_runtime.json"
  diffusionposer_commit: null
  unity_commit: null
  command: null
tests: []
result:
  metrics: {}
  conclusion: null
  failure_reason: null
---

# {{EXPERIMENT_ID}}｜{{SUMMARY}}

## 实验目的

{{MOTIVATION}}

## 改动说明

- DiffusionPoser：{{DIFFUSIONPOSER_CHANGE_SUMMARY}}
- Unity：{{UNITY_CHANGE_SUMMARY}}

## 版本定位

| 仓库 | 实验前版本 | 实验版本 | 参与方式 |
|---|---|---|---|
| DiffusionPoser | `{{DIFFUSIONPOSER_BASELINE_SHA}}` | `{{DIFFUSIONPOSER_EXPERIMENT_SHA}}` | `primary` |
| Unity | `{{UNITY_BASELINE_SHA}}` | `{{UNITY_EXPERIMENT_SHA}}` | `{{UNITY_PARTICIPATION}}` |

## 执行与测试

- 实验脚本：`experiments/{{EXPERIMENT_ID}}/run.ps1`
- 启动命令：{{RUN_COMMAND}}
- 测试结果：{{TEST_RESULTS}}

## 数据与产物路径

路径以 YAML 元数据为准。即使目录或文件被 `.gitignore` 忽略，也必须保留实际路径；未使用的项目必须写明原因。

## 指标与结论

{{RESULT_SUMMARY}}
