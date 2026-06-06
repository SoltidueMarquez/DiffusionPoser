---
name: diffusionposer-experiment-branch
description: Branch-and-document workflow for DiffusionPoser model, diffusion, loss, training, sampling, evaluation, or export experiments. Use when Codex is asked to start an ablation branch, preserve a baseline tag, create or update experiment records, document model/loss changes, add smoke tests, or launch a short training run while keeping dataset/runs/output artifacts out of commits.
---

# DiffusionPoser Experiment Branch

## Core Rule

Use this together with `diffusionposer-repro` when editing this repository. Keep communication in Chinese, keep code changes narrowly scoped, and never stage or commit `dataset/`, `runs/`, `output/`, `save/`, checkpoint files, or generated binary data unless the user explicitly overrides the artifact policy.

## Workflow

1. Check repository state before changing anything:
   - Run `git status --short --branch`.
   - Inspect existing branches and tags when creating a new experiment.
   - If unrelated user changes exist, leave them alone and work around them.

2. Preserve the baseline before the experiment:
   - Use an annotated tag named `baseline/<experiment-name>` unless the user specifies another name.
   - Example: `git tag -a baseline/loss-ablation-v1 -m "Baseline before loss ablation v1"`.
   - Record the tag and commit in the experiment document.

3. Create or switch to the experiment branch:
   - Prefer `codex/<experiment-name>` for new branches.
   - Example: `git switch -c codex/loss-ablation-v1`.
   - If the branch already exists, inspect it instead of recreating it.

4. Create the experiment record before or alongside implementation:
   - Put records under `documents/experiments/`.
   - Use `<experiment-name>.md`.
   - Include baseline tag, baseline commit, branch name, date, motivation, artifact policy, changed files, test commands, training command, run directory, metrics, and conclusion.

5. Implement the model or loss change through the existing module boundaries:
   - Diffusion/loss logic belongs in `diffusion/`.
   - CLI options belong in `utils/parser_util.py`.
   - Diffusion construction and parameter pass-through belong in `utils/model_util.py`.
   - Training loop changes belong in `train/` only when the loop behavior itself changes.
   - Add focused smoke tests under `tests/smoke/<domain>/`.

6. Validate with the narrowest relevant smoke tests first:
   - For training/loss changes, run `conda run -n diffusionposer5070 pytest tests/smoke/train`.
   - For cross-module contract changes, run `conda run -n diffusionposer5070 pytest tests/smoke`.
   - Record the exact command and result in the experiment document.

7. Launch training only into ignored artifact directories:
   - Use `runs/<experiment-name>` or another ignored run root.
   - Keep `dataset/`, `runs/`, `output/`, and `save/` out of git.
   - If the user wants VSCode logs, add a reusable `.vscode/tasks.json` task or paste the command into the VSCode integrated terminal. Do not leave `runOptions.runOn: folderOpen` enabled after the launch.

8. Final status check:
   - Run `git status --short --branch`.
   - Run `git status --short --ignored dataset runs output` when artifact safety matters.
   - Summarize changed source/docs/tests separately from ignored training artifacts.

## Experiment Document Template

```markdown
# <Experiment Name> 实验记录

## 基线

- baseline tag: `baseline/<experiment-name>`
- baseline commit: `<short-sha> <subject>`
- experiment branch: `codex/<experiment-name>`
- 创建日期: `<YYYY-MM-DD>`

## 实验目的

- <what changed and why>

## 产物约定

- 不提交 `dataset/`、`runs/`、`output/`、`save/` 或 checkpoint/data 二进制产物。
- 训练、采样和评估命令默认使用 `conda run -n diffusionposer5070 <command>`。

## 实验条目

### 001 - <short title>

- 改动摘要:
- 关键文件:
- 训练配置:
- 测试命令:
- Run 目录:
- 结果指标:
- 结论:
```

## Current Branch Example

For `codex/loss-ablation-v1`, the workflow created:

- baseline tag: `baseline/loss-ablation-v1`
- branch: `codex/loss-ablation-v1`
- experiment record: `documents/experiments/loss-ablation-v1.md`
- reusable VSCode task: `loss-ablation-v1: train 20k nonhip huber`

The model/loss change was:

- In `diffusion/gaussian_diffusion.py`, change `sensor_reprojection_pos_loss` from per-axis absolute mean to vector-distance Huber.
- Exclude `HIP_TRACKER_INDEX` from the valid tracker mask so only head, hands, and feet constrain tracker position reprojection.
- Apply timestep weight `w_t = w_min + (1 - w_min) * progress^gamma`, where `progress = 1 - t / (T - 1)`.
- Add defaults `tracker_pos_huber_beta=0.05`, `tracker_pos_timestep_min_weight=0.1`, and `tracker_pos_timestep_gamma=2.0`.
- Pass the new options through `utils/parser_util.py` and `utils/model_util.py`.
- Add smoke tests that verify hip tracker errors do not affect the position loss and that low-noise timesteps produce larger loss than high-noise timesteps.

The first suggested training command for this branch uses:

```powershell
--num_steps 20000 `
--tracker_pos_loss_weight 10 `
--tracker_pos_huber_beta 0.05 `
--tracker_pos_timestep_min_weight 0.1 `
--tracker_pos_timestep_gamma 2.0
```
