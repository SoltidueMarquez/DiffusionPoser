---
name: realtime-poser-experiment
description: Record, run, close, locate, or roll back DiffusionPoser realtime-pose experiments without creating experiment branches or semantic experiment names. Use after code changes or loss-only adjustments to allocate an EXP-YYYYMMDD-NNN identifier, create a natural-language experiment record and reproducible PowerShell script, bind exact DiffusionPoser and Unity baseline/experiment commits, record dataset/checkpoint/run/output paths, validate the paired repository state, or summarize experiment results.
---

# Realtime Poser Experiment

## 核心约定

- 遵循仓库根目录 `AGENTS.md`，使用中文沟通，保持改动范围最小。
- 不创建实验分支或 baseline tag，不要求用户提供语义化实验名。
- 使用 `EXP-YYYYMMDD-NNN` 作为文件、脚本、run 和 output 的唯一稳定编号。
- 用一句简洁中文说明表达实验意图，但不要把说明写入文件名或目录名。
- 默认 Unity 仓库为 `../SIGGRAPH2024Unity`；用户可覆盖路径，但必须验证目标是 Git 仓库。
- 不提交 `dataset/`、`runs/`、`output/`、`outputs/`、`save/`、checkpoint 或生成的二进制产物；必须在实验记录中写明它们的实际路径。
- 保留已有历史实验记录，不迁移或改写旧的 branch/tag 字段。

## 可复用资源

- 开始实验前完整读取并使用 [assets/experiment-record.md](assets/experiment-record.md) 和 [assets/run-experiment.ps1](assets/run-experiment.ps1)。
- 使用 `scripts/experiment_record.py next-id` 分配编号，使用 `validate` 校验记录和双仓库状态。
- 所有 Python、pytest、训练、采样、评估和导出命令使用 `conda run --no-capture-output -n diffusionposer5070`。

## 开始实验

1. 检查 DiffusionPoser 与 Unity 的 repo root、remote、当前 branch、完整 HEAD、commit subject、`git status --short --branch` 和 staged/unstaged diff。
2. 判断实验类型：
   - `loss_only`：运行时代码不变，差异全部由实验脚本或配置中的 loss 参数表达。
   - `code_change`：任何源码、测试、运行时或 loss 实现发生变化。
3. 判断 Unity 参与方式：
   - `participating`：本轮会运行、验证或依赖 Unity；运行前必须提交相关改动并保持工作区干净。
   - `reference_only`：本轮不运行 Unity；记录当前 HEAD，明确排除未提交改动，不宣称 Unity 工作区可复现。
4. 运行：

   ```powershell
   conda run --no-capture-output -n diffusionposer5070 python .codex/skills/realtime-poser-experiment/scripts/experiment_record.py next-id --records-dir documents/experiments
   ```

5. 根据 diff 生成简洁中文说明，实例化 `scripts/experiments/<experiment-id>.ps1`。脚本必须固定完整命令、输入路径、输出路径和随机种子，并把运行时 commit 与退出状态写入 `runs/<experiment-id>/experiment_runtime.json`。
6. 向用户展示自然语言说明、两个仓库的参与方式、待提交文件清单和 commit 方案。只有得到确认后才能提交。
7. 分别创建实验 commit：
   - DiffusionPoser commit 包含本轮源码/loss 配置、测试和实验 PowerShell 脚本。
   - Unity 仅在 `participating` 且存在相关改动时创建独立 commit。
   - 不得暂存或提交范围外文件；同一文件混有无法拆分的无关改动时停止并说明。
8. 获得稳定 SHA 后，实例化 `documents/experiments/<experiment-id>.md`。记录中的 `experiment_commit` 指向包含实验实现与脚本的 commit；记录文件随后单独提交，避免 commit 自引用。
9. 在启动前运行 `validate --phase pre-run`。DiffusionPoser 只允许实验记录文件尚未提交，HEAD 必须等于实验 commit；参与型 Unity 的 HEAD 必须匹配且工作区干净。
10. 运行最窄相关 smoke tests，记录命令和结果，再通过实验 PowerShell 脚本启动。
11. 启动成功后仅提交实验记录。长任务可以先记录 `running`；不要把训练产物加入 Git。

## 收尾实验

1. 读取实验脚本、runtime manifest、日志、checkpoint、评估摘要和实际产物路径。
2. 将状态更新为 `completed`、`failed` 或 `abandoned`；补全实际运行 commit、命令、测试结果、指标、结论或失败原因。
3. 运行 `validate --phase close`，确认 runtime manifest 与记录的 commit 一致。
4. 只提交更新后的实验记录；分别汇报源码/脚本、文档和忽略产物状态。

## 定位与回退

- 定位时从记录读取两个仓库的完整 SHA，使用 `git show`、`git diff <baseline>..<experiment>` 或新的隔离 worktree 查看版本。
- 回退前展示 DiffusionPoser 与 Unity 的目标 commit 和影响范围，并再次获得用户明确授权。
- 优先使用 `git revert <experiment_commit>` 保留线性历史；禁止使用 `git reset --hard`。
- Unity 为 `reference_only` 时，不把其未提交工作区视为本实验的一部分，也不对其执行回退。

## 校验命令

```powershell
conda run --no-capture-output -n diffusionposer5070 python .codex/skills/realtime-poser-experiment/scripts/experiment_record.py validate `
  --record documents/experiments/<experiment-id>.md `
  --phase pre-run
```

如 Unity 不在默认位置，追加 `--unity-root <path>`。校验失败时先修正记录或仓库状态，不要绕过检查启动实验。
