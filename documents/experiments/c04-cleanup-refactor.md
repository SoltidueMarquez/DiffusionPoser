# C04 Cleanup And Refactor

## Baseline

- 代码基线：`e8d93edc9d12e5725a6612a12891fc576538e8d6`
- 语义 tag：`baseline/c04-loss-v3-stable-rollout`
- 清理分支：`codex/c04-cleanup-refactor`
- 状态：`training_seed_unaccepted_stationary`

C04 可以作为后续训练的 `--init_checkpoint`，但不能在论文、导出验收或实验结论中表述为 stationary 已通过。历史 V4 分支只保留为研究线，不并入此活跃基线。

## Active Assets

所有活跃路径由 `configs/artifact_roots.local.json` 和
`configs/artifact_registry.json` 解析。C04 使用以下逻辑资产：

- `source.c04.causal_stationary`
- `task.c04.rollout9`
- `normalizer.c04.train`
- `longseq.c04.pre_stage_a`
- `checkpoint.c04.model`
- `checkpoint.c04.ema`
- `run.c04`
- `runtime.body_fbx_rest`
- `raw.amass`
- `raw.body_models`
- `raw.amass_archive`

`output.c04.longseq` and `summary.c04.longseq` have been compacted into
`realtime-loss-v3-causal-stationary-rollout-results.json`. They are deletion candidates only after a reviewed predelete manifest; the C04 source, task, normalizer, longseq set, model, EMA, args, and logs remain active.

`checkpoint.c04.optimizer` 已归入 `archive/2026-07-cleanup`，不参与新的训练恢复。C04 的 source、task、normalizer、longseq、run 和 output 已迁到短路径布局，迁移证据位于：

`artifactStore/DiffusionPoser/manifests/20260717_c04_short_path_relocation.json`

该 manifest 对每项资产保留文件数、字节数、tree SHA-256 和 JSONL 文件级清单。

## Experiment Entry

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m scripts.experiments.run_profile `
  --experiment-config configs/experiments/c04-loss-v3-stable-rollout.json `
  --stage validate
```

支持的阶段为 `validate`、`train`、`evaluate` 和 `summarize`。显式 CLI 参数覆盖 profile，profile 覆盖代码安全默认值。运行记录写入 artifact store 中的 run/output 目录，不提交二进制 checkpoint、数据或评估输出。

## Historical Archive

`raw.amass_archive` is retained as the active AMASS compressed archive and is explicitly excluded from deletion candidates. The Unity source `body_fbx_rest.json` is retained outside this repository and mirrored to the active artifact store; neither copy is a deletion candidate.

Body Rest 镜像已通过 24 骨骼、parent、tracker index 与 rest pose 格式校验：`7,196` bytes，源和镜像的 SHA-256 均为 `c59d0251623c42080e0c231aae534d353406d97b08d9959fc1ba948f70f201d2`。镜像与复核清单位于 `artifactStore/DiffusionPoser/manifests/20260717_body_fbx_rest_mirror.json` 和 `20260717_body_fbx_rest_verify.json`。

历史 generated 数据、旧 schema 数据、GlobalPose、旧 runs 和旧 save 已迁入：

`artifactStore/DiffusionPoser/archive/2026-07-cleanup`

迁移证据位于：

`artifactStore/DiffusionPoser/manifests/20260717_historical_archive_migration.json`

迁移工具在源与目标完整 hash、schema metadata 和依赖均通过后才移除源路径。下一轮删除必须先读取 `configs/cleanup_2026_07_deletion_candidates.json`，重新输出精确路径、文件数、大小和再生命令，不使用 glob。

`20260717_c04_cleanup_predelete.json` 已输出 51 项待人工审阅候选，共 631,128 个文件、236,802,390,989 bytes（220.539 GiB）；本轮没有执行删除。实际删除时传入 `--remove-cleanup-config --purge-deletion-audit`，会在候选、registry 更新均成功后移除本轮 predelete JSON、其 `.files` JSONL 目录和候选配置。C04 的真实 CUDA 1-step warm-start 位于 `active/runs/experiments/c04-followup/20260717_165815_c04_warm_start_20260717`，使用 `model000005000.pt` 初始化，产出 model、EMA、optimizer checkpoint，最后记录的 `loss=0.00067988655064255`。

## Code Boundaries

- `diffusion/realtime_pose/config.py`：实时 pose Loss 配置、默认值和校验。
- `diffusion/realtime_pose/resolver.py`：可微 RuntimeRootResolver。
- `diffusion/realtime_pose/loss_terms.py`：实时 pose 几何、tracker、stationary、contact 和 velocity Loss。
- `diffusion/gaussian_diffusion.py`：通用扩散流程和 Loss 组件调用。
- `train/training_loop.py`：batch、rollout、反向传播和日志。
- `train/realtime_loss_calibration.py`：梯度标定。

历史 V2/V3 PowerShell 编排、独立 summarizer 和失效的 `scripts/export_smpl_source_rest.py` 已删除；可复现入口统一为 experiment profile runner。
