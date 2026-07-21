# Realtime Loss v2 与 Rollout8 实验记录

## 基线与分支

- baseline tag: `baseline/realtime-loss-v2-rollout8`
- baseline commit: `4b67af9 feat: complete realtime sparse tracker baseline`
- experiment branch: `codex/realtime-loss-v2-rollout8`
- 创建日期: `2026-07-15`
- 统一初始化权重: `runs/realtime_pose_stationary5_v1/v2_stage_a_rotation_fix_20k_20260714/20260714_140005_stage_a_rotation_fix_20k_seed10/model000020000.pt`

## 已完成实现

- Loss v2：统一局部旋转、pelvis-relative/body-heading-local 几何、锚点相对 tracker、no-Hip yaw/height、stationary、contact 和真实单位 temporal loss。
- 可微 Resolver：覆盖 Hip、no-Hip head correction、height clamp、continuous-Hip filter、reconnect 与 reset boundary；与 NumPy `RuntimeRootResolver` 做数值对齐测试。
- 所有辅助项统一 timestep attenuation；`L_x0` 的 feature-weighted 分母改为权重和。
- 辅助几何固定在 FP32 中反向，避免 BF16 identity rotation 的非有限梯度。
- RPM 风格 rollout：`rollout_steps=9` 表示 base + 8 个相邻窗口；每次事件均匀采样 `H∈[1,8]`，前缀 DDIM10/Resolver 全部 `no_grad`，只在终点反向。
- rollout 历史传播完整的 214 维帧，包含 154 维预测结果和当时实际使用的 tracker reference；不会在后续滑窗退回 GT-state 编码。
- task generator 默认 base + 1 个相邻窗口；H=8 实验显式使用 `--rollout_steps 9`。mask timeline 与额外 dropout 增强均按 source absolute frame 对齐。
- 评估新增 MPJRE、MPJVE、Jitter、PJ/AUJ，并按 full-six、standard-three、左右单脚缺失、Hip 缺失、6→3、3→6/reconnect 和 no-Hip 持续时间分组。

## 梯度标定

标定配置为 seed 10、batch 16、Stage A rotation-fix 20k、四类 mask × 四个 timestep 区间，共 16 组。完整输入、逐组梯度范数与公式结果见 `realtime-loss-v2-rollout8-calibration.json`。

| loss group | target ratio | 固化 λ |
|---|---:|---:|
| local rotation | 0.20 | 3.4085423061 |
| body geometry | 0.25 | 0.2473900482 |
| tracker relative position | 0.15 | 0.0635098506 |
| tracker relative rotation | 0.05 | 0.0834529156 |
| no-Hip yaw | 0.05 | 9.3916115557 |
| no-Hip height | 0.05 | 0.2236590567 |
| stationary | 0.10 | 0.0278058304 |
| contact height | 0.025 | 0.0196552916 |
| contact velocity | 0.025 | 0.0001（触发下限） |
| joint velocity | 0.04 | 0.0003982433 |
| rotation velocity | 0.04 | 0.0006147233 |
| yaw velocity | 0.02 | 0.0010540296 |

contact velocity 因 `[1e-4,100]` 下限得到实测梯度比例 `0.0754623`，是目标 `0.025` 的 `3.018×`，超出目标的 `0.5×–2×`；其他组等于标定目标。

复现入口：

```powershell
conda run --no-capture-output -n diffusionposer5070 python -u -m train.realtime_loss_calibration `
  --calibration_output documents/experiments/realtime-loss-v2-rollout8-calibration.json `
  --schema realtime_pose_stationary5_v1 --model_arch target_dit `
  --input_feats 214 --seq_len 61 --max_seq_len 61 `
  --data_dir dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_v2_base_tasks/20260714_001222_v2_base_seed10 `
  --normalizer_dir dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_v2_train/20260714_003252_v2_base_online_seed10 `
  --save_dir runs/realtime-loss-v2-rollout8/calibration `
  --batch_size 16 --seed 10 --layers 8 --heads 8 --latent_dim 512 --diffusion_steps 50 `
  --tracker_mask_seed 10 --snr_gamma 0 `
  --init_checkpoint runs/realtime_pose_stationary5_v1/v2_stage_a_rotation_fix_20k_20260714/20260714_140005_stage_a_rotation_fix_20k_seed10/model000020000.pt `
  --cuda true --device 0
```

## 验证结果

- `pytest tests/smoke/train`: passed。
- `pytest tests/smoke/data_pipeline`: passed。
- `pytest tests/smoke/eval`: passed。
- `pytest tests/smoke`: `349 passed, 1 skipped`（筛选实验完成后的最终状态）。
- 真实 CUDA H=1 探针：batch 16、DDIM10、Stage A 20k 初始化；单步 loss `0.01725`，pre-clip grad norm `0.493`，loss/gradient 均 finite，未触发裁剪。
- H=8 smoke：8 个预测历史帧及其 tracker 条件全部回填；history tensor 无梯度。

## 筛选实验矩阵

四个 5k 筛选 run 必须互相独立从同一 20k checkpoint warm start，不能串行继承：

1. `A00 legacy-h1`：在 baseline tag 上运行旧 loss，H=1。
2. `A01 gate-clean-h1`：门控/归一化/stationary 清理快照，H=1。
3. `A02 loss-v2-h1`：当前 Loss v2，`rollout_steps=2, rollout_prob=0.5`。
4. `A03 loss-v2-h1-8`：当前 Loss v2，`rollout_steps=9, rollout_prob=0.125`。

共同配置：seed 10、batch 16、LR `1e-5`、`snr_gamma=0`、四类 mask `30/30/20/20`、`--init_checkpoint model000020000.pt`。A03 task 必须用下列方式独立生成，source、normalizer 和 schema 不变：

```powershell
conda run --no-capture-output -n diffusionposer5070 python -m data_loaders.generate_realtime_pose_tasks `
  --source_dir dataset/generated/sources/realtime_pose_stationary5_v1/amass_60hz_v2 `
  --output_dir dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_v2_rollout8_tasks `
  --splits train test --samples_per_source 2 --rollout_steps 9 `
  --schema realtime_pose_stationary5_v1 --seed 10 --run_name loss_v2_rollout8_seed10
```

## 实际筛选执行

rollout9 task 于 `2026-07-15` 生成完成：

- task run: `dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_v2_rollout9_tasks_screening/20260715_214141_loss_v2_r9_seed10`
- train/test manifest: `17,526 / 2,608`
- 文件数与体积: `181,211` 个文件，`20.53 GiB`
- 生成耗时: `751.682 s`

四组 5k 训练均从相同 Stage A 20k checkpoint 独立 warm start，未继承前一组 optimizer、global step 或 EMA：

| 实验 | 训练耗时(min) | last-1k loss | simple | aux | pre-clip grad | 裁剪率 |
|---|---:|---:|---:|---:|---:|---:|
| A00 legacy-h1 | 38.0 | 0.020256 | 0.003797 | 0.008580 | 6.3966 | 100.0% |
| A01 gate-clean-h1 | 36.9 | 0.013775 | 0.003855 | 0.004140 | 5.0685 | 100.0% |
| A02 loss-v2-h1 | 36.5 | 0.007462 | 0.003882 | 0.000266 | 0.2621 | 0.3% |
| A03 loss-v2-h1-8 | 108.4 | 0.012331 | 0.004258 | 0.000273 | 0.2695 | 1.7% |

固定长序列评估使用 `20260714_pre_stage_a_seed10` 的 18 条序列、49,326 帧，统一为 DDIM10、predicted history、无 IK、同一动态 mask timeline。额外 8 帧 Hip 掉线只在移除 Hip 后仍至少有 3 个 tracker 有效的完整窗口上施加，保持 schema 契约。完整结果见：

- `documents/experiments/realtime-loss-v2-rollout8-results.json`
- `documents/experiments/realtime-loss-v2-rollout8-results.md`

为缩短墙钟时间，评估按 A00+A03、A01+A02 两路并行执行。精度、速度误差、jitter、掉线和 stationary 指标不依赖墙钟计时，可直接用于筛选；summary 中的 DDIM/end-to-end 延迟与 FPS 受 GPU 竞争影响，不作为单模型实时性能基准。四组模型结构和推理图完全相同。

关键指标：

| 指标 | A00 | A01 | A02 | A03 |
|---|---:|---:|---:|---:|
| full-six MPJPE (cm) | 4.222 | 3.902 | 4.344 | 4.095 |
| full-six MPJRE (deg) | 12.625 | 11.696 | 13.658 | 12.293 |
| full-six MPJVE (cm/s) | 63.869 | 61.142 | 62.337 | 66.021 |
| standard-three MPJPE (cm) | 20.171 | 19.751 | 20.242 | 18.463 |
| standard-three MPJRE (deg) | 21.672 | 21.264 | 22.221 | 19.629 |
| standard-three MPJVE (cm/s) | 175.241 | 161.522 | 162.980 | 161.046 |
| reconnect MPJVE (cm/s) | 155.919 | 156.241 | 159.141 | 185.321 |
| reconnect Jitter (m/s³) | 9801.62 | 9765.73 | 9884.52 | 12694.60 |
| reconnect PJ | 1833.70 | 1991.53 | 1822.06 | 1765.78 |
| reconnect AUJ | 9586.67 | 9733.21 | 9461.67 | 11997.00 |
| stationary F1 | 0.2144 | 0.2318 | 0.2234 | 0.2722 |
| stationary 越界率 | 53.25% | 48.53% | 46.15% | 42.11% |

## 筛选结论

总判定为 **FAIL**：

- PASS：所有训练 loss、梯度和长序列输出 finite。
- PASS：A03 相对 A00 的 full-six/standard-three MPJPE、MPJRE、MPJVE 和 tracker 误差均满足不退化超过 5% 的守门条件；其中 full-six MPJVE 退化 `3.37%`，其余列出的守门指标均改善。
- FAIL：A03 stationary 越界率为 `42.11%`，未达到 `<5%`；F1 相对 A00 只提高 `0.0578`，未达到 `+0.10`。
- FAIL：A03 相对 A02 的 reconnect MPJVE、Jitter、AUJ 分别恶化 `16.45% / 28.43% / 26.80%`，PJ 只改善 `3.09%`；没有两项达到至少 5% 改善。
- PASS：A03 最近 1k step 梯度裁剪率为 `1.7%`，显著低于 80%；A02 为 `0.3%`。
- FAIL：contact-velocity 梯度比例为标定目标的 `3.018×`，超出 `0.5×–2×`；其余 loss group 为 `1×`。

因此不启动筛选后的 3-seed `0–30k` 长训，也不进入方案一的 Sensor-only MLP Proposal。下一轮应先处理 stationary 输出范围/目标强度，以及 A03 长 rollout 对 reconnect 速度与 jitter 的负迁移，再重新做短筛选。

## 产物约定

- 不提交 `dataset/`、`runs/`、`output/`、`save/` 或 checkpoint。
- source、normalizer、schema、ONNX/Sentis 输出和 Unity runtime 接口保持不变。
