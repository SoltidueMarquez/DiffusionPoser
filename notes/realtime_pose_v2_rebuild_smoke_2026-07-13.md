# Realtime Pose v2 数据重建与短训练记录（2026-07-13/14）

> 历史记录：本文保留当时使用的 manifest、hash、latest 指针和旧 Task Store 数据，不能作为当前主链路操作指南。当前目录式流程请阅读 `README.md`、`documents/复现.md` 和 `contract.md`。

## 运行边界

- Git HEAD：`250ee5df5c6c1e9204f9af9eadef9f3863e08075`。
- 在现有 dirty working tree 中直接运行；未创建 worktree，未覆盖或回滚用户改动。
- 环境：`diffusionposer5070`，Python 3.11.15，PyTorch 2.7.0+cu128，CUDA 12.8，RTX 5070（12,820,480,000 bytes）。
- 只使用本地已有 AMASS 与 SMPL 资产，没有下载或补抓任何授权数据。
- 本轮数据/训练验证没有启用 Physics，也没有 GORP 或力可视化内容。

## 原始数据与 source v2

- `dataset/AMASS`：14,228 个 NPZ，14,230 个文件，26,649,296,208 bytes。
- SMPL body models：6 个文件，197,382,593 bytes。
- split：train 21,460 行 / 17,604 unique；test 2,706 行 / 2,624 unique。
- 新 source：`dataset/generated/sources/realtime_pose_stationary5_v1/amass_60hz_v2`。
- source 结果：28,188 个可复用 NPZ；manifest 28,192 条，其中 28,188 skipped/reused、4 条预期失败。
- source manifest SHA256：`8a1f1e59e4a04cf19ead40de0da5e0b4b91cc427081a401269fb74a7b4404c32`。
- 4 条失败来自两个不足 3 帧的原始动作及其镜像：`EKUT/265/MTR03_poses.npz`、`M/EKUT/265/MTR03_poses.npz`、`KIT/9/WalkingStraightBackwards08_poses.npz`、`M/KIT/9/WalkingStraightBackwards08_poses.npz`。
- 修复了 legacy source 升级规则：仅当 8 个 runtime-contract 字段全部缺失时允许按当前契约补齐；部分缺失、显式版本冲突、`raw_device_world` 或数组/schema 错误仍拒绝。
- source 重建日志：`notes/realtime_pose_v2_rebuild_2026-07-13/source_rebuild_clean.log`。

## base task 策略

- 每个窗口只物化一个完整六点 base task；`rollout_steps=2` 时另存一个相邻 `r01`。
- 30/30/20/20 mask 不预展开到磁盘，由 Dataset 按固定 seed 在线生成。
- 训练窗口额外保存 32 帧窗口前 Resolver context，用于跨窗口恢复 Hip 丢失/重连状态。
- 同一 base task 连续 10 次在线访问固定得到：3 full-six、3 standard-three、2 static-sparse、2 dynamic-dropout。
- main 与 `r01` 使用同一完整序列 mask timeline；抽检 `main[1:] == r01[:-1]`，Head 全程有效，最大同时缺失 3。

## smoke 数据（只用于连通性）

- task run：`dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_v2_base_smoke_tasks/20260714_000705_v2_base_smoke_seed10`。
- train/test 各 256 个 base task + 256 个 `r01`，共 1,024 NPZ，124,244,416 NPZ bytes。
- train manifest SHA256：`db0d1691025710debd725ecc5ae2a370d4edad93acbb5714c8a337c567bab2b6`。
- test manifest SHA256：`d37f004f320706a2a3a886078bf255074c4a3997026536bd448d6f6fcf1cc82a`。
- smoke normalizer：`dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_v2_base_smoke_train/20260714_000852_v2_base_smoke_seed10`。
- normalizer 使用 256 × 10 = 2,560 个在线 mask 样本，156,160 帧；类别比例精确为 30/30/20/20；validity/stationary 均保持 mean=0、std=1；无效 Tracker 归一化后为零。
- 2-step 训练：`runs/realtime_pose_stationary5_v1/v2_base_smoke_20260714/20260714_001026_v2_base_smoke_2step/model000000002.pt`；32 samples，step 2 loss=0.195，无 NaN。

## formal base tasks

- run：`dataset/generated/tasks/realtime_pose_stationary5_v1/amass_60hz_v2_base_tasks/20260714_001222_v2_base_seed10`。
- train 70,216 个 base tasks；test 10,456 个；总计 80,672 个 manifest entries。
- 每个 base task 都有一个 `r01`，共 161,344 NPZ，19,564,668,992 NPZ bytes。
- train/test 因 `required_frames=62` 分别跳过 50/10 个短 source 窗口。
- train manifest SHA256：`b45b0a2d75f92bc4b0c1bcc0ff8c97f3794e02ea129b0a325c764c4101f951d6`。
- test manifest SHA256：`f7a760e5a0b034d0e1507edbb0a8cbb066f5d282a6545cba32413c72babdd748`。
- 完成时 D 盘余量：461,973,495,808 bytes。
- 先前误启动的预展开目录经计数确认未切换 latest pointer 后，由主线程安全删除。

## formal normalizer

- 输出：`dataset/generated/normalizers/realtime_pose_stationary5_v1/amass_60hz_v2_train/20260714_003252_v2_base_online_seed10`。
- train 70,216 × 10 = 702,160 个在线 mask 样本，42,831,760 帧；运行耗时 49:38。
- 类别计数：full-six 210,648、standard-three 210,648、static-sparse 140,432、dynamic-dropout 140,432，精确为 30/30/20/20。
- task manifest SHA256 与 formal train manifest 一致：`b45b0a2d75f92bc4b0c1bcc0ff8c97f3794e02ea129b0a325c764c4101f951d6`。
- codec/reference-policy hash：`ba0d881d5a8c57ecd4dd2e07d7efe8ec1256e840b61a3728a7b2c24a12bcd29f`。
- `mean.pt` SHA256：`511ca2bd8129d2e5721182330ccdbbbd18d3658be65b01e507405b53a1a7eb07`。
- `std.pt` SHA256：`5ec680598aeeeeec482cb0bf54ab2d3c46940962ab51c105d27f7077e5602dc0`。
- 检查通过：mean/std 全有限、std 全正；validity/stationary 通道严格 mean=0/std=1；在线 10 样本中的 23 个无效 Tracker position/rotation 通道归一化后严格为零。
- 发现原实现会对同一 task 的 10 次采样重复解压 NPZ；增加 normalizer-only 的只读 last-task base-array cache，训练 Dataset 默认关闭。
- 缓存前后 normalizer mean/std 在 `atol=1e-7` 内一致；测试覆盖连续 10 次类别/张量独立与不同 task 0→1→0 不串数据。
- 首次未缓存尝试在低于 1% 时停止，未写 latest pointer；成功 run 才是上面列出的 latest formal normalizer。

## 200-step Stage A smoke

- 必须使用上面的 formal tasks 与 formal normalizer；不能使用 smoke normalizer。
- 初始化权重：`runs/realtime_pose_stationary5_v1/stationary5_target_dit/20260626_192142_s5_tdit_seed10/model000100000.pt`，通过 `--init_checkpoint` 严格初始化普通模型权重。
- 设置：200 steps，学习率 `5e-5`，full-six / standard-three 精确 50/50，不启用 rollout。
- run：`runs/realtime_pose_stationary5_v1/v2_stage_a_smoke_200_20260714/20260714_012443_v2_stage_a_formal_200`。
- checkpoint：`model000000200.pt`，SHA256 `535cc4ed0ebbe6239d17fe05c59d18c67fd99764a3c0ca57e3e5626f83aa153c`；另有 step 100 checkpoint 和对应 EMA/optimizer 状态。
- step 10 loss：0.0637517177；step 200 loss：0.0257937708；共 3,200 samples；20 条日志行的全部数值均有限，无 NaN/Inf。
- 端到端训练墙钟约 34.64 秒（包含启动与 step 100/200 checkpoint 写盘）。
- WDDM 下 `nvidia-smi` 无法给出 per-process memory；用同配置 batch=16 的 2-step backward probe 采样整卡 memory.used：baseline 3,553 MiB、peak 4,494 MiB、增量 941 MiB。该值是整卡增量，不等同于 PyTorch allocator 精确峰值。
- 普通 `model000000200.pt` 已按当前模型结构严格检查加载，无 missing/unexpected keys。
- 使用 formal test task、formal normalizer 和 DDIM10 完成 1-window inference：exit 0，包含启动/加载/推理/写盘共 2.137 秒；输出 8 个数值数组全部 finite。

## 测试状态

- 实施在线 base-task 策略后的完整 smoke 基线：311 passed，1 skipped（加入两项 cache 测试后重跑）。
- normalizer cache focused tests：4 passed。
- checkpoint strict-load + DDIM10 单窗口 smoke：通过。
- `git diff --check`：通过；仅有现有 Windows LF/CRLF 提示，没有 whitespace error。
