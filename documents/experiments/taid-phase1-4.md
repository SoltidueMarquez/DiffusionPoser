# TAID Phase 1～4 实验记录

## 基线

- baseline tag: `baseline/taid-phase1-4`
- baseline commit: `c397ff8 Merge branch 'codex/dynamic-topology-inpainting-v2' into WWJ-改进`
- experiment branch: `codex/taid-phase1-4`
- 创建日期: `2026-08-03`
- 启动时用户未提交内容: `.codex/skills/diffusionposer-taid-experiment/` 下的 SKILL、agent 配置与 references；本实验不改写或提交这些内容。

## 实验目的

- 先锁定 Root yaw 与 13D Tracker 契约，再实现可独立消融的角色管理、Anchor Prior、FK Innovation、TargetDiT 条件注入、固定区域路由和连续 U→A 转换。
- 保持 144D full-pose `x0`、24 关节 TargetDiT、全身 self-attention、Diffusion schedule、Projected DDIM、raw/deployed 与 rollout 主契约不变。

## 产物约定

- 不提交 `dataset/`、`runs/`、`output/`、`save/`、checkpoint 或其他二进制训练产物。
- 本轮不重建全量数据、不运行长训练、不修改 Unity 稳定接口。
- smoke、mini-batch forward/backward 与短过拟合检查使用 `conda run -n diffusionposer5070`。

## 固定基线

- code commit: `c397ff8`
- B0 ablation: 现有 TargetDiT + 安全 Projected DDIM
- seed: `10`
- projection mode: `all_steps`
- 初始 smoke: `conda run -n diffusionposer5070 python -m pytest -p no:cacheprovider tests/smoke -q`
- 初始 smoke 结果: `91 passed, 13 warnings`，耗时 `14.05s`
- task / normalizer / longseq eval / checkpoint: 当前工作区不可访问，因此尚无可登记的 hash、EMA 和真实长序列指标。
- 证据边界: 当前 B0 仅为 `implemented + tested`，不是 `observed` 效果基线。

## 实验条目

### 000 - Phase 0 契约与 B0

- 状态: `implemented + tested`，尚未形成真实评测 `observed` 结果。
- 改动摘要: 明确 source Actor Root yaw 与 target Pelvis forward heading 是两种语义；task GT、长序列评估和 runtime 均从 144D target/prediction 解码 Pelvis heading，统一 wrap 到 `[-π,π)`。稳定 Tracker 字段明确保持 13D，方案草案中的 velocity/`delta_e` 改为内部派生，不扩展 task 或 Unity 字段。
- 关键文件: `data_loaders/realtime_pose_geometry.py`、`data_loaders/generate_realtime_pose_tasks.py`、`sample/evaluate_longseq_eval_set.py`、`sample/realtime_pose_runtime.py`、`contract.md`。
- 测试命令: `python -m pytest tests/smoke/data_pipeline/test_dynamic_task_contract.py tests/smoke/sample/test_realtime_pose_runtime.py tests/smoke/sample/test_evaluate_longseq_eval_set.py -q`
- 测试结果: `17 passed`；π 邻域样例中 source Actor yaw 与 target Pelvis heading 相差约 `3.1216 rad`，而 target 解码自洽误差约 `2.38e-7 rad`。
- 结果指标: 真实评测产物缺失，待固定产物可访问后补录。
- 结论: π 偏差根因是语义混用，不需要改变 source 分解；B0 继续保持原 144D TargetDiT/DDIM 路径。

### 001 - Phase 1 角色管理器

- 状态: `implemented + tested`。
- 改动摘要: 增加 Unconfigured/Missing/Uncertain/Anchor 四身份、`K0/K1/KR=5/15/15`、连续 `alpha/beta`、hard beta 和五区域 Anchor coverage；NumPy/Torch 使用同一公式，Head 固定 Anchor。
- 关键文件: `data_loaders/tracker_roles.py`、`data_loaders/realtime_pose_config.py`。
- 测试命令: `python -m pytest tests/smoke/data_pipeline/test_taid_tracker_roles.py -q`
- 测试结果: `4 passed`；覆盖 invalid 严格零、未配置与 Missing 区分、5/15 帧边界、Head 约束和 NumPy/Torch 一致性。
- 结论: 角色和连续权重可独立审计，未在本阶段接入模型。

### 002 - Phase 2 / B1 Anchor Prior

- 状态: `implemented + synthetic forward/backward tested`；未进行真实 mini-batch 过拟合，因为工作区无 task/normalizer。
- 改动摘要: 新增轻量 history GRU、逐 Tracker 独立 token 后乘 `alpha` 的 Anchor 聚合、144D pose residual head、Head-relative Root head、contact head、训练期 joint-velocity 辅助 head 和可微 FK；B1 只训练 Prior。损失使用区域 coverage 加权 SO(3)、FK L1、Root xyz/yaw、真实 `target_joints-prev_joints` 速度与 contact BCE。
- 输入隔离: 改变 Uncertain 当前 LeftHand 前 9 维不会改变 Prior pose/root/contact；不得在 Tracker 融合后再 mask。
- checkpoint: 增加 `init_checkpoint`；只允许 B0→B1，保留新 Prior 初始化且不恢复 step/optimizer。
- 关键文件: `model/taid_conditioning.py`、`model/realtime_pose_target_dit.py`、`diffusion/realtime_pose_losses.py`、`diffusion/gaussian_diffusion.py`、`train/training_loop.py`。
- 测试结果: B1 专用 loss 全部有限，backward 后只有 `taid_conditioner.prior.*` 获得梯度。
- 结论: B1 已具备独立短训入口；当前证据只证明契约与梯度边界正确，不代表 Prior 精度上限。

### 003 - Phase 3 / B2～B4 Posterior

- 状态: `implemented + tested`。
- B2: stop-gradient Prior pose/root/coverage/role 通过小 adapter 注入 24 joint token。
- B3: Uncertain Tracker 以绝对观测 token 进入全局区域条件，作为 B4 因果对照。
- B4: 计算 `observed-FK(prior)` 的 position meter 与 rotation radian SO(3) Log residual；上一帧位置 residual 从 `C_(n-1)` 重表达到 `C_n` 后形成 `delta_e`。固定尺度、`tanh` 截断、共享 `12→128→64` MLP、type/duration/contact context；Prior 参数与用于条件的输出均 stop-gradient。
- 训练约束: Tracker position/rotation consistency 使用 `alpha+beta`；B2～B4 继续预测完整 144D raw x0。
- checkpoint: B2～B4 只允许从 B1 初始化；同构 B1～B6 state dict 内保存 ablation code、角色阈值、innovation 配置/尺度与路由权重，resume 任一配置不一致时拒绝。
- 测试结果: B2 Prior 无梯度；零 FK residual/零 token/零区域注入严格成立；SO(3) Log 在 0 与 π 邻域 forward/backward 有限；B2～B4 输出均为 `[B,144]`。
- 结论: B3/B4 只切换绝对/innovation 表达，保持同一模型主干、参数预算入口、采样与评估路径。

### 004 - Phase 4 / B5～B6 路由与连续转换

- 状态: `implemented + tested`。
- B5: 固定 Hip→Torso/弱双腿、Hand→对应 Arm/弱 Torso、Foot→对应 Leg/contact-gated 弱 Torso；Head route row 严格为零。路由只约束直接注入，不改全身 self-attention。
- B6: posterior 条件和观测一致性从 Uncertain hard beta 切换为互补连续 `alpha/beta`，避免第 15 帧路径突变。
- 调用链: training、offline sampling、single runtime 与 batch runtime 都传入 raw 13D、骨架 offsets 和 pose/tracker normalizer；Prior/innovation 每目标帧只准备一次，所有 DDIM step 复用。
- 测试命令: `python -m pytest tests/smoke/train/test_taid_conditioning.py tests/smoke/data_pipeline/test_taid_tracker_roles.py tests/smoke/sample/test_realtime_pose_runtime.py -q`
- 测试结果: TAID/角色/runtime 最终定向集 `29 passed`；LeftHand 不直接进入 Leg、contact Foot 到 Torso 权重大于 swing Foot、B6 连续权重小于 B5 hard 权重、三次模拟 DDIM forward 只调用一次 Prior，真实小型 B6 模型可走通 single/batch runtime。
- 结论: Phase 4 链路具备可审计的确定性路由和连续 U→A 过渡；是否改善掉线/重连必须等固定评测产物后验证。

## 最终静态验证

- 完整命令: `conda run -n diffusionposer5070 python -m pytest -p no:cacheprovider tests/smoke -q`
- 结果: `119 passed, 33 warnings`，pytest 耗时 `11.76s`；warning 均为 PyTorch Transformer nested-tensor 提示。
- 输出契约: B0～B6 均保持 144D；没有修改 Unity 接口，没有创建 dataset/checkpoint/runs/output 产物。
- 未执行: 全量数据重建、长时间训练、固定长序列指标对比。原因是本轮明确禁止前两项，且当前工作区没有可访问的 task/normalizer/checkpoint/longseq 产物。
