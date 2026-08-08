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
- 正式 B0: `runs/taid/b0/20260804_152543_taid_b0_seed10/model000120000.pt`，SHA256 `F52A35989B0BF5E89BB498C79C82F6019EB300E741973EE7A1D6310E0B91FFE0`。
- 正式 B0 EMA: `ema000120000.pt`，SHA256 `3A8008AA4060868341835F7506063BC30464BA4D5D0B7C6AAE1F8C9763F35B83`。
- 固定 train task: `20260804_142644_taid_full_seed10`；`generation_plan.sha256=0269100c95d9ab5732f32daa4b95c94782dd058ecb192c587b91985ebb74c802`，`.realtime_pose_tasks.json` SHA256 `76ACFBA2E691B47FD459FDB99D07FD04D45131DFB884EFA701ACB0A21B4F5173`。
- 固定 normalizer: `20260804_152246_taid_full_seed10/normalizer_meta.json`，SHA256 `A25E6840A5A3EADCB04D940BE735E8C71AC5661EB903A6387714855430E2C95C`。
- 固定 longseq eval: `taid_fixed_test_stress_long_seed10`；`manifest.jsonl` SHA256 `EF145F4690B7784CED13DF26AD4698C330CEE48C1601D0932F5D2F26A10F32DA`，`config.json` SHA256 `9714259C9455B00A1687BFE8AA72213A88ECF351C954FCE34A4BA4CA3D723C11`。
- B0 全集汇总: `output/l/120000e-a-b2-c5-a2bb63c2a3/longseq_eval_summary.json`，SHA256 `B9788CE1109605C3114AA7A6EBC3372E8D1FEEFE1F98F77251F0E7ABBB05C734`；使用 EMA、seed/timeline seed `10`、5-step DDIM、`all_steps` projection、18 条序列。
- 证据边界: B0 已升级为 `observed` 效果基线；B1/B2 仍只有实现证据，尚无真实训练改善结论。

## 实验条目

### 000 - Phase 0 契约与 B0

- 状态: `observed`。
- 改动摘要: 明确 source Actor Root yaw 与 target Pelvis forward heading 是两种语义；task GT、长序列评估和 runtime 均从 144D target/prediction 解码 Pelvis heading，统一 wrap 到 `[-π,π)`。稳定 Tracker 字段明确保持 13D，方案草案中的 velocity/`delta_e` 改为内部派生，不扩展 task 或 Unity 字段。
- 关键文件: `data_loaders/realtime_pose_geometry.py`、`data_loaders/generate_realtime_pose_tasks.py`、`sample/evaluate_longseq_eval_set.py`、`sample/realtime_pose_runtime.py`、`contract.md`。
- 测试命令: `python -m pytest tests/smoke/data_pipeline/test_dynamic_task_contract.py tests/smoke/sample/test_realtime_pose_runtime.py tests/smoke/sample/test_evaluate_longseq_eval_set.py -q`
- 测试结果: `17 passed`；π 邻域样例中 source Actor yaw 与 target Pelvis heading 相差约 `3.1216 rad`，而 target 解码自洽误差约 `2.38e-7 rad`。
- 结果指标: fixed-six MPJPE `3.41 cm`、Root yaw `0.38°`；fixed-three MPJPE `19.64 cm`、Root yaw 均值 `101.93°`。fixed-three 的 `steady_state_60_plus` Root yaw 中位数 `161.13°`、P90 `177.68°`、P95 `178.87°`、`>90°` 比例 `56.94%`、`>150°` 比例 `55.32%`，18 条中有 9 条为 π 模态占多数。
- 结论: 旧的 Actor Root yaw/Pelvis heading 语义错误已修复；当前 B0 仍真实存在三点输入下的 Pelvis heading 相反朝向模态，因此后续以 Anchor Prior 的单一 yaw 真值进行模型层修复，而不是运行时按 180°翻转。

### 001 - Phase 1 角色管理器

- 状态: `implemented + tested`。
- 改动摘要: 增加 Unconfigured/Missing/Uncertain/Anchor 四身份、`K0/K1/KR=5/15/15`、连续 `alpha/beta`、hard beta 和五区域 Anchor coverage；NumPy/Torch 使用同一公式，Head 固定 Anchor。
- 关键文件: `data_loaders/tracker_roles.py`、`data_loaders/realtime_pose_config.py`。
- 测试命令: `python -m pytest tests/smoke/data_pipeline/test_taid_tracker_roles.py -q`
- 测试结果: `4 passed`；覆盖 invalid 严格零、未配置与 Missing 区分、5/15 帧边界、Head 约束和 NumPy/Torch 一致性。
- 结论: 角色和连续权重可独立审计，未在本阶段接入模型。

### 002 - Phase 2 / B1 Anchor Prior

- 状态: `implemented + synthetic forward/backward tested`；正式 task/normalizer 已锁定，但尚未启动本次 Root yaw 修复后的 B1 训练。
- 改动摘要: 新增轻量 history GRU、逐 Tracker 独立 token 后乘 `alpha` 的 Anchor 聚合、144D pose residual head、Head-relative Root xyz head、contact head、训练期 joint-velocity 辅助 head 和可微 FK；B1 只训练 Prior。Root yaw 不再由独立 MLP 预测，而是从 `prior_pose_raw` 的 Pelvis rotation 解码并拼回 `[B,4]` 条件接口。损失使用区域 coverage 加权 SO(3)、FK L1、pose-derived circular Root yaw、真实 `target_joints-prev_joints` 速度与 contact BCE。
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

### 005 - B0 π 模态诊断与 B1 Root yaw 单一真值修复

- 状态: `implemented + pilot observed`；旧 B1 5k pilot 已完成，但后续发现其 FK 训练—部署契约不一致，禁止作为新阶段初始化。
- B0 诊断: 长序列汇总新增 Root yaw 中位数、P90/P95、`>90°`/`>150°` 比例、π 模态首次出现绝对帧、独立进入/退出次数、总 transition 数和 π 模态占多数的序列数；所有字段均自动按 condition、cold-start/steady-state 与既有子分组拆分。阶段门槛只读取 `steady_state_60_plus`。
- 单一真值: Prior Root MLP 输出由 4D 改为 3D xyz；对外 `prior_root_head` 仍为 `[B,4]`，其 yaw 严格由 144D `prior_pose_raw` 的 Pelvis heading 派生。circular yaw loss、FK、innovation、TargetDiT adapter 和 runtime 因而消费同一 yaw，梯度直接进入部署所用 Pelvis rotation。
- 定向测试: `python -m pytest -p no:cacheprovider tests/smoke/train/test_taid_conditioning.py tests/smoke/eval/test_realtime_pose_eval.py tests/smoke/sample/test_evaluate_longseq_eval_set.py -q`。
- 定向结果: `35 passed, 21 warnings`；覆盖 pose/yaw 严格一致、Root xyz 对 heading 隔离、±π 连续有限梯度、B1 仅 Prior 获得梯度、B0→B1 初始化，以及合成 π 比例/transition/首次帧确定值。
- pilot 配置: `.vscode/launch.json` 新增 `TAID 29`（从唯一正式 B0 初始化 B1 5k）和 `TAID 29A`（EMA、fixed-three、前 3 条、5-step DDIM、`all_steps`）。manifest 前三条确认为 `CMU_55_55_13`、`CMU_80_80_69`、`CMU_40_40_12`；B0 steady `>150°` 合并比例为 `67.78%`，因此 pilot 必须降至 `33.89%` 以下才可进入正式 B1。
- 旧 pilot 结果: checkpoint 为 `runs/taid/pilot/b1_yaw/20260805_203224_taid_b1_yaw_pilot_seed10/model000005000.pt`（SHA256 `35C2F24D40B7E863735D78592C9EFA24895B214B6C892848B0AFCD700B3677B8`）。fixed-three 前三条 steady `>150°=13.23%`、中位数 `67.59°`、π 多数序列数 `0`，通过基础 π 门槛；但同期 deployed MPJPE 为 `49.28 cm`，不能据此进入正式 B1。
- 证据边界: 该结果只说明 pose-derived yaw 降低了前三条的 π 比例；由于旧 FK loss 未使用部署 Resolver，它不是有效的完整 B1 候选。

### 006 - B1-CS 冷启动分层采样对照

- 状态: `implemented + pilot rejected`；B1-CS 5k 已完成并因冷启动 π 模态灾难性回退而淘汰。
- 假设来源: B0 前三条 fixed-three 在 `0～59` 帧没有 `>150°`，但 Root yaw P95 已达 `92.10°`；`60～119` 帧 `>150°=45.56%`，`120～299` 帧升至 `66.48%`。待证伪假设是启动期方向不确定性进入 deployed 历史后逐渐锁入 π 模态。
- 单变量实现: 保留普通 B1 的 `cold_start_prob=0.1` 与 `0～59` 均匀采样；B1-CS 只把冷启动概率改为 `0.5`，历史桶权重改为 `0.25/0.20/0.25/0.15/0.15`，冷启动场景权重改为 `0.10/0.60/0.10/0.10/0.10`。模型、loss、rollout、seed、B0、task 与 normalizer 均不变，`taid_ablation` 仍为 `B1`。
- 兼容边界: 新参数缺省时复用旧的 deterministic uniform 抽样；`cold_start_scenario_weights` 只改变部分历史样本，完整 60 帧样本继续使用基础 `scenario_weights` 和旧 RNG 上下文。不修改 task store、13D Tracker、144D 输出或 Unity 接口。
- 诊断: 新增 `by_startup_phase` 六段：`0～14`、`15～29`、`30～59`、`60～119`、`120～299`、`300+`；每段复用完整 Root yaw 分位数和 π 模态统计，并自动进入 condition 汇总。
- 历史 Launch: `TAID 29B/29C/30B` 曾用于该对照；Root/FK 契约修复后已从活动链移除，只在本记录保留历史语义。
- 定向测试: `python -m pytest -p no:cacheprovider tests/smoke/data_pipeline/test_dynamic_task_contract.py tests/smoke/train/test_train_entrypoint.py tests/smoke/eval/test_realtime_pose_eval.py tests/smoke/sample/test_evaluate_longseq_eval_set.py -q`。
- 定向结果: `41 passed`；覆盖旧抽样精确复现、五桶边界、两类非法权重拒绝、冷启动场景隔离、seed/epoch/task 确定性、CLI 解析和六段绝对帧边界/condition 汇总。
- 真实 task 只读检查: 从固定 train task/normalizer 读取首个 B1-CS batch，得到 `pose_history=[32,60,144]`、`tracker_history=[32,60,6,13]`；样本同时覆盖 `H=0`、分层部分历史和 `H=60`，所有无效 pose padding 的非零元素数为 `0`。
- 选择规则: 候选先满足前三条 fixed-three steady `>150°≤33.89%`；两者都通过时，B1-CS 还必须满足计划中的启动 P95、`60～119`/`120～299` π 比例、steady 不回退和 fixed-six `≤10 cm` 门槛，才能取代普通 B1。
- 真实结果: checkpoint 为 `runs/taid/pilot/b1_cold_start/20260805_232514_taid_b1_cs_pilot_seed10/model000005000.pt`（SHA256 `CA0C3923DEE457682367B44B088696FF23EFBC17BA0EB6F2FC45A18C438C54BE`）。虽然 steady `>150°=10.71%`，但 `0～14`、`15～29`、`30～59` 帧的 `>150°` 分别为 `66.67%/66.67%/61.11%`，Root yaw P95 分别为 `176.99°/177.16°/179.05°`，违反冷启动门槛，故淘汰且不得续训。

### 007 - B1 Root/FK 训练—部署契约修复

- 状态: `implemented + verified`；未启动训练、未重建 source/task/normalizer。
- 触发证据: 旧普通 B1 fixed-six 前三条结果 `output/l/5000e-a-b2-c1217d-f09d989dd3/longseq_eval_summary.json`（SHA256 `5480C40C940C4B9BBF1BE940872362AE3D538121C7110A7504A4F40D0300B15D`）的 steady MPJPE 为 `42.75 cm`、MPJRE 为 `76.42°`、非 Head Tracker 位置误差为 `0.684 m`，而 hard rotation 仍为 `1.38e-6°`、Root yaw 无 π 模态。训练日志中的内部 `prior_fk_loss≈0.012 m` 与 deployed 几何严重矛盾，定位为 Root/FK 双路径契约错误。
- 根因: 旧 `prior_fk_loss` 使用 `prior_root_head.xyz + prior_pose_raw` 平移后的内部 FK；runtime 则只用 hard-projected 144D pose，经 Head-Anchored Resolver 固定 Head 和 floor。内部 Root MLP xyz 从未进入部署解析，所以低内部 FK loss 不能保证 deployed 关节正确。
- 修复: B1 主 FK 改为 `deployed_pred_xstart → 反归一化 → rotation6D → Torch Head-Anchored Resolver → target_joints_head_ref`。内部 FK 保留为 `prior_internal_fk_loss`，并新增 Root 3D/XZ gap、joint resolver gap，只读诊断不进入总损失。Root xyz/yaw、velocity、contact 权重保持不变。
- 三级诊断: 单步 loss 直接报告五项 FK/Root 指标；4步 rollout 使用 `rollout_step_1_*`～`rollout_step_3_*` 表示第2～4帧；Python runtime/长序列只读导出 Prior root/joints，并按 condition、cold-start/steady-state 与六段 startup phase 汇总七项 Prior/deployed 指标。B0 的可用率为0，其余指标为 null，不伪造零误差。
- 稳定边界: 部署 Root 继续由稳定 Resolver 唯一决定，`prior_root_head.xyz` 只用于 TAID 内部 Prior/FK innovation/TargetDiT 条件；保持13D Tracker、144D输出、TargetDiT/DDIM、Projected DDIM和Unity接口不变。这是相对 proposal“Root xyz辅助Resolver”的受控差异。
- 基线锚点: 唯一初始化仍是 `runs/taid/b0/20260804_152543_taid_b0_seed10/model000120000.pt`（SHA256 `F52A35989B0BF5E89BB498C79C82F6019EB300E741973EE7A1D6310E0B91FFE0`）；EMA SHA256 为 `3A8008AA4060868341835F7506063BC30464BA4D5D0B7C6AAE1F8C9763F35B83`。旧 B1/B1-CS checkpoint 和 optimizer/EMA 均禁止初始化新 pilot。
- 活动链: `TAID 29` 写入 `runs/taid/pilot/b1_root_fk_contract`；`29A` 固定评估三点前三条，只有通过后才运行 `29D` 六点前三条；`TAID 30` 标明仅在两组门槛全部通过后运行。B1-CS 活动入口已移除，历史产物未删除。

### 008 - B1 Teacher-Forced 与多 Horizon 闭环诊断

- 状态: `implemented + observed`；只新增离线诊断，不训练、不生成 source/task/normalizer，不改变13D Tracker、144D输出、TargetDiT/DDIM、Projected DDIM或Unity接口。
- 当前 checkpoint: `runs/taid/pilot/b1_root_fk_contract/20260806_134128_taid_b1_root_fk_pilot_seed10/model000005000.pt`，SHA256 `E0810140D8C1FFE55C558273C8B4A92947DD4AE8A74AF010A865F029949C6BDC`；EMA SHA256 `32E9EDE6D723371915BD6E0AA8F534B2B76189848E9357DCEB45F7F6F395938D`。
- 29A 已观察结果: `output/l/5000e-a-b2-c1ed27-e9d192b29b/longseq_eval_summary.json`，SHA256 `424CC0E229EBA212DF0CAE15D1502DE5C0C3C6BCCDE3E6BCB3308D85F750B003`。fixed-three steady MPJPE `43.656 cm`，Root yaw 中位数 `55.158°`、P95 `160.205°`、`>150°=9.098%`、π多数序列 `0/3`；只通过 π 比例 pilot 门槛，几何仍明显失效。
- 29D 已观察结果: `output/l/5000e-a-b2-c1217d-e9d192b29b/longseq_eval_summary.json`，SHA256 `E9435A650404B3B5088D21769A132B30B746F9F048199A0A86E9FA0DFA8DF788`。fixed-six steady MPJPE `41.750 cm`、MPJRE `72.487°`、非 Head Tracker position error `0.670 m`，hard rotation mean `1.381e-6°`；明确违反 `≤10 cm` 门槛，因此不得运行正式 B1/B2。
- Teacher-forced 协议: 前60帧只推进现有 runtime；每个 `t≥60` 在采样前用 source GT world rotations 覆写 `[t-60,t)` pose history。Tracker/trajectory/duration/Head 状态保持连续，GT 仅进入该离线诊断，不改变部署入口。
- 闭环协议: 在绝对帧 `60/120/180/...` 前注入前60帧 GT，随后连续运行60帧；只汇总完整块。保存 `diagnostic_horizon_frame=1...60`、GT reset mask、完整块数和未计分尾帧数，并单独汇总 `1/4/15/30/60` 全指标。
- 公平性: teacher-forced 与 closed-loop 分别重建 runtime，但对相同 sequence/absolute frame 重建同一个逐序列 diffusion generator；h1 同帧 deployed pose、world joints和Root必须在 `1e-6` 内一致，否则结论分支固定为诊断契约错误。
- 结果入口: `TAID 29E | 诊断 B1 GT-history 与 1/4/15/30/60 闭环`；默认使用前三条、fixed-three+fixed-six、EMA、seed/timeline seed 10、5-step DDIM和`all_steps`。产物写入 `output/diagnostics/taid_history_horizon/<checkpoint-and-config-id>`，包含汇总JSON、完整曲线JSON和每协议/condition/sequence压缩NPZ。
- 自动分支: teacher fixed-six `≤10 cm`、closed h1 `≤10 cm`、hard mean `≤1e-5°`、teacher fixed-three `>150°≤33.89%`且无π多数序列、h1配对一致全部通过后，才检查h15绝对/相对恶化和h30/h60继续恶化；满足才允许另行规划15帧 rollout task。任一 Prior 单步门槛失败都回到模型/监督审计。
- 活动链: `TAID 30` 已改名为“禁止运行（等待29E诊断结论）”；现有 pilot、29A/29D和后续29E产物只作诊断证据，不作为新的训练初始化。
- 定向验证: `tests/smoke/sample tests/smoke/eval` 为 `27 passed, 2 warnings`；新入口聚焦测试为 `8 passed`，覆盖60帧GT覆写、状态隔离、协议mask、尾块排除、逐帧GT替换、协议同噪声、单/批一致、B0 Prior null、JSON/NPZ与自动分支。
- 完整验证: `conda run -n diffusionposer5070 python -m pytest -p no:cacheprovider tests/smoke -q` 为 `145 passed, 38 warnings`；warning仍仅为既有PyTorch Transformer nested-tensor提示。`TAID 29E` Launch参数解析、模块`--help`、`git diff --check`和新增文件尾随空白检查均通过。
- 真实资产只读检查: 使用29E固定参数成功加载当前5k checkpoint同目录EMA，解析到18条固定长序列、首条`CMU_55_55_13_poses`共4869帧、5个DDIM timestep及normalizer run `20260804_152246_taid_full_seed10`；GT world state可覆盖全部4869帧。该检查未执行采样、训练或产物写入。
- 29E真实结果: 汇总为 `output/diagnostics/taid_history_horizon/5000e-h60-b2-7a51b0c480/history_horizon_diagnostic_summary.json`，SHA256 `6CFE9CA9985C6BBA9B6A34518F61C0C4762A439A8EDE8D6B7F095F0E9636249F`；完整曲线 SHA256 `86B1ED101B22128550B5A4F5398B7AD134DC8DD56A855FD99001EC142E6B4EBC`。
- 单步能力: teacher-forced fixed-three/fixed-six MPJPE 分别为 `0.692/0.612 cm`；fixed-three `>150°=0.068%` 且 π 多数序列为0，fixed-six hard rotation mean 为 `1.381e-6°`。teacher-forced 与 closed-loop h1 在438个配对样本上的 pose/joint/Root 差均为0，诊断实现有效。
- 闭环曲线: fixed-three h1/h4/h15/h30/h60 MPJPE 为 `0.677/2.586/8.602/14.985/21.912 cm`；fixed-six为 `0.608/2.314/7.536/13.054/18.960 cm`。teacher-forced和h1均远低于10 cm，而误差随预测历史连续注入迅速扩大，自动分支为 `plan_15_frame_rollout_experiment`，确认当前主要失败属于 exposure bias。
- 结论: 允许进入受控 K15 小型 task 实验，但29E checkpoint只提供诊断证据，仍禁止续训、正式B1、TAID 30和B2。

### 009 - B1 15帧闭环训练配对实验

- 状态: `implemented + observed + rejected`；R4-Control与R15均已完成5k训练和固定29E曲线评估，R15改善长horizon但未通过h15配对门槛，禁止继续旧29M/29N或作为后续初始化。
- 实现边界: 新统一上限 `REALTIME_POSE_MAX_ROLLOUT_STEPS=15`；生成器、绝对 Tracker timeline 与训练循环统一接受 `1…15`，默认仍为4。K15 shard动态物化15个 target、75帧五场景状态，Dataset返回基础帧加14个 rollout item，Transformer窗口仍为61。
- source抽样: 新增 `limit_selection=prefix/stratified`；默认 prefix 保持旧行为。本实验固定从已排序 train split 的1024个等距区间按 seed10各选一个 source，`base_windows_per_source=1`，不重建 AMASS/source。
- normalizer: 新入口 `data_loaders.reuse_realtime_pose_normalizer` 严格核对正式 K4 normalizer plan hash、K4/K15 task source、144D/13D/五场景及真实 shard 形状，逐字节复制六个统计文件，并以 K15 plan hash 写新 metadata和全部来源 SHA256；没有 plan mismatch 绕过。
- 显存预检: 新入口 `train.validate_realtime_pose_rollout` 依次尝试 batch `32→16→8→4`，每次从正式 B0 构建 B1并对真实15帧batch执行 forward/backward；不 optimizer step、不保存 checkpoint，只有 loss有限、`step_14`诊断齐全、仅 Prior有有限非零梯度且CUDA peak reserved不超过14 GiB才接受。
- 配对训练: `TAID 29I` R4-Control 与 `TAID 29J` R15 使用同一 K15 task、复用normalizer、正式B0、seed10、5k、EMA、batch、场景和loss；两份Launch除目录/名称外唯一模型训练变量为 `rollout_steps=4/15`，均独立从B0初始化。
- 真实数据产物: K15 task为 `dataset/taid_rollout15/tasks/20260807_000135_taid_rollout15_seed10`，train共1018个source/task、K=15、13D Tracker，generation plan SHA256为 `F20DE90C5C89C6AED8435E9D48FE833B4CB24F2C54AAA1765B7AA0DCBE2BCF4C`；`task_store.json` SHA256为 `BE9552F79607FE6A0DB33856D6DA7515CF74A280F51598D657D25B95420D6C1A`。
- 真实normalizer: `dataset/taid_rollout15/normalizer/20260807_000230_taid_rollout15_seed10`，metadata SHA256为 `3EFEEF8E9E76359FBD92A65A44BFADC295492B2C2A7C7A5EB69DC408AD2310F2`；绑定同一K15 plan，六个统计文件逐字节复用正式normalizer，没有重算统计。29H以`rollout_prob=1.0`完成15帧真实batch forward/backward，最终选择batch=4，因此R4、R15及后续P100均固定使用4。
- 真实checkpoint: R4为 `runs/taid/pilot/r4_control/20260807_000344_taid_b1_r4_control_seed10/model000005000.pt`，model/EMA SHA256分别为 `D57BD1D1D0B4BBC0DB764A981CCE0256DEBC9974AD5012FF00F771962AE01C11` / `DCE75CC57378CC8E603147C9D6D0858417D62440A51F888FA441FFE4BE7E5B09`；R15为 `runs/taid/pilot/r15/20260807_020133_taid_b1_r15_seed10/model000005000.pt`，model/EMA SHA256分别为 `597BA4FAB473AF57DB8414CAB5BF939C5564901EA10A44C1135E39412E07CC63` / `BBD021B02A01F11363FDE7E4C0781438C5B1F1845E7B29AAD70302CCCD28A228`。
- 29K/29L证据: R4 summary/curve SHA256为 `3017D12AD2A8584FA311CBBBCC8D58475F00799761309C955385ECDBFA6A0E9B` / `873BD77180258C9936981F15F5D7B2E03159BEE809ED51E880DB1C4CED01B7A2`；R15为 `8F94B21FD941F0066A229B6704788B5D5E139CF7B190E2035144B0C480B49B54` / `BE1F0CFF29431C6972B54886E6A8865F2C06CAD638CF4C0A1D7EFF70C9BB837A`。两者teacher-forced与closed h1的pose/joint/Root配对差均为0，hard rotation约 `1.4e-6°`。
- 曲线结论: fixed-three的R4→R15 h1/h15/mean(h30,h60)为 `1.011→0.880 cm`、`11.293→10.079 cm`、`27.244→19.334 cm`，改善 `13.01%/10.75%/29.04%`；fixed-six为 `0.905→0.820 cm`、`9.887→9.227 cm`、`23.375→17.681 cm`，改善 `9.42%/6.67%/24.36%`。两条件长horizon门槛通过，但h15均未达到至少25%的改善，故R15整体淘汰。
- 训练审计: rollout batch内R4三个后续帧各占frame loss的`1/3`，R15十四个后续帧各占`1/14`；二者`rollout_prob=0.5`时，每个R15后续状态的期望直接监督系数仅为 `0.5/14`，约为R4早期状态的21.4%。该事实支持先做闭环曝光率单变量实验，不足以直接证明需要改loss或模型。
- 活动链: 历史`29M/29N`已明确标记为R15曲线失败、禁止运行；后续由条目010的`29O～29R`接管，`TAID 30`继续禁止运行。
- smoke证据: data pipeline `48 passed`，train `83 passed, 37 warnings`，完整 `tests/smoke` 为 `159 passed, 39 warnings`。覆盖默认K4、显式K15/K16拒绝、K15各动态字段与75帧状态、同task K4前缀逐字段一致、H=0/59/60历史推进、future-leg额外3帧、15帧时序loss/逐step诊断/Prior-only backward、normalizer字节SHA与三类不匹配拒绝、预检候选/梯度摘要和Launch配对参数。
- 证据边界: 结果证明K15 uniform训练在5k预算下能减缓30～60帧漂移，但不能证明15帧内的exposure bias已经解决；不运行旧R15长序列门槛，也不据此进入正式B1/B2。

### 010 - B1 R15-P100 闭环曝光单变量实验

- 状态: `observed + rejected`；P100增加30～60帧闭环曝光后改善长horizon，但fixed-three/fixed-six的h15均未过门槛，旧29Q/29R已禁止运行，不作为后续初始化。
- 唯一变量: 从正式B0独立训练B1 5k，完全复用条目009的K15 task、normalizer、batch=4、seed10、EMA、cold-start、五场景、loss和`detach_rollout_history=true`；相对R15只把 `rollout_prob=0.5` 改为 `1.0`。输出固定写入 `runs/taid/pilot/r15_p100`，不得续训R15。
- 真实checkpoint: `runs/taid/pilot/r15_p100/20260807_174735_taid_b1_r15_p100_seed10/model000005000.pt`；args/model/EMA/progress SHA256分别为 `D48DEF848DCD753043BC3AD6848A6D23B076C00892ED52167C8D21FD1E795E95` / `4C6A2DC60FF4A21C1593166BFC54D49A0F4CC9C85421049561A91C4ECE1184B1` / `913445BA75350AF2360633C5126AFEEB94215C9D4F530EB1C813DFEE8D37D84E` / `8E7B679E9E686D4C2AC5F6B710BD543CBE4FA408D9B52EF6EC27626AA4D84588`。
- 29P证据: 结果目录 `output/diagnostics/taid_history_horizon/5000e-h60-b2-c01bcb7c34`；summary/curve SHA256为 `555A360A9AB18339DF26BEE48AF4D01A00E27954B6BB7D46C283BD2E98EBCC7E` / `28E06DE31D33A3542C859AFDA1536BE1B1B7AF78BDBD3ADAFAFCD8C502BD5C29`。teacher/closed h1三类配对差均为0，hard rotation约 `1.38e-6°`；teacher fixed-three/fixed-six MPJPE为 `0.934/0.810 cm`，两条件均无π多数序列，实现有效性通过。
- 曲线结论: fixed-three h1/h15/mean(h30,h60)为 `0.973/10.060/18.554 cm`，相对R4改善 `3.72%/10.91%/31.90%`，相对R15为h15改善 `0.19%`、长horizon改善 `4.03%`；fixed-six为 `0.848/8.594/15.577 cm`，相对R4改善 `6.26%/13.07%/33.36%`，相对R15为h15改善 `6.86%`、长horizon改善 `11.90%`。两条件h15仍高于 `8.470/7.415 cm` 上限，29P整体失败。
- Launch结论: 29O/29P保留为历史证据，29Q/29R标记为P100曲线未通过、禁止运行；后续由条目011的29S～29W接管。
- 29P实现门槛: teacher/closed h1三类配对差为0；hard rotation `≤1e-5°`；teacher fixed-six MPJPE `≤1.096 cm`；teacher fixed-three `>150°≤33.89%`且π多数序列为0。
- 29P曲线门槛: fixed-three/fixed-six h1分别 `≤1.264/1.131 cm`，h15分别 `≤8.470/7.415 cm`，mean(h30,h60)分别 `≤20.300/18.565 cm`，两条件h15 π多数序列均为0。任一失败即停止，不运行29Q/29R；下一轮才允许单独规划后段线性加权。
- 证据边界: P100说明“增加出现闭环batch的数量”能继续抑制30～60帧漂移，但不能把15帧误差压到配对门槛；因此结束 rollout probability 调参，只允许再做一次后续帧内部时间权重单变量实验。

### 011 - B1 R15-LW 后段线性加权实验

- 状态: `observed + rejected`；29S/29T/29U均已完成，29U的h15未过门槛，因此29V/29W已标记禁止运行，旧checkpoint不得续训或初始化后续实验。
- 唯一变量: 新增 `rollout_frame_weighting=uniform/linear_late`，默认uniform直接调用原 `mean`，确保旧总loss和梯度行为不变。R15-LW从正式B0独立训练5k，相对原R15只把策略从uniform改为linear_late，仍固定K15、`rollout_prob=0.5`、batch4、seed10、EMA、cold-start、五场景、detach及全部loss权重。
- 权重契约: 对后续step 1～14使用 `s/sum(1…14)`，即首尾权重 `1/105` 与 `14/105`，总和为1。只加权逐帧 `rollout_loss`；joint velocity、rotation velocity、foot-slide和每帧B1 loss组成保持不变。日志同时保留原始 `rollout_step_i_loss`，新增 `rollout_uniform_frame_loss`、`rollout_step_i_weight` 与 `rollout_step_i_weighted_loss`。
- 预检: `TAID 29S`固定真实K15 batch=4、linear_late和正式B0，执行forward/backward但不optimizer step、不写checkpoint；除有限loss、Prior-only有限非零梯度和CUDA peak reserved `≤14 GiB`外，还必须程序化核对14个实际权重与理论值一致。
- Launch顺序: `29S`预检已通过；`29T`从正式B0完成R15-LW 5k；`29U`完成曲线后因h15失败停止，29V/29W、正式B1、TAID30和B2继续禁止运行。
- 29U门槛: teacher/closed h1的pose/joint/Root差为0，hard rotation `≤1e-5°`，teacher fixed-six `≤1.096 cm`且teacher fixed-three无π多数序列；fixed-three/fixed-six的h1分别 `≤1.264/1.131 cm`、h15分别 `≤8.470/7.415 cm`、mean(h30,h60)分别 `≤20.300/18.565 cm`，两条件h15 π多数序列均为0。
- 长序列门槛: 29V steady fixed-three要求MPJPE `≤30.56 cm`、Root yaw中位数 `<45°`、`>150°≤10%`、π多数序列0；通过后29W steady fixed-six要求MPJPE `≤10 cm`、无Root yaw π模态、hard rotation `≤1e-5°`。全部通过后仍不续训pilot或自动启动正式B1/B2。
- 真实checkpoint: `runs/taid/pilot/r15_linear_late/20260807_230647_taid_b1_r15_linear_late_seed10/model000005000.pt`；args/model/EMA SHA256分别为 `E1ED31C51D6D3730A1531636E8DF48AEE5AA1EA54642C17E1F04953C51792E81` / `EB5A4CD99A365017F515C8E0F8494EE2E37A2CDC938BD8D923B4B860BC17202C` / `6223FFCDBBE8537211892B4852BB26382FA101E4938AA45934D306B5B5A618F8`。
- 29U证据: 结果目录 `output/diagnostics/taid_history_horizon/5000e-h60-b2-837c4b0327`；summary/curve SHA256为 `9F39600ACDCFB4C0C07610CDC0E58633F4BD7780F0EEA98C5B5F9019BA5AC60F` / `151640188FC65FC0C0F69C8528DD247B735B383968C6084FF190543B127B0417`。teacher fixed-three/fixed-six MPJPE为 `0.880/0.779 cm`，h1为 `0.898/0.796 cm`，hard rotation约 `1.4e-6°`，实现有效性通过；h15为 `9.719/8.634 cm`，均高于 `8.470/7.415 cm` 门槛；mean(h30,h60)为 `19.127/16.868 cm`，长期门槛通过。
- 结论: 后段线性加权继续改善30～60帧漂移，但不能解决15帧误差；结束rollout概率/时间权重调参，转入Prior结构与监督能力审计。

### 012 - B1 Prior能力审计与 Tracker-history 单变量结构修复

- 状态: `29X/29Y/29Z/29AA completed + curve gate failed`；旧结构审计严格先于 `tracker_fusion` 形状修改执行，29Z从正式B0独立训练，未续训任何旧B1。
- 29X固定证据: 使用条目011的R15-LW EMA、K15 task、复用normalizer、batch2和现有29U NPZ。输出目录为 `output/diagnostics/taid_prior_capacity/model000005000-c23cf6f9cf84`；summary/input/gradient/regional SHA256分别为 `951D6FF9521C40B2437EDD2AC30AB5E53784679B47F50C9615C9D81CF83A074B` / `E4F40460A83084000D94E51695BFF325CB5C295E9181278941EA99E79546844B` / `BC686A97FD67AB6CCA30A73722EDDB42A34AF5C1D979F7EEAF7E6DF73A452F70` / `42B31BD9F1663E712174220485819CF4C46345E66196B97162C76ED292254FB8`；审计时代码diff指纹为 `1BE26099E437FF9B57FF83AD990C606AB72EDCE1B2352661842DA22FF7582242`。regional文件随后只读补算raw Prior经runtime Resolver的MPJPE、hard projection增益及完整MPJVE/MPJAE端点，未重跑模型。
- 输入审计结论: 交换 `pose_history` 时Prior raw pose最大变化 `1.11337`、deployed joint平均变化 `0.25763 m`；交换整个 `tracker_history` 时pose/joint/root/contact/velocity全部严格为0。current Tracker和trajectory均有非零敏感度；强制 `alpha=0` 的Tracker历史扰动严格零泄漏。由此确认当前Prior没有消费Observation Encoder已经计算的60帧 `history_summary:[B,6,D]`。
- 梯度审计结论: auxiliary `prior_velocity_loss` 对velocity head和shared fusion均有有限非零梯度，对pose head为0；rollout joint/rotation velocity loss对pose head均有有限非零梯度。B1只有 `taid_conditioner.prior.*` 可训练，Observation Encoder冻结，说明本轮不需要改velocity监督定义。
- 区域审计结论: closed-loop前一帧与当前MPJPE的序列内相关系数约为fixed-three `0.960`、fixed-six `0.959`，反馈污染非常明显。h15的平均区域MPJPE（Torso/左臂/右臂/左腿/右腿）约为fixed-three `4.03/10.15/11.93/15.84/15.41 cm`、fixed-six `3.81/10.11/11.90/13.15/12.60 cm`；腿和手臂明显高于Torso。teacher raw→deployed rotation仅由 `1.366→1.135°`（三点）和 `1.362→1.015°`（六点）；同一runtime Resolver下hard projection的teacher joint MPJPE增益约为三点 `0.000 cm`、六点 `0.096 cm`，不能解释闭环h15增长。
- 单变量修复: `AnchorPriorRegressor.tracker_fusion` 输入由 `3D` 改为 `4D`，每个Tracker独立融合 `[state,position,rotation,history_summary]` 后才乘 `alpha` 并跨Tracker加权平均。pose history GRU、coverage、trajectory、所有heads、Resolver、loss、K15 rollout和训练参数保持不变；Observation Encoder继续冻结。
- checkpoint边界: 正式B0仍按缺失全部TAID参数初始化当前B1；旧B1/R15/P100/LW因 `tracker_fusion` 权重形状不同不再兼容，禁止恢复或续训，不增加兼容开关。固定六槽拼接与pose-coupled velocity监督继续作为受控剩余变量，本轮不处理。
- 活动链: `29X`为已完成旧结构审计记录；`29Y`从正式B0执行真实K15 batch=4 forward/backward并额外验证active history梯度非零、alpha=0 history梯度严格为0；`29Z`从正式B0独立训练5k；`29AA`重跑29E曲线。因29AA未通过，29AB/29AC已封存，29V/29W、TAID30和B2继续禁止运行。
- 29Y真实预检: 从正式B0成功初始化新结构；batch4、15步linear-late forward/backward的全部loss/gradient有限，`step_14`诊断齐全，CUDA peak reserved `3.5625 GiB`。34个有梯度的参数张量全部位于Prior；active Head history输入梯度范数 `1.1455966887e-4`，强制 `alpha=0` 的Tracker history输入梯度严格为0，Observation Encoder冻结。未执行optimizer step，未写checkpoint。
- 29Z产物: `runs/taid/pilot/r15_tracker_history/20260808_153405_taid_b1_r15_tracker_history_seed10/`；args/model/EMA SHA256分别为 `652836B3887167B9938898CA48BA53134667EC683C4E13F22E74D099521FD5E3` / `97C0F9822FCB4456C439AC3CC6C5AF8A53966B6246D51BFB817C9950D6755263` / `E3D81E5413758F86A8A2DF9CE4E872770EC9CF8F5B2FAE6D2ED89A49E3320385`。
- 29AA证据: 结果目录 `output/diagnostics/taid_history_horizon/5000e-h60-b2-6b031e0088`；summary/curve SHA256为 `18DE1773FF9598BB8B297B1A7465BE4F5C45FBEBF7083D3AE4DBE064AD601E47` / `3C2E631E740181EA88CD5ECCDBE8B4AAFD57607F8D656320476613759E6A7789`。teacher fixed-three/fixed-six MPJPE为 `0.943/0.836 cm`，h1为 `0.952/0.850 cm`，hard rotation约 `1.4e-6°`，实现有效性通过；h15为 `10.578/9.361 cm`，相对R15-LW的 `9.719/8.634 cm` 分别恶化约 `8.84%/8.42%`；mean(h30,h60)为 `21.016/18.716 cm`，分别恶化约 `9.88%/10.96%`。teacher与h15均无π多数序列。
- 结论: Tracker history 已正确接入且隔离契约成立，但当前“六个Tracker压成一个平均token”的用法没有改善闭环误差。29AB/29AC不得运行；下一轮只审计跨Tracker聚合方式，不同时修改velocity监督或训练预算。

### 013 - B1 固定六槽 Tracker 聚合单变量实验

- 状态: `implementation complete + smoke pending + awaiting 29AD`；未运行预检、训练或评估。
- 单变量修复: 每个Tracker仍独立融合 `[state,position,rotation,history_summary]→D`，乘 `alpha` 并除以 `sum(alpha)`；随后按 Head、LeftHand、RightHand、Hip、LeftFoot、RightFoot 固定顺序保留为 `[B,6,D]`，展平后由无bias的 `anchor_slot_projection:[D,6D]` 压回 `[B,D]`。不再直接求和。
- 初始等价性: `anchor_slot_projection.weight` 初始化为六个单位矩阵横向拼接，形状固定为 `[512,3072]`。因此开始训练前与29Z的加权平均逐元素等价，不引入随机初始化差异；训练后才允许六个身体槽位形成不同权重。
- 隔离与连续性: `alpha` 在固定槽投影之前相乘，所以 `alpha=0` 的current/history输入及输入梯度严格为0；`alpha=0/0.5/1.0` 继续连续变化。Observation Encoder保持冻结。
- 参数与checkpoint边界: `args.json`固定记录 `taid_prior_tracker_aggregation=fixed_slots`。新state dict增加固定槽投影参数，29Z及更早B1 checkpoint不得加载或续训；正式B0仍可按缺失全部TAID参数初始化新B1。13D Tracker、144D输出、Root/Resolver、K15 rollout、loss、DDIM和Unity接口不变。
- 活动链: `29AD`执行正式B0+真实K15 batch4的15步forward/backward，只读核对投影形状、六槽梯度、alpha=0泄漏、step14、Prior梯度边界和14 GiB显存门槛；通过后`29AE`从正式B0独立训练5k。`29AF`重跑29E曲线，只有绝对门槛全部通过才运行29AG/29AH。旧29AB/29AC、TAID30和B2继续禁止运行。
- 曲线门槛: fixed-three/fixed-six的h1分别 `≤1.264/1.131 cm`，h15分别 `≤8.470/7.415 cm`，mean(h30,h60)分别 `≤20.300/18.565 cm`，h15 π多数序列均为0；相对29AA的10%方向门槛为h15 `≤9.520/8.425 cm`。任一条件改善不足10%或继续恶化即停止固定槽方向，下一轮再单独规划pose-coupled velocity监督。

## 最终静态验证

- 完整命令: `conda run -n diffusionposer5070 python -m pytest -p no:cacheprovider tests/smoke -q`
- 结果: 固定六槽实现后的完整 `tests/smoke` 为 `181 passed, 47 warnings`，最终复跑耗时 `10.05s`；warning均为PyTorch Transformer nested-tensor提示。train分域为 `105 passed, 45 warnings`，sample+eval为 `27 passed, 2 warnings`。
- 真实 checkpoint 兼容: 使用正式 B0 `model000120000.pt` 构建固定六槽 B1 后，阶段检查为 `B0→B1`；59个缺失项全部为预期的 `taid_conditioner.*`，主干无 missing/unexpected key。真实29Z `model000005000.pt` 因缺少固定槽参数/配置契约变化被拒绝加载，符合禁止续训边界。
- 真实 task 前向: 从固定 train task/normalizer 读取 `[1,144]` batch，并在 RTX 5070 Ti 上从正式 B0 初始化完整 B1。单步 `prior_fk_loss=0.001124 m`；4步前向 `loss=0.590819`，第2～4帧全部 FK/Root gap 诊断键齐全、所有张量有限，且可训练参数仍仅为 `taid_conditioner.prior.*`。未执行 backward、optimizer step 或 checkpoint 保存。
- 静态检查: `.vscode/launch.json` 可解析、共52个配置且名称唯一，`git diff --check`无错误。K15历史链为 `29F～29L`；旧`29M/29N`、29Q/29R、29V/29W、29AB/29AC分别因R15/P100/R15-LW/Tracker-history曲线失败封存；新活动链为29AD→29AE→29AF，只有29AF通过才运行29AG/29AH。29AE与29Z除run信息和固定槽聚合标记外训练参数一致；`TAID 30`继续禁止运行。
- 输出契约: B0～B6 均保持144D，Tracker保持13D，对外 `prior_root_head` 保持4D；没有修改Unity接口，没有创建或修改dataset/runs/checkpoint。只新增ignored的29X只读诊断output，不纳入提交。
- 本节记录的是 Root/FK 代码修复完成时的静态验证；随后完成的 B1 pilot 与29A/29D实测见条目008。仍未执行全量数据重建、正式B1或B2，历史失败 pilot 也不得作为后续初始化。
