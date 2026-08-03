---
name: diffusionposer-taid-experiment
description: DiffusionPoser“动态可信锚点 Prior + FK Innovation 条件扩散”方案的审计、实现和消融实验规范。用于根据新算法文档实现或评审 Tracker Anchor/Uncertain/Missing 角色状态机、轻量 Prior heads、Head-relative Root、可微 FK innovation、固定区域路由、TargetDiT 条件注入、分阶段训练、掉线重连数据与事件指标；也用于建立 B0-B10 消融、修复相关契约缺口、添加 smoke tests，或判断该方案是否真正改善多 Tracker 配置、掉线与重连效果。
---

# DiffusionPoser TAID 实验

## 核心原则

把方案视为待证伪的研究假设，不把文档公式当作已实现或效果改善的证据。始终区分：

- `proposal`：算法文档建议做什么；
- `contracted`：`contract.md` 当前允许什么；
- `implemented`：当前分支实际执行什么；
- `tested`：自动测试实际覆盖什么；
- `observed`：固定 checkpoint 和评测产物证明什么。

同时应用 `$diffusionposer-repro`、`$diffusionposer-experiment-branch` 和 `$diffusionposer-code-review`。只做方案审阅时保持只读；用户要求实施时再创建实验分支、修改代码和测试。

将 `TAID` 只作为本实验的简称。未经用户明确确认，不新增或修改稳定 schema 名称、数据版本名或 Unity runtime 接口版本。

## 必读材料

开始前完整读取：

1. [taid-proposal.md](references/taid-proposal.md)：同学提供的完整算法方案；
2. 仓库根目录 `AGENTS.md`、`contract.md`、`documents/算法架构.md`；
3. 当前入口涉及的 source、task、Dataset、normalizer、model、diffusion、train、sample 和 eval 实现；
4. 基线 checkpoint 的 `args.json`、task/normalizer metadata、训练日志和评估 JSON。

不得用 reference 替代当前源码。发现 proposal、contract、代码或实验产物冲突时并列记录并先解决契约，不静默选择一方。

## 启动审计

1. 运行 `git status --short --branch` 和 `git log -n 5 --oneline`，保留用户已有改动。
2. 锁定 AMASS source、task、normalizer、checkpoint、EMA、seed、采样步数、projection mode 和评测集。
3. 用当前 baseline 重跑相关 smoke tests，并保存 B0 的长序列指标；smoke 通过只证明链路可运行。
4. 按 `$diffusionposer-experiment-branch` 创建 baseline tag、`codex/` 实验分支和 `documents/experiments/` 记录。
5. 先输出一份“文档要求 → 当前实现 → 差距 → 拟改文件 → 验证方式”的映射，再开始编码。

## 两项实施前门槛

### Root yaw 契约

先验证以下 GT 自洽不变量：

```text
wrap(
  current_head_yaw_world
  + decode_target_head_rotations_np(current_target_raw).pelvis_heading
) == target_root_yaw_world
```

当前已观察到一个 source 序列中左右两侧稳定相差约 π。根因未修复并由 smoke test 锁定前：

- 不训练 Prior Root yaw head；
- 不启用 Root yaw loss 或用该指标判断模型优劣；
- 不把 180° 误差归因于 checkpoint；
- 不扩展依赖该字段的闭环 Root 逻辑。

### Tracker 维度契约

proposal 写的是 `[61,6,15]`，当前 `contract.md` 是 `position3 + rotation6D + configured + measured_valid + d_off + d_on = 13D`。以当前 contract 为基线：

- 优先从 tracker history、当前帧和 innovation history 派生速度或 `delta_e`；
- 不为对齐文档而静默增加两个稳定字段；
- 如果算法确实必须改变稳定字段或 Unity 输入，先列出生产端、normalizer、checkpoint、runtime、export 和测试影响，并征得用户确认。

## 第一版允许的算法范围

只实现可独立消融的最小组合：

1. 确定性 Tracker 角色管理器；
2. 复用现有编码器的轻量 Anchor Prior heads；
3. Head-relative Root 内部状态和可微 FK；
4. 共享 MLP 的 FK Innovation Encoder；
5. 固定 Tracker-to-region 路由；
6. 向现有 TargetDiT 注入 Prior、innovation 和 role 条件；
7. 分阶段训练和事件评估。

保持不变：

- 144D full-pose `x0` 预测目标；
- 现有 24 关节 TargetDiT 主干和全身 self-attention；
- 现有 diffusion schedule 与 DDIM 主流程；
- 现有 raw/deployed、rollout 和 checkpoint 语义，除非某个受控消融明确改变它们。

第一版禁止同时加入：

- learned reliability gate；
- 概率不确定性头；
- Root/Leg/Arm 三个独立 DiT；
- residual diffusion；
- 每个 DDIM step 重算 FK guidance；
- 复杂 motion codebook；
- 额外 IK 优化器。

不要直接删除现有 Projected DDIM。把“无 hard projection”和“保留安全投影”做成受控消融，确保 B0 仍可复现。

## 算法不变量

### 角色和连续权重

为每个已配置 Tracker 计算 `Missing / Uncertain / Anchor`，同时保留 `configured=0` 与 `configured=1, measured_valid=0` 的区别。Head 始终为 Anchor。

第一版默认：

```text
K0 = 5
K1 = 15
KR = 15
alpha = valid * clip((d_on - K0) / (K1 - K0), 0, 1)
beta  = valid * (1 - alpha) * min(1, d_on / KR)
```

保证同一测量不会以完整权重同时进入 Prior 和 innovation。不要只用第 15 帧的硬切换代替连续权重。

### Anchor Prior

让 Prior 只读取：

- deployed pose history；
- Head；
- 乘过 `alpha` 的 Anchor Tracker token；
- Anchor coverage。

必须在每 Tracker 原始输入或独立 token 尚未融合前应用 `alpha`。不得先融合全部 Tracker 再 mask，否则 Uncertain 测量已经泄漏进 Prior。

输出至少包含：

```text
prior_pose_raw: [B,144]
prior_root_head: [B,4]   # Head-relative xyz + yaw；仅在 yaw 契约通过后启用完整监督
prior_contact: [B,2]
region_coverage: [B,R]
```

### FK Innovation

对 Uncertain Tracker 计算：

```text
e_pos = observed_position_head - prior_fk_position_head        # [B,6,3]
e_rot = Log(prior_fk_rotation_head.T @ observed_rotation_head) # [B,6,3]
delta_e = e_t - e_previous
```

按 Tracker 类型尺度归一化，对异常 residual 使用确定性 `tanh`/Huber 截断。Innovation Encoder 使用共享 MLP 和 Tracker type embedding，读取 residual、`delta_e`、`d_on/d_off` 与 prior contact。

在 posterior 阶段对 Prior 使用 `stop-gradient`。不得允许 Prior 主动制造特殊 residual 作为隐藏通信编码。

### 固定区域路由

第一版使用可审计的固定路由：

- Hip → Root/Torso，次级影响双腿连接；
- Left/Right Foot → 对应 Leg，按 prior contact 弱路由到 Root；
- Left/Right Hand → 对应 Arm，弱路由到 Torso；
- Head → Prior，不进入重连 innovation。

区域路由只限制直接条件注入，不截断 TargetDiT 的全身 self-attention。

### TargetDiT 条件

每个目标帧只计算一次 Prior 和 innovation，然后在所有 DDIM steps 复用。不得对带噪 `x_t` 每步做 FK 并重新定义 innovation。

先通过小型 adapter 将 Prior、region coverage、role 和 innovation 注入现有 joint tokens。继续预测完整 144D `x0`，不要在同一实验里改变输出空间。

## 分阶段实施与消融

每完成一阶段先运行对应测试并更新实验记录；前一阶段没有通过，不叠加后一阶段。

### Phase 0：契约与基线 B0

- 修复并测试 Root yaw GT 自洽问题；
- 固定 13D Tracker 语义；
- 重跑完整 smoke；
- 在固定长序列评测集记录 B0 raw/deployed 与事件指标。

### Phase 1：角色管理器

- 实现 M/U/A、`alpha/beta` 和 region coverage；
- 不接模型，先用纯函数和 deterministic timeline 测试边界与重连连续性。

### Phase 2：Prior only，B1

- 增加轻量 Prior heads、Root/FK/contact 输出和独立训练配置；
- 验证 Uncertain Tracker 不泄漏；
- 用小批次 forward/backward、过拟合检查和独立评估判断 Prior 上限。

### Phase 3：Posterior，B2/B3/B4

- B2：TargetDiT + Prior absolute condition；
- B3：Prior + Uncertain Tracker absolute condition；
- B4：Prior + FK innovation；
- 复用同一套实现和显式实验开关，不复制三套模型代码。

B3 对 B4 是核心因果比较。除 absolute/innovation 表达外，保持数据、参数量、训练预算、seed、checkpoint 初始化、采样和评估完全一致。

### Phase 4：路由与平滑转换，B5/B6

- B5：在 B4 上加入固定区域路由；
- B6：加入连续 `U → A` 的互补 `alpha/beta`；
- 报告目标区域收益与非目标区域扰动。

### Phase 5：闭环事件训练

- 使用 deployed 预测历史做 15～30 帧 rollout；
- 按 Hip、单脚、双脚、手、跨类型和 6→3→6 分开采样事件；
- 比较预测与 GT 的速度、加速度和 jerk，不直接最小化运动幅度；
- 保持 Prior 冻结，除非证据表明其误差限制上限。

只有 B6 在真实评测上成立后，才考虑 B7 learned gate、B8 hard projection、B9 residual diffusion 和 B10 Root-first adapter。

## 数据与训练规则

- 把设备 `configured` 与运行事件 `measured_valid` 分开生成。
- 让事件在 source-absolute 时间线上确定，重叠窗口不得得到矛盾状态。
- 分别覆盖 5、15、30、60、120 帧掉线；不要用“随机两个 Tracker mask”替代分类型事件。
- 第一阶段只处理掉线和重连；spike、jitter、bias、delay 和 calibration error 留到后续实验。
- 继续使用 deployed/predicted history 和 rollout，不得回退到只用 GT history。
- checkpoint `args.json` 必须保存所有 role、prior、innovation、routing、loss 和 ablation 参数。
- 数据、normalizer、runs、save、output 与 checkpoint 继续留在忽略目录，不提交二进制产物。
- 未经用户明确要求，不启动完整数据重建或长时间训练；实现阶段只运行 smoke、mini-batch forward/backward 和必要的短检查。

## 强制测试

至少覆盖：

1. `valid=0` 时该 Tracker 的 `alpha/beta/innovation token` 严格为零；
2. `configured=0` 与临时 Missing 的身份语义不混淆；
3. `d_on=5/15` 附近 `alpha/beta` 连续且无路径跳变；
4. 任意改变 Uncertain 当前测量不会改变 Prior；
5. GT target 解码后的 pelvis heading 与 `target_root_yaw_world` 自洽；
6. 零 innovation 不产生直接区域注入；
7. LeftHand innovation 不直接注入 Leg；
8. swing Foot 到 Root 的权重弱于 contact Foot；
9. posterior FK residual 的计算、单位、旋转 Log 和梯度有限；
10. Prior/innovation 每目标帧只准备一次，DDIM steps 复用；
11. 训练和 runtime 使用相同 role、reference frame 和 normalization 语义；
12. 新 checkpoint 可保存、恢复并拒绝不兼容配置。

跨模块改动完成后运行：

```powershell
conda run -n diffusionposer5070 python -m pytest -p no:cacheprovider tests/smoke -q
```

## 效果验收

固定并报告 commit、task/normalizer hash、checkpoint/EMA、seed、Tracker timeline、DDIM steps 和 projection mode。至少拆分：

- steady 3/4/5/6 点和未见组合；
- Hip、单脚、双脚、手、跨类型掉线与重连；
- raw/deployed；
- cold-start/steady-state；
- `d_off/d_on` 和角色阶段。

除 MPJPE、MPJRE、Root XZ/yaw、Tracker error、contact 和 foot slide 外，报告：

- 重连首帧跳变 `J0`；
- 峰值误差 `E_peak`；
- `AUC_30`；
- 恢复时间 `T_settle`；
- Root overshoot；
- 非目标区域扰动；
- 重连后的速度与加速度误差。

只有固定评测集上的 B0→对应消融结果可以支持“效果改善”。shape 正确、smoke 通过、loss 下降或单样本视频都不能替代效果证据。

## 交付要求

最终按以下顺序汇报：

1. 已实现到哪个 Phase/B 编号；
2. proposal 与当前 contract 的差异及处理；
3. 修改的源文件、测试和实验记录；
4. 实际执行的测试与短检查；
5. 真实评测结果或“尚无效果证据”；
6. 剩余风险和下一项单变量实验；
7. `git status --short --branch`，并单列忽略的训练产物。
