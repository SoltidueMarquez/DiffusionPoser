---
name: diffusionposer-code-review
description: DiffusionPoser 实时稀疏姿态重建项目的代码审阅、导师问答与证据核对规范。用于梳理完整任务目标和应用场景，追踪 AMASS/source/task/normalizer/TargetDiT/diffusion/Projected DDIM/runtime/eval 链路，解释张量维度、坐标系、Tracker 状态、可靠性、raw/deployed 输出、rollout 和 loss，核对算法文档、contract、论文、代码、测试与实验结果的一致性，定位实现问题并提出可验证的改进方向，或生成可直接用于导师沟通的中文 review 回答。
---

# DiffusionPoser 代码审阅

## 核心目标

先确认当前入口和实际代码，再解释任务、实现和效果。始终把以下五类内容分开：

- `需求`：论文、用户目标与 `documents/算法架构.md` 声明要解决什么。
- `契约`：`contract.md` 规定字段、形状、坐标系、时序和不变量。
- `实现`：当前分支源码在具体入口中实际做什么。
- `证据`：测试、日志、checkpoint 参数、评估产物和可复现实验实际证明了什么。
- `假设`：尚未由运行证据验证的原因判断或改进方向。

同时遵守 `$diffusionposer-repro`。只做解释或诊断时保持只读；只有用户明确要求修改时才改代码。

## 开始前核查

1. 读取仓库根目录 `AGENTS.md`、`contract.md` 和 `documents/算法架构.md`。
2. 用 `git status --short --branch`、`git log -n 5 --oneline` 锁定分支、commit 和未提交改动。
3. 确认问题属于哪条入口：离线数据生成、训练、单窗口重建、闭环长序列 runtime、评估或导出。不得跨入口借用变量含义。
4. 做项目概览或跨模块 review 时读取 [references/project-map.md](references/project-map.md)。把该文件当作导航，不替代当前源码。
5. 涉及 checkpoint 或实验效果时读取对应 `args.json`、normalizer/task metadata、评估 JSON 和日志；不得根据目录名猜配置。

## 强制审阅流程

### 1. 还原任务

先用一段话回答：谁提供什么输入、系统在什么时限内预测什么、输出供谁消费、允许哪些 Tracker 变化、哪些问题明确不在当前范围内。

区分离线监督任务和在线部署任务。离线窗口含 GT target；在线 runtime 只能使用 Tracker 测量、过去 deployed 预测和内部状态，不能把离线 GT history 当作部署输入。

### 2. 锁定数据血缘

用 `rg -n -w` 查目标变量的定义、生产、归一化、布局变换、缓存更新、最终消费和评估位置。读取最小完整代码块，不只看变量名或文档公式。

为关键张量建立语义卡：

```text
名称：
形状与每一维：
物理量与单位：
坐标系/参考帧：
raw 或 normalized：
padding 与有效区：
状态极性或 True 的含义：
生产位置 -> 变换 -> 消费位置：
是否参与梯度、投影、缓存或评估：
```

出现 `pose/history/current/raw/deployed/configured/measured_valid/d_off/d_on/hard` 等相似变量时必须填写语义卡，不凭记忆合并。

### 3. 核对跨模块契约

逐段核对：

```text
AMASS/SMPL + Unity rest
-> realtime source
-> source-absolute Tracker events
-> mmap task store
-> Dataset/cold-start batch
-> normalizer
-> observation + motion conditioning
-> raw TargetDiT x0
-> SO(3)/hard projection
-> deployed pose / rollout history
-> Head-anchored world resolver
-> reconstruction artifact
-> scenario/duration/history/latency metrics
```

对每个边界至少核对形状、dtype、单位、坐标系、归一化、padding、状态极性和旧产物拒绝策略。发现文档与代码冲突时并列报告，不静默选择一方。

### 4. 核对训练—部署对称性

重点验证：

- 历史使用上一帧参考系，当前 target/Tracker 使用当前参考系。
- 完整 144D 都参与前向加噪；新链路不存在 `known_mask`、`inpaint_mask` 或 `inpaint_cond`。
- raw 输出承担生成和 Tracker rotation 学习；deployed 输出承担硬约束、FK、位置与 rollout。
- Projected DDIM 用投影后的 `x0` 重新计算 epsilon，hard 集合在一帧内固定。
- `prepare_conditioning()` 每个目标帧只计算一次，不在每个 DDIM step 重算历史编码。
- 下一步历史只追加 `deployed_pred_xstart.detach()`，不得读取后续 GT pose history。
- 冷启动 padding 在归一化后仍是字面量零；duration 从虚拟会话起点重新推进。
- 训练窗口事件与长序列 runtime 使用一致的 source-absolute Tracker 状态语义。

把任何 GT 泄漏、参考系错位、raw/deployed 混用、mask 极性错误或训练/runtime 不对称优先视为高风险问题。

### 5. 核对算法对应

需要对照论文或设计公式时，分别回答：

- 语义是否一致：解决的物理或概率问题是否相同。
- 结构是否一致：张量组织、网络模块、采样公式是否相同。
- 当前工程增加了什么：可靠性、区域路由、trajectory、projection、resolver 或训练场景。
- 当前工程删去了什么：旧 inpainting、旧 target 字段或未同步 runtime 接口。

不得把“借鉴 StableMotion 的扩散骨架”写成“复现了 StableMotion 全部算法”，也不得把设计文档中的公式直接当成代码已实现的证据。

### 6. 核对效果证据

使用以下证据等级：

1. `contracted`：文档规定。
2. `implemented`：当前调用链可确认。
3. `tested`：相关自动测试在当前环境实际通过。
4. `observed`：真实 checkpoint、固定评测集和结果文件支持。
5. `hypothesis`：待证伪解释或改进建议。

只有 `observed` 能支持“效果改善”。前向 shape、smoke test、训练 loss 下降或代码结构合理，都不能替代长序列质量指标和可视化。

比较实验时固定并报告：代码 commit、数据/task/normalizer hash、checkpoint 与 EMA、seed、Tracker timeline、projection mode、推理步数、评测序列和指标聚合口径。至少拆分 raw/deployed、五类场景、`d_off/d_on`、hard/soft、cold-start/steady-state 与延迟。

## Findings 输出规范

先给 findings，按严重度排序：

- `P0`：实验无效、GT 泄漏、坐标系/契约根本错误、无法部署。
- `P1`：会明显损害训练、采样、runtime 正确性或主要指标。
- `P2`：测试、诊断、配置复现或维护性缺口，可能掩盖后续问题。
- `P3`：局部可读性或低风险清理。

每条 finding 写清：具体文件/函数、触发条件、当前行为、后果、证据边界、最小修复方向和验证方法。没有可执行问题时明确写“未发现已证实缺陷”，并列出剩余风险或缺失证据。

不要把个人偏好写成 bug；不要因为方案复杂就建议重写；不要在同一实验中同时更换数据、模型、loss 和采样协议。

## 导师问答格式

先直接回答导师最后一句，再给必要证据。单变量问题通常控制在 4～10 句：

```text
X 是什么。
X 的当前形状/值/单位/坐标系是什么。
X 在当前入口的哪一步产生并在哪一步起作用。
它不是什么，避免最容易发生的误解。
```

维度问题必须写出实际变换链，例如：

```text
tracker_history [B,60,6,13]
-> measurement/state 分支编码
-> 每 Tracker 的 history summary [B,6,D]
-> 区域 motion prior 与 TargetDiT context
```

用户要转发给导师时，最后附一段自然、确定、无需拼接上下文的中文话术。代码尚未核准时直说需要继续核对，不使用“我记得”“应该是”“大概”等措辞。

## 交付自检

- 是否锁定了当前 commit、入口和 active 配置？
- 是否直接回答了任务目标或导师原问题？
- 是否区分了上一帧/当前参考系、raw/normalized、raw/deployed？
- 是否解释了每一维和状态极性，而非只报 shape？
- 是否同时检查生产端和消费端？
- 是否区分文档声明、代码实现、测试通过、真实效果和假设？
- 是否给出了最小、可证伪的验证步骤？
- 是否避免修改用户未要求修改的代码或产物？
