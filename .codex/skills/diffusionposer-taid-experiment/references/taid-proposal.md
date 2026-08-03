# 动态可信锚点先验 + Innovation 条件扩散

## 面向多 Tracker 配置、掉线与重连的两阶段实时姿态恢复方案

## 一、总体结论

这个两阶段方向是可行的，而且与当前任务的矛盾匹配得比较好：
$$
\boxed{
\text{稳定 Tracker}
\rightarrow
\text{简单回归 Prior}
}
$$
不过第一版应该刻意保持克制。建议只新增三个组件：

1. **Tracker 角色管理器**：区分 Anchor、Uncertain、Missing；
2. **轻量 Anchor Prior Regressor**：输出完整人体基础姿态；
3. **Innovation Encoder**：把不确定 Tracker 与 Prior 的运动学残差编码给现有 TargetDiT。

第一版暂时不要同时加入：

- learned reliability gate；
- 概率不确定性头；
- 三个独立的 Root/Leg/Arm DiT；
- 每一步 DDIM 的 FK guidance；
- residual diffusion 输出形式；
- 复杂运动 codebook；
- 额外 IK 优化器。

否则即使实验提升，也很难判断到底是哪一个设计有效。

这套方案最准确的定位不是“两个网络串联”，而是：

> **一个低方差的状态预测器，配合一个处理缺失自由度和不确定测量的生成式后验模型。**

------

# 二、相关工作梳理与思路来源

这里的“相关工作”限定为会直接影响本方案设计和新颖性判断的核心技术线，而不是机械罗列所有稀疏姿态论文。

## 2.1 稀疏 Tracker 到完整姿态的回归模型

早期和主流稀疏姿态方法通常直接学习：
$$
X_t=F(O_{1:t})
$$
AvatarPoser 使用头和双手时序恢复完整人体，并在网络预测之后加入手臂 IK；HMD-Poser 面向多种 HMD/IMU 配置，通过缺失输入补零和轻量时序、空间模块统一处理多个配置；DynaIP 则使用分部身体建模和递归网络恢复稀疏惯性观测下的运动。它们共同说明，MLP、LSTM、GRU 或小型 Transformer 已足以提供一个稳定的全身姿态中心估计，不需要为 Prior 再构建复杂生成模型。

这条技术线为你的方案提供的直接依据是：
$$
\boxed{
\text{Prior 可以是简单回归模型}
}
$$
但这些模型大多把输出本身视为最终姿态，未把它进一步解释为供生成模型修正的预测状态。

------

## 2.2 任意配置、缺失设备和间歇观测

DiffusionPoser 使用一个扩散模型支持任意传感器组合，不需要为每种配置重新训练；其主要机制是将当前已观测特征作为 inpainting 区域，将未观测特征交给扩散模型生成。HMD-Poser 通过固定输入槽位和缺失补零处理不同配置；EgoPoser 显式模拟了手部追踪因视野受限而间歇丢失的情况；WHIP 使用按模态划分的条件模块和 modality dropout 支持任意模态子集；DragPoser 则允许动态增加或减少姿态约束，但通过测试时的潜空间优化满足约束。

这些方法说明：
$$
\boxed{
\text{多配置可以通过统一网络和动态观测集合建模}
}
$$
但现有方法主要采用：

- mask；
- zero padding；
- modality dropout；
- 测试时优化；
- observed/unobserved 二元 inpainting。

它们通常没有显式区分：
$$
\text{刚重连但暂不可信}
$$
和：
$$
\text{已经连续稳定很久}
$$
因此，“存在”与“可信”仍然经常被压缩为一个有效性 mask。

------

## 2.3 两阶段“基础估计—约束修正”

与你的两阶段结构最接近的工作包括：

- AvatarPoser：全身回归后，用 IK 调整手臂；
- SparsePoser：先由生成器得到完整人体，再通过 learned IK 调整手脚末端；
- MobilePoser：先获得运动学姿态，再通过物理优化提高运动合理性；
- EnvPoser：先生成带不确定性的初始人体，再根据环境几何和语义做第二阶段修正；
- DSPoser：先补全间歇缺失的手部轨迹并估计不确定性，再由条件扩散恢复完整人体。

它们共同支持一种思路：
$$
\boxed{
\text{先生成一个稳定基础解，再利用额外约束缩小解空间}
}
$$
其中 DSPoser 是结构上非常接近的参考：第一阶段负责补全和不确定性估计，第二阶段使用扩散恢复完整人体。但它的第一阶段主要补全间歇手部轨迹，并没有让同一个 Tracker 根据时序可信状态在 Prior 分支和 posterior 分支之间迁移，也没有通过人体 FK 构造预测—测量 innovation。

EnvPoser 也提供了“初始预测不确定性决定第二阶段修正强度”的先例，但其第二阶段条件来自环境，不是动态重连的 Tracker。

------

## 2.4 Diffusion 在稀疏人体恢复中的作用

AGRoL 和 BoDiffusion 证明了条件扩散可以从稀疏头手观测恢复完整人体；DiffusionPoser进一步处理任意传感器配置；SAGE 使用先上半身、再下半身的分层扩散结构；Ultra Diffusion Poser 则在采样过程中根据 FK 几何误差对扩散结果进行显式 guidance。

这些工作为你的 Diffusion 模块提供三类依据：
$$
\text{缺失自由度的条件补全}
$$
但 Ultra Diffusion Poser 面向固定传感器布局，并在采样过程中反复使用几何 guidance；你当前更关注的是动态配置和重连，而且需要实时运行。因此更适合：

> 每帧只根据固定 Prior 计算一次 innovation，然后在所有 DDIM 步骤中复用，而不是每一步重新 FK 和反向 guidance。

------

## 2.5 Prediction–Innovation–Update 的理论来源

Kalman 滤波的经典结构是：
$$
x_t^-=f(x_{t-1})
$$
KalmanNet 保留了“状态预测—测量 innovation—状态更新”的可解释结构，同时使用递归网络学习复杂系统中的更新增益，而不是完全依赖手工噪声模型。

你的方案可以视为这一结构在人体姿态生成中的非线性扩展：

| 状态估计概念        | 本方案对应                         |
| ------------------- | ---------------------------------- |
| 预测状态 $x_t^-$    | Anchor Prior $S_t^-$               |
| 测量函数 $h(x)$     | Differentiable FK                  |
| 测量创新 $y-h(x^-)$ | Tracker position/rotation residual |
| Kalman gain         | 固定区域路由与后续 learned adapter |
| 后验状态 $x_t^+$    | TargetDiT 输出姿态                 |
| 多峰/高维后验       | Diffusion 条件分布                 |

这只是**结构类比**，不是严格的线性高斯 Kalman 滤波。你的状态空间是高维旋转流形，后验也由 Diffusion 隐式建模。

------

## 2.6 不确定性和区域层次建模

DynaIP 已经使用分部身体动态；SAGE 使用上半身到下半身的分层生成；EnvPoser 显式估计初始预测不确定性；FisherPoser 使用旋转概率分布和区域、关节层次结构表达可观测性差异。

因此下面这些概念不能单独作为核心创新：

- 人体区域划分；
- 层次化更新；
- 预测不确定性；
- 两阶段生成；
- 任意配置；
- 条件 Diffusion。

真正可以强调的是它们在**动态 Tracker 角色迁移**中的组合方式。

------

# 三、从已有工作到本方案的思路演进

整个逻辑可以写成五步。

## 第一步：简单回归网络可以给出稳定中心

稀疏 Tracker 回归工作已经说明：
$$
O^{sparse}+History
\rightarrow
\text{Full-body estimate}
$$
是可学习的。

但三点条件下存在多解，所以这个基础估计不应被视为最终答案，而应视为：
$$
S_t^- \approx \text{稳定的条件中心或模式}
$$

------

## 第二步：基础姿态之后可以利用观测做局部修正

AvatarPoser、SparsePoser、MobilePoser 和 EnvPoser 都表明，先得到基础人体，再利用运动学、物理或环境信息细化，是合理的两阶段路线。

但传统 IK 或优化通常把观测视为强约束，不适合刚重连时的渐进吸收。

------

## 第三步：多配置可以视为观测集合变化

DiffusionPoser、HMD-Poser 和 WHIP 表明，同一模型可以通过 mask、缺失填充或模态 dropout 适应不同观测集合。

因此：
$$
3\text{点、4点、5点、6点}
$$
不必分别训练多个网络，可以统一为：
$$
\mathcal O_t=\{O_{t,i}\mid i\text{ 当前可用}\}
$$

------

## 第四步：掉线和重连不能只表示成二值 mask

相同的 `measured_valid=1` 可能对应：

- 已稳定跟踪几百帧；
- 刚重连第一帧；
- 连续抖动；
- 突然跳变；
- 存在慢性偏置。

因此需要引入：
$$
\text{是否可用}
\neq
\text{是否适合作为强先验}
$$

------

## 第五步：用稳定观测构造 Prior，用不确定观测构造 Innovation

最终得到：
$$
\boxed{
\text{多配置}
=
\text{Anchor 集合不同}
}
$$
在本次检索覆盖的核心工作中，我没有找到一篇工作同时实现：

1. 同一个 Tracker 根据时序稳定性在 Anchor、Uncertain、Missing 之间迁移；
2. 稳定 Tracker 先生成每帧固定的完整人体 Prior；
3. 不确定 Tracker 通过 FK 计算相对 Prior 的 innovation；
4. innovation 以 Tracker 类型相关的区域路径进入 Diffusion；
5. 用重连后长期闭环后果训练观测吸收速度。

因此这个组合是可以重点发展的方向，但论文中应写成“由多条已有思路推演得到”，而不是声称整个两阶段思想从未出现。

------

# 四、建议的最简模型：TAID-V1

可以暂时命名为：

> **Trusted-Anchor Innovation Diffusion，TAID**

第一版只改三个位置。

```
Tracker Role Manager
        │
        ├── Anchor Tracker ──→ 轻量 Prior Regressor ──→ 固定人体 Prior
        │                                              │
        ├── Uncertain Tracker ──→ FK Innovation ───────┤
        │                                              ↓
        └── Missing Tracker ──→ 不输入          现有 TargetDiT
                                                       │
                                              最终姿态 / Root / Contact
```

------

# 五、输入、输出与坐标系

## 5.1 输入保持不变

人体历史：
$$
H_t
=
X_{t-60:t-1}
\in
\mathbb R^{60\times144}
$$
Tracker 窗口：
$$
O_t
\in
\mathbb R^{61\times6\times15}
$$
六个固定 Tracker 槽位：
$$
\{
Head,LHand,RHand,Hip,LFoot,RFoot
\}
$$
继续保留：

- `configured`；
- `measured_valid`；
- $d^{off}$；
- $d^{on}$；
- Tracker identity；
- 当前已有的运动速度、旋转、位置特征。

------

## 5.2 统一使用 Head-relative 坐标

位置：
$$
p_{t,i}^{H}
=
(R_{t,H}^{W})^\top
\left(
p_{t,i}^{W}-p_{t,H}^{W}
\right)
$$
旋转：
$$
R_{t,i}^{H}
=
(R_{t,H}^{W})^\top R_{t,i}^{W}
$$
这样 innovation 不受世界原点和全局朝向变化直接影响。

------

# 六、Tracker 角色管理器

## 6.1 三种观测角色

对已配置 Tracker 定义：
$$
s_{t,i}\in
\{
M,U,A
\}
$$
其中：
$$
M=\text{Missing}
$$
具体规则：
$$
s_{t,i}
=
\begin{cases}
M,
&
measuredValid_{t,i}=0
\\[2mm]
U,
&
measuredValid_{t,i}=1,\quad
d^{on}_{t,i}<K_A
\\[2mm]
A,
&
measuredValid_{t,i}=1,\quad
d^{on}_{t,i}\ge K_A
\end{cases}
$$
`configured=0` 与临时掉线仍应通过原始 metadata 区分：

- `configured=0`：设备本来不存在；
- `configured=1, valid=0`：设备临时缺失。

Head 按当前假设始终为 Anchor。

第一版建议沿用当前经验：
$$
K_A=15\text{帧}
$$
即约0.25秒。

------

## 6.2 不建议硬切换

若第14帧进入 innovation 分支，第15帧突然进入 Prior 分支，仍可能产生姿态跳变。

建议使用连续 Anchor 权重：
$$
\alpha_{t,i}
=
measuredValid_{t,i}
\operatorname{clip}
\left(
\frac{d^{on}_{t,i}-K_0}
{K_1-K_0},
0,1
\right)
$$
初始可以设：
$$
K_0=5,\qquad K_1=15
$$
其中：

- $\alpha=0$：完全不作为 Prior Anchor；
- $0<\alpha<1$：逐渐进入 Prior；
- $\alpha=1$：稳定 Anchor。

不确定观测的更新权重使用：
$$
\beta_{t,i}
=
measuredValid_{t,i}
(1-\alpha_{t,i})
\min
\left(
1,
\frac{d^{on}_{t,i}}{K_R}
\right)
$$
其中 $K_R$ 也可先设为15。

因此重连时：
$$
\beta:0\rightarrow\text{较大}\rightarrow0
$$
Tracker 先以弱 residual correction 进入，再逐渐成为 Prior Anchor。

这样同一测量不会在两个分支中被完整重复注入。

------

# 七、阶段一：Anchor Prior Regressor

## 7.1 Prior 的职责

Prior 回答：

> 根据姿态历史、Head 和当前稳定 Tracker，当前人体最合理的基础状态是什么？

定义：
$$
S_t^-
=
\left(
X_t^-,
r_t^-,
c_t^-
\right)
$$
其中：
$$
X_t^-\in\mathbb R^{24\times6}
$$
是24关节 rotation6D；
$$
r_t^-
=
[
p_{root,t}^{H,-},
\psi_{root,t}^{H,-}
]
\in\mathbb R^4
$$
是 Head-relative Root 位置和 Root yaw；
$$
c_t^-\in[0,1]^2
$$
是左右脚先验接触概率。

最终运行时仍可保持144维姿态接口，$r_t^-$ 只作为内部 FK 和 Root Resolver 辅助状态。

------

## 7.2 最简单的实现：复用现有编码器

不建议重新增加一个大型 GRU 或 Transformer。

你当前已有：

- 60帧姿态历史编码；
- 61帧 Tracker 动态观测编码；
- 每个 Tracker 的独立 token；
- Tracker identity 和时序有效性特征。

直接从现有编码器分出 Prior 支路即可。

对每个 Tracker：
$$
z_{t,i}^{obs}
=
E_{\mathrm{obs}}
\left(
O_{t-60:t,i}
\right)
$$
在 Prior 分支中，必须在原始输入或 token 级别乘 Anchor 权重：
$$
z_{t,i}^{A}
=
\alpha_{t,i}
z_{t,i}^{obs}
$$
不能先将所有 Tracker 全量融合，再区分 Anchor 和 Uncertain，否则刚重连测量已经泄露进 Prior。

将固定六个槽位拼接：
$$
z_t^A
=
\operatorname{MLP}_A
\left(
[
z_{t,Head}^A,
z_{t,LH}^A,
z_{t,RH}^A,
z_{t,Hip}^A,
z_{t,LF}^A,
z_{t,RF}^A
]
\right)
$$
历史运动 latent：
$$
z_t^H
=
E_{\mathrm{history}}
\left(
X_{t-60:t-1},
Head_{t-60:t}
\right)
$$
融合：
$$
z_t^-
=
\operatorname{MLP}_{fusion}
\left(
[z_t^H,z_t^A]
\right)
$$
输出：
$$
X_t^-
=
Head_{pose}(z_t^-)
$$

------

## 7.3 推荐的轻量尺寸

| 模块               | 推荐尺寸                          |
| ------------------ | --------------------------------- |
| 每个 Tracker token | 64D                               |
| 六 Tracker 拼接    | $6\times64=384$D                  |
| Anchor 压缩 MLP    | $384\rightarrow128$               |
| History latent     | 256D                              |
| Fusion MLP         | $384\rightarrow256$               |
| Pose head          | $256\rightarrow256\rightarrow144$ |
| Root head          | $256\rightarrow128\rightarrow4$   |
| Contact head       | $256\rightarrow64\rightarrow2$    |

如果现有编码器 latent 维度不同，只增加线性投影即可。

第一版不增加 learned uncertainty head，而使用确定性的区域观测覆盖度。

------

## 7.4 确定性区域覆盖度

令 $M_{i,r}$ 表示 Tracker $i$ 是否覆盖区域 $r$：
$$
r\in
\{
Torso,LArm,RArm,LLeg,RLeg
\}
$$
定义：
$$
\rho_{t,r}
=
1-
\prod_i
\left(
1-\alpha_{t,i}M_{i,r}
\right)
$$
例如：

- Hip 对 Torso 覆盖高；
- LeftFoot 对 LeftLeg 覆盖高；
- LeftHand 对 LeftArm 覆盖高；
- Head 对 Torso 提供基础覆盖。

$\rho_{t,r}$ 是完全可解释的，不需要第一版就学习不确定性。

------

# 八、Prior 必须包含内部 Root 状态

仅有144维局部旋转无法完整预测 Tracker 世界或 Head-relative 位置。

因此 Prior 的预测状态必须是：
$$
S_t^-=(X_t^-,r_t^-,c_t^-)
$$
对 Tracker $i$，预测其位姿：
$$
\hat T_{t,i}^-
=
T_{root}^{H}(r_t^-)
FK_i(X_t^-)
T_i^{calib}
$$
其中：

- $FK_i$：到 Tracker 对应骨骼节点的前向运动学；
- $T_i^{calib}$：骨骼节点到实际 Tracker 安装位姿的标定变换。

否则 Hip position residual、Foot position residual 没有完整的作用对象。

------

# 九、阶段二：Innovation-conditioned TargetDiT

## 9.1 计算位置 innovation

对 Uncertain Tracker：
$$
e_{t,i}^{p}
=
p_{t,i}^{obs,H}
-
\hat p_{t,i}^{-,H}
$$
按 Tracker 类型的数据尺度归一化：
$$
\tilde e_{t,i}^{p}
=
\frac{e_{t,i}^{p}}
{\sigma_{p,i}}
$$

------

## 9.2 计算旋转 innovation

推荐使用 SO(3) 对数映射：
$$
e_{t,i}^{R}
=
\operatorname{Log}
\left(
(\hat R_{t,i}^{-,H})^\top
R_{t,i}^{obs,H}
\right)
\in\mathbb R^3
$$
归一化：
$$
\tilde e_{t,i}^{R}
=
\frac{e_{t,i}^{R}}
{\sigma_{R,i}}
$$
完整 innovation：
$$
\tilde e_{t,i}
=
[
\tilde e_{t,i}^{p},
\tilde e_{t,i}^{R}
]
\in\mathbb R^6
$$

------

## 9.3 加入 residual 的时序一致性

仅看单帧 innovation 无法区分：

- 真实重连后的持续偏差；
- 单帧 Tracker 跳点；
- 高频抖动。

因此加入：
$$
\Delta\tilde e_{t,i}
=
\tilde e_{t,i}
-
\tilde e_{t-1,i}
$$
Innovation Encoder 输入：
$$
q_{t,i}
=
E_{\mathrm{inn}}
\left(
[
\tilde e_{t,i},
\Delta\tilde e_{t,i},
d_{t,i}^{on},
d_{t,i}^{off},
type_i,
c_t^-
]
\right)
$$
第一版使用一个共享两层 MLP 加 Tracker type embedding：
$$
E_{\mathrm{inn}}:
d_{in}\rightarrow128\rightarrow64
$$
而不是为 Hip、Feet、Hands 分别建网络。

------

## 9.4 对异常 residual 做确定性截断

第一版不建议直接学习 gate。

可以先使用：
$$
\bar e^p
=
\tau_p
\tanh
\left(
\frac{e^p}{\tau_p}
\right)
$$
避免单个异常测量产生无限大的更新 token。

这比一开始增加 learned outlier rejection 更容易解释和消融。

------

# 十、固定的 Tracker—区域更新路由

定义固定路由矩阵：
$$
M^{route}_{i,r}
$$
第一版采用：

| Tracker   | 主要更新区域  | 次级更新区域      |
| --------- | ------------- | ----------------- |
| Hip       | Root/Torso    | 双腿的骨盆连接    |
| LeftFoot  | LeftLeg       | contact介导的Root |
| RightFoot | RightLeg      | contact介导的Root |
| LeftHand  | LeftArm       | 很弱的Torso       |
| RightHand | RightArm      | 很弱的Torso       |
| Head      | 主要进入Prior | 不进入重连更新    |

区域 innovation token：
$$
C_{t,r}^{inn}
=
\sum_i
M^{route}_{i,r}
\beta_{t,i}
q_{t,i}
$$
Feet 到 Root 的路径使用先验接触概率：
$$
M^{route}_{Foot\rightarrow Root}
=
\eta c_{t,foot}^-
$$
其中 $\eta<1$。

因此：
$$
Foot_{\mathrm{contact}}
\rightarrow
\text{较强 Root 约束}
$$
第一版使用 $c_t^-$ 避免形成：
$$
Contact_t
\rightarrow Pose_t
\rightarrow Contact_t
$$
的因果循环。

------

# 十一、尽量不修改 TargetDiT 主体

## 11.1 新增条件

对关节 $j$，所在区域为 $r(j)$，构造：
$$
C_{t,j}^{prior}
=
E_{prior}
\left(
X_{t,j}^-,
r_t^-,
\rho_{t,r(j)}
\right)
$$
现有 DiT token 初始化改为：
$$
Z_{t,j}^{0}
=
E_x(X_{t,j}^{n})
+
E_n(n)
+
C_{history,j}
+
C_{t,j}^{prior}
+
C_{t,j}^{inn}
+
C_{role,j}
$$
后续仍使用现有24关节 self-attention。

输出：
$$
\hat X_t^0
=
D_\theta
\left(
X_t^n,n,
H_t,
S_t^-,
C_t^{inn}
\right)
$$
同时从 Root/Torso token 增加一个很小的 Root correction head：
$$
\Delta r_t
=
Head_{\Delta root}(Z_{torso})
$$
最终 contact 继续使用现有 contact head：
$$
c_t^+
=
Head_{contact}^{post}(Z_{legs})
$$

------

## 11.2 每帧固定 Prior 和 Innovation

每帧只计算一次：
$$
S_t^-=F_{prior}(\cdots)
$$
在所有 DDIM 步骤中复用：
$$
\hat X_0^{(n)}
=
D_\theta
\left(
X_n,n,S_t^-,E_t
\right)
$$
不要使用：
$$
E_t^{(n)}
=
O_t-FK(X_n)
$$
因为 $X_n$ 是带噪旋转状态，其 FK 没有稳定的物理含义，也会显著增加每步去噪开销。

------

# 十二、第一版不要改成 Residual Diffusion

理论上可以定义：
$$
\delta\omega_{t,j}^{GT}
=
\operatorname{Log}
\left(
(R_{t,j}^-)^\top R_{t,j}^{GT}
\right)
$$
让 Diffusion 预测：
$$
\Delta X_t
$$
再组合：
$$
R_{t,j}^+
=
R_{t,j}^-
\operatorname{Exp}
(\hat\delta\omega_{t,j})
$$
但这会同时改变：

- 训练目标；
- 输出维度；
- 噪声空间；
- 旋转投影；
- DDIM 更新；
- Prior 和 Diffusion 的误差分布。

第一版建议保持现有：
$$
\hat X_0\in\mathbb R^{144}
$$
只把 Prior 作为干净条件输入。

等证明“Anchor/Uncertain 分工”和“innovation”有效后，再对比：
$$
\text{full-pose diffusion}
$$
与：
$$
\text{residual diffusion}
$$
这样实验结论更干净。

------

# 十三、Inpainting 的最终处理

第一版建议取消每个 DDIM step 的硬关节写回：
$$
X_n
\leftarrow
M\odot X^{obs}
+
(1-M)\odot X_n
$$
原因是 Tracker 的观测空间是：
$$
SE(3)\text{末端位姿}
$$
而模型生成空间是：
$$
24\times SO(3)\text{局部旋转}
$$
两者不是逐维对应关系。

保留：

- 观测角色 mask；
- 区域覆盖度；
- innovation condition；
- posterior FK consistency loss。

部署阶段可以保留一次高可信度安全投影：
$$
X_t^{deploy}
=
P_{safe}
\left(
X_t^+,O_t^{high-confidence}
\right)
$$
但它只作为安全边界和消融项，而不是主要恢复机制。

DiffusionPoser 的硬 inpainting 适用于其已观测特征与生成特征可以直接对齐的表示；Ultra Diffusion Poser 则使用测量几何误差进行采样 guidance。你的模型位于两者之间：不做逐维覆盖，也不在每一步进行昂贵 guidance，而是用一次 FK innovation 形成条件。

------

# 十四、完整推理流程

```
输入：
    PoseHistory [60, 144]
    TrackerWindow [61, 6, 15]

1. 根据 configured、valid、d_on 更新每个 Tracker 的角色
       Missing / Uncertain / Anchor

2. 计算连续权重
       alpha_i：进入 Prior 的权重
       beta_i ：进入 Innovation 分支的权重

3. Anchor Prior
       S^- = Prior(PoseHistory, alpha_i * Tracker_i)

4. Differentiable FK
       O_hat_i^- = FK_i(S^-)

5. 对 Uncertain Tracker 计算
       e_pos_i
       e_rot_i
       delta_e_i

6. Innovation Encoder
       q_i = MLP(e_i, delta_e_i, type_i, d_on, d_off, contact^-)

7. 固定区域路由
       Hip   → Root/Torso
       Foot  → Leg + contact-limited Root
       Hand  → Arm

8. 固定 S^- 和 innovation tokens

9. 执行 K 步 DDIM
       X_0 = TargetDiT(X_n, History, S^-, Innovation)

10. 输出
       144D pose
       root correction
       contact

11. 可选部署安全投影
```

------

# 十五、训练数据构造

必须把“设备配置”和“运行时观测事件”分开。

## 15.1 设备配置层

`configured` 决定设备是否存在。

训练覆盖：
$$
3pt=
\{Head,LHand,RHand\}
$$
以及部分4点、5点配置。

建议4、5点配置主要用于提高组合泛化，不必与三点、六点使用同样高的采样概率。

------

## 15.2 观测事件层

在已配置 Tracker 内构造：
$$
A\rightarrow M
$$
事件类型应分开采样：

1. Hip 单独掉线和重连；
2. 单脚掉线和重连；
3. 双脚掉线和重连；
4. 单手或双手掉线和重连；
5. 跨类型组合，如 Hip+Foot、Hand+Foot；
6. 任意一到两个非 Head Tracker 掉线；
7. 六点降三点和三点恢复六点。

不要只使用“随机两个 Tracker 同时 mask”，因为那会模糊 Hip、Feet、Hands 各自的作用。

------

## 15.3 推荐的事件时长

数据为60 fps，可以先覆盖：
$$
5,\ 15,\ 30,\ 60,\ 120\text{帧}
$$
分别对应：

- 短暂闪断；
- 当前重连阈值；
- 中等掉线；
- 1秒掉线；
- 2秒掉线。

重连监督窗口重点覆盖：
$$
15\sim30\text{帧}
$$

------

## 15.4 第二阶段再加入观测异常

基础掉线重连验证完成后，再增加：

- 单帧位置 spike；
- 单帧旋转 spike；
- 高频抖动；
- 持续位置偏置；
- 持续旋转偏置；
- 1～3帧延迟；
- 标定误差。

否则第一版同时解决掉线、重连、异常检测和标定鲁棒性，变量过多。

------

# 十六、分阶段训练流程

## 阶段一：单独训练 Prior

输入只允许使用：

- 历史姿态；
- Head；
- Anchor Tracker；
- Anchor 权重 $\alpha$。

Uncertain Tracker 当前测量不得进入 Prior。

损失：
$$
L_{prior}
=
\lambda_{rot}L_{rot}^-
+
\lambda_{FK}L_{FK}^-
+
\lambda_{root}L_{root}^-
+
\lambda_vL_v^-
+
\lambda_cL_c^-
$$
旋转损失：
$$
L_{rot}^-
=
\sum_j
w_{r(j)}
d_{SO(3)}
\left(
R_{t,j}^-,
R_{t,j}^{GT}
\right)
$$
FK 关节位置损失：
$$
L_{FK}^-
=
\sum_j
\left\|
p_{t,j}^--p_{t,j}^{GT}
\right\|_1
$$
Root 损失：
$$
L_{root}^-
=
\left\|
p_{root,t}^{H,-}
-
p_{root,t}^{H,GT}
\right\|_1
+
d_{SO(2)}
\left(
\psi_t^-,
\psi_t^{GT}
\right)
$$
速度损失：
$$
L_v^-
=
\sum_j
\left\|
v_{t,j}^--v_{t,j}^{GT}
\right\|_1
$$
接触损失：
$$
L_c^-=
BCE(c_t^-,c_t^{GT})
$$

------

## 16.1 区域覆盖加权监督

Prior 仍然要输出完整人体，但对未观测区域不应施加和直接观测区域完全相同的强度。

定义：
$$
w_r
=
w_{min}
+
(w_{max}-w_{min})
\rho_r
$$
其中：
$$
w_{min}>0
$$
保证三点模式下 Prior 仍学习合理下肢，而不是完全不监督。

直观上：

- 有稳定 Foot 时，腿部回归监督较强；
- 没有 Foot 时，腿部保留较弱的姿态、速度和接触监督；
- Root、Torso、速度连续性始终保持较强监督。

继续使用你现有的预测历史训练和 rollout，而不是只用纯 GT history。

------

## 阶段二：冻结 Prior，训练 Innovation 与 TargetDiT

先固定 Prior：
$$
S_t^-
=
\operatorname{sg}
\left(
F_{prior}(\cdots)
\right)
$$
其中 $\operatorname{sg}$ 表示 stop-gradient。

再计算：
$$
E_t
=
O_t-FK(S_t^-)
$$
这样可以防止 Prior 故意制造某种特殊 residual，作为两个网络之间的隐藏通信编码。

TargetDiT 继续使用当前 $x_0$ 预测目标：
$$
L_{diff}
=
D
\left(
\hat X_t^0,
X_t^{GT}
\right)
$$
加入 posterior FK consistency：
$$
L_{obs}
=
\sum_i
w_{t,i}^{obs}
\left[
\rho_p
\left(
\hat p_{t,i}^{+}-p_{t,i}^{obs}
\right)
+
\lambda_R
\rho_R
\left(
\operatorname{Log}
\left(
(\hat R_{t,i}^{+})^\top
R_{t,i}^{obs}
\right)
\right)
\right]
$$
其中 $\rho_p,\rho_R$ 使用 Huber 或 Charbonnier loss。

权重：
$$
w_{t,i}^{obs}
=
\alpha_{t,i}+\beta_{t,i}
$$
稳定 Anchor 保持较强一致性；刚重连 Tracker 的一致性约束渐进增加。

这一阶段可以用当前 TargetDiT 参数初始化，只新增 Prior condition 和 innovation adapter。

------

## 阶段三：事件感知的闭环微调

使用模型自己的历史输出进行15～30帧 rollout。

重连事件损失：
$$
L_{event}
=
\sum_{\tau=0}^{T_R-1}
\gamma^\tau
\left[
\lambda_yL_{rootYaw}
+
\lambda_pL_{rootXZ}
+
\lambda_vL_{velocity}
+
\lambda_aL_{acceleration}
+
\lambda_jL_{jerk}
+
\lambda_fL_{footSlide}
\right]_{t+\tau}
$$
需要注意：
$$
L_{acceleration}
=
\|
\ddot p^{pred}
-
\ddot p^{GT}
\|
$$
不要直接最小化：
$$
\|\ddot p^{pred}\|,
\qquad
\|\dddot p^{pred}\|
$$
否则快速动作和脚落地冲击会被过度平滑。

这一阶段可以继续冻结 Prior，先验证两阶段分工。如果发现 Prior 误差明显限制上限，再使用较小学习率联合微调，但 innovation 的 FK 路径仍建议对 Prior 使用 stop-gradient。

------

## 阶段四：可选 learned gate

只有 V1 已经证明有效后，再把固定 $\beta_i$ 升级为：
$$
g_{t,i}
=
\sigma
\left(
G(
\tilde e_{t,i},
\Delta\tilde e_{t,i},
d_{t,i}^{on},
d_{t,i}^{off},
motion_t,
contact_t,
type_i
)
\right)
$$
最终：
$$
C_{t,r}^{inn}
=
\sum_i
M_{i,r}^{route}
\beta_{t,i}
g_{t,i}
q_{t,i}
$$
这样可以清晰证明：

1. 固定状态机本身是否有效；
2. learned gate 是否进一步提升异常观测鲁棒性。

------

# 十七、总损失

完整训练目标可以写为：
$$
L
=
L_{diff}
+
\lambda_P L_{prior}
+
\lambda_O L_{obs}
+
\lambda_C L_{contact}
+
\lambda_F L_{futureLeg}
+
\lambda_E L_{event}
$$
不同阶段启用不同子项：

| 阶段           | 启用损失                                     |
| -------------- | -------------------------------------------- |
| Prior 预训练   | $L_{prior},L_{contact}$                      |
| Posterior 训练 | $L_{diff},L_{obs},L_{contact},L_{futureLeg}$ |
| 闭环微调       | 上述全部 + $L_{event}$                       |
| Learned gate   | 加异常观测和 gate 相关训练                   |

------

# 十八、配对训练应该怎样使用

可以对同一运动构造：
$$
O^{base}
$$
和：
$$
O^{base+i}
$$
其中 $i$ 是刚重连 Tracker。

但不要强制：
$$
\hat X^{base+i}
\approx
\hat X^{base}
$$
因为新观测本来就应该改变对应区域。

更合理的是“非作用区域一致性”：
$$
L_{locality}
=
\sum_{r\notin Scope(i)}
D
\left(
\operatorname{sg}
(\hat X_r^{base}),
\hat X_r^{base+i}
\right)
$$
例如 LeftFoot 重连时：

- LeftLeg 允许明显改变；
- Root 允许有限改变；
- RightArm 不应发生明显跳变。

这项损失直接增强区域更新的可解释性。

对于 full 与 hip-off，也只约束：

- contact；
- Root 速度；
- 低频运动 latent；
- 不受 Hip 直接影响的区域。

不建议对完整144维做强一致性蒸馏。

------

# 十九、必须执行的消融实验

建议按以下顺序验证。

| 编号 | 模型                                     | 目的                   |
| ---- | ---------------------------------------- | ---------------------- |
| B0   | 当前平坦 TargetDiT                       | 原始基线               |
| B1   | 简单 Prior only                          | 判断回归上限           |
| B2   | TargetDiT + Prior absolute condition     | 判断 Prior 是否有价值  |
| B3   | Prior + 不确定 Tracker 绝对观测          | 对比 innovation        |
| B4   | Prior + FK innovation                    | 核心方法               |
| B5   | B4 + 固定区域路由                        | 判断路由价值           |
| B6   | B5 + 连续 U→A 转换                       | 判断重连平滑价值       |
| B7   | B6 + learned gate                        | 判断学习可靠性价值     |
| B8   | B6 + hard inpainting                     | 判断硬约束是否反而有害 |
| B9   | full-pose diffusion → residual diffusion | 后续增强               |
| B10  | 单层次 → Root-first adapter              | 后续增强               |

其中最关键的比较是：
$$
B3:\text{绝对不确定观测}
$$
对比：
$$
B4:\text{相对 Prior 的 FK innovation}
$$
它直接验证你的核心主张：

> 重连 Tracker 更适合作为状态修正量，而不是普通绝对条件。

------

# 二十、评估协议

## 20.1 静态配置

分别评估：

- 稳定三点；
- 稳定六点；
- 4点、5点；
- 训练中未出现的 Tracker 组合。

标准指标：

- MPJPE；
- MPJRE；
- Root yaw error；
- Root XZ error；
- Tracker position error；
- Tracker rotation error；
- foot slide；
- contact precision/recall。

------

## 20.2 掉线事件

分别评估：

- Hip 掉线；
- 单脚掉线；
- 双脚掉线；
- 单手掉线；
- 两个跨类型 Tracker 掉线；
- 六点降三点。

关注：

- 掉线后误差增长速度；
- 累计 Root 漂移；
- 失去 Foot 后脚接触保持；
- 未掉线区域是否受扰动。

------

## 20.3 重连事件

除普通 MPJPE 外，需要报告事件指标。

重连首帧跳变：
$$
J_0
=
\left\|
(\hat p_{t_0}-\hat p_{t_0-1})
-
(p_{t_0}^{GT}-p_{t_0-1}^{GT})
\right\|
$$
重连峰值误差：
$$
E_{peak}
=
\max_{0\le\tau<T}
E_{t_0+\tau}
$$
30帧累计恢复误差：
$$
AUC_{30}
=
\sum_{\tau=0}^{29}
E_{t_0+\tau}
$$
恢复时间：
$$
T_{settle}
=
\min
\left\{
\tau:
E_{t_0+k}<\epsilon,
\forall k\ge\tau
\right\}
$$
还应报告：

- Root overshoot；
- Foot sliding 峰值；
- 更新非目标区域的扰动；
- 重连后速度和加速度误差。

这些指标比只报告全序列平均 MPJPE 更能证明你的方法确实解决了重连。

------

# 二十一、可解释性设计

每帧保存以下中间量：
$$
s_{t,i}
$$
Tracker 角色；
$$
\alpha_{t,i},\beta_{t,i}
$$
Prior 和 innovation 权重；
$$
\|e_{t,i}^{p}\|,
\quad
\|e_{t,i}^{R}\|
$$
测量 innovation；
$$
\|\Delta X_{t,r}\|
$$
各区域更新幅度；
$$
\|e_{t,i}^{pre}\|,
\quad
\|e_{t,i}^{post}\|
$$
更新前后 Tracker residual；
$$
c_{t,foot}^-
$$
Feet→Root 更新所使用的接触概率。

可以构造一张重连可视化：

```
帧号
 │
 ├── Tracker角色：M → U → U → U → A
 ├── d_on：       0   1   2   ... 15
 ├── innovation：大 → 中 → 小
 ├── beta：       小 → 大 → 小
 ├── alpha：      0  → 0.2 → 1
 ├── Leg更新量：  渐进下降
 └── Root误差：   平滑收敛
```

这会比仅展示动作视频更有算法说服力。

------

# 二十二、应当通过的可解释性单元测试

1. `valid=0` 时，该 Tracker 的 posterior 更新必须严格为零；
2. innovation 接近零时，相应区域更新应接近零；
3. LeftHand innovation 不应大幅修改双腿；
4. Foot 处于 swing 时，Feet→Root 更新应明显弱于 contact 时；
5. 同方向 residual 连续多帧出现时，模型应逐渐吸收；
6. 单帧异常 spike 应被截断，不能导致全身跳变；
7. Tracker 从 U 转 A 时，姿态不应出现输入路径切换跳变；
8. 加入一个 Tracker 后，非作用区域变化应小于对应区域变化；
9. posterior FK residual 应小于 prior FK residual；
10. 三点时 Prior 的腿部误差可以较大，但最终 Diffusion 应显著修正。

------

# 二十三、主要风险及对应处理

| 风险                    | 表现                                    | 第一版处理                                      |
| ----------------------- | --------------------------------------- | ----------------------------------------------- |
| Prior 太强              | Diffusion 只剩微小修正                  | Prior 保持轻量；未覆盖区域使用较弱监督          |
| Prior 太弱              | innovation 过大，posterior 近似重新生成 | 加 Root、FK、速度和接触监督                     |
| U→A 切换跳变            | 第15帧 popping                          | 使用连续 $\alpha,\beta$ 和滞回                  |
| 同一观测重复注入        | 过度跟踪 Tracker 噪声                   | Prior 与 innovation 使用互补权重                |
| 稳定 Tracker 也可能异常 | Prior 被污染                            | 第一版 residual clipping，后续 learned gate     |
| Contact 因果循环        | Contact 与姿态互相依赖                  | Feet→Root 使用 prior contact $c^-$              |
| Root 定义不完整         | Hip position residual 无更新对象        | Prior 和 posterior 都增加内部 Root head         |
| Learned gate 退化       | 始终全信或全拒绝                        | 第一版先用确定性状态机                          |
| 结构一次改太多          | 无法解释提升来源                        | 保持 TargetDiT 和144D输出不变                   |
| 区域路由过硬            | 攀爬、撑地动作受限                      | 保留全身 self-attention，路由只约束直接注入路径 |

------

# 二十四、论文创新点应该怎样表述

不建议把创新写成：

- 提出两阶段人体恢复；
- 提出区域层次建模；
- 提出不确定性条件 Diffusion；
- 提出任意 Tracker 配置；
- 提出 prediction–update。

这些单独都有较接近的相关工作。

更准确的三项贡献是：

## 贡献一：动态观测角色建模

> 将多配置、掉线和重连统一建模为 Tracker 在 Anchor、Uncertain 和 Missing 三种观测角色之间的时序转换；多配置对应 Anchor 集合的变化，掉线对应 Anchor 退出，重连对应不确定观测逐步转化为稳定锚点。

------

## 贡献二：FK Innovation 条件化的 Diffusion 后验

> 稳定 Tracker 与姿态历史首先产生每帧固定的完整人体 Prior；刚重连 Tracker 不作为普通绝对条件，而是通过可微分 FK 计算相对 Prior 的位置和旋转 innovation，并以 Tracker 类型相关的区域路径完善 Root、腿部和手臂状态。

------

## 贡献三：事件感知的重连训练与评估

> 通过分 Tracker 类型的掉线—重连事件、15～30帧闭环恢复监督以及首帧跳变、峰值误差、收敛时间和非目标区域扰动等事件指标，显式训练和评估 Tracker 的渐进吸收过程。

------

# 二十五、给导师的完整概括

> 当前模型通过 Tracker 身份、有效性状态和区域路由支持多种传感器配置，但不同 Tracker 仍作为并列绝对条件进入 Diffusion，稳定观测、刚重连观测和临时缺失观测之间缺少明确的状态语义。拟提出一个轻量两阶段框架：首先根据人体历史和长期稳定的 Tracker 锚点，由简单回归网络预测当前帧完整姿态、Head-relative Root 状态和脚部接触，形成在本帧所有扩散步骤中固定的基础先验；随后通过可微分前向运动学预测各不确定 Tracker 的先验位姿，计算实际测量相对先验的位置和旋转 innovation，并依据 Hip、Feet 和 Hands 的运动学作用域，将 innovation 以区域化条件输入现有 TargetDiT。Tracker 掉线时从锚点集合退出，重连后先以渐增的 innovation 权重修正先验，稳定后再平滑转化为 Prior 锚点。第一版保持现有 TargetDiT、144D输出、区域划分和 DDIM 流程不变，仅新增角色管理器、轻量 Prior 输出头和 Innovation Encoder，以便通过明确消融验证动态角色划分、预测—测量 residual 和区域更新路径的独立作用。

## 最终推荐的第一版

$$
\boxed{
\text{现有编码器}
+
\text{轻量 Prior Heads}
+
\text{确定性角色状态机}
+
\text{FK Innovation MLP}
+
\text{现有 TargetDiT}
}
$$

第一版的核心研究问题只保留为：
$$
\boxed{
\text{稳定 Tracker 用于建立姿态中心，
不稳定 Tracker 是否更适合以 innovation 完善这个中心？}
}
$$
这是最简单、最容易验证，也最能直接对应多配置、掉线和重连问题的一版设计。