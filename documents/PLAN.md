# DiffusionPoser Fix-Only 训练路线计划

## 目标概述

本阶段目标是把当前项目推进到正式的 DiffusionPoser 训练流程：使用已经转换完成的 X277 动作数据，追加 6 维传感器丢失标签，训练一个只做 fix/inpainting 的扩散模型。

训练任务可以理解为：给定一段动作序列，当 6 个传感器中的若干个在某个连续时间段内缺失时，模型根据剩余传感器信息、缺失标签和上下文，补全被遮盖的传感器输入维度。这个任务和 StableMotion 的修复模式高度相似，但不需要 StableMotion 的检测模式。

本计划直接更新当前 `documents/PLAN.md`，后续实现按本文逐步推进。

## 数据格式与传感器维度

模型训练输入统一为 `[B, 283, T]`：

- 前 `277` 维：X277 动作/传感器特征。
- 后 `6` 维：传感器丢失标签，顺序与 6 个 tracker 一一对应。
- `T`：帧数，训练时由数据集裁剪或 padding 后得到。

X277 维度定义如下：

```text
[0:144)    body_rot_root_fwd_up_prev       24 x 6
[144:216)  body_vel_root_prev              24 x 3
[216:234)  tracker_pos_root_now             6 x 3
[234:270)  tracker_rot_root_fwd_up_now      6 x 6
[270:272)  waist_delta_xz                   2
[272:273)  waist_yaw_delta_degree           1
[273:277)  contact_cur                      4
```

6 个传感器顺序固定为：

```text
0 head
1 left_wrist
2 right_wrist
3 waist/pelvis
4 left_foot
5 right_foot
```

本项目中“某个传感器丢失”只遮盖该传感器在 tracker 条件中的位置和旋转维度：

```text
tracker_pos_root_now:
sensor k -> [216 + 3*k : 216 + 3*(k+1))

tracker_rot_root_fwd_up_now:
sensor k -> [234 + 6*k : 234 + 6*(k+1))
```

例如 `left_wrist` 的 `k=1`，对应：

```text
position: [219:222)
rotation: [240:246)
```

注意：`body_rot_root_fwd_up_prev`、`body_vel_root_prev`、`waist_delta_xz`、`waist_yaw_delta_degree`、`contact_cur` 暂时不因为传感器丢失而遮盖。它们可以作为动作上下文继续保留。后续如果实验发现需要让模型同时重建身体状态，再单独扩展 body 维度的 mask 策略。

## 训练数据生成

训练前需要编写一个真实数据集 loader 和遮盖生成脚本。数据来源为：

```text
dataset/AMASS_x277_60hz/**/*.npz
```

每个 `.npz` 里读取：

```text
x: [T, 277]
```

训练时在线或离线生成缺失传感器任务：

- 对每个数据文件随机挑选一个或多个连续帧区间。
- 每个区间随机选择缺失传感器数量，范围为 `1-4`。
- 必须保证至少还有 `2` 个传感器未缺失。
- 对被选中的传感器，在对应帧区间内遮盖它的 tracker position 和 tracker rotation 维度。
- 同时在追加的 6 维传感器丢失标签中，把对应传感器通道置为 `1`。

生成后的 batch 字段建议为：

```text
x: [B, 283, T]
valid_frame_mask: [B, T]
sensor_missing_labels: [B, 6, T]
inpaint_mask: [B, 283, T]
```

其中：

- `x[:, :277, :]` 是 X277 特征。
- `x[:, 277:283, :]` 是 6 维传感器丢失标签。
- `inpaint_mask=True` 表示该位置需要扩散模型补全并参与 loss。
- `inpaint_mask[:, 277:283, :]` 必须固定为 `False`，传感器丢失标签只作为条件输入，不参与预测损失。
- 被遮盖的 tracker 维度可以置零，或保留原值但通过 `inpaint_mask` 控制加噪；实现时优先采用和 StableMotion 一致的方式：条件位置保留原值，待补全位置参与扩散加噪与 loss。

## StableMotion 参考方式

可以重点参考 `D:/Desktop/动画项目/StableMotion-改进模型训练` 中的多维标签训练思路，尤其是：

- 7 维损坏标签如何拼接到动作特征后面。
- `inpaint_cond` 如何表达哪些维度由模型生成，哪些维度作为条件。
- `y["mask"]` 如何只在需要修复的位置计算 masked loss。
- 多维标签和特征维度之间如何保持明确映射。

但本项目需要做以下简化：

- 不做 detection 模式。
- 不做 `fraction` batch 切分。
- 不训练单维或 7h 损坏检测分支。
- 不预测传感器丢失标签。
- 只训练 fix/inpainting 模式。

因此，`StableMotion-改进模型训练` 应作为 mask 组织方式和多维条件设计的参考，而不是整体照搬训练 loop。

## 实现步骤

1. 接入 X277 数据集
   - 新增真实数据集 loader，读取 `dataset/AMASS_x277_60hz/**/*.npz` 中的 `x: [T, 277]`。
   - 支持从 `manifest.jsonl` 使用 `stablemotion_split_key`，兼容 StableMotion 风格 split。
   - 保留 `M/...` 镜像样本路径，继续复用 StableMotion 命名规则。

2. 实现传感器遮盖生成器
   - 新增 `data_loaders/sensor_masking.py`。
   - 提供传感器索引到 X277 维度的映射函数。
   - 提供随机帧区间和随机传感器选择函数。
   - 输出 `sensor_missing_labels` 和 `inpaint_mask`。

3. 调整数据 collate
   - 将可变长度 `[T, 277]` 样本 padding 成 `[B, 277, T]`。
   - 拼接 6 维传感器丢失标签，得到 `[B, 283, T]`。
   - 同步生成 `valid_frame_mask`，padding 帧不参与 loss。

4. 调整模型配置
   - 将默认 `input_feats` 从临时 smoke 骨架的 `190` 改为 `283`。
   - 保留当前 `[B, C, T]` DiT 接口。
   - 更新注释和参数名，使其明确表示 X277 + 6 维传感器缺失标签。

5. 调整训练 loop
   - `mask_manager` 不再内部随机逐特征 mask。
   - 直接使用数据集提供的 `inpaint_mask`。
   - 将 `model_kwargs["inpaint_cond"]` 与 `model_kwargs["y"]["mask"]` 都设为 `inpaint_mask`。
   - 将 `model_kwargs["valid_frame_mask"]` 设为 batch 的 `valid_frame_mask`。
   - label 维 mask 固定为 False，确保 loss 只来自被遮盖的 tracker 维度。

6. 迁移 robust yaw 改进
   - 参考 `D:/Desktop/动画项目/StableMotion-FSQ集成/data_loaders/amasstools/globsmplrifke_feats.py` 与 `globsmplrifke_base_feats.py`。
   - 将 `data_converter/amass_to_x277.py` 中当前基于 pelvis rotation 的 yaw 提取，替换为肩膀/髋部融合的 robust yaw。
   - 保留退化 fallback：融合向量退化时沿用上一帧 yaw，第一帧退化则置 0。

## 测试计划

- 数据集 smoke test：读取 2 个真实 `.npz`，确认输出 `x: [B, 283, T]`。
- 维度映射 test：固定传感器索引，确认只遮盖 `[216:234)` 和 `[234:270)` 中对应切片。
- mask test：固定随机种子，确认每条样本缺失传感器数量为 `1-4`，且至少 2 个传感器保留。
- label test：确认 `x[:, 277:283, :]` 只包含 0/1，且对应缺失传感器和帧区间。
- inpaint loss test：确认 label 维不参与 loss，被遮盖 tracker 维参与 loss，padding 帧不参与 loss。
- model smoke test：用 `[2, 283, 60]` 前向，输出仍为 `[2, 283, 60]`。
- training smoke test：用真实 X277 数据跑 `num_steps=1`，确认 loss、反向传播和 checkpoint 保存正常。
- yaw regression test：选择包含 crawl/prone 的样本，对比旧 yaw 与 robust yaw，确认 yaw 更连续且无 NaN。

## 当前默认假设

- 当前转换数据已经位于 `dataset/AMASS_x277_60hz`。
- `.npz` 内部字段为 `x: [T, 277]`。
- 镜像数据已经按 StableMotion 风格放在 `M/` 下。
- 6 维传感器标签只表达“该传感器在该帧是否缺失”，不表达检测概率或损坏类别。
- 第一个可训练版本只做 fix/inpainting，不做检测、不做 CFG、不做多任务检测分支。
