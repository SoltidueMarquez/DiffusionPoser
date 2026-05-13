# DiffusionPoser Fix-Only 训练路线计划

## Summary
目标是把当前项目从早期 smoke 训练骨架推进到正式的 DiffusionPoser 修复训练：使用已转换完成的 X277 数据，追加 6 维传感器丢失标签，训练一个只做 fix/inpainting 的扩散模型。当 6 个传感器中的一部分在某段时间内缺失时，模型根据剩余传感器和丢失标签补全对应动作特征。计划文件实施时写入 `docs/diffusionposer_fix_training_plan.md`。

## Key Changes
- 数据输入统一为 `[B, 283, T]`，其中前 `277` 维为 X277 动作/传感器特征，后 `6` 维为传感器丢失标签。
- 传感器顺序固定沿用转换脚本中的 `TRACKER_NAMES`：`head, left_wrist, right_wrist, waist, left_foot, right_foot`。
- 训练只保留 fix 模式，不再迁移 StableMotion 的 detection 模式、fraction 切分、单维/多维损坏检测 loss。
- 每个样本训练时随机生成遮盖任务：随机选择连续帧区间，随机选择 `1-4` 个缺失传感器，保证至少还有 `2` 个传感器可用。
- `inpaint_mask=True` 仅标记需要扩散模型补全的位置；6 维丢失标签始终作为条件输入，不参与 loss。
- X277 朝向计算改用 FSQ 项目中肩膀+髋部融合的 robust yaw，替换当前 `extract_yaw_from_rotations(pelvis_rotations)` 逻辑，避免爬行/俯卧姿态朝向不稳定。

## Implementation Steps
1. 创建计划文档
   - 新建 `docs/diffusionposer_fix_training_plan.md`，写入本计划。
   - 文档中明确当前版本训练目标是“传感器缺失修复”，不是损坏检测。

2. 接入 X277 数据集
   - 新增真实数据集 loader，读取 `dataset/AMASS_x277_60hz/**/*.npz` 中的 `x: [T, 277]`。
   - 支持从 `manifest.jsonl` 读取 `stablemotion_split_key`，并兼容 StableMotion 风格 split，包括 `M/...` 镜像样本。
   - collate 输出 `x: [B, 283, T]`、`valid_frame_mask: [B, T]`、`sensor_missing_labels: [B, 6, T]`、`inpaint_mask: [B, 283, T]`。

3. 实现传感器遮盖生成器
   - 新增 `data_loaders/sensor_masking.py`，封装随机区间和随机传感器选择。
   - 默认每条样本生成 1 个连续缺失区间，区间长度默认在有效长度的 `10%-60%` 之间，至少 5 帧。
   - 对缺失传感器，将其 6 维标签在对应帧置 1，并把对应 X277 特征加入 `inpaint_mask`。
   - 传感器到 X277 特征的默认映射：
     `head` 对应 head/neck 关节 rot+vel 和 head tracker pos/rot；
     `left_wrist/right_wrist` 对应左右手臂关节 rot+vel 和对应 tracker pos/rot；
     `waist` 对应 pelvis/spine 关节 rot+vel、waist delta/yaw 和 waist tracker pos/rot；
     `left_foot/right_foot` 对应左右腿脚关节 rot+vel、对应 foot contact 和对应 tracker pos/rot。

4. 调整模型和训练接口
   - 将默认 `input_feats` 从临时的 `190` 改为 `283`。
   - 保留当前 DiT 的 `[B, C, T]` 接口，但把注释和参数名改成 X277+sensor-label 语义。
   - 训练 loop 的 `mask_manager` 改为直接使用 dataset 提供的 `inpaint_mask`，不再用随机逐特征 `mask_ratio`。
   - `diffusion.training_losses` 继续使用 `y["mask"]` 做 masked loss，label 维的 mask 固定为 False。

5. 迁移 robust yaw
   - 在 `data_converter/amass_to_x277.py` 中新增 `compute_robust_root_yaw_from_joints`。
   - 使用肩膀/髋部横向向量与髋部 forward 融合，计算稳定水平 forward，再得到 yaw。
   - `build_root_frames` 改为从 `joint_positions` 计算 yaw，而不是从 pelvis rotation 的 forward 投影计算。
   - 保留现有 fallback：如果融合向量退化，则沿用上一帧 yaw，第一帧退化则置 0。

## Test Plan
- 数据集 smoke test：读取 2 个真实 `.npz`，确认 `x` 输出为 `[B, 283, T]`，label 维只有 0/1。
- mask test：固定随机种子，确认每个样本缺失传感器数为 `1-4`，且至少两个传感器未缺失。
- inpaint mask test：确认 label 维不参与 loss，缺失传感器对应的 X277 特征维参与 loss。
- model smoke test：用 `[2, 283, 60]` 前向，输出保持 `[2, 283, 60]`。
- training smoke test：用真实 X277 数据跑 `num_steps=1`，确认 loss、反向传播和 checkpoint 保存正常。
- yaw regression test：选包含 crawl/prone 的样本，对比旧 yaw 与 robust yaw，确认 yaw 连续性更稳定且无 NaN。

## Assumptions
- 当前数据转换已经生成 `dataset/AMASS_x277_60hz`，并且 `.npz` 内部字段为 `x: [T, 277]`。
- 镜像数据已经按 StableMotion 风格放在 `M/` 下，manifest 中已有 `M/...` 的 `stablemotion_split_key`。
- 6 个传感器标签只表达“该传感器在该帧是否缺失”，不表达检测概率或损坏类别。
- 第一个可训练版本不做 classifier-free guidance、不做检测分支、不做 StableMotion 的 7h 检测训练。
