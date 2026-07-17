# Stationary 五通道有界投影

## 基线与范围

- 日期：2026-07-18
- 工作分支：`codex/c04-cleanup-refactor`
- 基线提交：`ea925a2`（`refactor: finalize C04 artifact cleanup`）
- 基线标签：`baseline/stationary-bounded-projection`
- 实施方式：直接使用现有 cleanup 工作树，不新建工作树或实验分支。

本次只修改 `target_dit` 的 stationary 输出语义和与之重复的 range loss 默认配置。模型仍为 `214→154`，不新增 `StationaryHead`，ONNX/Sentis 仍只有一个 `pred_x0` 输出，schema、normalizer、Resolver 和 Unity C# 接口均不变。

## 产物约定

- 本次不生成训练 run、dataset、output、save 或 checkpoint，也不提交任何二进制产物。
- 未启动训练；run 目录与训练命令均为不适用。后续姿态实验继续使用独立实验配置，不改写历史 C04 profile。

## 关键文件

- `model/realtime_pose_target_dit.py`：五通道 sigmoid 投影。
- `diffusion/realtime_pose/config.py`：range 默认权重和梯度标定目标。
- `tests/smoke/train/test_realtime_pose_target_dit.py`：概率范围、梯度、inpaint 与 ONNX 图契约。
- `tests/smoke/train/test_realtime_loss_calibration.py`：默认阈值、margin 与标定目标。
- `tests/smoke/train/test_realtime_pose_training.py`：默认辅助 loss 回归值。

## 决策

`output_proj` 的 `149:154` 五个值解释为 logits，并在与 inpaint 已知值混合前执行：

```text
stationary_prob_5 = sigmoid(stationary_logits_5)
```

stationary 通道在 normalizer 契约中固定为 `mean=0, std=1`，因此该概率可直接写回归一化的 `x0`。其余 149 个目标通道不做 sigmoid。被 inpaint mask 关闭的已知 stationary 值保留输入原值。

Loss 约束如下：

- 保持 runtime threshold `0.7` 和 margin `0.1`。
- 保持 active/inactive 两类分别归一化、等权相加，不引入 false-lock/missed-lock 非对称代价。
- `stationary_range_loss` 继续作为可显式启用的诊断项，但默认权重改为 `0`，并移出梯度标定目标。
- 原始输出越界率不再作为模型筛选的硬验收条件；有界投影本身保证当前 `target_dit` 的预测概率位于 `[0,1]`。
- 当前阶段仍以 pose、tracker、rollout 指标选择模型；stationary 指标只记录，不作为姿态实验的淘汰条件。

## Checkpoint 与运行时说明

本次没有新增参数或 buffer，旧 C04 `target_dit` checkpoint 可 strict load，state dict 和导出输入/输出名称不变。旧 checkpoint 的五个 raw stationary 输出会被重新解释为 logits，因此 stationary 数值行为会变化；需要用相同评估集记录变化，不能把代码兼容等同于指标等价。姿态的 `0:149` 模型输出未被投影修改。

## 验证记录

- 定向 train：`conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/train/test_realtime_pose_target_dit.py tests/smoke/train/test_realtime_loss_calibration.py tests/smoke/train/test_realtime_pose_training.py -q`，结果 `36 passed, 1 skipped`。
- export + schemas + sample：`conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke/export tests/smoke/schemas tests/smoke/sample -q`，结果 `128 passed`。
- 完整 smoke：`conda run --no-capture-output -n diffusionposer5070 pytest tests/smoke -q`，结果 `354 passed, 1 skipped`。
- 唯一 skip：当前 `diffusionposer5070` 环境未安装 `onnx`；ONNX smoke 已加入图中必须存在 `Sigmoid` 的断言，依赖可用时执行。
- 真实 C04 `model000005000.pt`：110 个 state-dict key 全部匹配，无 missing/unexpected；前向检查确认 `0:149` 与线性输出逐值一致，`149:154` 等于对应 logits 的 sigmoid。
