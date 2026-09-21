# 椅子扶手碰撞后处理部署

此功能默认关闭，仅用于 Unity 部署实验。当前采用分段 ONNX 与碰撞 Compute Shader，
整条五步链一次提交、最终一次读回。接口、坐标和参数语义统一见
[contract.md](../contract.md#椅子扶手的可选残差碰撞后处理)。
没有修改训练、常规 Python 离线采样或原有普通 sampler 的输入。

## 标定与导出

1. 在相邻 FLUIDUnity 工程执行菜单 `FLUID/Collision/Setup chair scene and calibrate proxies`。
   它打开并保存 `VRSeated` 场景，绑定两个现有扶手 BoxCollider，读取蒙皮一次，
   写入 `Assets/FLUID/Models/collision_profile.json`。选中组件可查看 Gizmos。
   代理包络以覆盖对应蒙皮顶点为目标，三球近似可能偏保守；先检查 Gizmos，
   调整代理后必须重新导出模型。该菜单用于编辑模式，勿用于覆盖未保存的场景编辑。
2. 在 DiffusionPoser 根目录导出碰撞命令链的三个网络与配置：

```powershell
conda run -n diffusionposer5070 python -m export.export_sentis_denoiser `
  --model_path artifacts/unity_demo/dit/ema000100000.pt `
  --predictor_model_path artifacts/unity_demo/predictor/model_latest.pt `
  --normalizer_dir artifacts/unity_demo/normalizer `
  --ik_calibration_path artifacts/unity_demo/ik_calibration.json `
  --body_fbx_rest_json ../FLUIDUnity/Assets/StreamingAssets/FLUID/Inference/body_fbx_rest.json `
  --collision_profile ../FLUIDUnity/Assets/FLUID/Models/collision_profile.json `
  --output_dir output/collision_deployment --ts_respace 5 --sampler_only `
  --collision_postprocess `
  --postedit_h 0.0002 --postedit_m 0.005 --collision_tracking_tolerance 0.08
```

当前采用单一保真项的确定性梯度更新，移除额外混合回拉；具体公式和默认参数见契约。
手掌球使用与退化线段等价的直接距离，胶囊仍覆盖整条轴线。
速度优化利用左右臂损失独立性，把每轮 49 个候选合并为 25 个，仍计算全部
24 维中心差分。内层只维护 24 维活动状态，几何准备仅保留必要的脊柱/肩部
链路，复用旋转投影。部署时由 Shader 合并执行这些几何计算，Python 保留作参考。
五步采样、每步五次更新、四个插值位置和所有优化参数保持不变。
需要重新实验时使用 `--postedit_steps`、
`--postedit_h`、`--postedit_m`、`--postedit_difference_step`、
`--collision_margin`、`--collision_tracking_tolerance` 显式改常量并重导出。
保持 `--sampler_only` 可避免覆盖已有 Predictor 和普通运行时配置。

3. 将生成的 `pose_sampler_condition.onnx`、`pose_sampler_step.onnx`、
   `pose_sampler_projection.onnx` 和 `collision_compute.json` 复制到
   `FLUIDUnity/Assets/FLUID/Models`，执行 `FLUID/Collision/Bind exported sampler`。
4. `VRSeated` 中的 `FluidCollisionPostprocess.Enable Collision Postprocess`
   在启动推理时生效。关闭时仍加载原五步 sampler；开启后缺模型、Shader 或配置会明确报错。
   原有坐姿、脚部、历史反馈和显示过渡继续工作。

## 数值与回放检查

独立后处理图和几何夹具：

```powershell
conda run -n diffusionposer5070 python -m export.check_collision_postprocess `
  --body_fbx_rest_json ../FLUIDUnity/Assets/StreamingAssets/FLUID/Inference/body_fbx_rest.json `
  --collision_profile ../FLUIDUnity/Assets/FLUID/Models/collision_profile.json `
  --output_dir output/collision_checks
```

将导出命令的模块名改为 `export.check_collision_sampler` 并去掉 `--sampler_only`，
可生成 `sampler_fixtures.json`。分段 ONNX 使用 ORT，碰撞用 Python 参考计算，
与完整 PyTorch sampler 比较；此检查使用录制条件及人为放置的接触盒体。
再运行 `export.check_collision_chain --output_dir <导出目录> --fixtures <sampler_fixtures.json>`，
生成 `shader_fixtures.json`。该目录需要包含 `collision_compute.json` 和三个 ONNX。
ORT 对照关闭图优化：默认融合在一个深接触请求上造成了差分链的明显偏离。

把 `geometry_fixtures.json`、`sampler_fixtures.json`、`shader_fixtures.json` 复制到 Unity 的
`Library/CollisionChecks`。先执行 `FLUID.Editor.FluidCollisionComputeCheck.Run`，
再运行 `python -m export.check_collision_shader_results --fixtures <shader_fixtures.json>`，
检查最终投影、每轮更新和 GPU 实际迭代点的几何/梯度。退化零旋转只要求有限输出、
固定分量和最终投影一致；不要求不可微奇点附近的候选梯度一致。
真实 Tracker 的关闭／开启对照使用菜单 `FLUID/Collision/Check geometry and GPU sampler`，
或批处理 `-executeMethod FLUID.Editor.FluidCollisionCheck.Run`。此入口只比较普通模型、
分段旁路和当前 Compute Shader 链；旧碰撞 ONNX 已删除，不再参与检查。
保留图形设备（Windows 可加 `-force-d3d11`），不要加 `-nographics` 或 `-quit`；
检查结束自行退出，不保存测试中的场景变化。

检查读取同一段输入分别运行关闭/开启路径，保持相同噪声种子，模拟 30 Hz
推理时间轴和 90 Hz 显示过渡，记录全链路平均/P95、实际显示代理穿透帧比例、
最大/P95 穿透、腕误差及三阶位置差分抖动。默认由现有站立录制构造下坐与
扶手接近轨迹并保存 `chair_probe.json`；这是构造用例，不能称为实录。
提供 `-fluidCollisionRecording <tracker录制.json>` 可使用指定录制。
产物 `unity_report.json`、`display_path*_run*.json`、`poses_path*_run*.json`、`raw_poses_path*_run*.json` 都在上述 Library
目录；最终模型姿态和实际显示姿态分开保存，方便复查过渡中的剩余穿透。
真实 VR 的渲染帧时间、显示延迟和丢帧仍须在现场运行中用原有性能日志检查。

```powershell
conda run -n diffusionposer5070 pytest tests/smoke
```

## 实测结果与限制

环境：Unity 6000.6.0f1 / Inference Engine 2.6.1 / RTX 5070，编辑器、无 XR。
固定输入为 `output/collision_comparison/before/chair_probe.json`，由站立录制构造的下坐轨迹，
不是现场椅子实录。各路径预热 5 次、测量 60 次，重复三轮并重置历史和噪声。
完整耗时从输入上传到最终姿态取得，包含一次读回和协程等待；20 ms 仅为参考。

| 路径 | 三轮平均耗时 | 三轮 P95 |
| --- | ---: | ---: |
| 普通五步完整 ONNX | 14.49–14.65 ms | 15.74–16.02 ms |
| 分段链路，旁路碰撞编辑 | 17.11–17.69 ms | 19.11–19.93 ms |
| 旧碰撞 ONNX | 67.56–68.11 ms | 69.72–69.74 ms |
| 分段链路＋碰撞 Shader | 18.38–18.60 ms | 19.79–20.15 ms |

分段本身有约 2.6–3.0 ms 代价；融合碰撞相对旁路只增加约 0.7–1.5 ms。
这是完整链路差值，不是精确的单 kernel GPU 时间。已加入条件/KV、每步 DiT、几何准备、
候选评估、更新、DDIM 与最终投影的 CommandBuffer GPU Profiler 标记，避免把 CPU
提交耗时误认为 GPU 执行时间。五步采样、每步五次更新、四个插值位置及优化参数未变。

新路径三轮构建命令链/Worker 约 58–86 ms，首次执行约 29–42 ms；这些值不是冷启动保证，
进程此前已执行数值检查并编译 Shader。Unity 图形驱动内存统计运行期间约 147.8 MB，
旧碰撞路径约 125.3 MB；退出后都回到约 25.0 MB，三轮未出现递增。
这是 Unity 驱动分配统计，不是整个显卡的峰值显存。五个独立 Worker 增加了约 22.5 MB。

### 动作与数值

| 同段显示指标 | 关闭 | 旧碰撞 ONNX | 碰撞 Shader |
| --- | ---: | ---: | ---: |
| 穿透帧比例 | 68.89% | 68.33% | 68.33% |
| 最大穿透 | 94.48 mm | 69.84 mm | 72.27 mm |
| P95 穿透 | 93.37 mm | 63.63 mm | 69.99 mm |
| 平均腕误差 | 17.03 mm | 42.05 mm | 38.24 mm |
| 手部三阶差分抖动 | 620.05 m/s³ | 761.29 m/s³ | 689.42 m/s³ |

不能把旧 GPU 输出当作正确性真值：捕获该路径的 65 个相同请求快照，使用完整 PyTorch
重算，旧 ONNX 的 Unity GPU 最大几何偏差约 **15.56 mm**；新 Shader 全部满足原归一化
容差 `atol=1e-4, rtol=1e-3`，最大几何偏差约 **0.0165 mm**。因此回放指标略有变化，
不是通过改损失权重或修正幅度换取速度。旧图的具体出错算子尚未定位，产品已不使用该路径。
旧图此前修复过仅带上界的 Clip，浅接触夹具通过不足以覆盖全部坐姿请求。

51 组单步 Shader 夹具覆盖肘/腕掌、中段、边缘、下方接近、坐姿、交叉及退化旋转。
在 GPU 实际迭代点重算参考，非退化候选位置最大误差约 0.0036 mm，最终位置最大误差
约 0.0181 mm。梯度比较使用 L2 相对误差（2.5%，绝对容差 0.035）和非微小梯度方向
余弦 >0.99，同时检查每次更新与最终投影；不能把不同迭代轨迹上的单分量差当作求导错误。
完整 Python smoke 回归：194 项通过。

诊断文件：

- `output/collision_compute/benchmark_report.json`：四条路径三轮性能、显示指标、内存统计。
- `output/collision_compute/compute_report.json`、`shader_check.json`：分段 GPU 及独立 Shader 检查。
- `output/collision_compute/replay_reference_check.json`：65 帧相同输入的 PyTorch/ORT/GPU 对照。
- `captured_replay_inputs.json` 保存旧 GPU 请求和结果，`verified_replay_inputs.json` 保存完整 PyTorch 真值。

重现逐帧检查：在历史回放命令加 `-fluidCollisionCapture` 捕获输入（诊断运行不作性能结果），
运行 `export.check_collision_replay`，使用导出命令相同参数并增加 `--replay_inputs` 和
`--gpu_report`；GPU 报告来自 `FluidCollisionComputeCheck.Run -fluidCollisionReplay`。
该模式读取 Library 中的 `replay_sampler_inputs.json`。用完整 PyTorch 真值替换其 expected 后，
可执行独立的 Unity 回归。CPU 参考使用 ORT 关闭图优化，生产 Unity 不运行 ORT。

仍有大量剩余穿透，四个插值采样不保证连续零穿透。上述显示指标只有 180 帧，抖动按模拟
90 Hz 计算；真实 VR 渲染负载、显示延迟和现场丢帧尚未验收。
