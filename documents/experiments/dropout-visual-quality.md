# 掉线视觉质量补实验

本实验针对审稿人 hADe 对掉线后迟缓和漂浮的疑问。固定 Ours、RPM-Reactive、
RPM-Smooth 的 checkpoint 和推理参数，不训练、不调参，不安排原论文指标复现检查。

## 比较口径

- 使用 A-P2 的 138 条测试序列，30 fps；沿用官方 hand_tracking gap，seed=6。
- 官方 gap 索引基于删掉首帧后的 tracking feature，换成 source 帧时加 1。
- Ours 沿用原实验 seed=10、EMA、DDIM10、重连 smoothstep 10 帧。
- RPM 输入在缺失手对应的全部 18 个通道置零，严格遵循官方 evaluator；
  不套用 Ours 对跨 gap 速度通道的遮挡规则。
- Ours 的 AUJ 曲线直接引用已保存的论文实验结果。RPM 新结果使用对应官方模型输出，
  经现有转换函数重定向到同一 source body.fbx 骨架，并仅利用始终在线的头部位置确定平移。
  新增的共享骨架指标不能标成 RPM 论文原生 SMPL 评估数字。
- 所有比较按绝对 source 帧对齐，跳过 source frame 30 之前的计分区间。
  每个阶段统计 0.5 秒，事件边界沿用官方 transition evaluator。

## 指标

AUJ 按发布代码计算：先在每帧取 22 个关节的 jerk 模长最大值，再对事件平均，
最后累加预测与 GT 曲线绝对差，乘以 0.01。此实现没有乘帧间隔，因此报告中不把
离散和标成物理时间积分。分别报告高于 GT 和低于 GT 的差额。

失联手腕的 MPJPE、MPJVE、预测平均速度和 GT 平均速度先按事件计算，再等权平均。
速度比取这两个平均速度的比值，不平均各事件的比值；GT 平均速度为零时记为 N/A。
速度比只能说明动态幅度，不能当成延迟指标。阶段内的速度使用后向差分，并保留窗口前一帧。

GT 双脚关节高度第 5 百分位作为序列参考地面。脚部距地不超过 5 cm 且三维速度不超过
0.1 m/s 时视作接触代理；所有方法共享此掩码。报告接触帧覆盖率、预测脚高相对 GT 的
正向误差，以及预测脚部水平滑动速度。两个手部事件重叠的阶段帧取并集，不重复计数脚部。
没有有效接触帧时指标为 N/A，而不是零。

## 视频

选择 GT 腕部平均速度处于中位数、90% 分位、最大值附近的三个事件，
以及剩余事件中 Ours 腕部误差最大的一个。要求具有完整的掉线前后各 0.5 秒上下文；
排序相同时按序列名、掉线起始帧、手部编号确定顺序。双手同时掉线且区间相同的片段只选一次。

复用项目已有的逆重定向、SMPL-H FK、相机和场景渲染，四种方法使用相同身形，
以在线头部定位，保留预测头部残差；不把 GT 根平移直接赋给预测。四栏共享相机和 GT 地面。
原速 30 fps，不增加姿态平滑或触地修正。定量指标仍来自 source 骨架，SMPL-H 视频用于
统一体型的定性展示，其脚部几何与 source 骨架并不完全相同；视频不能替代感知用户研究。

## 运行

在仓库根目录使用 `diffusionposer5070` 环境。以下命令生成新的实验目录；已有完整轨迹时
可直接运行分析、渲染命令，无需重新推理。

```bash
REVIEW_ROOT=/home/asus/Datasets/DiffusionPoserArtifacts/predictor/output/dropout_visual_quality_20261108
ARTIFACT_ROOT=/home/asus/Datasets/DiffusionPoserArtifacts
BASELINE_ROOT=/home/asus/Datasets/DiffusionPoserBaselines
REFERENCE_REPORT="$ARTIFACT_ROOT/predictor/output/prior_residual_rpm_p2_hand_recovery_soft10f_dropout_trained_full.json"

conda run --no-capture-output -n diffusionposer5070 python -m sample.evaluate_hand_signal_recovery \
  --source_dir "$ARTIFACT_ROOT/source_30hz" \
  --split_dir data_loaders/splits/RPM-P2 \
  --normalizer_dir "$ARTIFACT_ROOT/predictor/normalizer" \
  --predictor_model_path "$ARTIFACT_ROOT/predictor/runs/predictor_rpm_hand_dropout/model_latest.pt" \
  --dit_model_path "$ARTIFACT_ROOT/predictor/runs/dit_rpm_hand_dropout/20260901_132605_prior_residual_20f_tracker_ik_hand_dropout/model000100000.pt" \
  --gap_config "$BASELINE_ROOT/RPM/datasets_processed/amass_p2/eval_gap_configs/hand_tracking.json" \
  --activation_blend_frames 10 --ts_respace 10 --seed 10 --use_ema true \
  --diagnostic_dir "$REVIEW_ROOT/ours" --output_json "$REVIEW_ROOT/ours_run.json"

conda run --no-capture-output -n diffusionposer5070 python -m sample.prepare_dropout_baseline_inputs \
  --source_dir "$ARTIFACT_ROOT/source_30hz" --split_dir data_loaders/splits/RPM-P2 \
  --amass_dir /home/asus/Datasets/incoming/AMASS \
  --smpl_model_dir /home/asus/Datasets/incoming/body_models \
  --rpm_stats_dir "$BASELINE_ROOT/RPM/datasets_processed/amass_p2/new_format_data" \
  --gap_config "$BASELINE_ROOT/RPM/datasets_processed/amass_p2/eval_gap_configs/hand_tracking.json" \
  --output_dir "$REVIEW_ROOT/inputs"

for RPM_SETTING in reactive smooth; do
  conda run --no-capture-output -n diffusionposer5070 python -m sample.run_official_rpm_dropout \
    --rpm_repo "$BASELINE_ROOT/RPM" --human_body_prior_repo "$BASELINE_ROOT/human_body_prior" \
    --model_path "$BASELINE_ROOT/RPM/checkpoints/amass_p2/$RPM_SETTING/model_latest.pt" \
    --input_dir "$REVIEW_ROOT/inputs" --output_dir "$REVIEW_ROOT/rpm_$RPM_SETTING"
done

conda run --no-capture-output -n diffusionposer5070 python -m sample.analyze_dropout_visual_quality \
  --experiment_dir "$REVIEW_ROOT" --reference_report "$REFERENCE_REPORT" \
  --gap_config "$BASELINE_ROOT/RPM/datasets_processed/amass_p2/eval_gap_configs/hand_tracking.json"

conda run --no-capture-output -n diffusionposer5070 python -m sample.render_dropout_visual_quality \
  --experiment_dir "$REVIEW_ROOT" --smpl_model_dir /home/asus/Datasets/incoming/body_models
```

输出包括 `metrics.json/csv`、`metrics_table.png/pdf`、`jerk_comparison.png/pdf`、
`case_wrist_speed.png/pdf`、`events.json`、`selected_events.json`、
`dropout_comparison.mp4` 和四个独立案例视频。`aligned/` 保存可重用的共享骨架轨迹。

新增轨迹导出是独立诊断产物，不改变 source、checkpoint 或 Unity runtime 接口。
检查新增指标、gap 通道、帧对齐、坐标转换和轨迹导出：

```bash
conda run -n diffusionposer5070 pytest tests/smoke/eval tests/smoke/sample
```
