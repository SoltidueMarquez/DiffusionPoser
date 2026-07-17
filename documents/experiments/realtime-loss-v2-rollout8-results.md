# Realtime Loss v2 A00-A03 筛选结果

## 训练末 1k step

| 实验 | loss | simple | aux | grad norm | 裁剪比例 | 耗时(min) |
|---|---|---|---|---|---|---|
| A00_legacy_h1 | 0.020256 | 0.003797 | 0.008580 | 6.3966 | 100.0% | 38.0 |
| A01_gate_clean_h1 | 0.013775 | 0.003855 | 0.004140 | 5.0685 | 100.0% | 36.9 |
| A02_loss_v2_h1 | 0.007462 | 0.003882 | 0.000266 | 0.2621 | 0.3% | 36.5 |
| A03_loss_v2_h1_8 | 0.012331 | 0.004258 | 0.000273 | 0.2695 | 1.7% | 108.4 |

## 固定长序列评估

> 本次评估按 A00+A03、A01+A02 两路并行；DDIM/end-to-end 延迟和 FPS 受 GPU 竞争影响，不作为单模型实时性能基准，其余误差与分组指标用于本次筛选。

| 指标 | A00_legacy_h1 | A01_gate_clean_h1 | A02_loss_v2_h1 | A03_loss_v2_h1_8 |
|---|---|---|---|---|
| full_six_mpjpe_cm | 4.22239 | 3.90217 | 4.34409 | 4.09479 |
| full_six_mpjre_deg | 12.6252 | 11.6956 | 13.6581 | 12.2925 |
| full_six_mpjve_cmps | 63.8685 | 61.142 | 62.3372 | 66.0211 |
| standard_three_mpjpe_cm | 20.1711 | 19.7511 | 20.2422 | 18.463 |
| standard_three_mpjre_deg | 21.6724 | 21.2641 | 22.2214 | 19.6288 |
| standard_three_mpjve_cmps | 175.241 | 161.522 | 162.98 | 161.046 |
| hip_missing_mpjve_cmps | 175.835 | 163.599 | 161.111 | 163.754 |
| hip_missing_jitter_mps3 | 7234.11 | 6682.4 | 6871.28 | 10228.4 |
| transition_3_to_6_reconnect_mpjve_cmps | 155.919 | 156.241 | 159.141 | 185.321 |
| transition_3_to_6_reconnect_jitter_mps3 | 9801.62 | 9765.73 | 9884.52 | 12694.6 |
| transition_3_to_6_reconnect_pj | 1833.7 | 1991.53 | 1822.06 | 1765.78 |
| transition_3_to_6_reconnect_auj | 9586.67 | 9733.21 | 9461.67 | 11997 |
| tracker_reprojection_pos_mean_cm | 5.00941 | 4.55012 | 4.82355 | 4.63665 |
| tracker_reprojection_rot_mean_deg | 23.5574 | 22.4206 | 23.5167 | 20.4961 |
| stationary_f1 | 0.214402 | 0.231765 | 0.223423 | 0.272164 |
| stationary_clamp_pre_out_of_bounds_ratio | 0.532544 | 0.485255 | 0.461481 | 0.421085 |
| ddim_p50_ms | 58.3624 | 60.5763 | 60.0641 | 61.0025 |
| end_to_end_p50_ms | 61.2656 | 63.522 | 62.9856 | 63.9667 |
| fps | 16.2912 | 15.5305 | 15.6257 | 15.4501 |

## 验收

- PASS — `finite`
- PASS — `pose_guard_a03_vs_a00`
- FAIL — `stationary`
- FAIL — `rollout_a03_vs_a02`
- PASS — `last_1k_gradient_clipping`
- FAIL — `gradient_ratio_calibration`

总判定：**FAIL**。

原始结构化结果：`D:\Projects\SchoolWorkProjects\firstPaperRalated\01_当前主线项目\DiffusionPoser\documents\experiments\realtime-loss-v2-rollout8-results.json`。
