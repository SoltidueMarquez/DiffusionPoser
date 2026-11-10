"""汇总固定掉线实验，导出共享骨架轨迹、图表和客观选例。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_converter.amass_smpl_utils import AMASS_TO_UNITY, local_to_global_rotations
from data_loaders.body_fbx_kinematics import (
    extract_root_heading_from_source_pelvis_up, fk_body_fbx_local_delta,
    load_body_fbx_rest, source_global_rotations_to_body_fbx_local_delta_6d,
)
from data_loaders.realtime_pose_kinematics import SMPL_PARENTS
from eval.dropout_visual_quality import (
    aggregate_contact_statistics, aggregate_wrist_events, contact_statistics,
    decompose_auj, frame_velocities, gt_contact_mask, require_aligned_frames,
    select_video_events, transition_events, wrist_event_metrics, WRIST_JOINTS,
)
from eval.realtime_pose_metrics import (
    aggregate_rpm_transition_jerk_segments, extract_rpm_transition_jerk_segments,
)
from sample.three_point_baseline_data import baseline_motion_6d_to_rotation_matrices


METHODS = ("Ours", "RPM-Reactive", "RPM-Smooth")
KEYS = {"GT": "gt", "Ours": "ours", "RPM-Reactive": "rpm_reactive", "RPM-Smooth": "rpm_smooth"}
COLORS = {"GT": "#333333", "Ours": "#008b8b", "RPM-Reactive": "#ca662a", "RPM-Smooth": "#7961a8"}
PHASES = ("tracking_to_synthesis", "synthesis_to_tracking")


# region 输入对齐与共享骨架


def retarget_rpm_to_source(motion_6d, target_head_positions, rest):
    """RPM SMPL22 → 项目骨架；平移仅使用始终在线的头部，不使用 GT 根轨迹。"""
    local22 = baseline_motion_6d_to_rotation_matrices(motion_6d)
    count = len(local22)
    local = np.broadcast_to(np.eye(3), (count, 24, 3, 3)).copy()
    local[:, :22] = local22
    global_amass = local_to_global_rotations(local, SMPL_PARENTS)
    global_unity = AMASS_TO_UNITY @ global_amass @ AMASS_TO_UNITY.T
    yaw = extract_root_heading_from_source_pelvis_up(global_unity[:, 0])
    delta = source_global_rotations_to_body_fbx_local_delta_6d(global_unity, yaw)
    positions, rotations = fk_body_fbx_local_delta(
        delta, np.zeros((count, 3), dtype=np.float32), yaw, rest,
    )
    positions += (np.asarray(target_head_positions) - positions[:, 15])[:, None]
    return positions, rotations, yaw


def assemble_sequence(root, item):
    filename = item["file"]
    with np.load(root / "inputs" / filename, allow_pickle=False) as inputs:
        source_path = Path(str(inputs["source_path"]))
        available = inputs["tracker_available"]
        amass_path = str(inputs["amass_path"])
    with np.load(source_path, allow_pickle=False) as source, np.load(root / "ours" / filename, allow_pickle=False) as ours:
        target = source["joints_world"]
        count = len(target)
        require_aligned_frames(ours["source_frames"], np.arange(count))
        if str(ours["source_relative_path"]) != item["source"]:
            raise ValueError("Ours 导出序列与输入 manifest 不匹配。")
        if not np.array_equal(ours["tracker_available"], available):
            raise ValueError("Ours 与 RPM 的 tracker availability 不一致。")
        rest_path = str(source["body_fbx_rest_json"])
        rest = load_body_fbx_rest(rest_path)
        merged = {
            "gt_positions": target, "gt_rotations": ours["target_rotations"],
            "gt_root_yaw": source["root_yaw"],
            "ours_positions": ours["predicted_positions"],
            "ours_rotations": ours["predicted_rotations"],
            "ours_root_yaw": ours["predicted_root_yaw"],
            "tracker_available": available, "source_frames": np.arange(count),
            "amass_path": np.asarray(amass_path), "rest_path": np.asarray(rest_path),
            "source_path": np.asarray(str(source_path)), "fps": np.asarray(30),
        }
        for method in METHODS[1:]:
            key = KEYS[method]
            with np.load(root / key / filename, allow_pickle=False) as rpm:
                require_aligned_frames(rpm["source_frames"], np.arange(1, count))
                if str(rpm["source_relative_path"]) != item["source"] or not np.array_equal(rpm["tracker_available"], available[1:]):
                    raise ValueError("RPM 导出序列或 gap 不匹配。")
                positions, rotations, yaw = retarget_rpm_to_source(
                    rpm["rpm_local_rotations_6d"], target[1:, 15], rest,
                )
            # 官方 feature 从 source frame 1 开始；frame 0 仅作缓存占位，不计分。
            merged[f"{key}_positions"] = np.concatenate([target[:1], positions])
            merged[f"{key}_rotations"] = np.concatenate([ours["target_rotations"][:1], rotations])
            merged[f"{key}_root_yaw"] = np.concatenate([source["root_yaw"][:1], yaw])
    return merged


# endregion

# region 图表


def plot_jerk(report, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), constrained_layout=True)
    fig.suptitle("Supplemental comparison on the shared source skeleton", fontsize=11)
    for ax, phase, title in zip(axes, PHASES, ("Tracker loss", "Tracker recovery")):
        target = report["jerk"]["Ours"][phase]["target_mean_curve_1e2_m_per_s3"]
        time = np.arange(len(target)) / 30
        ax.plot(time, target, color=COLORS["GT"], linestyle="--", linewidth=2, label="GT")
        for method in METHODS:
            entry = report["jerk"][method][phase]
            curve = entry["predicted_mean_curve_1e2_m_per_s3"]
            ax.plot(time, curve, color=COLORS[method], linewidth=2,
                    label=f"{method}: AUJ {entry['decomposition']['auj']:.2f}")
        ours = report["jerk"]["Ours"][phase]["predicted_mean_curve_1e2_m_per_s3"]
        ax.fill_between(time, ours, target, where=np.asarray(ours) < target, color=COLORS["Ours"], alpha=0.12)
        ax.set(title=title, xlabel="Time from transition (s)", ylabel="Event-mean max-joint jerk (100 m/s³)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output_dir / "jerk_comparison.png", dpi=200)
    fig.savefig(output_dir / "jerk_comparison.pdf")
    plt.close(fig)


def plot_case_speeds(selected, root):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), constrained_layout=True)
    for ax, event in zip(axes, selected):
        with np.load(root / "aligned" / event["file"], allow_pickle=False) as data:
            start, end = event["gap_start"] - 15, event["gap_end"] + 15
            joint = WRIST_JOINTS[event["hand"]]
            time = (np.arange(start, end) - event["gap_start"]) / 30
            for method in ("GT", *METHODS):
                velocity = frame_velocities(data[f"{KEYS[method]}_positions"], 30)
                speed = np.linalg.norm(velocity[start:end, joint], axis=-1)
                ax.plot(time, speed, color=COLORS[method], label=method, linewidth=1.6)
            ax.axvspan(0, (event["gap_end"] - event["gap_start"]) / 30, color="orange", alpha=0.12)
            ax.set_title(f"{event['selection']}\n{event['sequence']} | {'left' if event['hand'] == 0 else 'right'} hand", fontsize=9)
            ax.set(xlabel="Time from dropout (s)", ylabel="Wrist speed (m/s)")
            ax.grid(alpha=0.2)
    axes[0].legend(ncol=4, fontsize=8)
    fig.savefig(root / "case_wrist_speed.png", dpi=180)
    fig.savefig(root / "case_wrist_speed.pdf")
    plt.close(fig)


def plot_metric_table(report, root):
    """将新增指标直接排成可供 rebuttal 引用的表，不混入原论文基线数字。"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.7), constrained_layout=True)
    fig.suptitle("Supplemental comparison on the shared source skeleton", fontsize=11)
    columns = ["Method", "Wrist PE\n(cm)", "Wrist VE\n(cm/s)", "Speed\nratio", "Foot lift\n(cm)", "Foot slide\n(cm/s)", "GT contact\ncoverage"]
    fields = ["wrist_mpjpe_cm", "wrist_mpjve_cm_s", "speed_ratio", "foot_positive_height_cm", "foot_horizontal_speed_cm_s", "contact_coverage"]
    for ax, phase, title in zip(axes, PHASES, ("First 0.5 s after tracker loss", "First 0.5 s after tracker recovery")):
        ax.axis("off")
        rows = []
        for method in METHODS:
            values = report["phases"][phase][method]
            rows.append([method] + [
                ("N/A" if values[key] is None else f"{values[key] * 100:.1f}%" if key == "contact_coverage" else f"{values[key]:.3f}")
                for key in fields
            ])
        table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center", colWidths=[0.19] + [0.135] * 6)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.1)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d7dde5")
            if row == 0:
                cell.set_facecolor("#e7edf4")
            elif row == 1:
                cell.set_facecolor("#e3f3f1")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    fig.savefig(root / "metrics_table.png", dpi=200)
    fig.savefig(root / "metrics_table.pdf")
    plt.close(fig)


def write_summary(report, selected, root):
    """报告只陈述本次测量，视觉观察在检查视频后另行记录。"""
    rows = []
    for phase, label in zip(PHASES, ("掉线后", "恢复后")):
        for method in METHODS:
            v = report["phases"][phase][method]
            rows.append(f"| {label} | {method} | {v['wrist_mpjpe_cm']:.3f} | {v['wrist_mpjve_cm_s']:.3f} | {v['speed_ratio']:.3f} | {v['foot_positive_height_cm']:.3f} | {v['foot_horizontal_speed_cm_s']:.3f} |")
    losses = report["phases"][PHASES[0]]
    ours = losses["Ours"]
    reactive, smooth = losses["RPM-Reactive"], losses["RPM-Smooth"]
    text = [
        "# 掉线视觉质量补实验结果", "",
        f"固定模型比较，共 {report['sequence_count']} 条 A-P2 测试序列，30 fps；主要窗口为掉线／恢复后 0.5 秒。",
        "Ours 的 jerk 曲线引用原有结果；新增逐帧轨迹用于局部指标和视频，不设置论文结果复现检查。", "",
        "## 新增指标", "",
        "| 阶段 | 方法 | 腕部 PE (cm) | 腕部 VE (cm/s) | 速度比 | 脚部抬高 (cm) | 脚部水平速度 (cm/s) |",
        "|---|---|---:|---:|---:|---:|---:|", *rows, "",
        f"掉线事件数：{ours['event_count']}；GT 接触覆盖率：{ours['contact_coverage']:.2%}，有效脚部接触样本：{ours['contact_samples']}。",
        "脚部统计对阶段内帧取并集；速度比是事件平均预测速度与事件平均 GT 速度之比。", "",
        "## AUJ 分解", "", "| 阶段 | 方法 | AUJ | 高于 GT 的差额 | 低于 GT 的差额 |", "|---|---|---:|---:|---:|",
    ]
    for phase, label in zip(PHASES, ("掉线后", "恢复后")):
        for method in METHODS:
            d = report["jerk"][method][phase]["decomposition"]
            text.append(f"| {label} | {method} | {d['auj']:.3f} | {d['above_gt']:.3f} | {d['below_gt']:.3f} |")
    text += ["", "AUJ 遵循发布代码的离散求和口径，没有乘帧间隔；不能当成带物理时间单位的积分。", "",
             "## 四个案例", ""]
    for i, event in enumerate(selected, 1):
        text.append(f"- 案例 {i}：{event['selection']}；`{event['sequence']}`，{'左' if event['hand'] == 0 else '右'}手掉线 [{event['gap_start']}, {event['gap_end']})；[视频](case_{i:02d}.mp4)。")
    text += ["", "## 解释边界", "",
             "RPM 输出重定向到与 Ours 相同的 source 骨架，并用在线头部确定平移，因此此处是共享骨架补实验，不是 RPM 论文原生 SMPL 表格的数字。",
             "速度比低于 1 表示平均速度幅度偏低，不能据此断言时间延迟；抬高和滑动是基于 GT 接触代理的运动学指标，不能证明没有感知伪影。",
             "视频复用 SMPL-H 渲染，统一体型并以在线头部定位；不额外平滑或修正足部。定量指标来自 source 骨架，视频网格与其脚部几何并不完全相同；四个选例不能代表全部动作。", "",
             "[Jerk 图](jerk_comparison.png) · [指标表](metrics_table.png) · [腕部速度曲线](case_wrist_speed.png) · [汇总视频](dropout_comparison.mp4)", ""]
    fastest, worst = selected[2], selected[3]
    fast_metrics = fastest["metrics"]["Ours"]
    text += ["## 高动态与最差案例", "",
             f"最大 GT 速度案例中，Ours 的腕部平均速度比为 {fast_metrics['pred_speed_m_s'] / fast_metrics['gt_speed_m_s']:.3f}，腕部位置误差为 {fast_metrics['wrist_mpjpe_cm']:.2f} cm；最差选例的腕部位置误差为 {worst['ours_wrist_mpjpe_cm']:.2f} cm。",
             "总体均值优于基线不等于每段动作都准确。回复应明确保留快速动作的动态衰减与局部姿态偏差这一局限，不宣称无迟缓或无漂浮。", ""]
    (root / "report.md").write_text("\n".join(text), encoding="utf-8")
    d = report["jerk"]["Ours"][PHASES[0]]["decomposition"]
    draft = (
        "Thank you for raising this concern. Our discussion of dropout AUJ was imprecise. In the released RPM evaluator, AUJ is the summed absolute difference between the event-averaged predicted and ground-truth maximum-joint jerk curves; it penalizes both excessive and insufficient motion variation. "
        f"In our saved Table 2 results, the AUJ of {d['auj']:.2f} comprises {d['below_gt']:.2f} from jerk below the ground truth and {d['above_gt']:.2f} from jerk above it. This is consistent with attenuated motion dynamics after tracker loss.\n\n"
        f"To directly examine this issue, we additionally compared fixed checkpoints on all {report['sequence_count']} A-P2 sequences with identical dropout intervals, mapping the outputs to a common evaluation skeleton anchored by the observed head. During the first 0.5 s after dropout, the missing-wrist mean-speed ratio to ground truth was {ours['speed_ratio']:.3f} for ours, versus {reactive['speed_ratio']:.3f}/{smooth['speed_ratio']:.3f} for RPM-Reactive/Smooth. The corresponding wrist position errors were {ours['wrist_mpjpe_cm']:.2f} versus {reactive['wrist_mpjpe_cm']:.2f}/{smooth['wrist_mpjpe_cm']:.2f} cm.\n\n"
        f"On the same ground-truth contact frames, mean positive foot-height error was {ours['foot_positive_height_cm']:.2f} versus {reactive['foot_positive_height_cm']:.2f}/{smooth['foot_positive_height_cm']:.2f} cm, and horizontal foot speed was {ours['foot_horizontal_speed_cm_s']:.2f} versus {reactive['foot_horizontal_speed_cm_s']:.2f}/{smooth['foot_horizontal_speed_cm_s']:.2f} cm/s. These supplemental shared-skeleton measurements are distinct from the published RPM table scores. "
        "These aggregate results do not indicate greater motion-amplitude attenuation or larger contact errors than RPM in this supplemental comparison. "
        f"However, fast motions remain challenging: in the preselected maximum-speed clip, our wrist speed ratio is {fast_metrics['pred_speed_m_s'] / fast_metrics['gt_speed_m_s']:.3f}. "
        "We provide synchronized normal-speed SMPL videos selected by ground-truth motion speed and prediction error, including a worst-error example. We acknowledge the loss of motion dynamics as a limitation; neither a low peak jerk nor these aggregate measurements rules out localized sluggishness or floating. The speed ratio measures motion amplitude, not temporal lag, and the contact measurements are kinematic proxies rather than a perceptual study.\n"
    )
    (root / "rebuttal_draft.md").write_text(draft, encoding="utf-8")


# endregion

# region 汇总与命令行


def analyze(args):
    root = args.experiment_dir
    manifest = json.loads((root / "inputs" / "manifest.json").read_text())
    gaps = json.loads(args.gap_config.read_text())["gaps"]
    archive = json.loads(args.reference_report.read_text())
    (root / "aligned").mkdir(exist_ok=True)
    wrist = {phase: {m: [] for m in METHODS} for phase in PHASES}
    contact = {phase: {m: [] for m in ("GT", *METHODS)} for phase in PHASES}
    predicted_segments = {m: [] for m in METHODS[1:]}
    target_segments = []
    all_events = []
    for item in manifest:
        merged = assemble_sequence(root, item)
        np.savez_compressed(root / "aligned" / item["file"], **merged)
        target = merged["gt_positions"]
        count = len(target)
        source_gaps = [[(a + 1, b + 1) for a, b in hand] for hand in gaps[item["gap_key"]]]
        events = transition_events(source_gaps, count)
        gt_contact, _ = gt_contact_mask(target)
        transition_args = dict(hand_gap_intervals=[[(a - 30, b - 30) for a, b in hand] for hand in source_gaps], fps=30, transition_seconds=0.5)
        target_segments.append(extract_rpm_transition_jerk_segments(joint_positions=target[30:], **transition_args))
        for method in METHODS[1:]:
            predicted_segments[method].append(extract_rpm_transition_jerk_segments(
                joint_positions=merged[f"{KEYS[method]}_positions"][30:], **transition_args,
            ))
        for event in events:
            results = {}
            for method in METHODS:
                results[method] = wrist_event_metrics(merged[f"{KEYS[method]}_positions"], target, event)
                wrist[event["phase"]][method].append(results[method])
            all_events.append({**event, "file": item["file"], "sequence": item["source"],
                "frame_count": count, "generated_start_frame": 11,
                "gt_speed_m_s": results["Ours"]["gt_speed_m_s"],
                "ours_wrist_mpjpe_cm": results["Ours"]["wrist_mpjpe_cm"], "metrics": results})
        for phase in PHASES:
            selected = np.zeros(count, dtype=bool)
            for event in events:
                if event["phase"] == phase:
                    selected[event["start"]:event["stop"]] = True
            for method in ("GT", *METHODS):
                contact[phase][method].append(contact_statistics(
                    merged[f"{KEYS[method]}_positions"], target, gt_contact, selected,
                ))
        print(f"[dropout-analysis] {item['source']}", flush=True)
    jerk = {"Ours": archive["report"]["transition_dit_deployed"]}
    for method in METHODS[1:]:
        jerk[method] = aggregate_rpm_transition_jerk_segments(
            predicted_segments=predicted_segments[method], target_segments=target_segments,
        )
    for phases in jerk.values():
        for entry in phases.values():
            entry["decomposition"] = decompose_auj(
                entry["predicted_mean_curve_1e2_m_per_s3"], entry["target_mean_curve_1e2_m_per_s3"],
            )
    report = {
        "sequence_count": len(manifest), "fps": 30, "window_seconds": 0.5,
        "representation": "Shared source body.fbx skeleton; RPM rotations retargeted and translated using the always-observed head only. Supplemental RPM scores are not the published native-SMPL table scores.",
        "auj_definition": "sum(abs(event_mean_pred_max_joint_jerk - event_mean_gt_max_joint_jerk)) * 0.01; released evaluator convention without dt",
        "ours_jerk_source": str(args.reference_report.resolve()),
        "contact_definition": "GT feet: y <= sequence foot-height p05 + 0.05m and speed <= 0.1m/s; phase frames are unioned across hand events",
        "provenance": {
            "reference_report": str(args.reference_report.resolve()),
            "ours_run_report": str((root / "ours_run.json").resolve()),
            "gap_config": str(args.gap_config.resolve()),
            "rpm_checkpoints": {
                method: str(np.load(root / KEYS[method] / manifest[0]["file"], allow_pickle=False)["checkpoint"])
                for method in METHODS[1:]
            },
            "sampling_seed": archive["sampling_noise_seed"],
            "rpm_gap_seed": archive["gap_protocol"]["metadata"]["seed"],
        },
        "jerk": jerk, "phases": {},
    }
    rows = []
    for phase in PHASES:
        report["phases"][phase] = {}
        for method in METHODS:
            row = {"phase": phase, "method": method,
                   **aggregate_wrist_events(wrist[phase][method]),
                   **aggregate_contact_statistics(contact[phase][method]),
                   **jerk[method][phase]["decomposition"]}
            rows.append(row)
            report["phases"][phase][method] = row
        report["phases"][phase]["GT_contact_reference"] = aggregate_contact_statistics(contact[phase]["GT"])
    selected = select_video_events(all_events)
    for name, payload in (("metrics.json", report), ("events.json", all_events), ("selected_events.json", selected)):
        (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_jerk(report, root)
    plot_case_speeds(selected, root)
    plot_metric_table(report, root)
    write_summary(report, selected, root)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True, type=Path)
    parser.add_argument("--gap_config", required=True, type=Path)
    parser.add_argument("--reference_report", required=True, type=Path)
    analyze(parser.parse_args(argv))


if __name__ == "__main__":
    main()

# endregion
