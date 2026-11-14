"""场景引导的固定快照调参、计时和结果导出。"""
from __future__ import annotations

from collections import defaultdict
import csv
import html
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

from eval.scene_guidance import GuidanceConfig, sample_scene_guidance
from eval.scene_guidance_setup import create_scene_sampler, guidance_configs, prepare_output_directory, validate_inputs
from export.unity_onnx_models import SAMPLER_INPUTS


INPUT_SHAPES = {
    "motion_context": (1, 10, 144), "predictor_pose_horizon": (1, 11, 144),
    "current_tracker_raw": (1, 6, 10), "ik_residual": (1, 24, 6),
    "ik_gap": (1, 24), "ik_confidence": (1, 24), "denoise_strength": (1, 24),
    "constraint_type": (1, 24), "noise": (1, 144),
    "armrest_geometry": (1, 2, 15), "contact_mode": (1, 2, 2),
}
LABELS = {"none": "No correction", "postedit": "PostEdit-style", "lm": "LM adaptation", "pigdm": "PiGDM SDF adaptation"}
COLORS = {"none": "#667085", "postedit": "#2563eb", "lm": "#ea580c", "pigdm": "#16a34a"}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_snapshots(path, device):
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    snapshots = []
    for index, row in enumerate(rows):
        feed = {}
        for name, shape in INPUT_SHAPES.items():
            dtype = torch.long if name in {"constraint_type", "contact_mode"} else torch.float32
            value = torch.as_tensor(row["inputs"][name], dtype=dtype, device=device).reshape(shape)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"第 {index} 帧 {name} 含非有限值")
            feed[name] = value
        boxes = feed["armrest_geometry"]
        axes = boxes[..., 3:12].reshape(1, 2, 3, 3)
        if not bool((boxes[..., 12:15] > 0).all()):
            raise ValueError(f"第 {index} 帧盒体半尺寸必须为正")
        torch.testing.assert_close(axes @ axes.transpose(-1, -2), torch.eye(3, device=device).expand_as(axes), atol=1e-4, rtol=1e-4)
        snapshots.append(feed)
    return snapshots


def parameter_grid(methods):
    configs = [GuidanceConfig()]
    if "postedit" in methods:
        configs.extend(GuidanceConfig("postedit", iterations=n, step_size=h)
                       for n, h in itertools.product((1, 3, 5), (1e-5, 1e-4, 1e-3)))
    if "lm" in methods:
        configs.extend(GuidanceConfig("lm", iterations=n, damping=d)
                       for n, d in itertools.product((1, 3, 5), (1e-3, 1e-2, 1e-1)))
    if "pigdm" in methods:
        configs.extend(GuidanceConfig("pigdm", strength=s, covariance_scale=c)
                       for s, c in itertools.product((.1, 1., 10.), repeat=2))
    return configs


def split_snapshot_indices(count, tune_frames, strategy, max_frames=0):
    """固定保留调参帧；短程验证均匀抽取测试帧，覆盖回放后半段的碰撞。"""
    if not 0 < tune_frames < count:
        raise ValueError("调参帧数必须大于零且小于快照总数。")
    tuning = (np.linspace(0, count - 1, tune_frames, dtype=int).tolist()
              if strategy == "uniform" else list(range(tune_frames)))
    testing = [index for index in range(count) if index not in set(tuning)]
    if max_frames and max_frames < len(testing):
        testing = [testing[index] for index in np.linspace(0, len(testing) - 1, max_frames, dtype=int)]
    return tuning, testing


def seeded_feed(snapshot, seed, frame):
    # 噪声只由种子和原始帧号决定；筛选配置、失败或切分不会移动随机数序列。
    generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_000 + frame)
    noise = torch.randn((1, 144), generator=generator).to(snapshot["noise"].device)
    return dict(snapshot, noise=noise)


def summarize(records):
    groups = defaultdict(list)
    baseline = {(r["frame"], r["seed"], r["repeat"]): r for r in records if r["method"] == "none" and r["status"] == "ok"}
    for row in records:
        if row["status"] == "ok":
            base = baseline.get((row["frame"], row["seed"], row["repeat"]))
            row["extra_sampler_ms"] = row["sampler_ms"] - base["sampler_ms"] if base else None
            row["baseline_penetrating"] = base["penetrating"] if base else None
        groups[row["config"]].append(row)
    summaries = []
    for name, rows in groups.items():
        good = [r for r in rows if r["status"] == "ok"]
        summary = {"config": name, "method": rows[0]["method"], "runs": len(rows), "failures": len(rows) - len(good)}
        if good:
            for key in ("sampler_ms", "guidance_ms", "cache_prepare_ms", "geometry_evaluations", "extra_sampler_ms", "peak_allocated_mb", "extra_peak_mb",
                        "penetration_mean_mm", "clearance_mean_mm", "tracker_error_mm", "pose_displacement_mm"):
                values = [r[key] for r in good if r.get(key) is not None]
                summary[key] = float(np.mean(values)) if values else None
            summary["sampler_p95_ms"] = float(np.percentile([r["sampler_ms"] for r in good], 95))
            summary["penetration_max_mm"] = max(r["penetration_max_mm"] for r in good)
            summary["clearance_max_mm"] = max(r["clearance_max_mm"] for r in good)
            summary["penetrating_frame_percent"] = float(np.mean([r["penetrating"] for r in good]) * 100)
            collided = [r for r in good if r.get("baseline_penetrating")]
            summary["originally_colliding_runs"] = len(collided)
            summary["colliding_sampler_ms"] = float(np.mean([r["sampler_ms"] for r in collided])) if collided else None
            summary["colliding_guidance_ms"] = float(np.mean([r["guidance_ms"] for r in collided])) if collided else None
            summary["colliding_extra_sampler_ms"] = float(np.mean([r["extra_sampler_ms"] for r in collided])) if collided else None
            summary["colliding_penetration_mean_mm"] = float(np.mean([r["penetration_mean_mm"] for r in collided])) if collided else None
        summaries.append(summary)
    return summaries


def select_pareto(configs, summaries):
    """各方法分别在调参集筛选；失败配置不入选，保留全部速度—穿透非劣点。"""
    selected = []
    lookup = {c.name: c for c in configs}
    for row in summaries:
        if row["failures"]:
            continue
        competitors = [r for r in summaries if r["method"] == row["method"] and not r["failures"]]
        dominated = any(r["sampler_ms"] <= row["sampler_ms"] and r["penetration_mean_mm"] <= row["penetration_mean_mm"]
                        and (r["sampler_ms"] < row["sampler_ms"] or r["penetration_mean_mm"] < row["penetration_mean_mm"])
                        for r in competitors)
        if not dominated:
            selected.append(lookup[row["config"]])
    return selected


def run_phase(sampler, scene, snapshots, configs, frames, seeds, repeats, warmup, scale, directory, phase):
    records, poses, baseline_poses = [], {}, {}
    with (directory / f"{phase}_raw.jsonl").open("w", encoding="utf-8") as stream:
        for config in configs:
            print(f"[{phase}] {config.name}: {len(frames)} frames x {len(seeds)} seeds x {repeats}", flush=True)
            for index in range(warmup):
                # 预热覆盖整段回放中的不同接触状态，避免始终使用无穿透的第一帧。
                frame = frames[index * (len(frames) - 1) // max(warmup - 1, 1)]
                try:
                    sample_scene_guidance(sampler, scene, seeded_feed(snapshots[frame], seeds[0], frame), config, scale)
                except (FloatingPointError, torch.linalg.LinAlgError):
                    break
            for seed in seeds:
                for frame in frames:
                    feed = seeded_feed(snapshots[frame], seed, frame)
                    for repeat in range(repeats):
                        row = dict(phase=phase, config=config.name, method=config.method, frame=frame, seed=seed, repeat=repeat)
                        try:
                            pose, timing = sample_scene_guidance(sampler, scene, feed, config, scale, measure=True)
                            if config.method == "none":
                                baseline_poses[(seed, frame)] = pose
                            base = baseline_poses[(seed, frame)]
                            row.update(status="ok", **timing, **scene.metrics(pose, base, feed))
                            if repeat == 0:
                                poses[f"{config.name}__s{seed}__f{frame}"] = pose.cpu().numpy()
                        except (FloatingPointError, torch.linalg.LinAlgError) as error:
                            row.update(status="failed", error=str(error))
                        records.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
            good = [r for r in records if r["config"] == config.name and r["status"] == "ok"]
            print(f"  ok={len(good)}, mean_ms={np.mean([r['sampler_ms'] for r in good]) if good else 'failed'}", flush=True)
    summaries = summarize(records)
    write_json(directory / f"{phase}_summary.json", summaries)
    write_csv(directory / f"{phase}_frames.csv", records)
    write_csv(directory / f"{phase}_summary.csv", summaries)
    np.savez_compressed(directory / f"{phase}_poses.npz", **poses)
    return records, summaries, poses


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(directory, summaries, representatives, poses, records, snapshots, scene, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for method, label in LABELS.items():
        rows = [r for r in summaries if r["method"] == method and "sampler_ms" in r]
        if rows:
            rows.sort(key=lambda r: r["sampler_ms"])
            ax.plot([r["sampler_ms"] for r in rows], [r["penetration_mean_mm"] for r in rows],
                    "o-", color=COLORS[method], label=label, linewidth=1.4, markersize=6)
    ax.set(xlabel="Full sampler latency (ms, synchronized wall time)", ylabel="Mean capsule-box penetration (mm)",
           title="Fixed replay: speed vs. penetration (test split)")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(directory / "speed_penetration.png", dpi=180)
    plt.close(fig)

    # 以无修正穿透最深的三帧展示；方法配置在看测试集之前已由调参集选定。
    baseline = [r for r in records if r["method"] == "none" and r["seed"] == seed and r["repeat"] == 0 and r["status"] == "ok"]
    chosen = sorted(baseline, key=lambda r: r["penetration_max_mm"], reverse=True)[:3]
    if not chosen:
        return
    fig = plt.figure(figsize=(4 * len(representatives), 3.8 * len(chosen)), layout="constrained")
    for row_index, row in enumerate(chosen):
        frame = row["frame"]
        feed = snapshots[frame]
        joints_by_config = {}
        for config in representatives:
            key = f"{config.name}__s{seed}__f{frame}"
            if key in poses:
                pose = torch.as_tensor(poses[key], device=scene.pose_mean.device)
                with torch.no_grad():
                    joints_by_config[config.name] = scene.positions_and_rotations(pose, feed["current_tracker_raw"])[0][0].cpu().numpy()
        if not joints_by_config:
            continue
        all_joints = np.concatenate(list(joints_by_config.values()))
        center = (all_joints.min(0) + all_joints.max(0)) * .5
        extent = max(float(np.ptp(all_joints, axis=0).max()) * .55, .5)
        for column, config in enumerate(representatives):
            ax = fig.add_subplot(len(chosen), len(representatives), row_index * len(representatives) + column + 1, projection="3d")
            points = joints_by_config.get(config.name)
            if points is None:
                ax.set_title(f"{LABELS[config.method]}: failed")
                continue
            for joint, parent in enumerate(scene.parents):
                if parent >= 0:
                    pair = points[[parent, joint]]
                    ax.plot(pair[:, 0], pair[:, 2], pair[:, 1], color=COLORS[config.method], linewidth=2)
            for box in feed["armrest_geometry"][0].cpu().numpy():
                signs = np.array(list(itertools.product((-1, 1), repeat=3)))
                corners = (signs * box[12:15]) @ box[3:12].reshape(3, 3) + box[:3]
                for i in range(8):
                    for j in range(i + 1, 8):
                        if np.count_nonzero(signs[i] != signs[j]) == 1:
                            pair = corners[[i, j]]
                            ax.plot(pair[:, 0], pair[:, 2], pair[:, 1], color="#78716c", alpha=.7, linewidth=.8)
            ax.set(xlim=(center[0]-extent, center[0]+extent), ylim=(center[2]-extent, center[2]+extent),
                   zlim=(center[1]-extent, center[1]+extent), xlabel="X", ylabel="Z", zlabel="Y")
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=15, azim=-60)
            ax.set_title(f"Frame {frame} | {LABELS[config.method]}", fontsize=10)
    fig.savefig(directory / "pose_comparison.png", dpi=150)
    plt.close(fig)


def write_report(directory, metadata, summaries, representatives):
    headers = [("config", "配置"), ("sampler_ms", "Sampler ms"), ("sampler_p95_ms", "P95 ms"),
               ("guidance_ms", "引导 ms"), ("extra_sampler_ms", "较基线增加 ms"),
               ("penetration_mean_mm", "平均穿透 mm"), ("penetration_max_mm", "最大穿透 mm"),
               ("penetrating_frame_percent", "穿透帧 %"), ("tracker_error_mm", "Tracker mm"),
               ("pose_displacement_mm", "姿态改变量 mm"), ("failures", "失败次数")]
    def cell(value):
        return f"{value:.3f}" if isinstance(value, float) else str(value if value is not None else "—")
    table = "<tr>" + "".join(f"<th>{label}</th>" for _, label in headers) + "</tr>"
    for row in summaries:
        table += "<tr>" + "".join(f"<td>{html.escape(cell(row.get(key)))}</td>" for key, _ in headers) + "</tr>"
    assumptions = "".join(f"<li>{html.escape(item)}</li>" for item in metadata["scope"])
    body = f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>场景后验修正对照</title>
<style>body{{font:15px/1.65 system-ui;margin:36px;color:#172033}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border:1px solid #ddd;text-align:right}}th{{background:#f1f5f9}}img{{max-width:1200px;width:100%}}code{{background:#f1f5f9}}</style>
<h1>场景后验修正效率对照</h1><p>{html.escape(metadata['device'])} · FP32 · batch 1 · 5 步 DDIM</p>
<p>保留调参 {metadata['tune_frames']} 帧，测试 {metadata['test_frames']} 帧；本次调参：{metadata['tuning_performed']}；测试种子 {metadata['seeds']}，重复 {metadata['repeats']} 次。</p>
<ul>{assumptions}</ul><table>{table}</table><h2>速度与穿透</h2><img src="speed_penetration.png">
<h2>姿态对照</h2><p>展示配置：{html.escape(', '.join(c.name for c in representatives))}</p><img src="pose_comparison.png">
<p>逐帧：<a href="test_frames.csv">test_frames.csv</a>；汇总：<a href="test_summary.csv">test_summary.csv</a>；
配置与来源：<a href="experiment.json">experiment.json</a>；调参筛选：<a href="selection.json">selection.json</a>。</p></html>"""
    (directory / "report.html").write_text(body, encoding="utf-8")


def run_experiment(args):
    sources = validate_inputs(args)
    directory = prepare_output_directory(args)
    sampler, scene, _, diffusion, weight_kind = create_scene_sampler(args)
    device = scene.pose_mean.device
    snapshots = load_snapshots(args.replay_inputs, device)
    tuning_indices, test_indices = split_snapshot_indices(len(snapshots), args.tune_frames,
                                                          args.split_strategy, args.max_frames)
    configs = guidance_configs(args)
    if args.tune:
        configs = parameter_grid([config.method for config in configs])
    allocated_before_graphs = torch.cuda.memory_allocated(device)
    execution = scene.prepare_execution(snapshots[0]) if any(c.method != "none" for c in configs) else None
    metadata = {"arguments": vars(args), "command": [sys.executable, "-m", "eval.evaluate_scene_guidance", *sys.argv[1:]],
                "protocol": "snapshot", "tuning_performed": args.tune, "short_run": bool(args.max_frames),
                "device": torch.cuda.get_device_name(device), "torch": torch.__version__, "weight_kind": weight_kind,
                "timesteps": list(reversed(diffusion.timestep_map)), "tune_frames": args.tune_frames,
                "test_frames": len(test_indices), "seeds": args.seeds, "repeats": args.repeats,
                "tuning_indices": tuning_indices, "test_indices": test_indices,
                "sources": sources,
                "configs": [c.to_dict() for c in configs],
                "shared_geometry_cache": {
                    "backend": "CUDA graphs for geometry forward/VJP/Jacobian" if execution else "unused",
                    "initialization_ms": execution.initialization_ms if execution else 0.,
                    "graph_allocated_mb": (torch.cuda.memory_allocated(device) - allocated_before_graphs) / 2**20,
                    "resident_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
                    "frame_preparation_included": True,
                    "same_point_forward_reused": True,
                },
                "scope": ["固定快照与缓存 Predictor 输出；Sampler 时间含条件准备、五次 DiT、引导及最终投影，不含 Predictor、读取或绘图。",
                          "仅使用录制的两个扶手 OBB 与当前标定的八个手臂/手掌胶囊；各有效配对等权，评估最终姿态。",
                          "mode 1/2 均表示 SDF 约束存在；不使用旧座面字段、顶面规则、显示插值、时序或高度约束。",
                          "PostEdit/LM 共享干净残差保真目标；ΠGDM 使用局部线性 SDF 似然和 DiT VJP，不是完整 InPose。",
                          "ΠGDM 采样增量按 r_t² Jᵀ(r_t²JJᵀ+γ²I)⁻¹ 计算，r_t²=c(1−α_t)；与 InPose 的噪声除以 r_t² 写法等价。",
                          "沿用逐关节 denoise_strength：PostEdit 梯度预条件、LM 变量缩放、ΠGDM 带噪状态分数预条件。",
                          "默认直接测量固定配置；仅 --tune 在调参集筛选耗时—平均穿透 Pareto 配置，测试不参与筛选。",
                          "三种修正共用帧常量与几何自动微分执行缓存；帧准备计入引导时间，图捕获初始化单独记录。",
                          "同一点复用几何前向，更新姿态后重新求值；不跨步复用 SDF/Jacobian，ΠGDM 保留 DiT VJP。",
                          "所有组在同一共享缓存驻留条件下测量峰值显存；预热在本阶段帧范围内均匀取样。",
                          "原顺序前20帧无碰撞；当前划分策略与固定帧号见 experiment.json。",
                          "穿透均值对有效胶囊—盒体配对取均值；帧阈值为 0.001 mm，安全间隙违例另报。",
                          "额外耗时是同帧同种子同重复的 sampler 时间差；guidance 时间包含几何/Jacobian/VJP，DiT 建图开销计入 sampler。",
                          "峰值显存为 PyTorch allocated；测试帧来自同一短回放，不表示独立场景泛化或连续闭环平滑性。"]}
    write_json(directory / "experiment.json", metadata)
    # 唯一实际模型一致性检查；其余数值检查直接并入调参执行。
    with torch.no_grad():
        reference = sampler(*(snapshots[0][key] for key in SAMPLER_INPUTS))
        actual, _ = sample_scene_guidance(sampler, scene, snapshots[0], GuidanceConfig(), args.measurement_scale)
    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
    write_json(directory / "baseline_check.json", {"max_abs": float((actual-reference).abs().max()), "passed": True})
    selected, representatives = configs, configs
    if args.tune:
        _, tuning, _ = run_phase(sampler, scene, snapshots, configs, tuning_indices,
                                 [args.seeds[0]], 1, args.warmup, args.measurement_scale, directory, "tuning")
        selected = select_pareto(configs, tuning)
        representatives = []
        lookup = {r["config"]: r for r in tuning}
        for method in LABELS:
            candidates = [c for c in selected if c.method == method]
            if candidates:
                representatives.append(min(candidates, key=lambda c: lookup[c.name]["penetration_mean_mm"]))
    write_json(directory / "selection.json", {"selected": [c.to_dict() for c in selected],
               "pose_representatives": [c.name for c in representatives],
               "selection_split": "tuning only" if args.tune else "fixed configuration"})
    print("Selected: " + ", ".join(c.name for c in selected), flush=True)
    records, summaries, poses = run_phase(sampler, scene, snapshots, selected, test_indices,
                                          args.seeds, args.repeats, args.warmup, args.measurement_scale, directory, "test")
    plot_results(directory, summaries, representatives, poses, records, snapshots, scene, args.seeds[0])
    write_report(directory, metadata, summaries, representatives)
    print(f"Report: {(directory / 'report.html').resolve()}", flush=True)
    if any(row["failures"] for row in summaries):
        raise RuntimeError("实验存在失败采样；详情已保存到 test_raw.jsonl。")
    return summaries
