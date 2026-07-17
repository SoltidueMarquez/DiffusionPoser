from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.realtime_pose_contract import load_source_metadata, validate_realtime_source_contract
from data_loaders.realtime_pose_kinematics import JOINT_INDEX, SMPL_JOINT_NAMES, SMPL_PARENTS, joint_center_speed
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAMES,
    STATIONARY_JOINT_INDICES,
    STATIONARY_JOINT_NAMES,
    get_schema_spec,
)


JOINT_COLORS = ("#2b6cb0", "#d64545", "#2f9e44", "#7c3aed", "#f97316")
ROOT_COLOR = "#0f766e"
HEIGHT_COLORS = ("#2b6cb0", "#d64545", "#2f9e44")


def build_motion_review_for_sources(
    source_root: str | Path,
    output_dir: str | Path,
    max_sources: int = 20,
    seed: int = 0,
    fps: float | None = None,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    clip_frames: int = 600,
    clip_start: int | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    schema = get_schema_spec(schema_name)
    source_paths = discover_source_npz_files(source_root)
    selected_paths = select_source_paths(source_paths=source_paths, max_sources=max_sources, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_dir = output_dir / "sources"
    detail_dir.mkdir(parents=True, exist_ok=True)

    source_summaries = []
    for index, source_path in enumerate(selected_paths):
        review = load_source_motion_review(
            source_path=source_path,
            source_root=source_root,
            schema_name=schema.name,
            fps_override=fps,
            clip_frames=clip_frames,
            clip_start=clip_start,
            seed=seed,
        )
        detail_name = detail_file_name(index=index, relative_path=review["relative_path"])
        detail_path = detail_dir / detail_name
        detail_path.write_text(render_detail_html(review), encoding="utf-8", newline="\n")

        summary = dict(review["summary"])
        summary["detail_html"] = str(Path("sources") / detail_name).replace("\\", "/")
        source_summaries.append(summary)

    summary_payload = {
        "schema_name": schema.name,
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "total_source_count": len(source_paths),
        "selected_source_count": len(selected_paths),
        "max_sources": int(max_sources),
        "seed": int(seed),
        "fps_override": None if fps is None else float(fps),
        "clip_frames": int(clip_frames),
        "clip_start": None if clip_start is None else int(clip_start),
        "sources": source_summaries,
    }
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.html"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    index_path.write_text(render_index_html(summary_payload), encoding="utf-8", newline="\n")
    return {
        "summary_path": str(summary_path),
        "index_path": str(index_path),
        "selected_source_count": len(selected_paths),
        "total_source_count": len(source_paths),
    }


def discover_source_npz_files(source_root: Path) -> list[Path]:
    if not source_root.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_root}")
    paths = sorted(path for path in source_root.rglob("*.npz") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"source_dir contains no .npz files: {source_root}")
    return paths


def select_source_paths(source_paths: list[Path], max_sources: int, seed: int) -> list[Path]:
    if int(max_sources) <= 0 or int(max_sources) >= len(source_paths):
        return list(source_paths)
    rng = np.random.default_rng(int(seed))
    indices = sorted(int(index) for index in rng.choice(len(source_paths), size=int(max_sources), replace=False))
    return [source_paths[index] for index in indices]


def load_source_motion_review(
    source_path: Path,
    source_root: Path,
    schema_name: str,
    fps_override: float | None,
    clip_frames: int,
    clip_start: int | None,
    seed: int,
) -> dict[str, Any]:
    schema = get_schema_spec(schema_name)
    with np.load(source_path, allow_pickle=False) as data:
        validate_realtime_source_contract(data, schema=schema, source=str(source_path))
        metadata = load_source_metadata(data, source=str(source_path))
        joints_world = np.asarray(data["joints_world"], dtype=np.float32)
        stationary_prob = np.asarray(data["stationary_prob_5"], dtype=np.float32)
        root_pos_world = np.asarray(data["root_pos_world"], dtype=np.float32)
        pelvis_height = np.asarray(data[schema.pelvis_height_key], dtype=np.float32).reshape(-1)

    fps = float(fps_override) if fps_override is not None else float(metadata.get("target_fps", 60.0))
    frame_count = int(joints_world.shape[0])
    start, stop = choose_clip_range(
        frame_count=frame_count,
        clip_frames=int(clip_frames),
        clip_start=clip_start,
        seed=seed,
        relative_path=safe_relative_path(source_path=source_path, source_root=source_root),
    )
    clip = slice(start, stop)

    stationary_joints = joints_world[:, np.asarray(STATIONARY_JOINT_INDICES, dtype=np.int64)]
    stationary_speed = joint_center_speed(stationary_joints, fps=fps).astype(np.float32)
    root_speed = root_center_speed(root_pos_world=root_pos_world, fps=fps)
    left_foot_height = joints_world[:, JOINT_INDEX["left_foot"], 1].astype(np.float32)
    right_foot_height = joints_world[:, JOINT_INDEX["right_foot"], 1].astype(np.float32)
    relative_path = safe_relative_path(source_path=source_path, source_root=source_root)

    review_data = {
        "relative_path": relative_path,
        "fps": fps,
        "frame_count": frame_count,
        "clip_start_frame": start,
        "clip_frame_count": stop - start,
        "joint_names": list(SMPL_JOINT_NAMES),
        "skeleton_edges": [[int(parent), int(child)] for child, parent in enumerate(SMPL_PARENTS.tolist()) if parent >= 0],
        "stationary_joint_indices": [int(index) for index in STATIONARY_JOINT_INDICES],
        "stationary_joint_names": list(STATIONARY_JOINT_NAMES),
        "stationary_joint_colors": list(JOINT_COLORS),
        "joints_world": array_to_list(joints_world[clip], decimals=5),
        "stationary_prob_5": array_to_list(stationary_prob[clip], decimals=5),
        "joint_center_speed_5": array_to_list(stationary_speed[clip], decimals=5),
        "root_speed": array_to_list(root_speed[clip], decimals=5),
        "pelvis_height": array_to_list(pelvis_height[clip], decimals=5),
        "left_foot_height": array_to_list(left_foot_height[clip], decimals=5),
        "right_foot_height": array_to_list(right_foot_height[clip], decimals=5),
    }
    summary = {
        "source_path": str(source_path),
        "relative_path": relative_path,
        "frame_count": frame_count,
        "fps": fps,
        "clip_start_frame": start,
        "clip_end_frame": stop - 1,
        "clip_frame_count": stop - start,
        "stationary_prob_mean": round_list(np.mean(stationary_prob[clip], axis=0)),
        "stationary_prob_p05": round_list(np.percentile(stationary_prob[clip], 5, axis=0)),
        "stationary_prob_p95": round_list(np.percentile(stationary_prob[clip], 95, axis=0)),
        "joint_speed_mps_mean": round_list(np.mean(stationary_speed[clip], axis=0)),
        "joint_speed_mps_p95": round_list(np.percentile(stationary_speed[clip], 95, axis=0)),
        "root_speed_mps_p95": round_float(np.percentile(root_speed[clip], 95)),
        "pelvis_height_min": round_float(np.min(pelvis_height[clip])),
        "pelvis_height_max": round_float(np.max(pelvis_height[clip])),
        "left_foot_height_min": round_float(np.min(left_foot_height[clip])),
        "left_foot_height_max": round_float(np.max(left_foot_height[clip])),
        "right_foot_height_min": round_float(np.min(right_foot_height[clip])),
        "right_foot_height_max": round_float(np.max(right_foot_height[clip])),
    }
    return {
        "relative_path": relative_path,
        "metadata": metadata,
        "review_data": review_data,
        "summary": summary,
    }


def choose_clip_range(
    frame_count: int,
    clip_frames: int,
    clip_start: int | None,
    seed: int,
    relative_path: str,
) -> tuple[int, int]:
    if frame_count <= 0:
        raise ValueError("source has no frames")
    if clip_frames <= 0 or clip_frames >= frame_count:
        return 0, frame_count
    max_start = frame_count - clip_frames
    if clip_start is not None:
        start = min(max(int(clip_start), 0), max_start)
        return start, start + clip_frames
    digest = hashlib.sha1(f"{seed}:{relative_path}".encode("utf-8")).digest()
    start = int.from_bytes(digest[:4], byteorder="little") % (max_start + 1)
    return start, start + clip_frames


def root_center_speed(root_pos_world: np.ndarray, fps: float) -> np.ndarray:
    root = np.asarray(root_pos_world, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"root_pos_world should be [T,3], got {root.shape}")
    speed = np.zeros((root.shape[0],), dtype=np.float64)
    if root.shape[0] > 1:
        speed[1:] = np.linalg.norm(root[1:] - root[:-1], axis=-1) * float(fps)
        speed[0] = speed[1]
    return speed.astype(np.float32)


def render_detail_html(review: dict[str, Any]) -> str:
    title = review["relative_path"]
    summary = review["summary"]
    review_json = json.dumps(review["review_data"], ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)} stationary5 motion review</title>
  <style>{base_css()}</style>
</head>
<body>
  <main>
    <a href="../index.html">Back to index</a>
    <header>
      <h1>{escape(title)}</h1>
      <p class="muted">frames={summary["frame_count"]}, fps={summary["fps"]:.3f}, clip={summary["clip_start_frame"]}-{summary["clip_end_frame"]}</p>
    </header>
    <section class="viewer">
      <canvas id="motionCanvas" width="980" height="520"></canvas>
      <div class="controls">
        <button id="playButton" type="button">Play</button>
        <input id="frameSlider" type="range" min="0" value="0">
        <span id="frameLabel"></span>
      </div>
      <canvas id="chartCanvas" width="1180" height="660"></canvas>
    </section>
    <section>
      <h2>Summary</h2>
      <pre>{escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
    </section>
  </main>
  <script id="reviewData" type="application/json">{escape_script_json(review_json)}</script>
  <script>{viewer_js()}</script>
</body>
</html>
"""


def render_index_html(summary_payload: dict[str, Any]) -> str:
    rows = []
    for source in summary_payload["sources"]:
        rows.append(
            "<tr>"
            f"<td><a href=\"{escape(source['detail_html'])}\">{escape(source['relative_path'])}</a></td>"
            f"<td>{source['frame_count']}</td>"
            f"<td>{source['clip_start_frame']}-{source['clip_end_frame']}</td>"
            f"<td>{source['root_speed_mps_p95']:.4f}</td>"
            f"<td>{', '.join(f'{value:.3f}' for value in source['stationary_prob_mean'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>stationary5 source motion review</title>
  <style>{base_css()}</style>
</head>
<body>
  <main>
    <h1>stationary5 source motion review</h1>
    <p class="muted">source_root={escape(summary_payload["source_root"])}</p>
    <p class="muted">selected={summary_payload["selected_source_count"]} / total={summary_payload["total_source_count"]}</p>
    <table>
      <thead>
        <tr><th>source</th><th>frames</th><th>clip</th><th>root speed p95</th><th>stationary prob mean</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def viewer_js() -> str:
    return r"""
const data = JSON.parse(document.getElementById("reviewData").textContent);
const motionCanvas = document.getElementById("motionCanvas");
const chartCanvas = document.getElementById("chartCanvas");
const playButton = document.getElementById("playButton");
const frameSlider = document.getElementById("frameSlider");
const frameLabel = document.getElementById("frameLabel");
const motionCtx = motionCanvas.getContext("2d");
const chartCtx = chartCanvas.getContext("2d");
let frame = 0;
let playing = false;
let lastTick = 0;
const frameCount = data.clip_frame_count;
frameSlider.max = Math.max(frameCount - 1, 0);

function resizeCanvas(canvas, ctx) {
  const ratio = window.devicePixelRatio || 1;
  const displayWidth = canvas.clientWidth || canvas.width;
  const displayHeight = canvas.clientHeight || canvas.height;
  if (canvas.width !== Math.round(displayWidth * ratio) || canvas.height !== Math.round(displayHeight * ratio)) {
    canvas.width = Math.round(displayWidth * ratio);
    canvas.height = Math.round(displayHeight * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function projectionBounds() {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const joints of data.joints_world) {
    for (const joint of joints) {
      const x = joint[0] + joint[2] * 0.35;
      const y = joint[1];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
  }
  if (!Number.isFinite(minX) || Math.abs(maxX - minX) < 1e-6) {
    minX = -1;
    maxX = 1;
  }
  if (!Number.isFinite(minY) || Math.abs(maxY - minY) < 1e-6) {
    minY = 0;
    maxY = 2;
  }
  return { minX, maxX, minY, maxY };
}

const bounds = projectionBounds();

function project(joint, width, height) {
  const margin = 42;
  const x2d = joint[0] + joint[2] * 0.35;
  const y2d = joint[1];
  const scaleX = (width - margin * 2) / Math.max(bounds.maxX - bounds.minX, 1e-6);
  const scaleY = (height - margin * 2) / Math.max(bounds.maxY - bounds.minY, 1e-6);
  const scale = Math.min(scaleX, scaleY);
  const centerX = (bounds.minX + bounds.maxX) * 0.5;
  const centerY = (bounds.minY + bounds.maxY) * 0.5;
  return {
    x: width * 0.5 + (x2d - centerX) * scale,
    y: height * 0.5 - (y2d - centerY) * scale,
  };
}

function drawMotion() {
  resizeCanvas(motionCanvas, motionCtx);
  const width = motionCanvas.clientWidth || 980;
  const height = motionCanvas.clientHeight || 520;
  motionCtx.clearRect(0, 0, width, height);
  motionCtx.fillStyle = "#f8fafc";
  motionCtx.fillRect(0, 0, width, height);
  motionCtx.strokeStyle = "#d1d5db";
  motionCtx.lineWidth = 1;
  motionCtx.beginPath();
  motionCtx.moveTo(32, height - 42);
  motionCtx.lineTo(width - 32, height - 42);
  motionCtx.stroke();

  const joints = data.joints_world[frame];
  const points = joints.map((joint) => project(joint, width, height));
  motionCtx.lineCap = "round";
  motionCtx.lineJoin = "round";
  for (const edge of data.skeleton_edges) {
    const a = points[edge[0]];
    const b = points[edge[1]];
    motionCtx.strokeStyle = "#475569";
    motionCtx.lineWidth = 3;
    motionCtx.beginPath();
    motionCtx.moveTo(a.x, a.y);
    motionCtx.lineTo(b.x, b.y);
    motionCtx.stroke();
  }
  for (let i = 0; i < points.length; i += 1) {
    motionCtx.fillStyle = "#0f172a";
    motionCtx.beginPath();
    motionCtx.arc(points[i].x, points[i].y, 3.2, 0, Math.PI * 2);
    motionCtx.fill();
  }
  data.stationary_joint_indices.forEach((jointIndex, colorIndex) => {
    const p = points[jointIndex];
    motionCtx.fillStyle = data.stationary_joint_colors[colorIndex];
    motionCtx.beginPath();
    motionCtx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    motionCtx.fill();
    motionCtx.fillStyle = "#111827";
    motionCtx.font = "12px Segoe UI, Arial, sans-serif";
    motionCtx.fillText(data.stationary_joint_names[colorIndex], p.x + 9, p.y - 9);
  });
  motionCtx.fillStyle = "#111827";
  motionCtx.font = "13px Segoe UI, Arial, sans-serif";
  motionCtx.fillText(`absolute frame ${data.clip_start_frame + frame} / ${data.frame_count - 1}`, 18, 24);
}

function makePanels() {
  const stationary = data.stationary_joint_names.map((name, index) => ({
    name,
    values: data.stationary_prob_5.map((row) => row[index]),
    color: data.stationary_joint_colors[index],
  }));
  const speed = data.stationary_joint_names.map((name, index) => ({
    name: `${name} speed`,
    values: data.joint_center_speed_5.map((row) => row[index]),
    color: data.stationary_joint_colors[index],
  }));
  speed.push({ name: "root speed", values: data.root_speed, color: "#0f766e" });
  return [
    { title: "stationary_prob_5", unit: "probability", yMin: 0, yMax: 1, series: stationary },
    { title: "joint_center_speed_5 + root speed", unit: "m/s", yMin: 0, yMax: null, series: speed },
    {
      title: "pelvis height + foot height",
      unit: "m",
      yMin: null,
      yMax: null,
      series: [
        { name: "pelvis height", values: data.pelvis_height, color: "#2b6cb0" },
        { name: "left_foot height", values: data.left_foot_height, color: "#d64545" },
        { name: "right_foot height", values: data.right_foot_height, color: "#2f9e44" },
      ],
    },
  ];
}

const panels = makePanels();

function panelRange(panel) {
  const values = panel.series.flatMap((series) => series.values);
  let lo = panel.yMin === null ? Math.min(...values) : panel.yMin;
  let hi = panel.yMax === null ? Math.max(...values) : panel.yMax;
  if (!Number.isFinite(lo)) lo = 0;
  if (!Number.isFinite(hi)) hi = 1;
  if (Math.abs(hi - lo) < 1e-8) hi = lo + 1;
  return { lo, hi };
}

function drawCharts() {
  resizeCanvas(chartCanvas, chartCtx);
  const width = chartCanvas.clientWidth || 1180;
  const height = chartCanvas.clientHeight || 660;
  const left = 76;
  const right = 24;
  const top = 34;
  const gap = 42;
  const panelHeight = (height - top - 34 - gap * (panels.length - 1)) / panels.length;
  chartCtx.clearRect(0, 0, width, height);
  chartCtx.fillStyle = "#ffffff";
  chartCtx.fillRect(0, 0, width, height);
  panels.forEach((panel, panelIndex) => {
    const y = top + panelIndex * (panelHeight + gap);
    const plotW = width - left - right;
    const plotH = panelHeight - 30;
    const range = panelRange(panel);
    chartCtx.fillStyle = "#f8fafc";
    chartCtx.fillRect(left, y + 20, plotW, plotH);
    chartCtx.strokeStyle = "#cbd5e1";
    chartCtx.strokeRect(left, y + 20, plotW, plotH);
    chartCtx.fillStyle = "#111827";
    chartCtx.font = "13px Segoe UI, Arial, sans-serif";
    chartCtx.fillText(`${panel.title} (${panel.unit})`, left, y + 13);
    chartCtx.fillStyle = "#64748b";
    chartCtx.font = "11px Segoe UI, Arial, sans-serif";
    chartCtx.fillText(range.hi.toFixed(3), 8, y + 27);
    chartCtx.fillText(range.lo.toFixed(3), 8, y + 20 + plotH);

    for (const series of panel.series) {
      chartCtx.strokeStyle = series.color;
      chartCtx.lineWidth = 2;
      chartCtx.beginPath();
      series.values.forEach((value, valueIndex) => {
        const x = left + (frameCount <= 1 ? 0 : (valueIndex / (frameCount - 1)) * plotW);
        const yy = y + 20 + (1 - (value - range.lo) / (range.hi - range.lo)) * plotH;
        if (valueIndex === 0) {
          chartCtx.moveTo(x, yy);
        } else {
          chartCtx.lineTo(x, yy);
        }
      });
      chartCtx.stroke();
    }

    const cursorX = left + (frameCount <= 1 ? 0 : (frame / (frameCount - 1)) * plotW);
    chartCtx.strokeStyle = "#111827";
    chartCtx.lineWidth = 1;
    chartCtx.beginPath();
    chartCtx.moveTo(cursorX, y + 20);
    chartCtx.lineTo(cursorX, y + 20 + plotH);
    chartCtx.stroke();

    let legendX = left;
    const legendY = y + 20 + plotH + 21;
    for (const series of panel.series) {
      chartCtx.strokeStyle = series.color;
      chartCtx.lineWidth = 3;
      chartCtx.beginPath();
      chartCtx.moveTo(legendX, legendY - 4);
      chartCtx.lineTo(legendX + 18, legendY - 4);
      chartCtx.stroke();
      chartCtx.fillStyle = "#334155";
      chartCtx.font = "11px Segoe UI, Arial, sans-serif";
      chartCtx.fillText(series.name, legendX + 24, legendY);
      legendX += Math.max(120, chartCtx.measureText(series.name).width + 48);
    }
  });
}

function setFrame(nextFrame) {
  frame = Math.max(0, Math.min(frameCount - 1, Math.round(nextFrame)));
  frameSlider.value = String(frame);
  frameLabel.textContent = `frame ${data.clip_start_frame + frame} (${frame + 1}/${frameCount})`;
  drawMotion();
  drawCharts();
}

function tick(timestamp) {
  if (!playing) return;
  const frameMs = 1000 / Math.max(data.fps, 1);
  if (!lastTick || timestamp - lastTick >= frameMs) {
    lastTick = timestamp;
    setFrame(frame + 1 >= frameCount ? 0 : frame + 1);
  }
  window.requestAnimationFrame(tick);
}

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
  lastTick = 0;
  if (playing) window.requestAnimationFrame(tick);
});
frameSlider.addEventListener("input", () => {
  playing = false;
  playButton.textContent = "Play";
  setFrame(Number(frameSlider.value));
});
chartCanvas.addEventListener("click", (event) => {
  const rect = chartCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const width = chartCanvas.clientWidth || 1180;
  const left = 76;
  const right = 24;
  const plotW = width - left - right;
  if (x < left || x > width - right) return;
  setFrame(((x - left) / plotW) * Math.max(frameCount - 1, 0));
});
window.addEventListener("resize", () => setFrame(frame));
setFrame(0);
"""


def array_to_list(values: np.ndarray, decimals: int) -> list[Any]:
    return np.round(np.asarray(values, dtype=np.float32), decimals=decimals).tolist()


def detail_file_name(index: int, relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]
    stem = Path(relative_path).with_suffix("").as_posix().replace("/", "_").replace("\\", "_")
    clean = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in stem)
    return f"{index:04d}_{clean[:80]}_{digest}.html"


def safe_relative_path(source_path: Path, source_root: Path) -> str:
    try:
        return source_path.relative_to(source_root).as_posix()
    except ValueError:
        return source_path.name


def round_list(values: np.ndarray) -> list[float]:
    return [round_float(value) for value in np.asarray(values).reshape(-1)]


def round_float(value: Any) -> float:
    return round(float(value), 6)


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def escape_script_json(value: str) -> str:
    return value.replace("</", "<\\/")


def base_css() -> str:
    return """
body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #eef2f7; color: #111827; }
main { max-width: 1240px; margin: 0 auto; padding: 24px; background: white; min-height: 100vh; }
a { color: #0b57d0; text-decoration: none; }
header { margin: 12px 0 18px; }
h1 { font-size: 22px; margin: 0 0 8px; }
h2 { font-size: 17px; margin-top: 24px; }
.muted { color: #64748b; font-size: 13px; }
.viewer { display: grid; gap: 14px; }
#motionCanvas, #chartCanvas { width: 100%; border: 1px solid #cbd5e1; background: #f8fafc; }
#motionCanvas { height: 520px; }
#chartCanvas { height: 660px; }
.controls { display: grid; grid-template-columns: 92px 1fr 180px; gap: 12px; align-items: center; }
button { height: 34px; border: 1px solid #94a3b8; background: #ffffff; color: #111827; cursor: pointer; }
input[type="range"] { width: 100%; }
table { width: 100%; border-collapse: collapse; margin-top: 18px; }
th, td { text-align: left; border-bottom: 1px solid #ddd; padding: 8px; font-size: 13px; }
th { background: #f8f9fa; }
pre { background: #f8f9fa; border: 1px solid #ddd; padding: 12px; overflow: auto; font-size: 12px; }
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build interactive motion review HTML for stationary5 source NPZ files.")
    parser.add_argument("--source_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES)
    parser.add_argument("--max_sources", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fps", default=0.0, type=float, help="Override source metadata target_fps when > 0.")
    parser.add_argument("--clip_frames", default=600, type=int, help="Frames embedded per detail page; <=0 keeps full source.")
    parser.add_argument("--clip_start", default=-1, type=int, help="Optional absolute source frame for clip start; <0 samples deterministically.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    fps = float(args.fps) if float(args.fps) > 0.0 else None
    clip_start = int(args.clip_start) if int(args.clip_start) >= 0 else None
    result = build_motion_review_for_sources(
        source_root=args.source_dir,
        output_dir=args.output_dir,
        max_sources=int(args.max_sources),
        seed=int(args.seed),
        fps=fps,
        schema_name=str(args.schema),
        clip_frames=int(args.clip_frames),
        clip_start=clip_start,
    )
    print(f"[stationary5_source_motion_review] index: {result['index_path']}")
    print(f"[stationary5_source_motion_review] summary: {result['summary_path']}")
    print(
        "[stationary5_source_motion_review] "
        f"selected={result['selected_source_count']} total={result['total_source_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
