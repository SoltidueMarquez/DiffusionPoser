from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from data_loaders.realtime_pose_contract import load_source_metadata, validate_realtime_source_contract
from data_loaders.realtime_pose_kinematics import JOINT_INDEX, joint_center_speed
from data_loaders.sensor_masking import (
    DEFAULT_REALTIME_POSE_SCHEMA_NAME,
    REALTIME_POSE_SCHEMA_NAMES,
    STATIONARY_JOINT_INDICES,
    STATIONARY_JOINT_NAMES,
    get_schema_spec,
)


COLOR_CYCLE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
)
SVG_WIDTH = 1180
PANEL_HEIGHT = 210
PANEL_GAP = 36
MARGIN_LEFT = 74
MARGIN_RIGHT = 30
MARGIN_TOP = 34
MARGIN_BOTTOM = 34


def build_report_for_sources(
    source_root: str | Path,
    output_dir: str | Path,
    max_sources: int = 20,
    seed: int = 0,
    fps: float | None = None,
    schema_name: str = DEFAULT_REALTIME_POSE_SCHEMA_NAME,
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
        diagnostics = load_source_diagnostics(
            source_path=source_path,
            source_root=source_root,
            schema_name=schema.name,
            fps_override=fps,
        )
        detail_name = detail_file_name(index=index, relative_path=diagnostics["relative_path"])
        detail_path = detail_dir / detail_name
        detail_path.write_text(render_detail_html(diagnostics), encoding="utf-8", newline="\n")

        summary = dict(diagnostics["summary"])
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


def load_source_diagnostics(
    source_path: Path,
    source_root: Path,
    schema_name: str,
    fps_override: float | None,
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
    stationary_joints = joints_world[:, np.asarray(STATIONARY_JOINT_INDICES, dtype=np.int64)]
    stationary_speed = joint_center_speed(stationary_joints, fps=fps).astype(np.float32)
    root_speed = root_center_speed(root_pos_world=root_pos_world, fps=fps)
    left_foot_height = joints_world[:, JOINT_INDEX["left_foot"], 1].astype(np.float32)
    right_foot_height = joints_world[:, JOINT_INDEX["right_foot"], 1].astype(np.float32)

    relative_path = safe_relative_path(source_path=source_path, source_root=source_root)
    frames = np.arange(stationary_prob.shape[0], dtype=np.float32)
    series = {
        "stationary_prob": [
            make_series(name=joint_name, values=stationary_prob[:, index], color=COLOR_CYCLE[index])
            for index, joint_name in enumerate(STATIONARY_JOINT_NAMES)
        ],
        "speed": [
            make_series(name=f"{joint_name} speed", values=stationary_speed[:, index], color=COLOR_CYCLE[index])
            for index, joint_name in enumerate(STATIONARY_JOINT_NAMES)
        ]
        + [make_series(name="root speed", values=root_speed, color=COLOR_CYCLE[5])],
        "height": [
            make_series(name="pelvis height", values=pelvis_height, color=COLOR_CYCLE[0]),
            make_series(name="left_foot height", values=left_foot_height, color=COLOR_CYCLE[1]),
            make_series(name="right_foot height", values=right_foot_height, color=COLOR_CYCLE[2]),
        ],
    }
    summary = {
        "source_path": str(source_path),
        "relative_path": relative_path,
        "frame_count": int(stationary_prob.shape[0]),
        "fps": fps,
        "stationary_prob_mean": round_list(np.mean(stationary_prob, axis=0)),
        "stationary_prob_p05": round_list(np.percentile(stationary_prob, 5, axis=0)),
        "stationary_prob_p95": round_list(np.percentile(stationary_prob, 95, axis=0)),
        "joint_speed_mps_mean": round_list(np.mean(stationary_speed, axis=0)),
        "joint_speed_mps_p95": round_list(np.percentile(stationary_speed, 95, axis=0)),
        "root_speed_mps_p95": round_float(np.percentile(root_speed, 95)),
        "pelvis_height_min": round_float(np.min(pelvis_height)),
        "pelvis_height_max": round_float(np.max(pelvis_height)),
        "left_foot_height_min": round_float(np.min(left_foot_height)),
        "left_foot_height_max": round_float(np.max(left_foot_height)),
        "right_foot_height_min": round_float(np.min(right_foot_height)),
        "right_foot_height_max": round_float(np.max(right_foot_height)),
    }
    return {
        "relative_path": relative_path,
        "frames": frames,
        "metadata": metadata,
        "series": series,
        "summary": summary,
    }


def root_center_speed(root_pos_world: np.ndarray, fps: float) -> np.ndarray:
    root = np.asarray(root_pos_world, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"root_pos_world should be [T,3], got {root.shape}")
    speed = np.zeros((root.shape[0],), dtype=np.float64)
    if root.shape[0] > 1:
        speed[1:] = np.linalg.norm(root[1:] - root[:-1], axis=-1) * float(fps)
        speed[0] = speed[1]
    return speed.astype(np.float32)


def make_series(name: str, values: np.ndarray, color: str) -> dict[str, Any]:
    return {
        "name": str(name),
        "values": np.asarray(values, dtype=np.float32),
        "color": str(color),
    }


def render_detail_html(diagnostics: dict[str, Any]) -> str:
    title = diagnostics["relative_path"]
    summary = diagnostics["summary"]
    frames = diagnostics["frames"]
    svg = render_timeline_svg(
        frames=frames,
        panels=[
            ("stationary_prob_5", diagnostics["series"]["stationary_prob"], 0.0, 1.0, "probability"),
            ("joint center speed + root speed", diagnostics["series"]["speed"], 0.0, None, "m/s"),
            ("pelvis height + foot height", diagnostics["series"]["height"], None, None, "m"),
        ],
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)} stationary5 source label diagnostics</title>
  <style>{base_css()}</style>
</head>
<body>
  <main>
    <a href="../index.html">Back to index</a>
    <h1>{escape(title)}</h1>
    <p class="muted">frame_count={summary["frame_count"]}, fps={summary["fps"]:.3f}</p>
    {svg}
    <h2>Summary</h2>
    <pre>{escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
  </main>
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
            f"<td>{source['root_speed_mps_p95']:.4f}</td>"
            f"<td>{', '.join(f'{value:.3f}' for value in source['stationary_prob_mean'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>stationary5 source label diagnostics</title>
  <style>{base_css()}</style>
</head>
<body>
  <main>
    <h1>stationary5 source label diagnostics</h1>
    <p class="muted">source_root={escape(summary_payload["source_root"])}</p>
    <p class="muted">selected={summary_payload["selected_source_count"]} / total={summary_payload["total_source_count"]}</p>
    <table>
      <thead>
        <tr><th>source</th><th>frames</th><th>root speed p95</th><th>stationary prob mean</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def render_timeline_svg(
    frames: np.ndarray,
    panels: list[tuple[str, list[dict[str, Any]], float | None, float | None, str]],
) -> str:
    width = SVG_WIDTH
    total_height = MARGIN_TOP + MARGIN_BOTTOM + len(panels) * PANEL_HEIGHT + (len(panels) - 1) * PANEL_GAP
    parts = [
        f'<svg viewBox="0 0 {width} {total_height}" width="100%" role="img" aria-label="stationary label timeline">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
    ]
    for panel_index, (title, series_list, y_min, y_max, unit) in enumerate(panels):
        panel_y = MARGIN_TOP + panel_index * (PANEL_HEIGHT + PANEL_GAP)
        parts.append(render_panel(frames, series_list, title, panel_y, y_min, y_max, unit))
    parts.append("</svg>")
    return "\n".join(parts)


def render_panel(
    frames: np.ndarray,
    series_list: list[dict[str, Any]],
    title: str,
    panel_y: int,
    y_min: float | None,
    y_max: float | None,
    unit: str,
) -> str:
    plot_x = MARGIN_LEFT
    plot_y = panel_y + 28
    plot_w = SVG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = PANEL_HEIGHT - 62
    values = np.concatenate([np.asarray(series["values"], dtype=np.float32).reshape(-1) for series in series_list])
    actual_min = float(np.nanmin(values)) if values.size else 0.0
    actual_max = float(np.nanmax(values)) if values.size else 1.0
    lo = actual_min if y_min is None else float(y_min)
    hi = actual_max if y_max is None else float(y_max)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if abs(hi - lo) < 1e-8:
        hi = lo + 1.0

    parts = [
        f'<text x="{plot_x}" y="{panel_y + 15}" class="panel-title">{escape(title)} ({escape(unit)})</text>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#d0d0d0"/>',
        f'<text x="{plot_x - 8}" y="{plot_y + 4}" class="axis" text-anchor="end">{hi:.3g}</text>',
        f'<text x="{plot_x - 8}" y="{plot_y + plot_h}" class="axis" text-anchor="end">{lo:.3g}</text>',
    ]
    parts.extend(render_x_ticks(frames=frames, plot_x=plot_x, plot_y=plot_y, plot_w=plot_w, plot_h=plot_h))
    legend_x = plot_x + 8
    legend_y = plot_y + plot_h + 24
    for index, series in enumerate(series_list):
        polyline = series_polyline(
            frames=frames,
            values=np.asarray(series["values"], dtype=np.float32),
            plot_x=plot_x,
            plot_y=plot_y,
            plot_w=plot_w,
            plot_h=plot_h,
            y_min=lo,
            y_max=hi,
        )
        color = escape(series["color"])
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        lx = legend_x + (index % 4) * 260
        ly = legend_y + (index // 4) * 18
        parts.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 18}" y2="{ly - 4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 24}" y="{ly}" class="legend">{escape(series["name"])}</text>')
    return "\n".join(parts)


def render_x_ticks(frames: np.ndarray, plot_x: int, plot_y: int, plot_w: int, plot_h: int) -> list[str]:
    if frames.size == 0:
        return []
    last_frame = int(frames[-1])
    ticks = np.linspace(0, last_frame, num=min(6, max(2, last_frame + 1)), dtype=np.float32)
    parts = []
    for tick in ticks:
        x = plot_x if last_frame <= 0 else plot_x + float(tick) / float(last_frame) * plot_w
        parts.append(f'<line x1="{x:.2f}" y1="{plot_y + plot_h}" x2="{x:.2f}" y2="{plot_y + plot_h + 5}" stroke="#888"/>')
        parts.append(f'<text x="{x:.2f}" y="{plot_y + plot_h + 19}" class="axis" text-anchor="middle">{int(round(float(tick)))}</text>')
    return parts


def series_polyline(
    frames: np.ndarray,
    values: np.ndarray,
    plot_x: int,
    plot_y: int,
    plot_w: int,
    plot_h: int,
    y_min: float,
    y_max: float,
) -> str:
    if frames.size != values.size:
        raise ValueError(f"series length mismatch: frames={frames.size}, values={values.size}")
    frame_den = max(float(frames[-1]), 1.0)
    y_den = max(float(y_max - y_min), 1e-8)
    points = []
    for frame, value in zip(frames, values):
        x = plot_x + float(frame) / frame_den * plot_w
        y = plot_y + (1.0 - (float(value) - y_min) / y_den) * plot_h
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


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


def base_css() -> str:
    return """
body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f4f5f7; color: #202124; }
main { max-width: 1240px; margin: 0 auto; padding: 24px; background: white; min-height: 100vh; }
a { color: #0b57d0; text-decoration: none; }
h1 { font-size: 22px; margin: 0 0 8px; }
h2 { font-size: 17px; margin-top: 24px; }
.muted { color: #5f6368; font-size: 13px; }
table { width: 100%; border-collapse: collapse; margin-top: 18px; }
th, td { text-align: left; border-bottom: 1px solid #ddd; padding: 8px; font-size: 13px; }
th { background: #f8f9fa; }
pre { background: #f8f9fa; border: 1px solid #ddd; padding: 12px; overflow: auto; font-size: 12px; }
.panel-title { font-size: 14px; font-weight: 600; fill: #202124; }
.axis { font-size: 11px; fill: #5f6368; }
.legend { font-size: 12px; fill: #202124; }
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize stationary_prob_5 quality for generated realtime_pose source NPZ files.")
    parser.add_argument("--source_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--schema", default=DEFAULT_REALTIME_POSE_SCHEMA_NAME, choices=REALTIME_POSE_SCHEMA_NAMES)
    parser.add_argument("--max_sources", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--fps", default=0.0, type=float, help="Override source metadata target_fps when > 0.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    fps = float(args.fps) if float(args.fps) > 0.0 else None
    result = build_report_for_sources(
        source_root=args.source_dir,
        output_dir=args.output_dir,
        max_sources=int(args.max_sources),
        seed=int(args.seed),
        fps=fps,
        schema_name=str(args.schema),
    )
    print(f"[stationary5_source_label_visualization] index: {result['index_path']}")
    print(f"[stationary5_source_label_visualization] summary: {result['summary_path']}")
    print(
        "[stationary5_source_label_visualization] "
        f"selected={result['selected_source_count']} total={result['total_source_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
