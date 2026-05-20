import { Canvas } from "@react-three/fiber";
import { Grid, Html, Line, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import { ChevronDown, ChevronRight, Download, Pause, Play, Plus, RefreshCw, Save, SkipBack, SkipForward, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "./api";
import type { CompareFramesPayload, EditProject, FrameState, LibraryPayload, MotionAsset, MotionTrack, PaneSelection, Vec3 } from "./types";

const TRACKER_NAMES = ["head", "left_wrist", "right_wrist", "waist", "left_foot", "right_foot"];
const TRACKER_PATTERN_CATEGORIES = ["head-present", "hand-present", "foot-present", "upper-body", "lower-body", "mixed-sparse", "full-trackers"];
const EDIT_TARGETS = ["root", ...TRACKER_NAMES];
const ASSET_KINDS = ["source", "task", "result"] as const;
type AssetKind = MotionAsset["kind"];

interface ContextPreset {
  preset_id: string;
  label: string;
  description: string;
  panes: PaneSelection[];
}

function formatVec3(value: Vec3): string {
  return value.map((part) => part.toFixed(3)).join(", ");
}

function vectorForTarget(frame: FrameState | undefined, target: string): Vec3 {
  if (!frame) {
    return [0, 0, 0];
  }
  if (target === "root") {
    return frame.root;
  }
  const index = TRACKER_NAMES.indexOf(target);
  return index >= 0 ? frame.trackers[index] : [0, 0, 0];
}

function trackById(asset: MotionAsset | undefined, trackId: string): MotionTrack | undefined {
  return asset?.tracks.find((track) => track.track_id === trackId);
}

function availableTrack(asset: MotionAsset | undefined, trackId: string): MotionTrack | undefined {
  const track = trackById(asset, trackId);
  return track?.available ? track : undefined;
}

function makePane(asset: MotionAsset, track: MotionTrack, label?: string, frameOffset = 0): PaneSelection {
  return {
    asset_id: asset.asset_id,
    track_id: track.track_id,
    label: label ?? `${asset.label} / ${track.label}`,
    frame_offset: frameOffset,
  };
}

function makePaneFromTrack(asset: MotionAsset, trackId: string, label: string, frameOffset?: number): PaneSelection | undefined {
  const track = availableTrack(asset, trackId);
  if (!track) {
    return undefined;
  }
  const offset = Number(frameOffset ?? track.frame_offset ?? 0);
  return makePane(asset, track, label, offset);
}

function isSensorMissing(frame: FrameState | undefined, index: number): boolean {
  return !Boolean(frame?.sensor_valid?.[index]);
}

function SkeletonView({ frame, jointChains }: { frame: FrameState | undefined; jointChains: number[][] }) {
  if (!frame) {
    return null;
  }
  const segments = jointChains.flatMap((chain) =>
    chain.slice(0, -1).map((joint, index) => [frame.joints[joint], frame.joints[chain[index + 1]]] as [Vec3, Vec3]),
  );
  return (
    <>
      <PerspectiveCamera makeDefault position={[2.8, 2.25, 3.2]} fov={42} />
      <ambientLight intensity={0.72} />
      <directionalLight position={[3, 5, 2]} intensity={1.15} />
      <Grid args={[5, 5]} cellSize={0.25} cellThickness={0.55} sectionSize={1} sectionThickness={1.05} />
      {segments.map((segment, index) => (
        <Line key={index} points={segment} color={frame.inpaint_target ? "#ff5c8a" : "#8fb7ff"} lineWidth={2.2} />
      ))}
      {frame.joints.map((joint, index) => (
        <mesh key={`joint-${index}`} position={joint}>
          <sphereGeometry args={[0.017, 12, 12]} />
          <meshStandardMaterial color="#dce7ff" roughness={0.55} />
        </mesh>
      ))}
      {frame.trackers.map((tracker, index) => {
        const missing = isSensorMissing(frame, index);
        return (
          <group key={`tracker-${index}`} position={tracker}>
            <mesh>
              <sphereGeometry args={[0.052, 18, 18]} />
              <meshStandardMaterial color={missing ? "#fb923c" : "#35d0a7"} roughness={0.35} />
            </mesh>
            {missing ? (
              <>
                <mesh rotation={[Math.PI / 2, 0, 0]}>
                  <torusGeometry args={[0.085, 0.007, 8, 28]} />
                  <meshStandardMaterial color="#fb923c" emissive="#7c2d12" emissiveIntensity={0.35} roughness={0.35} />
                </mesh>
                <Html position={[0, 0.11, 0]} center distanceFactor={7} zIndexRange={[4, 0]}>
                  <div className="sensor-scene-label">{TRACKER_NAMES[index]}</div>
                </Html>
              </>
            ) : null}
          </group>
        );
      })}
      <mesh position={frame.root}>
        <boxGeometry args={[0.085, 0.04, 0.085]} />
        <meshStandardMaterial color="#facc15" roughness={0.4} />
      </mesh>
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} />
    </>
  );
}

function SensorStatusStrip({ frame }: { frame: FrameState | undefined }) {
  return (
    <div className="sensor-status-strip">
      {TRACKER_NAMES.map((name, index) => {
        const missing = isSensorMissing(frame, index);
        return (
          <span key={name} className={`sensor-status ${missing ? "missing" : ""}`} title={`${name}: ${missing ? "missing" : "observed"}`}>
            {name}
          </span>
        );
      })}
    </div>
  );
}

function assetLabel(asset: MotionAsset): string {
  return `${asset.kind.toUpperCase()} · ${asset.label}`;
}

function buildAssetPresets(asset: MotionAsset | undefined): ContextPreset[] {
  if (!asset) {
    return [];
  }
  const presets: ContextPreset[] = [];
  const pushPreset = (preset_id: string, label: string, description: string, panes: Array<PaneSelection | undefined>) => {
    const concretePanes = panes.filter((pane): pane is PaneSelection => Boolean(pane));
    if (concretePanes.length === panes.length && concretePanes.length > 0) {
      presets.push({ preset_id, label, description, panes: concretePanes });
    }
  };

  if (asset.kind === "source") {
    pushPreset("realtime_source", "Realtime Source", "Show realtime_pose_v1 converted source motion.", [makePaneFromTrack(asset, "realtime_source", "Source")]);
    return presets;
  }

  if (asset.kind === "task") {
    pushPreset("task_reference", "Task Reference", "Show realtime_pose_v1 materialized task.", [makePaneFromTrack(asset, "task_reference", "Task Reference")]);
    return presets;
  }

  if (asset.kind === "result") {
    pushPreset(
      "result_features",
      "Result Features",
      "Show available reconstruction feature tracks when joint arrays are present.",
      [makePaneFromTrack(asset, "reference_features", "Reference"), makePaneFromTrack(asset, "reconstructed_features", "Reconstructed")],
    );
    return presets;
  }

  return presets;
}

function defaultPresetForAsset(asset: MotionAsset | undefined): ContextPreset | undefined {
  const presets = buildAssetPresets(asset);
  if (!asset || !presets.length) {
    return undefined;
  }
  const preferredByKind: Record<string, string[]> = {
    source: ["realtime_source"],
    task: ["task_reference"],
    result: ["result_features"],
  };
  const preferredIds = preferredByKind[asset.kind] ?? [];
  return preferredIds.map((presetId) => presets.find((preset) => preset.preset_id === presetId)).find(Boolean) ?? presets[0];
}

function selectDefaultAsset(assets: MotionAsset[]): MotionAsset | undefined {
  const priority = ["source", "task", "result"];
  for (const kind of priority) {
    const asset = assets.find((candidate) => candidate.kind === kind && defaultPresetForAsset(candidate));
    if (asset) {
      return asset;
    }
  }
  return assets.find((asset) => defaultPresetForAsset(asset));
}

export function App() {
  const [library, setLibrary] = useState<LibraryPayload | null>(null);
  const [query, setQuery] = useState("");
  const [selectedAssetId, setSelectedAssetId] = useState("");
  const [selectedPresetId, setSelectedPresetId] = useState("");
  const [panes, setPanes] = useState<PaneSelection[]>([]);
  const [compare, setCompare] = useState<CompareFramesPayload | null>(null);
  const [compareError, setCompareError] = useState("");
  const [isCompareLoading, setIsCompareLoading] = useState(false);
  const [currentOffset, setCurrentOffset] = useState(0);
  const [windowStart, setWindowStart] = useState(0);
  const [windowCount, setWindowCount] = useState(180);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [activePane, setActivePane] = useState(0);
  const [editProject, setEditProject] = useState<EditProject | null>(null);
  const [editTarget, setEditTarget] = useState("root");
  const [positionText, setPositionText] = useState("0, 0, 0");
  const [exportStart, setExportStart] = useState(60);
  const [exportEnd, setExportEnd] = useState(60);
  const [exportStride, setExportStride] = useState(1);
  const [trackerPatterns, setTrackerPatterns] = useState<string[]>(["full-trackers"]);
  const [log, setLog] = useState("Connecting local API...");
  const [collapsedAssetKinds, setCollapsedAssetKinds] = useState<Record<AssetKind, boolean>>({
    source: false,
    task: false,
    result: false,
  });

  const jointChains = useMemo(
    () => [
      [0, 3, 6, 9, 12, 15],
      [9, 13, 16, 18, 20, 22],
      [9, 14, 17, 19, 21, 23],
      [0, 1, 4, 7, 10],
      [0, 2, 5, 8, 11],
    ],
    [],
  );

  const assetsById = useMemo(() => new Map((library?.assets ?? []).map((asset) => [asset.asset_id, asset])), [library]);
  const filteredAssets = useMemo(() => {
    const value = query.trim().toLowerCase();
    const assets = library?.assets ?? [];
    if (!value) {
      return assets;
    }
    return assets.filter((asset) => `${asset.kind} ${asset.label} ${asset.group}`.toLowerCase().includes(value));
  }, [library, query]);
  const groupedAssets = useMemo(() => {
    const groups: Record<AssetKind, MotionAsset[]> = { source: [], task: [], result: [] };
    for (const asset of filteredAssets) {
      groups[asset.kind].push(asset);
    }
    return groups;
  }, [filteredAssets]);
  const activeAssetKindCount = ASSET_KINDS.filter((kind) => !collapsedAssetKinds[kind]).length;

  const selectedAsset = assetsById.get(selectedAssetId);
  const contextPresets = useMemo(() => buildAssetPresets(selectedAsset), [selectedAsset]);
  const currentPreset = contextPresets.find((preset) => preset.preset_id === selectedPresetId);
  const activePaneFrames = compare?.panes[activePane];
  const activeFrame = activePaneFrames?.frames[currentOffset];
  const activeAsset = assetsById.get(panes[activePane]?.asset_id);
  const activeTrack = trackById(activeAsset, panes[activePane]?.track_id ?? "");
  const paneLengths = compare?.panes.map((pane) => pane.frames.length).filter((length) => length > 0) ?? [];
  const maxOffset = paneLengths.length ? Math.max(0, Math.min(...paneLengths) - 1) : 0;

  useEffect(() => {
    let cancelled = false;
    api
      .library()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setLibrary(payload);
        const initialAsset = selectDefaultAsset(payload.assets);
        const initialPreset = defaultPresetForAsset(initialAsset);
        setSelectedAssetId(initialAsset?.asset_id ?? "");
        setSelectedPresetId(initialPreset?.preset_id ?? "");
        setPanes(initialPreset?.panes ?? []);
        setActivePane(0);
        setCurrentOffset(0);
        setCompare(null);
        setCompareError("");
        setIsCompareLoading(false);
        setLog(
          initialAsset && initialPreset
            ? `Loaded ${payload.assets.length} assets. Selected ${assetLabel(initialAsset)} / ${initialPreset.label}`
            : `Loaded ${payload.assets.length} assets. No available presets.`,
        );
      })
      .catch((error) => setLog(`API failed: ${(error as Error).message}`));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!panes.length) {
      setCompare(null);
      setCompareError("");
      setIsCompareLoading(false);
      return;
    }
    setIsCompareLoading(true);
    setCompareError("");
    api
      .compareFrames(windowStart, windowCount, panes)
      .then((payload) => {
        setCompare(payload);
        setIsCompareLoading(false);
        setCurrentOffset((current) => Math.min(current, Math.max(0, payload.panes[0]?.frames.length - 1)));
        const frameCount = payload.panes[0]?.frame_count ?? 0;
        setExportStart(Math.min(60, Math.max(0, frameCount - 1)));
        setExportEnd(Math.min(Math.max(60, frameCount - 1), frameCount - 1));
        setLog(payload.warnings.length ? payload.warnings.join("\n") : `Compare panes: ${payload.panes.length}`);
      })
      .catch((error) => {
        const message = (error as Error).message;
        setCompare(null);
        setIsCompareLoading(false);
        setCompareError(message);
        setLog(`Compare failed: ${message}`);
      });
  }, [panes, windowStart, windowCount]);

  useEffect(() => {
    if (!isPlaying) {
      return;
    }
    const interval = window.setInterval(() => {
      setCurrentOffset((current) => {
        if (current >= maxOffset) {
          return 0;
        }
        return current + 1;
      });
    }, Math.max(16, 1000 / (60 * speed)));
    return () => window.clearInterval(interval);
  }, [isPlaying, maxOffset, speed]);

  useEffect(() => {
    setPositionText(formatVec3(vectorForTarget(activeFrame, editTarget)));
  }, [activeFrame?.source_frame, editTarget]);

  function updatePane(index: number, patch: Partial<PaneSelection>) {
    setPanes((current) => current.map((pane, paneIndex) => (paneIndex === index ? { ...pane, ...patch } : pane)));
    setEditProject(null);
  }

  function applyContextPreset(preset: ContextPreset) {
    setSelectedPresetId(preset.preset_id);
    setPanes(preset.panes);
    setActivePane(0);
    setCurrentOffset(0);
    setCompare(null);
    setCompareError("");
    setIsCompareLoading(false);
    setEditProject(null);
    setLog(`${preset.label}: ${preset.description}`);
  }

  function chooseAsset(asset: MotionAsset) {
    const preset = defaultPresetForAsset(asset);
    setCollapsedAssetKinds((current) => ({ ...current, [asset.kind]: false }));
    setSelectedAssetId(asset.asset_id);
    setSelectedPresetId(preset?.preset_id ?? "");
    setPanes(preset?.panes ?? []);
    setActivePane(0);
    setCurrentOffset(0);
    setCompare(null);
    setCompareError("");
    setIsCompareLoading(false);
    setEditProject(null);
    if (preset) {
      setLog(`${assetLabel(asset)} / ${preset.label}`);
    } else {
      setLog(`${assetLabel(asset)} has no available presets`);
    }
  }

  function toggleAssetKind(kind: AssetKind) {
    setCollapsedAssetKinds((current) => ({ ...current, [kind]: !current[kind] }));
  }

  function showOnlyAssetKind(kind: AssetKind) {
    setCollapsedAssetKinds({
      source: kind !== "source",
      task: kind !== "task",
      result: kind !== "result",
    });
    setLog(`Showing ${kind.toUpperCase()} assets`);
  }

  function expandAllAssetKinds() {
    setCollapsedAssetKinds({ source: false, task: false, result: false });
  }

  async function refreshLibrary() {
    try {
      const payload = await api.refreshLibrary();
      setLibrary(payload);
      const initialAsset = selectDefaultAsset(payload.assets);
      const initialPreset = defaultPresetForAsset(initialAsset);
      setSelectedAssetId(initialAsset?.asset_id ?? "");
      setSelectedPresetId(initialPreset?.preset_id ?? "");
      setPanes(initialPreset?.panes ?? []);
      setActivePane(0);
      setCurrentOffset(0);
      setCompare(null);
      setCompareError("");
      setIsCompareLoading(false);
      setEditProject(null);
      setLog(
        initialAsset && initialPreset
          ? `Refreshed ${payload.assets.length} assets. Selected ${assetLabel(initialAsset)} / ${initialPreset.label}`
          : `Refreshed ${payload.assets.length} assets. No available presets.`,
      );
    } catch (error) {
      setLog(`Refresh failed: ${(error as Error).message}`);
    }
  }

  async function ensureProject(): Promise<EditProject> {
    const pane = panes[activePane];
    if (!pane) {
      throw new Error("No active pane");
    }
    if (editProject && editProject.asset_id === pane.asset_id && editProject.track_id === pane.track_id) {
      return editProject;
    }
    const created = await api.createEditProject(pane.asset_id, pane.track_id);
    setEditProject(created);
    return created;
  }

  async function upsertKeyframe() {
    if (!activeFrame) {
      return;
    }
    try {
      const track = activeTrack;
      if (!track?.compatible_realtime_pose) {
        throw new Error("Active track is not realtime_pose_v1 exportable");
      }
      const position = positionText.split(",").map((part) => Number(part.trim()));
      if (position.length !== 3 || position.some((part) => !Number.isFinite(part))) {
        throw new Error("Position must be x, y, z");
      }
      const project = await ensureProject();
      const updated = await api.patchKeyframe(project.project_id, {
        target: editTarget,
        frame: activeFrame.source_frame,
        position,
        action: "upsert",
      });
      setEditProject(updated);
      await api.preview(updated.project_id, windowStart, windowCount);
      setLog(`Saved ${editTarget} keyframe at ${activeFrame.source_frame}`);
    } catch (error) {
      setLog(`Keyframe failed: ${(error as Error).message}`);
    }
  }

  async function deleteKeyframe() {
    if (!editProject || !activeFrame) {
      return;
    }
    try {
      const updated = await api.patchKeyframe(editProject.project_id, {
        target: editTarget,
        frame: activeFrame.source_frame,
        action: "delete",
      });
      setEditProject(updated);
      setLog(`Deleted ${editTarget} keyframe at ${activeFrame.source_frame}`);
    } catch (error) {
      setLog(`Delete failed: ${(error as Error).message}`);
    }
  }

  async function exportDataset() {
    try {
      const project = await ensureProject();
      const result = await api.exportProject(project.project_id, {
        frame_start: exportStart,
        frame_end: exportEnd,
        stride: exportStride,
        split: "train",
        tracker_patterns: trackerPatterns,
      });
      setLog(`Exported ${result.task_count} ${result.mask_policy} tasks: ${result.export_dir}`);
    } catch (error) {
      setLog(`Export failed: ${(error as Error).message}`);
    }
  }

  const renderedPanes = compare?.panes ?? [];

  return (
    <main className="studio-shell">
      <aside className="library-panel">
        <div className="topbar">
          <button className="icon-button" onClick={refreshLibrary} title="Refresh library">
            <RefreshCw size={17} />
          </button>
          <strong>RealtimePose Studio</strong>
        </div>
        <div className="library-search">
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search assets" />
        </div>
        <div className="asset-kind-switcher">
          <button className={activeAssetKindCount === ASSET_KINDS.length ? "kind-chip active" : "kind-chip"} onClick={expandAllAssetKinds}>
            All
          </button>
          {ASSET_KINDS.map((kind) => (
            <button
              key={kind}
              className={!collapsedAssetKinds[kind] && activeAssetKindCount === 1 ? "kind-chip active" : "kind-chip"}
              onClick={() => showOnlyAssetKind(kind)}
              title={`Show only ${kind.toUpperCase()} assets`}
            >
              {kind.toUpperCase()} <span>{groupedAssets[kind].length}</span>
            </button>
          ))}
        </div>
        <div className="asset-groups">
          {ASSET_KINDS.map((kind) => {
            const isCollapsed = collapsedAssetKinds[kind];
            return (
              <section key={kind} className={isCollapsed ? "asset-group collapsed" : "asset-group"}>
                <div className="asset-group-header">
                  <button
                    className="asset-group-toggle"
                    onClick={() => toggleAssetKind(kind)}
                    title={`${isCollapsed ? "Expand" : "Collapse"} ${kind.toUpperCase()} assets`}
                  >
                    {isCollapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
                    <span>{kind.toUpperCase()}</span>
                    <strong>{groupedAssets[kind].length}</strong>
                  </button>
                  <button className="asset-group-only" onClick={() => showOnlyAssetKind(kind)} title={`Show only ${kind.toUpperCase()} assets`}>
                    Only
                  </button>
                </div>
                {isCollapsed ? null : (
                  <div className="asset-group-body">
                    {groupedAssets[kind].slice(0, 600).map((asset) => (
                      <button
                        key={asset.asset_id}
                        className={`asset-row ${asset.asset_id === selectedAssetId ? "active" : ""}`}
                        onClick={() => chooseAsset(asset)}
                        title={asset.label}
                      >
                        <span>{asset.label}</span>
                        <small>{asset.tracks.map((track) => track.track_id).join(" / ")}</small>
                      </button>
                    ))}
                  </div>
                )}
              </section>
            );
          })}
        </div>
      </aside>

      <section className="compare-workspace">
        <header className="compare-toolbar">
          <div className="selected-asset-summary" title={selectedAsset?.label ?? "Select an asset"}>
            <strong>{selectedAsset ? selectedAsset.kind.toUpperCase() : "No Asset"}</strong>
            <span>{selectedAsset?.label ?? "Select an asset from the library"}</span>
          </div>
          <select
            value={selectedPresetId}
            onChange={(event) => {
              const preset = contextPresets.find((item) => item.preset_id === event.target.value);
              if (preset) {
                applyContextPreset(preset);
              }
            }}
            disabled={!contextPresets.length}
            title={currentPreset?.description ?? "No available presets"}
          >
            {contextPresets.length ? null : <option value="">No presets</option>}
            {contextPresets.map((preset) => (
              <option key={preset.preset_id} value={preset.preset_id}>
                {preset.label}
              </option>
            ))}
          </select>
          <div className="playback">
            <button className="icon-button" onClick={() => setCurrentOffset(0)} title="First frame">
              <SkipBack size={16} />
            </button>
            <button className="icon-button" onClick={() => setIsPlaying((current) => !current)} title="Play">
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <button className="icon-button" onClick={() => setCurrentOffset((current) => Math.min(maxOffset, current + 1))} title="Next frame">
              <SkipForward size={16} />
            </button>
            <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
              <option value={0.25}>0.25x</option>
              <option value={0.5}>0.5x</option>
              <option value={1}>1x</option>
              <option value={2}>2x</option>
            </select>
          </div>
          <div className="range-fields">
            <input type="number" value={windowStart} min={0} onChange={(event) => setWindowStart(Number(event.target.value))} />
            <input type="number" value={windowCount} min={1} max={240} onChange={(event) => setWindowCount(Number(event.target.value))} />
          </div>
        </header>

        <div className={`viewport-grid panes-${panes.length || 1}`}>
          {renderedPanes.length ? (
            renderedPanes.map((pane, index) => {
            const frame = pane.frames[currentOffset];
            return (
              <div
                key={`${pane.asset.asset_id}-${pane.track.track_id}-${index}`}
                className={`viewport-cell ${activePane === index ? "active" : ""}`}
                onClick={() => setActivePane(index)}
                role="button"
                tabIndex={0}
              >
                <div className="viewport-label">
                  <strong>{pane.label}</strong>
                  <span>{frame ? `f ${frame.source_frame} · ${pane.track.label}` : pane.track.unavailable_reason}</span>
                </div>
                <Canvas dpr={[1, 1.6]}>
                  <color attach="background" args={["#10141c"]} />
                  <SkeletonView frame={frame} jointChains={jointChains} />
                </Canvas>
                <SensorStatusStrip frame={frame} />
              </div>
            );
            })
          ) : (
            <div className="empty-viewport">
              <strong>{compareError ? "Compare failed" : isCompareLoading ? "Loading view" : selectedAsset ? "No view available" : "Select an asset"}</strong>
              <span>
                {compareError
                  ? compareError
                  : isCompareLoading
                  ? "Fetching frames for the selected preset."
                  : selectedAsset
                    ? "This asset has no available preset tracks."
                    : "Choose an item from the library on the left."}
              </span>
            </div>
          )}
        </div>
        <footer className="timeline">
          <input
            type="range"
            min={0}
            max={maxOffset}
            value={Math.min(currentOffset, maxOffset)}
            onChange={(event) => setCurrentOffset(Number(event.target.value))}
          />
          <span>{activeFrame ? `Frame ${activeFrame.source_frame}` : "No frame"}</span>
        </footer>
      </section>

      <aside className="control-panel">
        <section className="panel">
          <h2>Pane</h2>
          <label>
            Active
            <select value={panes.length ? String(activePane) : ""} onChange={(event) => setActivePane(Number(event.target.value))} disabled={!panes.length}>
              {panes.length ? (
                panes.map((pane, index) => (
                  <option key={index} value={index}>
                    {index + 1}: {pane.label || pane.track_id}
                  </option>
                ))
              ) : (
                <option value="">No pane</option>
              )}
            </select>
          </label>
          <div className="readonly-field">
            <span>Selected Asset</span>
            <strong title={selectedAsset?.label ?? ""}>{selectedAsset ? assetLabel(selectedAsset) : "None"}</strong>
          </div>
          <div className="readonly-field">
            <span>Preset</span>
            <strong title={currentPreset?.description ?? ""}>{currentPreset?.label ?? "None"}</strong>
          </div>
          <div className="readonly-field">
            <span>Pane Track</span>
            <strong title={activeAsset?.label ?? ""}>{activeTrack ? `${assetLabel(activeAsset!)} / ${activeTrack.label}` : "None"}</strong>
          </div>
          {activeTrack && !activeTrack.available ? <div className="meta-line warning">AMASS Raw: {activeTrack.unavailable_reason}</div> : null}
          <label>
            Offset
            <input
              type="number"
              value={panes[activePane]?.frame_offset ?? 0}
              onChange={(event) => updatePane(activePane, { frame_offset: Number(event.target.value) })}
            />
          </label>
        </section>

        <section className="panel">
          <h2>Edit</h2>
          <label>
            Target
            <select value={editTarget} onChange={(event) => setEditTarget(event.target.value)}>
              {EDIT_TARGETS.map((target) => (
                <option key={target} value={target}>
                  {target}
                </option>
              ))}
            </select>
          </label>
          <label>
            Position
            <input value={positionText} onChange={(event) => setPositionText(event.target.value)} />
          </label>
          <div className="button-row">
            <button onClick={upsertKeyframe} disabled={!activeTrack?.compatible_realtime_pose}>
              <Plus size={16} /> Save
            </button>
            <button onClick={deleteKeyframe} disabled={!editProject}>
              <Trash2 size={16} /> Delete
            </button>
            <button onClick={() => activeFrame && setPositionText(formatVec3(vectorForTarget(activeFrame, editTarget)))}>
              <Save size={16} /> Current
            </button>
          </div>
          <div className="meta-line">{editProject ? editProject.project_id : activeTrack?.compatible_realtime_pose ? "No edit project" : "Track is view-only"}</div>
        </section>

        <section className="panel">
          <h2>Export</h2>
          <div className="grid-form">
            <label>
              Start
              <input type="number" value={exportStart} min={60} onChange={(event) => setExportStart(Number(event.target.value))} />
            </label>
            <label>
              End
              <input type="number" value={exportEnd} min={60} onChange={(event) => setExportEnd(Number(event.target.value))} />
            </label>
            <label>
              Stride
              <input type="number" value={exportStride} min={1} onChange={(event) => setExportStride(Number(event.target.value))} />
            </label>
          </div>
          <div className="sensor-grid">
            {TRACKER_PATTERN_CATEGORIES.map((pattern) => (
              <label key={pattern} className="check-row">
                <input
                  type="checkbox"
                  checked={trackerPatterns.includes(pattern)}
                  onChange={(event) =>
                    setTrackerPatterns((current) => {
                      const next = event.target.checked ? [...current, pattern] : current.filter((item) => item !== pattern);
                      return next.length ? next : ["full-trackers"];
                    })
                  }
                />
                {pattern}
              </label>
            ))}
          </div>
          <div className="meta-line">Training masks are sampled by Dataset; exports here are fixed evaluation/debug patterns.</div>
          <button className="primary" onClick={exportDataset} disabled={!activeTrack?.compatible_realtime_pose}>
            <Download size={16} /> Export Tasks
          </button>
        </section>

        <section className="panel">
          <h2>Status</h2>
          <pre className="log">{log}</pre>
        </section>
      </aside>
    </main>
  );
}
