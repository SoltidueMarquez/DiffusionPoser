export type Vec3 = [number, number, number];

export interface MotionTrack {
  track_id: string;
  label: string;
  data_key: string;
  frame_count: number;
  fps: number;
  source_path: string;
  compatible_x277: boolean;
  available: boolean;
  unavailable_reason: string;
  frame_offset: number;
  meta: Record<string, unknown>;
}

export interface MotionAsset {
  asset_id: string;
  kind: "amass" | "x277" | "task" | "repair";
  label: string;
  source_path: string;
  frame_count: number;
  fps: number;
  group: string;
  meta: Record<string, unknown>;
  tracks: MotionTrack[];
}

export interface ComparePreset {
  preset_id: string;
  label: string;
  pane_count: number;
  description: string;
}

export interface LibraryPayload {
  schema_name: string;
  config: Record<string, unknown>;
  index: Record<string, unknown>;
  assets: MotionAsset[];
  presets: ComparePreset[];
}

export interface FrameState {
  asset_id: string;
  track_id: string;
  frame: number;
  source_frame: number;
  time: number;
  root: Vec3;
  root_yaw: number;
  trackers: Vec3[];
  joints: Vec3[];
  contact: number[];
  sensor_missing_labels: boolean[];
  inpaint_target: boolean;
  valid: boolean;
}

export interface PaneSelection {
  asset_id: string;
  track_id: string;
  label: string;
  frame_offset: number;
}

export interface PaneFrames {
  pane_index: number;
  label: string;
  asset: MotionAsset;
  track: MotionTrack;
  start: number;
  count: number;
  frame_count: number;
  fps: number;
  frames: FrameState[];
}

export interface CompareFramesPayload {
  schema_name: string;
  start: number;
  count: number;
  fps: number;
  panes: PaneFrames[];
  warnings: string[];
}

export interface EditProject {
  schema_name: string;
  project_id: string;
  name: string;
  asset_id: string;
  track_id: string;
  created_at: string;
  updated_at: string;
  keyframes: Record<string, Array<{ frame: number; position: Vec3 }>>;
}

export interface ExportResult {
  schema_name: string;
  created_at: string;
  export_dir: string;
  split: string;
  manifest_path: string;
  task_count: number;
  tasks: Array<{ task_id: string; target_frame: number; task_path: string }>;
  project_id: string;
}

export interface MeshPayload {
  available: boolean;
  reason?: string;
  frame?: number;
  vertices?: Vec3[];
  faces?: number[][];
  model_type?: string;
}
