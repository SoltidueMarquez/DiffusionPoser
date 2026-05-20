import type { CompareFramesPayload, EditProject, ExportResult, LibraryPayload, MeshPayload, PaneSelection } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8765";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  library: () => requestJson<LibraryPayload>("/api/library"),
  refreshLibrary: () => requestJson<LibraryPayload>("/api/library/refresh", { method: "POST" }),
  compareFrames: (start: number, count: number, panes: PaneSelection[]) =>
    requestJson<CompareFramesPayload>("/api/compare/frames", {
      method: "POST",
      body: JSON.stringify({ start, count, panes }),
    }),
  mesh: (assetId: string, trackId: string, frame: number) =>
    requestJson<MeshPayload>(`/api/assets/${assetId}/mesh?track_id=${trackId}&frame=${frame}`),
  createEditProject: (assetId: string, trackId: string, name?: string) =>
    requestJson<EditProject>("/api/edit/projects", {
      method: "POST",
      body: JSON.stringify({ asset_id: assetId, track_id: trackId, name }),
    }),
  patchKeyframe: (projectId: string, payload: { target: string; frame: number; position?: number[]; action: "upsert" | "delete" }) =>
    requestJson<EditProject>(`/api/edit/projects/${projectId}/keyframes`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  preview: (projectId: string, start: number, count: number) =>
    requestJson<CompareFramesPayload["panes"][number]>(`/api/edit/projects/${projectId}/preview`, {
      method: "POST",
      body: JSON.stringify({ start, count }),
    }),
  exportProject: (
    projectId: string,
    payload: {
      frame_start: number;
      frame_end: number;
      stride: number;
      split: string;
      missing_sensors: string[];
      export_name?: string;
    },
  ) =>
    requestJson<ExportResult>(`/api/edit/projects/${projectId}/export`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
