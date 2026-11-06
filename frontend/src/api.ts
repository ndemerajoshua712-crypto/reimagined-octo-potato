import type { AnalysisResult, CaseSummary, JobUpdate } from "./types";

// An empty base uses same-origin routes. Vite proxies these to FastAPI during
// local development; set VITE_API_BASE_URL when frontend and API are hosted
// on separate origins.
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Keep the status message when the API did not return JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function fetchCases(): Promise<CaseSummary[]> {
  const response = await request<{ items: CaseSummary[] }>("/api/v1/cases?limit=100&offset=0");
  return response.items;
}

export function fetchJob(jobId: string): Promise<JobUpdate> {
  return request<JobUpdate>(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
}

export function fetchResult(caseId: string): Promise<AnalysisResult> {
  return request<AnalysisResult>(`/api/v1/cases/${encodeURIComponent(caseId)}/result`);
}

export function apiUrl(path: string): string {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

export async function submitVideo(file: File, frameInterval: number) {
  const body = new FormData();
  body.append("file", file);
  body.append("frame_interval", String(frameInterval));
  return request<{ case_id: string; job_id: string; status: JobUpdate["status"] }>("/api/v1/cases", {
    method: "POST",
    body,
  });
}
