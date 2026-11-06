export type JobState = "queued" | "running" | "succeeded" | "failed";

export interface CaseSummary {
  case_id: string;
  filename: string;
  sha256: string;
  size_bytes: number;
  created_at: string;
  job_id: string;
  status: JobState;
  stage: string;
  progress: number;
}

export interface JobUpdate {
  job_id: string;
  case_id: string;
  status: JobState;
  stage: string;
  progress: number;
  error_code?: string | null;
  error_message?: string | null;
}

export interface FrameArtifact {
  timestamp: number;
  path: string;
}

export interface AudioWindow {
  window: number;
  start_time: number;
  end_time: number;
  real_probability: number;
  spoof_probability: number;
  prediction: string;
  confidence: number;
}

export interface AnalysisResult {
  case_id: string;
  filename: string;
  file_hash: string;
  provenance: Record<string, unknown>;
  scenes: Array<{ scene: number; start: number; end: number }>;
  frames: FrameArtifact[];
  video_results: Array<{
    frame: string;
    timestamp: number;
    face_count: number;
    faces: Array<Record<string, unknown>>;
  }>;
  temporal_result: Record<string, number | unknown[]>;
  audio_windows: {
    audio_duration: number;
    total_windows: number;
    average_spoof_probability: number;
    maximum_spoof_probability: number;
    spoof_window_percentage: number;
    windows: AudioWindow[];
  };
  audio_evidence: Record<string, unknown>;
  sync_result: Record<string, number | string>;
  lip_sync: Record<string, unknown>;
  multimodal_result: {
    primary_score: number;
    verdict: string;
    temporal_status: string;
    av_sync_status: string;
    video_evidence: number;
    audio_evidence: number;
    temporal_consistency: number;
  };
  calibrated_primary_score?: number | null;
  calibration?: {
    method: string;
    source_sha256?: string | null;
    train_rows?: number;
    test_rows?: number;
    raw_brier?: number;
    calibrated_brier?: number;
    calibrated_ece?: number;
    calibrated_auc?: number;
    subgroup_metrics?: Record<string, Record<string, {
      rows: number;
      raw_brier: number;
      calibrated_brier: number;
      calibrated_ece: number;
      raw_auc?: number;
      calibrated_auc?: number;
    }>>;
  } | null;
  animation_result: { detected: boolean; confidence: number; reason: string };
  report: string;
  frame_interval: number;
}
