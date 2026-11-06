import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  Activity,
  AlertCircle,
  ArrowDownToLine,
  ArrowUpRight,
  AudioLines,
  Check,
  ChevronRight,
  Clock3,
  FileVideo2,
  Fingerprint,
  Gauge,
  Image as ImageIcon,
  LoaderCircle,
  Play,
  Plus,
  RefreshCw,
  ScanFace,
  ShieldCheck,
  ShieldQuestion,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";
import { API_BASE, apiUrl, fetchCases, fetchJob, fetchResult, submitVideo } from "./api";
import type { AnalysisResult, CaseSummary, JobUpdate } from "./types";

const INTERVALS = [0.25, 0.5, 1, 2];

function formatDate(value?: string) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, {
    month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit",
  }).format(date);
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function percent(value?: unknown) {
  const score = typeof value === "number" ? value : 0;
  return `${Math.round(score * 100)}%`;
}

function stageName(stage: string) {
  return stage.replace(/_/g, " ").replace(/\b\w/g, (letter: string) => letter.toUpperCase());
}

function caseStatusName(status: CaseSummary["status"]) {
  return ({ queued: "Waiting", running: "Analyzing", succeeded: "Complete", failed: "Failed" })[status];
}

function stageDetails(stage: string) {
  const details: Record<string, string> = {
    queued: "Your video is uploaded and waiting for an analysis worker.",
    provenance: "Checking file metadata and creating its SHA-256 fingerprint.",
    scene_detection: "Finding scene changes and preparing sampled video frames.",
    face_analysis: "Checking faces in sampled frames. This step can take several minutes for longer videos.",
    temporal_analysis: "Comparing visual signals across the sampled frames.",
    audio_analysis: "Extracting and analyzing the audio track.",
    av_analysis: "Checking timing between the audio and video tracks.",
    report_generation: "Organizing the findings into your report.",
  };
  return details[stage] || "Your evidence is being analyzed. You can safely leave this page and return from case history.";
}

function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [job, setJob] = useState<JobUpdate | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [frameInterval, setFrameInterval] = useState(0.5);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");

  const selectedCase = cases.find((item) => item.case_id === selectedId) || null;
  const filteredCases = useMemo(() => cases.filter((item) =>
    item.filename.toLowerCase().includes(search.toLowerCase())
    || item.sha256.toLowerCase().includes(search.toLowerCase())), [cases, search]);

  async function refreshCases(chooseLatest = false) {
    const items = await fetchCases();
    setCases(items);
    if (chooseLatest && items[0]) setSelectedId(items[0].case_id);
    return items;
  }

  useEffect(() => {
    let alive = true;
    fetch(`${API_BASE}/ready`).then((response) => {
      if (!alive) return;
      setApiOnline(response.ok);
      if (!response.ok) throw new Error("Backend is not ready");
      return fetchCases();
    }).then((items) => {
      if (!alive || !items) return;
      setCases(items);
      if (items[0]) setSelectedId(items[0].case_id);
    }).catch(() => {
      if (alive) setApiOnline(false);
    });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    if (!selectedCase) {
      setResult(null);
      setJob(null);
      return;
    }
    setJob({
      job_id: selectedCase.job_id,
      case_id: selectedCase.case_id,
      status: selectedCase.status,
      stage: selectedCase.stage,
      progress: selectedCase.progress,
    });
    if (selectedCase.status === "succeeded") {
      fetchResult(selectedCase.case_id).then(setResult).catch((cause: Error) => setError(cause.message));
    } else {
      setResult(null);
    }
  }, [selectedId, selectedCase?.status]);

  useEffect(() => {
    if (!job?.job_id || job.status === "succeeded" || job.status === "failed") return;
    const source = new EventSource(`${API_BASE}/api/v1/jobs/${encodeURIComponent(job.job_id)}/events`);
    let pollingTimer: number | undefined;
    let stopped = false;
    const update = (event: MessageEvent<string>) => {
      try {
        const next = JSON.parse(event.data) as JobUpdate;
        setJob(next);
        if (next.status === "succeeded" || next.status === "failed") {
          stopped = true;
          source.close();
          if (pollingTimer) window.clearInterval(pollingTimer);
          refreshCases().catch(() => undefined);
        }
      } catch {
        // Ignore a malformed event and let the status endpoint provide the next snapshot.
      }
    };
    const poll = async () => {
      if (stopped) return;
      try {
        const next = await fetchJob(job.job_id);
        setJob(next);
        if (next.status === "succeeded" || next.status === "failed") {
          stopped = true;
          source.close();
          if (pollingTimer) window.clearInterval(pollingTimer);
          refreshCases().catch(() => undefined);
        }
      } catch {
        setApiOnline(false);
      }
    };
    source.addEventListener("job", update as EventListener);
    source.onerror = () => {
      if (!pollingTimer) {
        pollingTimer = window.setInterval(() => void poll(), 2500);
        void poll();
      }
    };
    return () => {
      stopped = true;
      source.close();
      if (pollingTimer) window.clearInterval(pollingTimer);
    };
  }, [job?.job_id]);

  useEffect(() => {
    if (job?.status !== "succeeded" || !selectedId || job.case_id !== selectedId) return;
    fetchResult(selectedId).then(setResult).catch((cause: Error) => setError(cause.message));
  }, [job?.status, job?.case_id, selectedId]);

  function acceptFile(file?: File) {
    if (!file) return;
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (!extension || !["mp4", "mov", "avi", "mkv", "webm"].includes(extension)) {
      setSelectedFile(null);
      if (inputRef.current) inputRef.current.value = "";
      setError("Choose an MP4, MOV, AVI, MKV, or WebM video.");
      return;
    }
    setError("");
    setSelectedFile(file);
  }

  async function startAnalysis() {
    if (!selectedFile || isUploading) return;
    setIsUploading(true);
    setError("");
    try {
      const accepted = await submitVideo(selectedFile, frameInterval);
      setResult(null);
      setJob({
        job_id: accepted.job_id,
        case_id: accepted.case_id,
        status: accepted.status,
        stage: "queued",
        progress: 0,
      });
      setSelectedId(accepted.case_id);
      setSelectedFile(null);
      if (inputRef.current) inputRef.current.value = "";
      await refreshCases();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not upload the video.");
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-icon"><span>◈</span></div>
          <div><div className="brand-name">DEEPTRACE</div><div className="brand-caption">MEDIA FORENSICS</div></div>
        </div>

        <button className="new-case-button" onClick={() => { setSelectedId(null); setSelectedFile(null); setResult(null); setError(""); if (inputRef.current) inputRef.current.value = ""; }}>
          <Plus size={16} strokeWidth={2.5} /> New analysis
        </button>

        <div className="sidebar-section-heading">
          <span>CASE HISTORY</span><span className="count-badge">{cases.length}</span>
        </div>
        <label className="search-field">
          <span className="sr-only">Search case history</span>
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Find an investigation" />
          <span className="search-key">⌕</span>
        </label>
        <div className="case-list">
          {filteredCases.map((item) => (
            <button key={item.case_id} className={`case-row ${selectedId === item.case_id ? "selected" : ""}`} aria-pressed={selectedId === item.case_id} title={`${item.filename} · ${caseStatusName(item.status)}`} onClick={() => { setSelectedId(item.case_id); setError(""); }}>
              <span className={`case-file-icon ${item.status}`}><FileVideo2 size={16} /></span>
              <span className="case-row-copy"><span className="case-row-name">{item.filename}</span><span className="case-row-meta">{formatDate(item.created_at)}</span></span>
              <span className={`case-state-label ${item.status}`}>{caseStatusName(item.status)}</span>
            </button>
          ))}
          {cases.length === 0 && <div className="sidebar-empty">Your investigations will appear here.</div>}
          {cases.length > 0 && filteredCases.length === 0 && <div className="sidebar-empty">No cases match that search.</div>}
        </div>

        <div className="sidebar-footer">
          <div className={`connection-indicator ${apiOnline ? "online" : apiOnline === false ? "offline" : "checking"}`} />
          <span>{apiOnline ? "Analysis service online" : apiOnline === false ? "Service unavailable" : "Connecting to service"}</span>
          <button className="icon-button sidebar-refresh" aria-label="Refresh case list" title="Refresh case list" onClick={() => refreshCases().catch((cause: Error) => setError(cause.message))}><RefreshCw size={14} /></button>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><ChevronRight size={13} /><strong>{selectedCase ? "Investigation" : "New analysis"}</strong></div>
          <div className="topbar-right"><span className="environment-label"><span className="env-dot" /> LOCAL WORKSPACE</span><div className="avatar">DT</div></div>
        </header>

        <div className="content-wrap">
          <section className="page-intro">
            <div className="eyebrow"><span className="eyebrow-line" /> DIGITAL EVIDENCE WORKSPACE</div>
            <div className="intro-row">
              <div><h1>Trace the signal.<br /><span>Find what changed.</span></h1><p>Examine video and audio evidence with a layered, explainable analysis.</p></div>
              <div className="intro-stamp"><ShieldCheck size={18} /><span>CHAIN OF CUSTODY<br /><b>SHA-256 VERIFIED</b></span></div>
            </div>
          </section>

          <section className="upload-panel">
            <div className="section-kicker"><span className="step-index">01</span><span>INGEST EVIDENCE</span><span className="kicker-rule" /></div>
            <div className={`dropzone ${isDragging ? "dragging" : ""} ${selectedFile ? "has-file" : ""}`}
              onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(event) => { event.preventDefault(); setIsDragging(false); acceptFile(event.dataTransfer.files[0]); }}>
              <input ref={inputRef} type="file" aria-label="Choose a video to analyze" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm,.mkv" onChange={(event) => acceptFile(event.target.files?.[0])} />
              <div className="upload-glyph">{selectedFile ? <FileVideo2 size={22} /> : <UploadCloud size={22} />}</div>
              <div className="drop-copy"><strong>{selectedFile ? selectedFile.name : "Drop evidence file here"}</strong><span>{selectedFile ? `${formatBytes(selectedFile.size)} · ${selectedFile.type || "Video file"}` : "or browse your device to select a video"}</span></div>
              {selectedFile ? <button className="clear-file" aria-label="Remove selected file" onClick={(event) => { event.stopPropagation(); setSelectedFile(null); if (inputRef.current) inputRef.current.value = ""; }}><X size={16} /></button> : <button className="browse-button" onClick={() => inputRef.current?.click()}>Browse files</button>}
            </div>
            <div className="upload-controls">
              <div className="interval-control"><span id="sampling-label">FRAME SAMPLING</span><div className="segmented-control" role="group" aria-labelledby="sampling-label">{INTERVALS.map((value) => <button type="button" key={value} className={frameInterval === value ? "active" : ""} aria-pressed={frameInterval === value} title={value >= 1 ? "Faster analysis with fewer sampled frames" : "More detailed sampling; analysis may take longer"} onClick={() => setFrameInterval(value)}>{value}s</button>)}</div></div>
              <div className="format-note"><ShieldQuestion size={15} /> Supported: MP4, MOV, AVI, MKV, WEBM</div>
              <button className="analyze-button" disabled={!selectedFile || isUploading || apiOnline === false} title={apiOnline === false ? "Start the API and worker to analyze a video" : undefined} onClick={startAnalysis}>{isUploading ? <LoaderCircle className="spin" size={16} /> : <Play size={15} fill="currentColor" />}<span>{isUploading ? "Uploading video…" : "Analyze video"}</span><ArrowUpRight size={15} /></button>
            </div>
            <div className="sampling-help" aria-live="polite">{frameInterval >= 1 ? "Faster scan: fewer frames are checked." : "Detailed scan: more frames are checked and analysis may take longer."} Sampling one frame every {frameInterval} seconds.</div>
            {error && <div className="error-banner" role="alert"><AlertCircle size={16} /><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError("")}><X size={14} /></button></div>}
          </section>

          {job && (job.status === "queued" || job.status === "running") && selectedId === job.case_id && (
            <section className="processing-card" aria-live="polite" aria-label="Video analysis status">
              <div className="processing-icon"><Activity size={19} className="pulse" /></div>
              <div className="processing-main"><div className="processing-title-row"><div><div className="processing-label">{job.status === "queued" ? "IN THE QUEUE" : "ANALYSIS IN PROGRESS"}</div><h2>{selectedCase?.filename || "Preparing evidence…"}</h2></div><span className="progress-number">{Math.round(job.progress)}<small>%</small></span></div>
                <div className="progress-track" role="progressbar" aria-label="Analysis progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(job.progress)}><div className="progress-fill" style={{ width: `${Math.max(4, job.progress)}%` }} /></div>
                <div className="processing-stage"><span className="live-dot" />{job.status === "queued" ? "Waiting for an available worker" : stageName(job.stage)}</div>
                <p className="processing-description">{stageDetails(job.status === "queued" ? "queued" : job.stage)}</p>
                <p className="progress-hint">Progress updates by analysis stage, so it may pause while a stage processes the video.</p>
                <div className="processing-meta"><span>Case history keeps this analysis available</span><span>CASE {job.case_id.slice(0, 8).toUpperCase()}</span></div>
              </div>
            </section>
          )}

          {job?.status === "failed" && selectedId === job.case_id && <div className="error-banner job-error" role="alert"><AlertCircle size={17} /><span>{job.error_message || "We couldn’t finish this analysis. Refresh the case list or try uploading the video again."}</span></div>}

          {result ? <ResultView result={result} selectedCase={selectedCase} /> : !job && (
            <section className="empty-analysis">
              <div className="empty-illustration"><div className="empty-ring ring-one" /><div className="empty-ring ring-two" /><div className="empty-core"><ScanFace size={27} /></div><span className="orbit-dot dot-a" /><span className="orbit-dot dot-b" /><span className="orbit-dot dot-c" /></div>
              <div className="empty-caption">READY FOR INVESTIGATION</div><h2>Start with a piece of evidence.</h2><p>Upload a video to inspect its provenance, visual frames, audio track, and synchronization signals.</p>
              <div className="empty-capabilities"><span><Fingerprint size={14} /> File integrity</span><span><ScanFace size={14} /> Visual analysis</span><span><AudioLines size={14} /> Audio analysis</span></div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

function ResultView({ result, selectedCase }: { result: AnalysisResult; selectedCase: CaseSummary | null }) {
  const score = result.multimodal_result.primary_score || 0;
  const evidenceClass = /high|fake|manipulat/i.test(result.multimodal_result.verdict) ? "high" : "low";
  const frames = result.frames.slice(0, 8);
  const duration = Number(result.provenance.duration_seconds || result.sync_result.video_duration || 0);
  const provenance = result.provenance;
  const metadata: Array<[string, string]> = [
    ["CONTAINER", String(provenance.format || "—")],
    ["DURATION", duration ? `${duration.toFixed(1)} sec` : "—"],
    ["VIDEO CODEC", String(provenance.video_codec || "—")],
    ["DIMENSIONS", provenance.width && provenance.height ? `${provenance.width} × ${provenance.height}` : "—"],
    ["AUDIO CODEC", String(provenance.audio_codec || "—")],
    ["FRAME SAMPLE", `${result.frame_interval}s`],
  ];

  return (
    <section className="results-section">
      <div className="section-kicker result-kicker"><span className="step-index">02</span><span>ANALYSIS FINDINGS</span><span className="kicker-rule" />{selectedCase && <span className="completed-date">{formatDate(selectedCase.created_at)}</span>}</div>
      <div className="result-heading"><div><div className="result-filename"><FileVideo2 size={18} /> {result.filename}</div><h2>Evidence overview</h2></div><a className="report-button" href={apiUrl(`/api/v1/cases/${result.case_id}/report`)} download><ArrowDownToLine size={15} /> Export report</a></div>

      <div className="summary-grid">
        <div className="verdict-card"><div className="card-label"><ShieldCheck size={15} /> MULTIMODAL ASSESSMENT</div><div className={`verdict-text ${evidenceClass}`}>{result.multimodal_result.verdict || "Assessment unavailable"}</div><div className="verdict-foot"><span className={`verdict-dot ${evidenceClass}`} /> Combined evidence signals</div></div>
        <div className="score-card"><div className="card-label"><Gauge size={15} /> PROTOTYPE EVIDENCE SCORE</div><div className="score-row"><div className={`score-number ${evidenceClass}`}>{Math.round(score * 100)}<span>%</span></div><div className="score-meter"><div className="meter-ring" style={{ "--score": `${Math.max(0, Math.min(100, score * 100))}%` } as CSSProperties}><div className="meter-inner"><Sparkles size={14} /></div></div></div></div><div className="score-foot">Raw model blend · not a probability</div>
          {typeof result.calibrated_primary_score === "number" && <div className="calibrated-score-note"><div><span>CALIBRATED ESTIMATE</span><strong>{percent(result.calibrated_primary_score)}</strong></div><small>Held-out ({result.calibration?.test_rows ?? "unknown"} labeled cases): Brier {result.calibration?.calibrated_brier?.toFixed(3) ?? "—"} · ECE {result.calibration?.calibrated_ece?.toFixed(3) ?? "—"} · AUC {result.calibration?.calibrated_auc?.toFixed(3) ?? "—"}. Applies only to data similar to that calibration set.</small></div>}
          {result.calibration?.subgroup_metrics && Object.entries(result.calibration.subgroup_metrics).some(([, groups]) => Object.keys(groups).length > 0) && <div className="calibration-breakdown"><span>HELD-OUT DATA BREAKDOWN</span>{Object.entries(result.calibration.subgroup_metrics).flatMap(([dimension, groups]) => Object.entries(groups).map(([group, metrics]) => <small key={`${dimension}-${group}`}>{dimension.replaceAll("_", " ")}: {group} · {metrics.rows} cases · Brier {metrics.calibrated_brier.toFixed(3)}{typeof metrics.calibrated_auc === "number" ? ` · AUC ${metrics.calibrated_auc.toFixed(3)}` : ""}</small>))}</div>}
        </div>
        <div className="signal-card"><div className="card-label"><Activity size={15} /> SIGNAL BREAKDOWN</div><SignalBar label="VIDEO" value={result.multimodal_result.video_evidence} /><SignalBar label="AUDIO" value={result.multimodal_result.audio_evidence} /><SignalBar label="TEMPORAL" value={result.multimodal_result.temporal_consistency} /></div>
      </div>

      <div className="detail-grid">
        <section className="data-card provenance-card"><div className="data-card-heading"><div><div className="card-label"><Fingerprint size={15} /> FILE PROVENANCE</div><h3>Source characteristics</h3></div><span className="verified-tag"><Check size={12} /> HASHED</span></div>
          <div className="metadata-grid">{metadata.map(([label, value]) => <div key={label}><span>{label}</span><strong>{String(value)}</strong></div>)}</div>
          <div className="hash-row"><span>SHA-256</span><code>{result.file_hash}</code><button className="copy-hash" onClick={() => navigator.clipboard?.writeText(result.file_hash)} title="Copy SHA-256"><Fingerprint size={14} /></button></div>
        </section>
        <section className="data-card sync-card"><div className="data-card-heading"><div><div className="card-label"><AudioLines size={15} /> AV SYNCHRONIZATION</div><h3>Track alignment</h3></div><span className="sync-value">{result.sync_result.sync_status || "UNKNOWN"}</span></div>
          <div className="sync-duration"><div><span>VIDEO TRACK</span><strong>{Number(result.sync_result.video_duration || 0).toFixed(2)}<small> sec</small></strong></div><div className="sync-link"><span /></div><div><span>AUDIO TRACK</span><strong>{Number(result.sync_result.audio_duration || 0).toFixed(2)}<small> sec</small></strong></div></div>
          <div className="sync-delta"><Clock3 size={13} /> Duration delta <strong>{Number(result.sync_result.duration_difference || 0).toFixed(3)} sec</strong></div>
          {result.lip_sync.available === true && <div className="lipsync-note">Lip-sync check: <b>{String(result.lip_sync.status || "available")}</b> · score {percent(result.lip_sync.lip_sync_score)}</div>}
        </section>
      </div>

      <section className="data-card frames-card"><div className="data-card-heading"><div><div className="card-label"><ImageIcon size={15} /> VISUAL EVIDENCE</div><h3>Sampled frames <span className="subtle-count">{result.frames.length} total</span></h3></div><span className="sample-rate">Every {result.frame_interval}s</span></div>
        {frames.length ? <div className="frame-grid">{frames.map((frame, index) => {
          const match = result.video_results.find((item) => Math.abs(item.timestamp - frame.timestamp) < 0.01);
          return <a className="frame-tile" key={`${frame.timestamp}-${index}`} href={apiUrl(frame.path)} target="_blank" rel="noreferrer"><img src={apiUrl(frame.path)} alt={`Video frame at ${frame.timestamp.toFixed(2)} seconds`} loading="lazy" /><span className="frame-timestamp">{frame.timestamp.toFixed(2)}s</span><span className="frame-face-count"><ScanFace size={12} /> {match?.face_count ?? 0}</span></a>;
        })}</div> : <div className="no-artifacts"><ImageIcon size={18} /> No sampled frame artifacts were returned.</div>}
      </section>

      <div className="detail-grid lower-grid">
        <section className="data-card audio-card"><div className="data-card-heading"><div><div className="card-label"><AudioLines size={15} /> AUDIO WINDOWS</div><h3>Spoof signal over time</h3></div><span className="audio-summary">{result.audio_windows.total_windows} windows</span></div>
          <div className="audio-legend"><span><i className="legend-real" /> Lower signal</span><span><i className="legend-spoof" /> Higher signal</span></div>
          <div className="audio-bars">{result.audio_windows.windows.map((window, index) => <div key={index} className={`audio-bar ${window.spoof_probability >= 0.5 ? "elevated" : ""}`} title={`${window.start_time.toFixed(1)}–${window.end_time.toFixed(1)}s · ${percent(window.spoof_probability)} spoof signal`}><span style={{ height: `${Math.max(5, window.spoof_probability * 100)}%` }} /></div>)}</div>
          <div className="audio-axis"><span>0:00</span><span>{duration ? `${Math.floor(duration / 2)}s` : "TIME"}</span><span>{duration ? `${duration.toFixed(0)}s` : "END"}</span></div>
          <div className="audio-stats"><div><span>AVERAGE</span><strong>{percent(result.audio_windows.average_spoof_probability)}</strong></div><div><span>PEAK</span><strong>{percent(result.audio_windows.maximum_spoof_probability)}</strong></div><div><span>FLAGGED WINDOWS</span><strong>{percent(result.audio_windows.spoof_window_percentage)}</strong></div></div>
        </section>
        <section className="data-card timeline-card"><div className="data-card-heading"><div><div className="card-label"><Activity size={15} /> SCENE STRUCTURE</div><h3>Detected segments</h3></div><span className="audio-summary">{result.scenes.length} scenes</span></div>
          {result.scenes.length ? <div className="scene-list">{result.scenes.slice(0, 6).map((scene, index) => {
            const width = duration > 0 ? Math.max(3, ((scene.end - scene.start) / duration) * 100) : 0;
            const left = duration > 0 ? (scene.start / duration) * 100 : 0;
            return <div className="scene-row" key={scene.scene}><div className="scene-label"><span>SCENE {String(scene.scene).padStart(2, "0")}</span><b>{scene.start.toFixed(1)}s — {scene.end.toFixed(1)}s</b></div><div className="scene-rail"><span style={{ left: `${left}%`, width: `${width}%` }} className={`scene-segment segment-${index % 4}`} /></div></div>;
          })}</div> : <div className="no-artifacts"><Activity size={18} /> No scene boundaries detected.</div>}
          <div className="temporal-footer"><span>TEMPORAL CONSISTENCY</span><strong>{percent(result.temporal_result.temporal_consistency)}</strong></div>
        </section>
      </div>

      <div className="interpretation-note"><ShieldQuestion size={16} /><p><b>Interpret findings as evidence signals.</b> The combined score is a prototype measure, not a calibrated probability or a definitive determination of authenticity.</p></div>
      <div className="result-footnote"><span>CASE {result.case_id.toUpperCase()}</span><span>{selectedCase ? formatBytes(selectedCase.size_bytes) : ""} · Analysis sampled {result.frames.length} frames</span></div>
    </section>
  );
}

function SignalBar({ label, value }: { label: string; value: unknown }) {
  const score = typeof value === "number" ? Math.max(0, Math.min(1, value)) : 0;
  return <div className="signal-row"><span>{label}</span><div className="signal-track"><i style={{ width: `${score * 100}%` }} /></div><b>{percent(score)}</b></div>;
}

export default App;
