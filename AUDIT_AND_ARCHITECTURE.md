# DeepTrace file audit and React + FastAPI architecture plan

**Audit date:** 2026-11-06  
**Scope:** Static review of the repository root, first-party Python modules, upload and case-history flows, and the vendored DeepfakeBench tree. No tests or models were run.

## Repository audit

### Current shape

- The product is a Streamlit application in `app.py` (about 2,500 lines) with page styling, upload handling, orchestration, session state, charts, and report rendering in one module.
- `analysis_pipeline.py` already exposes `analyze_video_file(...)`, which is the best starting boundary for a backend service. It coordinates video, face, audio, temporal, lip-sync, provenance, scoring, and report modules.
- First-party analysis code is split among `video_engine/`, `audio_engine/`, and root-level scoring/report/calibration/history modules.
- `case_history.py` persists a small summary to SQLite. Generated media and per-case artifacts live in directories at the repository root.
- `DeepfakeBench/` is a substantial vendored training/inference repository. The Xception model is imported by adding its training directory to `sys.path` at runtime.
- There is no root README, dependency manifest, API contract, or deployment configuration. Existing test files are present, but were not executed as part of this audit.

### Findings, in priority order

| Priority | Finding | Evidence and impact |
|---|---|---|
| High | Analysis is synchronous in the web request/UI flow. | Both the main upload handler and batch loop call the full inference pipeline inline. Long media and model startup block interaction; batch analysis has no durable job state or restart recovery. |
| High | Results/artifacts are not consistently isolated by case. | `analysis_pipeline.py` writes under `cases/<hash>/...`, but the main Streamlit path writes frames/faces to top-level directories and audio to a shared `audio/deeptrace_current.wav`. Concurrent requests could overwrite or mix intermediate files. |
| High | Upload size, duration, and resource limits are not enforced in application code. | The UI restricts extensions, but that is not media validation. Decoding is delegated to OpenCV/FFmpeg, which should be isolated and bounded when exposed through an API. |
| High | Model files and runtime requirements are not reproducible from the root. | Model/checkpoint files include multi-hundred-megabyte assets; the root has no dependency lock/manifest or model download/verification instructions. The model paths are hard-coded relative to source layout. |
| Medium | Model initialization repeats for each analysis. | `analyze_frames` creates and loads `DeepfakeDetector` each call; the top-level analysis path creates and loads `AudioDeepfakeModel` each call. Startup and memory costs are paid repeatedly. |
| Medium | The pipeline has duplicated orchestration. | `app.py` repeats most of the work in `analysis_pipeline.py`; changes can produce different outputs and failure handling between single and batch use. |
| Medium | Case history is summary-only and separate from generated artifacts. | SQLite stores filename, hash, timestamp, and summary JSON, but no durable status, full result pointer, artifact manifest, retention policy, or migration strategy. |
| Medium | Errors and progress are presentation-bound. | Progress is emitted by Streamlit UI callbacks; exception handling is broad in batch/lip-sync paths. A worker/API needs structured stages and typed failures. |
| Medium | Inference/runtime dependencies are loaded through global import state. | `audio_model.py` modifies `sys.path` to import `_net`; `deepfake_detector.py` inserts a vendored path. This makes module loading fragile and complicates packaging/workers. |
| Low | Report timestamps are local and not consistently tied to persisted case time. | `forensic_report.py` uses naive local `datetime.now()`, while case history uses UTC. Store UTC timestamps in structured results and format at presentation time. |
| Low | A user-provided filename is interpolated into custom HTML. | `app.py` builds an HTML block using `uploaded_file.name` and renders it with `unsafe_allow_html=True`. Escape the value or render it as text. |

### Preserve these useful pieces

- Keep the analysis functions as a domain/service layer; do not put FastAPI or React concerns into the model modules.
- Keep the evidence output explicit: provenance, sampled frames, per-frame/face findings, audio windows, temporal and A/V findings, score, and report.
- Preserve clear score semantics. The current report labels the combined score as a raw, uncalibrated score rather than a probability.
- Keep large generated media and model weights out of the web application bundle and ordinary source control workflow.

## Target architecture

```text
React browser
  ├─ upload, case list/detail, progress, evidence views, report download
  └─ REST + SSE (or WebSocket) over HTTPS
          │
FastAPI web service
  ├─ validates request metadata and creates a case/job
  ├─ serves case/result metadata and authorized artifact URLs
  └─ publishes job status/events
          │
Queue (Redis for first deployment; replaceable interface)
          │
Analysis worker (Python, CPU/GPU deployment profile)
  ├─ runs the existing analysis service
  ├─ writes artifacts to per-case storage
  └─ persists stage/result/failure updates
          │
PostgreSQL (case/job/result metadata) + object storage (uploads/artifacts)
```

### Backend boundaries

1. **`api/`**: FastAPI routers, request/response schemas, auth/dependency wiring, and HTTP error mapping only.
2. **`domain/`**: case/job state and stable result schemas, independent of HTTP and queue vendor.
3. **`services/analysis/`**: adapt `analyze_video_file` into a service accepting a case-scoped input and artifact directory. Extract the duplicated orchestration from `app.py` and route both single and batch operations through it.
4. **`workers/`**: queue task wrapper, retries only for transient infrastructure errors, progress updates, cleanup, and model lifecycle. Load each model once per worker process/device where safe.
5. **`storage/`**: database repository and artifact storage interfaces. Store original upload and outputs under opaque case IDs; never use the client filename as a path.

### Initial API contract

- `POST /api/v1/cases` — multipart upload plus analysis options; returns `202` with `{case_id, job_id, status}`.
- `GET /api/v1/cases` — paginated case summaries.
- `GET /api/v1/cases/{case_id}` — case metadata, job status, and result availability.
- `GET /api/v1/cases/{case_id}/result` — versioned structured findings and artifact references.
- `GET /api/v1/jobs/{job_id}` — current stage, progress, timestamps, and safe error code/message.
- `GET /api/v1/jobs/{job_id}/events` — SSE stream for queued/running/progress/completed/failed events; polling remains a fallback.
- `GET /api/v1/cases/{case_id}/report` — downloadable report.

Every endpoint that accesses a case must enforce ownership/authorization. Artifact URLs should be short-lived or proxied through the API after authorization.

### Data and job model

- `Case`: opaque ID, owner, original filename for display, SHA-256, size, media metadata, created time, retention state.
- `AnalysisJob`: ID, case ID, pipeline/version, options, status (`queued`, `running`, `succeeded`, `failed`, `cancelled`), stage, progress, attempt, timestamps, safe failure details.
- `AnalysisResult`: case/job ID, schema version, structured detector outputs, score semantics, model identifiers, report reference, artifact manifest.
- `Artifact`: case ID, kind, storage key, MIME type, byte size, checksum, retention metadata.

Use a database transaction plus an outbox or a recoverable enqueue pattern so a committed upload cannot become permanently stranded without a job. Enforce upload byte limits, accepted container/content checks, duration/frame caps, worker timeouts, and per-user concurrency limits. Use generated storage keys and isolate each job's scratch directory.

### React application

- **Upload page:** file selection, supported formats, limits, options, queued confirmation.
- **Case list:** server pagination/filtering and status/verdict summaries.
- **Case detail:** overview, provenance, video/frame timeline, audio windows, sync evidence, and report download.
- **Job status:** subscribe to SSE, reconnect with last event ID, and fall back to polling; do not keep analysis state only in browser memory.
- Use typed API schemas generated from OpenAPI, and keep rendering components separate from API transport/state.

### Migration sequence

1. **Stabilize the domain boundary:** make the single and batch UI paths call the same analysis service; pass explicit case-scoped artifact paths; return JSON-serializable results and structured progress callbacks.
2. **Package inference:** define pinned backend dependencies and model manifests/checksums; make checkpoint/device configuration explicit; remove runtime `sys.path` mutation in favor of packaged imports or an isolated inference package.
3. **Add persistence and worker:** introduce versioned schemas, durable job states, queue processing, per-case artifact storage, limits, and retention behavior.
4. **Expose FastAPI:** implement upload, case, result, job, event, and report endpoints with auth and OpenAPI schemas.
5. **Build React against the API:** migrate views incrementally, preserving the existing evidence/report structure and score caveats.
6. **Operate and cut over:** deploy API and worker separately, add health/readiness checks and structured logs, then remove Streamlit after feature parity and operational verification.

## Immediate next engineering decisions

- Pick the first deployment target and GPU availability; this determines worker image/device setup and whether local disk is acceptable for artifacts.
- Choose authentication/tenant model before exposing uploads beyond a trusted local environment.
- Establish explicit retention and deletion behavior for uploaded media and derived faces/frames/audio.
- Decide which model checkpoints are distributable and where they will be hosted; pin each by version and checksum.
- Define a versioned result schema before frontend work so the UI is not coupled to Python's current internal dictionaries.
