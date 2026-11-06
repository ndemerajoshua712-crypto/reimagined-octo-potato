# DeepTrace

DeepTrace is a local video and audio evidence analysis app. The React interface uploads video to a FastAPI service; a separate Python worker analyzes it and stores case metadata and results in PostgreSQL.

> DeepTrace produces investigative evidence signals, not definitive authenticity judgments. Current scores are not calibrated probabilities. The local app does not include user authentication and should remain restricted to trusted local use.

## What it includes

- Video upload for MP4, MOV, AVI, MKV, and WebM.
- Case history and job status updates.
- Sampled-frame, face, scene, audio, temporal, and audio/video synchronization analysis.
- Graceful handling when a video has no audio track; only available model signals contribute to the combined score, and unavailable evidence is labeled as such.
- Structured results, downloadable text reports, and artifact links.
- React, TypeScript, and Vite frontend; FastAPI API; PostgreSQL metadata database; Python analysis worker.

## Requirements

- Windows with PowerShell, or equivalent tools for the host OS.
- Python environment compatible with the analysis dependencies.
- Node.js and npm for the frontend.
- Docker Desktop with Docker Compose, or a local PostgreSQL server.
- FFmpeg available on `PATH` for audio extraction.
- The model checkpoints used by this repository. The current expected paths are `DeepfakeBench/training/weights/xception_best.pth` and `audio_model/aasist/AASIST-L.pth`.

`requirements-backend.txt` installs the FastAPI and PostgreSQL packages only. It does **not** install the full inference stack (such as PyTorch, OpenCV, librosa, and PySceneDetect). Run the worker from a Python environment that already has those analysis dependencies and the checkpoints above. The repository does not yet have a pinned, one-command installer for the complete inference environment.

Model weight files are excluded from Git and are not downloaded by the setup commands. Make sure you have the required checkpoints from an authorized source before running analysis.

## Run locally

Start each service in its own PowerShell terminal, from the repository root unless a step says otherwise.

### 1. Configure PostgreSQL

Create the local environment file once:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set a unique local `POSTGRES_PASSWORD`, then set the same password in `DATABASE_URL`. Keep the `.env` file private; it contains credentials.

Start the PostgreSQL container:

```powershell
docker compose -f compose.postgres.yaml up -d
```

The database listens on localhost port `5432` by default. Check that it is ready:

```powershell
docker compose -f compose.postgres.yaml ps
```

### 2. Prepare the Python environment

Activate the Python environment that contains the analysis packages and model dependencies. If you need to create an environment for the API packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-backend.txt
```

PowerShell may block activation scripts. In that case, use the environment's Python executable directly instead of changing the system execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt
```

The commands above install the API dependencies; they are not sufficient by themselves to run analysis. The worker also needs the full inference dependencies and model files described in [Requirements](#requirements).

### 3. Start the API

In a separate terminal, from the repository root and using the prepared Python environment:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Check API health and PostgreSQL readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Start the analysis worker

In another terminal, from the repository root, activate the same analysis-ready Python environment and run:

```powershell
python -m backend.worker
```

Leave the worker running while cases are being analyzed. The worker configures Numba's cache in `backend_data/numba_cache` so it does not try to write into the Python package directory.

### 5. Start the React app

In a separate terminal:

```powershell
Set-Location frontend
if (!(Test-Path .env)) { Copy-Item .env.example .env }
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). During local development, Vite proxies `/api` and `/ready` to FastAPI at port `8000`, so the browser uses the same origin for both frontend and API. Set `VITE_API_BASE_URL` in `frontend/.env` only when the frontend and API are hosted on separate origins.

If port `5173` is already in use, a Vite server may already be running. Open the existing address instead of starting a duplicate. The Vite proxy also lets a forwarded frontend tunnel reach the local API through the same origin. For a separately hosted API, set `VITE_API_BASE_URL` and add the frontend origin to `DEEPTRACE_CORS_ORIGINS` in the root `.env`, then restart the API.

## Use DeepTrace

1. Select **Browse files** or drop a supported video into the upload area.
2. Choose a frame-sampling interval. Shorter intervals check more frames and can take longer; longer intervals check fewer frames.
3. Select **Analyze video**. The case appears in the history while the worker processes it.
4. Select a case to view its status or results. Completed cases include source metadata, sampled frames, audio windows, scene information, synchronization findings, and a downloadable report.

Analysis can take several minutes for longer videos, especially during face analysis. The displayed percentage is stage-based and can pause while the current stage processes many frames. You can navigate to another case and return from case history; the job continues in the worker.

## API overview

| Method and path | Purpose |
| --- | --- |
| `POST /api/v1/cases` | Upload a video and queue analysis. Multipart fields: `file`, optional `frame_interval` (`0.25`, `0.5`, `1.0`, or `2.0`). Returns `202` with case and job IDs. |
| `GET /api/v1/config` | Read the public upload limit and accepted file extensions for frontend validation. |
| `GET /api/v1/cases` | List cases. Supports `limit` and `offset`. |
| `GET /api/v1/cases/{case_id}` | Get case metadata and job status. |
| `GET /api/v1/jobs/{job_id}` | Get job status, stage, progress, and safe error details. |
| `GET /api/v1/jobs/{job_id}/events` | Subscribe to job status updates with Server-Sent Events. |
| `GET /api/v1/cases/{case_id}/result` | Get structured findings after successful analysis. |
| `GET /api/v1/cases/{case_id}/artifacts` | List generated artifacts. |
| `GET /api/v1/cases/{case_id}/artifacts/{artifact_key}` | Retrieve a case artifact. |
| `GET /api/v1/cases/{case_id}/report` | Download the text report. |
| `GET /health`, `GET /ready` | Check API health and API/PostgreSQL readiness. |

## Data and configuration

- PostgreSQL stores case/job metadata and result JSON.
- By default, uploaded videos and generated artifacts are stored under `backend_data/uploads` and `backend_data/artifacts`.
- The local PostgreSQL container stores its database in the Docker volume `deeptrace_postgres_data`.
- Root `.env` controls PostgreSQL, `DATABASE_URL`, the data directory, upload-size limit, and allowed frontend origins. `frontend/.env` controls `VITE_API_BASE_URL`.
- `DEEPTRACE_MAX_ANALYSIS_FRAMES` and `DEEPTRACE_MAX_AUDIO_WINDOWS` bound model inference work per case (defaults: 600 frames and 300 audio windows). Long recordings use wider sample intervals and the result shows the effective sampling.
- The default upload limit is 2 GiB. Change `DEEPTRACE_MAX_UPLOAD_BYTES` in the root `.env` to override it.
- Keep model checkpoints, `.env` files, uploaded media, and generated evidence private. They are not intended for source control.

## Optional score calibration

The pipeline can apply a locally fitted isotonic mapping to new cases and show held-out calibration metrics broken down by dataset and manipulation type. No labeled calibration dataset is included, so the calibrated estimate stays disabled until you prepare a representative labeled video manifest, run the evaluation pipeline, and fit the mapping. Each manifest row needs `video_path,label,split`; `dataset` and `manipulation_type` are optional. Labels are `0` for real and `1` for manipulated. Relative video paths are resolved from the manifest file's directory.

```powershell
python -m backend.evaluate_dataset path\to\labeled_manifest.csv --output cases\labeled_predictions.csv
python -m backend.calibrate cases\labeled_predictions.csv
```

Keep near-duplicate source videos and their manipulations in the same split. The evaluator rejects duplicate media across splits and leaves combined scores blank when either modality is unavailable; inspect `analysis_error` and `modalities` before calibration. Use `train` plus held-out `test`/`holdout`/`validation` rows. The calibrator writes `cases/score_calibration.json`; only future analyses use it. See [the system guide](DEEPTRACE_SYSTEM_GUIDE.md) for requirements and interpretation.

If you have an older SQLite case database, migrate it after PostgreSQL is running:

```powershell
python -m backend.migrate_sqlite
```

The migrator reads `backend_data/deeptrace.sqlite3` by default, skips IDs already in PostgreSQL, and leaves the SQLite file in place.

## Troubleshooting

### The frontend says the analysis service is unavailable

Check that the API is running at port `8000` and that PostgreSQL is ready:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready
```

### A job stays at “Face analysis”

That stage includes face detection and model inference for sampled frames. The progress value is not updated for every individual frame, so it can remain unchanged while work continues. Check the worker terminal for activity and use `/api/v1/jobs/{job_id}` to confirm the current job state.

### A port is already in use

Check whether the API or Vite server is already running before starting another copy. Frontend development normally uses `5173`; the API uses `8000`; PostgreSQL uses `5432`.

### The worker reports a missing model or analysis package

Start it with the Python environment that has the full inference dependencies installed. Confirm both checkpoint paths listed under [Requirements](#requirements) exist. `requirements-backend.txt` does not install these model dependencies.

### PostgreSQL does not start or connect

Check `docker compose -f compose.postgres.yaml ps`, confirm port `5432` is available, and ensure the password in `DATABASE_URL` matches `POSTGRES_PASSWORD` in `.env`. Avoid `docker compose down -v` unless you intend to delete the local database volume and its data.

## Production and hosting

The current configuration is for trusted local use. Before public deployment, add authentication and per-user authorization, durable shared object storage, a managed/monitored PostgreSQL service, HTTPS, secret management, backups, resource and retention limits, and a production job queue. Deploy the API and worker separately and provide the worker with the media libraries and model checkpoints. More detail is in [BACKEND.md](BACKEND.md).

## More documentation

- [Backend setup, API, and hosting notes](BACKEND.md)
- [Frontend development notes](FRONTEND.md)
- [Repository audit and architecture plan](AUDIT_AND_ARCHITECTURE.md)
