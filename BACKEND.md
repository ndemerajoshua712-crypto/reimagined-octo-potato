# DeepTrace backend

The API and worker use PostgreSQL for case/job records. Uploaded media and generated analysis files remain in local file storage configured by `DEEPTRACE_DATA_DIR`; PostgreSQL stores only their paths, checksums, metadata, and JSONB results.

## Local PostgreSQL setup

In PowerShell from the repository root:

```powershell
if (!(Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env and set a unique POSTGRES_PASSWORD. Keep DATABASE_URL's password in sync.
docker compose -f compose.postgres.yaml up -d
python -m pip install -r requirements-backend.txt
```

`requirements-backend.txt` installs the API and PostgreSQL client packages. The analysis worker also needs the full inference environment (including PyTorch, OpenCV, librosa, PySceneDetect, FFmpeg, and the model checkpoints). This repository does not yet provide a pinned installer for all inference dependencies; run the worker with the Python environment already prepared for analysis.

The database port is bound to localhost. The backend reads `.env` automatically. Start the API and the worker in separate terminals, using the same backend/analysis-ready Python environment:

```powershell
python -m uvicorn backend.main:app --reload
python -m backend.worker
```

API docs: `http://127.0.0.1:8000/docs`.

See [README.md](README.md) for the full local setup, frontend startup steps, user workflow, and troubleshooting guide.

## Migrate existing local data

After PostgreSQL is running and the backend requirements are installed, copy the previous SQLite cases and jobs into PostgreSQL:

```powershell
python -m backend.migrate_sqlite
```

The command reads `backend_data/deeptrace.sqlite3` and preserves case/job IDs, results, and options. It is safe to rerun: existing IDs are skipped. Upload files remain at their existing paths under `backend_data/uploads`. The old SQLite file is left in place as a backup.

## API workflow

- `POST /api/v1/cases` — multipart `file` upload; optional `frame_interval` values: `0.25`, `0.5`, `1.0`, or `2.0` seconds.
- `GET /api/v1/cases?limit=50&offset=0` — paginated case summaries.
- `GET /api/v1/cases/{case_id}` — case and job status.
- `GET /api/v1/jobs/{job_id}` — job progress and safe failure details.
- `GET /api/v1/jobs/{job_id}/events` — Server-Sent Events for progress.
- `GET /api/v1/cases/{case_id}/result` — structured result; generated file paths are API URLs.
- `GET /api/v1/cases/{case_id}/artifacts` and `/artifacts/{artifact_key}` — artifact listing and retrieval.
- `GET /api/v1/cases/{case_id}/report` — text report download.
- `GET /health` — API health.
- `GET /ready` — readiness check including PostgreSQL connectivity.

The analysis worker uses PostgreSQL row locks to claim jobs, so multiple workers can safely share the queue. Jobs left running for more than six hours are requeued on worker startup. Local CORS origins cover React development on ports 5173 and 3000; set `DEEPTRACE_CORS_ORIGINS` for another frontend origin.

## Hosting readiness

The code can be hosted, but this local setup is not ready to expose publicly. Before deployment:

1. Add authentication and per-user authorization for cases, results, reports, and artifacts.
2. Move uploads and artifacts to durable object storage (for example S3-compatible storage) shared by API and workers; local disk only works when both run on one persistent host.
3. Use a managed PostgreSQL instance or a backed-up, monitored PostgreSQL service. Apply schema changes through versioned migrations.
4. Run API and worker as separate containers/services. Workers need the media libraries, model checkpoints, FFmpeg, and a GPU-capable host if GPU inference is required.
5. Add a production queue/broker for retries, cancellation, and independent scaling; PostgreSQL polling is suitable for the current small deployment.
6. Set HTTPS, upload and runtime quotas, retention/deletion policy, secret management, monitoring, backups, and restricted CORS origins.

The exact hosting instructions depend on the chosen platform and whether the worker must use a GPU. Model weights are not built into a deployment image by this setup.
