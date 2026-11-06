import asyncio
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from backend import repository
from backend.config import (
    ALLOWED_EXTENSIONS,
    CORS_ORIGINS,
    MAX_UPLOAD_BYTES,
    ARTIFACT_DIR,
    UPLOAD_DIR,
    ensure_data_dirs,
)
from backend.schemas import (
    ArtifactList,
    CaseAccepted,
    CaseDetail,
    CaseList,
    JobStatus,
)

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="DeepTrace API", version="0.1.0", description="Case and analysis job API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.on_event("startup")
def startup():
    repository.initialize_database()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def readiness():
    try:
        repository.check_database()
    except Exception as error:
        logging.getLogger("deeptrace.api").exception("database readiness check failed")
        raise HTTPException(503, "Database is unavailable.") from error
    return {"status": "ready", "database": "ok"}


@app.post("/api/v1/cases", status_code=202, response_model=CaseAccepted)
async def create_case(file: UploadFile = File(...), frame_interval: float = Form(0.5)):
    filename = Path(file.filename or "evidence").name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported media extension.")
    if frame_interval not in {0.25, 0.5, 1.0, 2.0}:
        raise HTTPException(422, "frame_interval must be one of 0.25, 0.5, 1.0, or 2.0 seconds.")

    ensure_data_dirs()
    case_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
    destination = UPLOAD_DIR / f"{case_id}{extension}"
    digest, size = hashlib.sha256(), 0
    try:
        with destination.open("xb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Upload exceeds the {MAX_UPLOAD_BYTES} byte limit.")
                digest.update(chunk)
                target.write(chunk)
        if size == 0:
            raise HTTPException(400, "Uploaded file is empty.")
        created_at = repository.create_case_and_job(
            case_id,
            job_id,
            filename,
            digest.hexdigest(),
            size,
            destination,
            options={"frame_interval": frame_interval},
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"case_id": case_id, "job_id": job_id, "status": "queued", "created_at": created_at}


@app.get("/api/v1/cases", response_model=CaseList)
def cases(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
    return {"items": repository.list_cases(limit, offset), "limit": limit, "offset": offset}


@app.get("/api/v1/cases/{case_id}", response_model=CaseDetail)
def case_detail(case_id: str):
    case = repository.get_case(case_id)
    if case is None:
        raise HTTPException(404, "Case not found.")
    case.pop("source_path", None)
    case.pop("result_json", None)
    return case


@app.get("/api/v1/cases/{case_id}/result", response_model=Dict[str, Any])
def case_result(case_id: str):
    case = repository.get_case(case_id)
    if case is None:
        raise HTTPException(404, "Case not found.")
    if case["status"] != "succeeded" or not case["result_json"]:
        raise HTTPException(409, "Analysis result is not ready.")
    result_data = case["result_json"]
    result = json.loads(result_data) if isinstance(result_data, str) else result_data
    result.pop("artifacts_root", None)
    artifact_root = _artifact_root(case)
    return JSONResponse(_replace_artifact_paths(result, artifact_root, case_id))


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    job = repository.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    return job


@app.get("/api/v1/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    if repository.get_job(job_id) is None:
        raise HTTPException(404, "Job not found.")

    async def stream():
        previous = None
        sequence = 0
        while not await request.is_disconnected():
            job = repository.get_job(job_id)
            if job is None:
                return
            event = {
                "job_id": job["job_id"],
                "case_id": job["case_id"],
                "status": job["status"],
                "stage": job["stage"],
                "progress": job["progress"],
                "error_code": job["error_code"],
                "error_message": job["error_message"],
            }
            fingerprint = json.dumps(event, sort_keys=True)
            if fingerprint != previous:
                sequence += 1
                yield f"id: {sequence}\nevent: job\ndata: {fingerprint}\n\n"
                previous = fingerprint
            if job["status"] in {"succeeded", "failed"}:
                return
            await asyncio.sleep(1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _artifact_root(case):
    return (ARTIFACT_DIR / case["case_id"] / "cases" / case["sha256"][:16]).resolve()


def _replace_artifact_paths(value, artifact_root, case_id):
    if isinstance(value, dict):
        return {key: _replace_artifact_paths(item, artifact_root, case_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_artifact_paths(item, artifact_root, case_id) for item in value]
    if isinstance(value, str):
        try:
            candidate = Path(value).resolve()
            relative = candidate.relative_to(artifact_root)
            if candidate.is_file():
                key = relative.as_posix()
                return f"/api/v1/cases/{case_id}/artifacts/{key}"
        except (OSError, ValueError):
            pass
    return value


@app.get("/api/v1/cases/{case_id}/artifacts", response_model=ArtifactList)
def case_artifacts(case_id: str):
    case = repository.get_case(case_id)
    if case is None:
        raise HTTPException(404, "Case not found.")
    if case["status"] != "succeeded":
        raise HTTPException(409, "Artifacts are not ready.")
    root = _artifact_root(case)
    if not root.is_dir():
        return {"items": []}
    items = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            key = path.relative_to(root).as_posix()
            items.append({
                "key": key,
                "size_bytes": path.stat().st_size,
                "url": f"/api/v1/cases/{case_id}/artifacts/{key}",
            })
    return {"items": items}


@app.get("/api/v1/cases/{case_id}/artifacts/{artifact_key:path}")
def case_artifact(case_id: str, artifact_key: str):
    case = repository.get_case(case_id)
    if case is None:
        raise HTTPException(404, "Case not found.")
    root = _artifact_root(case)
    candidate = (root / artifact_key).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(404, "Artifact not found.")
    if not candidate.is_file():
        raise HTTPException(404, "Artifact not found.")
    return FileResponse(candidate)


@app.get("/api/v1/cases/{case_id}/report")
def case_report(case_id: str):
    case = repository.get_case(case_id)
    if case is None:
        raise HTTPException(404, "Case not found.")
    if case["status"] != "succeeded" or not case["result_json"]:
        raise HTTPException(409, "Report is not ready.")
    result_data = case["result_json"]
    result = json.loads(result_data) if isinstance(result_data, str) else result_data
    report = result.get("report")
    if not report:
        raise HTTPException(404, "Report is unavailable.")
    return Response(
        content=report,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="deeptrace-{case_id}.txt"'},
    )
