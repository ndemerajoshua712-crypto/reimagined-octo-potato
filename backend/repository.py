import json
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from backend.config import DATABASE_URL, ensure_data_dirs


def now_utc():
    return datetime.now(timezone.utc)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported result value: {type(value).__name__}")


def _normalize_timestamps(values):
    for key, value in values.items():
        if isinstance(value, datetime):
            values[key] = value.isoformat(timespec="seconds")
    return values


def connect():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def check_database():
    with connect() as db:
        db.execute("SELECT 1")
    return True


def initialize_database():
    ensure_data_dirs()
    with connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
                created_at TIMESTAMPTZ NOT NULL,
                source_path TEXT NOT NULL,
                result_json JSONB
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
                status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed')),
                stage TEXT NOT NULL,
                progress DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK(progress BETWEEN 0 AND 100),
                created_at TIMESTAMPTZ NOT NULL,
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                error_code TEXT,
                error_message TEXT,
                options_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
        """)
        # Additive schema upgrade for databases created by earlier backend builds.
        db.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS options_json JSONB NOT NULL DEFAULT '{}'::jsonb")
        db.execute("CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status, created_at)")
        db.execute("CREATE INDEX IF NOT EXISTS cases_created ON cases(created_at DESC)")


def create_case_and_job(case_id, job_id, filename, sha256, size_bytes, source_path, options=None):
    created_at = now_utc()
    with connect() as db:
        db.execute(
            "INSERT INTO cases (case_id, filename, sha256, size_bytes, created_at, source_path) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (case_id, filename, sha256, size_bytes, created_at, str(source_path)),
        )
        db.execute(
            "INSERT INTO jobs (job_id, case_id, status, stage, progress, created_at, options_json) "
            "VALUES (%s, %s, 'queued', 'queued', 0, %s, %s)",
            (job_id, case_id, created_at, Jsonb(options or {})),
        )
    return created_at.isoformat(timespec="seconds")


def claim_job():
    with connect() as db:
        row = db.execute(
            "SELECT job_id, case_id, options_json FROM jobs WHERE status='queued' "
            "ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED"
        ).fetchone()
        if row is None:
            return None
        db.execute(
            "UPDATE jobs SET status='running', stage='analysis', progress=5, started_at=%s "
            "WHERE job_id=%s",
            (now_utc(), row["job_id"]),
        )
        case = db.execute(
            "SELECT filename, source_path, sha256 FROM cases WHERE case_id=%s",
            (row["case_id"],),
        ).fetchone()
        return {
            "job_id": row["job_id"],
            "case_id": row["case_id"],
            "options": row["options_json"] or {},
            **case,
        }


def update_job_progress(job_id, stage, progress):
    with connect() as db:
        db.execute(
            "UPDATE jobs SET stage=%s, progress=%s WHERE job_id=%s AND status='running'",
            (stage, max(5.0, min(99.0, float(progress))), job_id),
        )


def requeue_interrupted_jobs():
    """Recover jobs left running by a stopped worker process."""
    with connect() as db:
        db.execute(
            "UPDATE jobs SET status='queued', stage='queued', progress=0, started_at=NULL "
            "WHERE status='running' AND started_at < now() - interval '6 hours'"
        )


def finish_job(job_id, result):
    with connect() as db:
        row = db.execute("SELECT case_id FROM jobs WHERE job_id=%s", (job_id,)).fetchone()
        if row is None:
            return
        db.execute(
            "UPDATE cases SET result_json=%s WHERE case_id=%s",
            (Jsonb(result, dumps=lambda value: json.dumps(value, default=_json_default)), row["case_id"]),
        )
        db.execute(
            "UPDATE jobs SET status='succeeded', stage='complete', progress=100, finished_at=%s "
            "WHERE job_id=%s",
            (now_utc(), job_id),
        )


def fail_job(job_id, error_code, message):
    with connect() as db:
        db.execute(
            "UPDATE jobs SET status='failed', stage='failed', finished_at=%s, error_code=%s, error_message=%s "
            "WHERE job_id=%s",
            (now_utc(), error_code, message[:1000], job_id),
        )


def get_job(job_id):
    with connect() as db:
        row = db.execute("SELECT * FROM jobs WHERE job_id=%s", (job_id,)).fetchone()
        return _normalize_timestamps(dict(row)) if row else None


def list_cases(limit=50, offset=0):
    with connect() as db:
        rows = db.execute(
            "SELECT c.case_id, c.filename, c.sha256, c.size_bytes, c.created_at, "
            "j.job_id, j.status, j.stage, j.progress FROM cases c "
            "JOIN jobs j ON j.case_id=c.case_id ORDER BY c.created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        ).fetchall()
        return [_normalize_timestamps(dict(row)) for row in rows]


def get_case(case_id):
    with connect() as db:
        row = db.execute(
            "SELECT c.*, j.job_id, j.status, j.stage, j.progress, j.started_at, j.finished_at, "
            "j.error_code, j.error_message FROM cases c JOIN jobs j ON j.case_id=c.case_id "
            "WHERE c.case_id=%s ORDER BY j.created_at DESC LIMIT 1",
            (case_id,),
        ).fetchone()
        return _normalize_timestamps(dict(row)) if row else None


def migrate_from_sqlite(sqlite_path: Path):
    """Copy existing local SQLite rows into PostgreSQL, preserving identifiers/results."""
    import sqlite3

    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        return {"cases": 0, "jobs": 0}

    source = sqlite3.connect(sqlite_path)
    source.row_factory = sqlite3.Row
    try:
        cases = source.execute("SELECT * FROM cases").fetchall()
        jobs = source.execute("SELECT * FROM jobs").fetchall()
    finally:
        source.close()

    case_count = job_count = 0
    with connect() as target:
        for row in cases:
            values = dict(row)
            result = values.get("result_json")
            parsed_result = json.loads(result) if isinstance(result, str) and result else None
            cursor = target.execute(
                "INSERT INTO cases (case_id, filename, sha256, size_bytes, created_at, source_path, result_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (case_id) DO NOTHING",
                (
                    values["case_id"], values["filename"], values["sha256"],
                    values["size_bytes"], values["created_at"], values["source_path"],
                    Jsonb(parsed_result, dumps=lambda value: json.dumps(value, default=_json_default))
                    if parsed_result is not None else None,
                ),
            )
            case_count += cursor.rowcount
        for row in jobs:
            values = dict(row)
            options = values.get("options_json", "{}")
            status = values["status"]
            stage = values["stage"]
            progress = values["progress"]
            started_at = values.get("started_at")
            if status == "running":
                status, stage, progress, started_at = "queued", "queued", 0, None
            cursor = target.execute(
                "INSERT INTO jobs (job_id, case_id, status, stage, progress, created_at, started_at, "
                "finished_at, error_code, error_message, options_json) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (job_id) DO NOTHING",
                (
                    values["job_id"], values["case_id"], status, stage,
                    progress, values["created_at"], started_at,
                    values.get("finished_at"), values.get("error_code"), values.get("error_message"),
                    Jsonb(json.loads(options) if isinstance(options, str) else options or {}),
                ),
            )
            job_count += cursor.rowcount
    return {"cases": case_count, "jobs": job_count}
