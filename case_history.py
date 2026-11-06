import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DATABASE_PATH = Path(__file__).resolve().parent / "cases" / "case_history.sqlite3"


def save_case(
    filename,
    file_hash,
    result,
    database_path=DATABASE_PATH,
):
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    case_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "verdict": result["multimodal_result"].get("verdict"),
        "primary_score": result["multimodal_result"].get("primary_score"),
        "face_count": sum(
            frame.get("face_count", 0)
            for frame in result.get("video_results", [])
        ),
        "frames": len(result.get("video_results", [])),
        "audio_windows": result.get("audio_windows", {}).get("total_windows", 0),
    }

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    summary_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO cases (case_id, created_at, filename, sha256, summary_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (case_id, created_at, filename, file_hash, json.dumps(summary)),
            )

    return {
        "case_id": case_id,
        "created_at": created_at,
        "filename": filename,
        "sha256": file_hash,
        **summary,
    }


def list_cases(limit=100, database_path=DATABASE_PATH):
    database_path = Path(database_path)
    if not database_path.exists():
        return []

    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        with connection:
            rows = connection.execute(
                "SELECT case_id, created_at, filename, sha256, summary_json "
                "FROM cases ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()

    cases = []
    for row in rows:
        cases.append(
            {
                "case_id": row["case_id"],
                "created_at": row["created_at"],
                "filename": row["filename"],
                "sha256": row["sha256"],
                **json.loads(row["summary_json"]),
            }
        )
    return cases
