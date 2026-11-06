import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=False)
DATA_DIR = Path(os.getenv("DEEPTRACE_DATA_DIR", ROOT_DIR / "backend_data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
ARTIFACT_DIR = DATA_DIR / "artifacts"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://deeptrace:deeptrace@127.0.0.1:5432/deeptrace",
)
MAX_UPLOAD_BYTES = int(os.getenv("DEEPTRACE_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "DEEPTRACE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]


def ensure_data_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
