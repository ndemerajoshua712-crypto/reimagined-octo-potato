import logging
import os
import time
from pathlib import Path

from backend import repository
from backend.config import ARTIFACT_DIR, DATA_DIR

# Numba's default cache points inside site-packages, which is not writable in
# many virtual environments. Set a project-owned cache before importing the
# analysis pipeline (and its librosa/numba dependencies).
NUMBA_CACHE_DIR = DATA_DIR / "numba_cache"
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

logger = logging.getLogger("deeptrace.worker")


def run_one_job():
    job = repository.claim_job()
    if job is None:
        return False
    try:
        # Imported only in the worker so the API process does not load ML dependencies.
        from analysis_pipeline import analyze_video_file

        case_artifacts = ARTIFACT_DIR / job["case_id"]
        case_artifacts.mkdir(parents=True, exist_ok=True)

        def progress(stage, value):
            repository.update_job_progress(job["job_id"], stage, value)

        result = analyze_video_file(
            job["source_path"],
            job["filename"],
            frame_interval=job["options"].get("frame_interval", 0.5),
            output_root=case_artifacts,
            progress_callback=progress,
        )
        result["case_id"] = job["case_id"]
        repository.finish_job(job["job_id"], result)
        logger.info("job complete", extra={"job_id": job["job_id"], "case_id": job["case_id"]})
    except Exception:
        logger.exception("analysis job failed", extra={"job_id": job["job_id"]})
        repository.fail_job(job["job_id"], "analysis_failed", "The analysis pipeline failed. Check worker logs for details.")
    return True


def main(poll_seconds=2):
    repository.initialize_database()
    repository.requeue_interrupted_jobs()
    logger.info("DeepTrace worker started")
    while True:
        if not run_one_job():
            time.sleep(poll_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
