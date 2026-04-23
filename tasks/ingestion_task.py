from datetime import datetime, timezone

from core.logging import get_logger
from db.crud import update_training_job
from db.database import async_session

from .celery_app import celery_app, run_async

log = get_logger(__name__)


# update the TrainingJob row to failed
async def _mark_failed(job_id: int, message: str) -> None:
    async with async_session() as db:
        await update_training_job(db, job_id, {
            "status": "failed",
            "error": message[:2000],
            "completed_at": datetime.now(timezone.utc),
        })


@celery_app.task(name="tasks.run_ingestion", bind=True, max_retries=0)
def run_ingestion(self, job_id: int, file_path: str, file_name: str) -> dict:
    """Celery task: run the ADK training pipeline for an uploaded file.
    On uncaught exception the TrainingJob row is marked failed."""
    from core.adk.training_runner import run_training_pipeline

    try:
        return run_async(run_training_pipeline(
            job_id=job_id,
            file_path=file_path,
            file_name=file_name,
        ))
    except Exception as exc:
        log.exception("ingestion task failed job_id=%s", job_id)
        try:
            run_async(_mark_failed(job_id, str(exc)))
        except Exception:
            log.exception("failed to mark job failed job_id=%s", job_id)
        raise
