import asyncio

from .celery_app import celery_app


@celery_app.task(name="tasks.run_ingestion")
def run_ingestion(job_id: str, file_path: str, file_name: str):
    """Celery task: run the ADK training pipeline for an uploaded file."""
    from core.adk.training_runner import run_training_pipeline

    result = asyncio.run(run_training_pipeline(
        file_path=file_path,
        file_name=file_name,
    ))
    return result
