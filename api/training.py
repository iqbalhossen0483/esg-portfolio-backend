import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile

from core.auth.dependencies import require_admin
from db.crud import (
    create_training_job,
    get_training_job,
    list_training_jobs,
)
from db.database import get_db
from db.models import User
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/training", tags=["training"])

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".pdf"}


@router.post("/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Upload a data file and trigger the multi-agent ingestion pipeline."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {"error": f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}"}

    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}{ext}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = file_path.stat().st_size

    # Create tracking record
    await create_training_job(db, {
        "job_id": uuid.UUID(job_id),
        "file_name": file.filename,
        "file_size": file_size,
        "status": "processing",
    })

    # Trigger ingestion pipeline in background
    from tasks.ingestion_task import run_ingestion
    background_tasks.add_task(
        _run_ingestion_sync, job_id, str(file_path), file.filename
    )

    return {
        "job_id": job_id,
        "file_name": file.filename,
        "file_size": file_size,
        "status": "processing",
        "message": "File uploaded. Ingestion pipeline started.",
    }


def _run_ingestion_sync(job_id: str, file_path: str, file_name: str):
    """Wrapper to run ingestion — can be replaced with Celery .delay() in production."""
    import asyncio
    from core.adk.training_runner import run_training_pipeline

    asyncio.run(run_training_pipeline(file_path=file_path, file_name=file_name))


@router.get("/status/{job_id}")
async def get_ingestion_status(
    job_id: str,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Check status of an ingestion job."""
    job = await get_training_job(db, uuid.UUID(job_id))
    if not job:
        return {"error": "Job not found"}

    return {
        "job_id": str(job.job_id),
        "file_name": job.file_name,
        "status": job.status,
        "total_chunks": job.total_chunks,
        "chunks_processed": job.chunks_processed,
        "records_stored": job.records_stored,
        "quality_report": job.quality_report,
        "started_at": str(job.started_at) if job.started_at else None,
        "completed_at": str(job.completed_at) if job.completed_at else None,
    }


@router.get("/history")
async def get_ingestion_history(
    limit: int = 20,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List past ingestion jobs."""
    jobs = await list_training_jobs(db, limit=limit)
    return [
        {
            "job_id": str(j.job_id),
            "file_name": j.file_name,
            "status": j.status,
            "total_chunks": j.total_chunks,
            "chunks_processed": j.chunks_processed,
            "records_stored": j.records_stored,
            "started_at": str(j.started_at) if j.started_at else None,
            "completed_at": str(j.completed_at) if j.completed_at else None,
        }
        for j in jobs
    ]


@router.post("/recompute")
async def trigger_recompute(user: User = Depends(require_admin)):
    """Trigger metric recomputation (Sharpe, Sortino, etc.) without new data upload."""
    from tasks.pipeline_task import recompute_metrics
    recompute_metrics.delay()
    return {"status": "triggered", "message": "Metric recomputation started."}
