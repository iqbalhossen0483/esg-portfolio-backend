import hashlib
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from core.auth.dependencies import require_admin
from core.logging import get_logger
from core.response import success_response
from db.crud import create_training_job, get_training_job, list_training_jobs
from db.database import get_db
from db.models import User

log = get_logger(__name__)

router = APIRouter(prefix="/training", tags=["training"])

ALLOWED_EXTENSIONS = (".xlsx", ".xls", ".csv", ".pdf")
ALLOWED_MIME = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",                                            # .xls
    "application/octet-stream",                                            # some browsers send this for .xls/.xlsx
    "text/csv",
    "application/csv",
    "text/plain",                                                          # .csv from some clients
    "application/pdf",
}
READ_CHUNK = 1024 * 1024  # 1 MiB


def _ext_list_str() -> str:
    return ", ".join(ALLOWED_EXTENSIONS)


def _validate_magic(buf: bytes, ext: str) -> bool:
    """Lightweight magic-byte check to back up the MIME header."""
    if ext == ".pdf":
        return buf.startswith(b"%PDF-")
    if ext == ".xlsx":
        # xlsx is a ZIP container
        return buf.startswith(b"PK\x03\x04")
    if ext == ".xls":
        return buf.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    if ext == ".csv":
        # CSV has no signature; accept anything text-ish (no NUL byte)
        return b"\x00" not in buf
    return False


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_and_ingest(
    file: UploadFile = File(...),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Upload a data file and queue the multi-agent ingestion pipeline (Celery)."""
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {ext or '(none)'}. Allowed: {_ext_list_str()}",
        )

    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content-type: {file.content_type}",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f".tmp-{uuid4().hex}{ext}"

    sha = hashlib.sha256()
    written = 0
    first_bytes = b""

    try:
        async with aiofiles.open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(READ_CHUNK)
                if not chunk:
                    break
                if not first_bytes:
                    first_bytes = chunk[:16]
                written += len(chunk)
                if written > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File too large: {written} bytes "
                            f"(max {settings.MAX_UPLOAD_BYTES})"
                        ),
                    )
                sha.update(chunk)
                await out.write(chunk)

        if written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file",
            )

        if not _validate_magic(first_bytes, ext):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File contents do not match extension {ext}",
            )

        digest = sha.hexdigest()

        job = await create_training_job(db, {
            "file_name": filename,
            "file_size": written,
            "file_sha256": digest,
            "uploaded_by": user.id,
            "status": "queued",
        })

        final_path = upload_dir / f"{job.id}{ext}"
        tmp_path.rename(final_path)
        log.info("upload accepted job_id=%s sha256=%s size=%d user_id=%s",
                 job.id, digest, written, user.id)

        from tasks.ingestion_task import run_ingestion
        run_ingestion.delay(job.id, str(final_path), filename)

        return success_response(
            status_code=status.HTTP_202_ACCEPTED,
            data={
                "job_id": job.id,
                "file_name": filename,
                "file_size": written,
                "file_sha256": digest,
                "status": "queued",
            },
            message="File uploaded. Ingestion pipeline queued.",
        )

    except HTTPException:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        log.exception("upload failed user_id=%s file=%s", user.id, filename)
        raise


@router.get("/status/{job_id}")
async def get_ingestion_status(
    job_id: int,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Check status of an ingestion job."""
    job = await get_training_job(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return success_response(
        data={
            "job_id": job.id,
            "file_name": job.file_name,
            "status": job.status,
            "total_chunks": job.total_chunks,
            "chunks_processed": job.chunks_processed,
            "records_stored": job.records_stored,
            "quality_report": job.quality_report,
            "error": job.error,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        },
        message="Job status retrieved successfully",
    )


@router.get("/history")
async def get_ingestion_history(
    limit: int = 20,
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """List past ingestion jobs."""
    limit = max(1, min(limit, 100))
    jobs = await list_training_jobs(db, limit=limit)
    return success_response(
        data=[
            {
                "job_id": j.id,
                "file_name": j.file_name,
                "status": j.status,
                "total_chunks": j.total_chunks,
                "chunks_processed": j.chunks_processed,
                "records_stored": j.records_stored,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs
        ],
        message="Ingestion history retrieved successfully",
    )


@router.post("/recompute")
async def trigger_recompute(user: User = Depends(require_admin)):
    """Trigger metric recomputation without new data upload."""
    from tasks.pipeline_task import recompute_metrics
    task = recompute_metrics.delay()
    return success_response(
        data={"status": "triggered", "task_id": task.id},
        message="Metric recomputation started.",
    )
