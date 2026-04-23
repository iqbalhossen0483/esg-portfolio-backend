"""Training pipeline orchestrator.
Parses file → chunks → processes each chunk via ADK agent pipeline.
Owns the TrainingJob row state from start to finish.
"""

import re
from collections import defaultdict
from datetime import datetime, timezone

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from core.logging import get_logger
from core.parsers.base_parser import parse_file
from core.parsers.chunker import chunk_pages, count_tokens
from db.crud import update_training_job
from db.database import async_session

log = get_logger(__name__)

# Training sessions are ephemeral — InMemorySessionService is correct here
session_service: InMemorySessionService | None = None
runner: Runner | None = None


def init_training_runner():
    """Called once from lifespan, inside the running event loop."""
    global session_service, runner
    try:
        from .training_agents import chunk_pipeline

        session_service = InMemorySessionService()
        runner = Runner(
            agent=chunk_pipeline,
            app_name="esg_training",
            session_service=session_service,
        )
        log.info("training runner initialized")
    except Exception:
        log.exception("training runner initialization failed")
        raise


_STORAGE_RESULT_RE = re.compile(r'"records_stored"\s*:\s*(\d+)')


def _parse_records_stored(response_text: str) -> tuple[str | None, int]:
    """Pull (data_type, count) from the storage agent's JSON output (best effort)."""
    if not response_text:
        return None, 0
    m = _STORAGE_RESULT_RE.search(response_text)
    count = int(m.group(1)) if m else 0
    type_match = re.search(r'"type"\s*:\s*"([^"]+)"', response_text)
    return (type_match.group(1) if type_match else None), count


async def _mark_job(job_id: int, **fields) -> None:
    async with async_session() as db:
        await update_training_job(db, job_id, fields)


async def run_training_pipeline(
    *,
    job_id: int,
    file_path: str,
    file_name: str,
    sio=None,
    sid=None,
) -> dict:
    """Full training pipeline: parse → chunk → process each chunk via ADK agents.

    Args:
        job_id: TrainingJob.id (required so we can update status).
        file_path: Path to the uploaded file.
        file_name: Original file name.
        sio: Socket.IO server instance (optional, for progress emissions).
        sid: Socket.IO session ID (optional).

    Returns:
        dict with job results summary.
    """
    if runner is None or session_service is None:
        raise RuntimeError("Training runner not initialized. Call init_training_runner() in lifespan.")

    log.info("pipeline start job_id=%s file=%s", job_id, file_name)
    await _mark_job(job_id, status="processing")

    raw_pages = parse_file(file_path)
    log.info("parsed pages job_id=%s pages=%d", job_id, len(raw_pages))

    chunks = chunk_pages(raw_pages)
    log.info("created chunks job_id=%s chunks=%d", job_id, len(chunks))

    results: dict = {
        "job_id": job_id,
        "file_name": file_name,
        "total_chunks": len(chunks),
        "chunks_processed": 0,
        "records_stored": defaultdict(int),
        "warnings": [],
        "errors": [],
    }

    if not chunks:
        log.warning("no chunks created job_id=%s", job_id)
        await _mark_job(
            job_id,
            status="failed",
            total_chunks=0,
            quality_report={"reason": "empty_or_unparsable"},
            completed_at=datetime.now(timezone.utc),
        )
        results["status"] = "failed"
        return results

    if sio and sid:
        await sio.emit("training:job_started", {
            "job_id": job_id,
            "file_name": file_name,
            "total_chunks": len(chunks),
        }, room=sid)

    for i, chunk in enumerate(chunks):
        token_count = count_tokens(chunk)

        try:
            session = await session_service.create_session(
                app_name="esg_training",
                user_id=f"job-{job_id}",
            )

            chunk_response = ""
            data_type: str | None = None
            chunk_records = 0

            async for event in runner.run_async(
                user_id=f"job-{job_id}",
                session_id=session.id,
                new_message=types.Content(
                    parts=[types.Part(
                        text=f"Process this data chunk "
                             f"({token_count} tokens, chunk {i + 1}/{len(chunks)}):\n\n"
                             f"{chunk}"
                    )]
                ),
            ):
                if not (event.content and event.content.parts):
                    continue
                for part in event.content.parts:
                    if not part.text:
                        continue
                    chunk_response = part.text
                    author = getattr(event, "author", "")
                    if author == "DataStorer":
                        dt, cnt = _parse_records_stored(part.text)
                        if cnt:
                            data_type, chunk_records = dt, cnt

            log.debug("chunk done job_id=%s chunk=%d response=%s",
                      job_id, i + 1, (chunk_response[:200] if chunk_response else ""))

            results["chunks_processed"] += 1
            if data_type and chunk_records:
                results["records_stored"][data_type] += chunk_records

            if sio and sid:
                await sio.emit("training:chunk_progress", {
                    "job_id": job_id,
                    "chunk_index": i + 1,
                    "total_chunks": len(chunks),
                    "status": "completed",
                    "token_count": token_count,
                }, room=sid)

        except Exception as e:
            err = f"Chunk {i + 1}: {e}"
            log.exception("chunk failed job_id=%s chunk=%d", job_id, i + 1)
            results["errors"].append(err)

            if sio and sid:
                await sio.emit("training:chunk_error", {
                    "job_id": job_id,
                    "chunk_index": i + 1,
                    "error": err,
                }, room=sid)

    if any(v > 0 for v in results["records_stored"].values()):
        from core.tools.training_tools import trigger_metric_recomputation

        if sio and sid:
            await sio.emit("training:metrics_recomputing", {"status": "running"}, room=sid)

        trigger_metric_recomputation()

        if sio and sid:
            await sio.emit("training:metrics_done", {"status": "completed"}, room=sid)

    final_status = "completed" if not results["errors"] else "completed_with_errors"
    completed_at = datetime.now(timezone.utc)
    records_stored_dict = dict(results["records_stored"])

    await _mark_job(
        job_id,
        status=final_status,
        total_chunks=results["total_chunks"],
        chunks_processed=results["chunks_processed"],
        records_stored=records_stored_dict,
        quality_report={"errors": results["errors"], "warnings": results["warnings"]},
        completed_at=completed_at,
    )

    if sio and sid:
        await sio.emit("training:job_completed", {
            "job_id": job_id,
            "summary": {**results, "records_stored": records_stored_dict},
        }, room=sid)

    results["status"] = final_status
    results["records_stored"] = records_stored_dict
    results["completed_at"] = completed_at.isoformat()
    log.info("pipeline done job_id=%s status=%s chunks=%d/%d",
             job_id, final_status, results["chunks_processed"], results["total_chunks"])
    return results
