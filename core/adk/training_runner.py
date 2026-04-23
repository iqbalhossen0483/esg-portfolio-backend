"""Training pipeline orchestrator.
Parses file → chunks → processes each chunk via ADK agent pipeline.
"""

from datetime import datetime, timezone

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from core.parsers.base_parser import parse_file
from core.parsers.chunker import chunk_pages, count_tokens

# Training sessions are ephemeral — InMemorySessionService is correct here
session_service: InMemorySessionService | None = None
runner: Runner | None = None


def init_training_runner():
    """Called once from lifespan, inside the running event loop."""
    global session_service, runner
    try:
        from .training_agents import chunk_pipeline  # this is where it may be failing

        session_service = InMemorySessionService()
        runner = Runner(
            agent=chunk_pipeline,
            app_name="esg_training",
            session_service=session_service,
        )
        print("✅ Training runner initialized successfully")
    except Exception as e:
        print(f"❌ Training runner initialization FAILED: {e}")
        raise  # don't swallow it — crash loudly on startup


async def run_training_pipeline(
    file_path: str,
    file_name: str,
    sio=None,
    sid=None,
) -> dict:
    """Full training pipeline: parse → chunk → process each chunk via ADK agents.

    Args:
        file_path: Path to the uploaded file.
        file_name: Original file name.
        sio: Socket.IO server instance (optional, for progress emissions).
        sid: Socket.IO session ID (optional).

    Returns:
        dict with job results summary.
    """
    print("Running training pipeline...")
    if runner is None or session_service is None:
        raise RuntimeError("Training runner not initialized. Call init_training_runner() in lifespan.")
    
    print(f"Running training pipeline for {file_name} ({file_path})")

    job_id = "0"  # Will be set by caller from DB auto-increment

    # Step 1: Parse file into raw pages (pure Python, no LLM)
    raw_pages = parse_file(file_path)
    print(f"📄 Parsed {len(raw_pages)} raw pages")

    # Step 2: Apply tokenization + chunking rules
    chunks = chunk_pages(raw_pages)
    print(f"🔪 Created {len(chunks)} chunks")

    if not chunks:
        print("⚠️ NO CHUNKS CREATED — file may be empty or parser failed")
        return results

    # Emit: job started
    if sio and sid:
        await sio.emit("training:job_started", {
            "job_id": job_id,
            "file_name": file_name,
            "total_chunks": len(chunks),
        }, room=sid)

    results = {
        "job_id": job_id,
        "file_name": file_name,
        "total_chunks": len(chunks),
        "chunks_processed": 0,
        "records_stored": {
            "prices": 0,
            "esg_scores": 0,
            "company_meta": 0,
            "knowledge": 0,
        },
        "warnings": [],
        "errors": [],
    }

    # Step 3: Process each chunk through ADK agent pipeline
    for i, chunk in enumerate(chunks):
        token_count = count_tokens(chunk)

        try:
            session = await session_service.create_session(
                app_name="esg_training",
                user_id="admin",
            )

            response = ""
            async for event in runner.run_async(
                user_id="admin",
                session_id=session.id,
                new_message=types.Content(
                    parts=[types.Part(
                        text=f"Process this data chunk "
                             f"({token_count} tokens, chunk {i + 1}/{len(chunks)}):\n\n"
                             f"{chunk}"
                    )]
                ),
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            response = part.text

            print(f"🤖 Agent response: {response[:300] if response else 'NO RESPONSE'}")
            results["chunks_processed"] += 1

            # Emit: chunk progress
            if sio and sid:
                await sio.emit("training:chunk_progress", {
                    "job_id": job_id,
                    "chunk_index": i + 1,
                    "total_chunks": len(chunks),
                    "status": "completed",
                    "token_count": token_count,
                }, room=sid)

        except Exception as e:
            error_msg = f"Chunk {i + 1}: {str(e)}"
            print(f"error: {error_msg}")
            results["errors"].append(error_msg)

            if sio and sid:
                await sio.emit("training:chunk_error", {
                    "job_id": job_id,
                    "chunk_index": i + 1,
                    "error": error_msg,
                }, room=sid)

    # Step 4: Trigger metric recomputation if data was stored
    if results["records_stored"]["prices"] > 0 or results["records_stored"]["esg_scores"] > 0:
        from core.tools.training_tools import trigger_metric_recomputation

        if sio and sid:
            await sio.emit("training:metrics_recomputing", {"status": "running"}, room=sid)

        trigger_metric_recomputation()

        if sio and sid:
            await sio.emit("training:metrics_done", {"status": "completed"}, room=sid)

    # Emit: job completed
    if sio and sid:
        await sio.emit("training:job_completed", {
            "job_id": job_id,
            "summary": results,
        }, room=sid)

    results["status"] = "completed"
    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    return results
