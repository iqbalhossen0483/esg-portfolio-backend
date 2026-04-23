# Backend Review — ESG Portfolio Optimization API

**Scope:** full backend (FastAPI app at `esg-portfolio-backend/`), with priority focus on `POST /api/training/upload`.
**Reviewer perspective:** senior backend architect / production code review.

---

## 1. Findings (Issues, Risks, Inconsistencies)

### 1.1 `/api/training/upload` — End-to-end Flow Audit

Endpoint: `api/training.py:25-68` → `_run_ingestion_async` (background) → `core/adk/training_runner.py:run_training_pipeline` → `SequentialAgent` (4 LLM agents) → `core/tools/training_tools.py` → `db/crud.py` upserts.

| #   | Severity     | Location                                                           | Finding                                                                                                                                                                                                                                                                                                                      |
| --- | ------------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| F1  | **Critical** | `core/adk/training_runner.py:72`                                   | `return results` is referenced **before** `results = {...}` is initialized (line 82). Empty file → `NameError`.                                                                                                                                                                                                              |
| F2  | **Critical** | `core/tools/training_tools.py:144-148, 179-183, 200-217, 235-245`  | Storage tools call `asyncio.run(_store())` from within ADK's running event loop. Raises `RuntimeError: asyncio.run() cannot be called from a running event loop`. The pipeline cannot persist anything.                                                                                                                      |
| F3  | **Critical** | `api/training.py:71-80`                                            | `_run_ingestion_async` never updates the `TrainingJob` row. Status is locked at `"processing"` forever; `chunks_processed`, `records_stored`, `quality_report`, `completed_at` are never written. The `/training/status/{job_id}` endpoint always lies.                                                                      |
| F4  | **Critical** | `api/training.py:48-49`                                            | `shutil.copyfileobj(file.file, f)` is a **synchronous blocking I/O call inside an async route**. For multi-MB uploads it blocks the event loop and stalls every other request on the worker.                                                                                                                                 |
| F5  | **High**     | `api/training.py:56`                                               | `BackgroundTasks` runs **inside the FastAPI process**. Each chunk fans out 4 sequential `gemini-2.5-pro` calls. A 1000-row Excel ≈ 50 chunks × 4 agents × ~5–15s = 15–60 min while occupying a uvicorn worker. The dedicated Celery task `tasks/ingestion_task.py:run_ingestion` exists but is **not wired in** — dead code. |
| F6  | **High**     | `core/adk/training_runner.py:150`                                  | `if results["records_stored"]["prices"] > 0 …` — but `records_stored` counters are **never incremented anywhere** in this function. `trigger_metric_recomputation` is unreachable even on success.                                                                                                                           |
| F7  | **High**     | `core/tools/training_tools.py:235-246`                             | `store_knowledge_embedding` writes `embedding=None` ("will be generated later" — never wired up). Subsequent `search_knowledge` (`db/crud.py:323`) does `cosine_distance(NULL, …)` and returns junk / errors. The RAG layer is silently broken.                                                                              |
| F8  | **High**     | `api/training.py:33-38`                                            | Only **extension** is validated. No MIME sniffing, no magic-byte check, no file size limit — DoS / malicious file vector.                                                                                                                                                                                                    |
| F9  | **High**     | `api/training.py:19-20`                                            | `UPLOAD_DIR.mkdir(exist_ok=True)` runs at import time. Files are written to a CWD-relative path — fragile under different working directories or when run as a system service.                                                                                                                                               |
| F10 | **Medium**   | `api/training.py:44`                                               | `file_size: 0` is committed first, then the actual size is computed from the saved file. The DB row is wrong between the insert and (the never-occurring) update.                                                                                                                                                            |
| F11 | **Medium**   | `api/training.py:27-30`                                            | `background_tasks: BackgroundTasks = None` — the `= None` default is misleading. FastAPI injects regardless, but explicit `None` invites `.add_task()` on `NoneType` if the dependency tree changes.                                                                                                                         |
| F12 | **Medium**   | `api/training.py:53` (and throughout `core/adk/*`, `core/tools/*`) | `print()` instead of structured logging. Production observability is impossible.                                                                                                                                                                                                                                             |
| F13 | **Medium**   | `api/training.py:71-80`                                            | Errors in the background task only `print` and `traceback.print_exc()`. Job is never marked `failed`. No retry, no DLQ.                                                                                                                                                                                                      |
| F14 | **Medium**   | `api/training.py:19`                                               | No upload-dir cleanup; files accumulate indefinitely.                                                                                                                                                                                                                                                                        |
| F15 | **Medium**   | `core/parsers/excel_parser.py:12`                                  | `openpyxl.load_workbook(..., read_only=True)` — good — but full file is read in-process. No streaming for very large workbooks.                                                                                                                                                                                              |
| F16 | **Low**      | `api/training.py:37`                                               | `f"{ALLOWED_EXTENSIONS}"` prints a Python `set` repr — leaks internal type into API response.                                                                                                                                                                                                                                |
| F17 | **Low**      | `schemas/training.py:5`                                            | `UploadResponse.job_id: str`, but route returns `int`. Schema is also not bound as `response_model=` — silently divergent contract.                                                                                                                                                                                          |
| F18 | **Low**      | `core/adk/training_runner.py:60`                                   | `job_id = "0"` hard-coded. Caller never passes it in. Socket emissions and result dict use the wrong id.                                                                                                                                                                                                                     |

### 1.2 Cross-cutting Backend Issues

| #   | Severity     | Location                                                 | Finding                                                                                                                                                                                                                                          |
| --- | ------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| X1  | **Critical** | `config.py:24`                                           | `JWT_SECRET_KEY` defaults to `"dev-secret-key-change-in-production"`. No fail-fast in prod. If `.env` is missing the key, the server boots with a publicly-known secret.                                                                         |
| X2  | **High**     | `main.py:26`                                             | `IS_DEV = os.getenv("ENV", ...)` — `ENV` is never declared in `Settings`, so a misaligned env-var path. Default leaks **full tracebacks** to clients. Use the settings object.                                                                   |
| X3  | **High**     | `api/auth.py:55, 86, 89, 154, 168`                       | Mixed `datetime.now()` (naive, local TZ) with JWT's `datetime.now(timezone.utc)`. On any non-UTC server, refresh-token DB expiry vs JWT `exp` claim diverge. Use UTC consistently.                                                               |
| X4  | **High**     | `schemas/auth.py:5,12,17`                                | `email: str` instead of `EmailStr`; `password` has no min length, no complexity validators. Anyone can register with `"a"@""` and password `"x"`.                                                                                                |
| X5  | **High**     | `core/adk/training_socket.py:32`                         | Failed JWT decode `pass`es — connection allowed without auth context. The subscribe action checks role, but the connection itself is allowed. Should reject the connection.                                                                      |
| X6  | **High**     | `main.py:41-42`                                          | Both `register_chat_handlers(sio)` and `register_training_handlers(sio)` register their own `@sio.on("connect")` on the same default namespace. The second decorator overwrites the first. Whichever module registers last wins. Use namespaces. |
| X7  | **High**     | `tasks/pipeline_task.py:24-60`                           | Classic N+1: a `select` per company for prices and ESG. For 500 companies → 1000+ queries inside one Celery task. Will not scale.                                                                                                                |
| X8  | **Medium**   | `db/crud.py:81-89, 115-122, 147-155, 180-187`            | Single-record `upsert_*` functions each commit. Used inside loops elsewhere (e.g. `store_company_metadata` at `core/tools/training_tools.py:200-217` commits per row). Bulk variants exist for prices/ESG but not company.                       |
| X9  | **Medium**   | `db/database.py:5-11`                                    | Async engine pool size 20+10. No statement timeout, no connection lifetime. Long-lived connections to PG are a known footgun.                                                                                                                    |
| X10 | **Medium**   | `db/crud.py:208`                                         | `getattr(ComputedMetric, sort_by, ComputedMetric.composite_score)` — **untrusted user input from query string** mapped into SQL `ORDER BY`. Any attribute on the model is reachable. Whitelist allowed sort keys.                                |
| X11 | **Medium**   | `db/crud.py:232`                                         | Same untrusted-`sort_by` issue in `get_sector_rankings`.                                                                                                                                                                                         |
| X12 | **Medium**   | `db/models.py:42, 86`                                    | `onupdate=datetime.utcnow` is the deprecated naive form. Use `lambda: datetime.now(timezone.utc)` or DB-side `func.now()`.                                                                                                                       |
| X13 | **Medium**   | `api/admin.py:179-194`                                   | `trigger_training`, `change_user_role`, `deactivate_user` accept critical parameters as **query strings** (`role: str`, `esg_lambda: float`). State-changing endpoints should take a Pydantic body.                                              |
| X14 | **Medium**   | `api/companies.py`, `api/sectors.py`, `api/portfolio.py` | All call **synchronous** `core/tools/*` functions inside `async def` route handlers. If those tools touch the DB or filesystem, the event loop is blocked.                                                                                       |
| X15 | **Medium**   | `api/admin.py:243`                                       | `/pipeline/status` returns a hard-coded `{"status": "idle"}`. Stub left in production code.                                                                                                                                                      |
| X16 | **Medium**   | `api/auth.py:174-181`                                    | `/forgot-password` and `/reset-password` are no-ops returning success. Either implement or return 501; silent success is worse than no endpoint.                                                                                                 |
| X17 | **Medium**   | `db/models.py:154-164`                                   | `PortfolioAllocation.portfolio_id` is `Integer` with no FK and no unique/composite index, yet logically references "a portfolio". Schema integrity hole.                                                                                         |
| X18 | **Low**      | `main.py:18-24`                                          | Monkey-patching Google ADK's internal `BaseApiClient.aclose` at import time. Brittle, version-coupled, and silent — replace with a try/finally in your shutdown path or pin the SDK version.                                                     |
| X19 | **Low**      | `api/router.py`                                          | All routers exposed under one flat `/api` prefix; no versioning (`/api/v1/...`). Future breaking changes will hurt clients.                                                                                                                      |
| X20 | **Low**      | `tests/__init__.py`                                      | Test directory is empty. No coverage anywhere.                                                                                                                                                                                                   |
| X21 | **Low**      | `requirements.txt`                                       | No lockfile (`pip-compile`/`uv.lock`/`poetry.lock`). Deploys are non-reproducible; `>=` constraints on `torch`, `pgvector`, `google-adk` will drift.                                                                                             |
| X22 | **Low**      | `core/auth/security.py:13-15`                            | Truncating the password to 72 bytes silently is fine for bcrypt, but combined with no min-length validation users can create a 1-byte password without warning.                                                                                  |

---

## 2. Critical Problems (Must Fix)

These will cause functional failure or security incidents in production.

1. **`results` referenced before assignment** — `training_runner.py:72`. Fix the early-return ordering.
2. **`asyncio.run()` from within a running event loop** — all `store_*` tools (`training_tools.py`). The persistence layer is broken; no records ever land in PG. Convert tools to `async def`.
3. **`TrainingJob` row never updated** — `/status` endpoint returns stale `"processing"` indefinitely. Wire job-update calls into the runner with the real `job_id`.
4. **Synchronous `shutil.copyfileobj` in async handler** — use `await asyncio.to_thread(...)` or `aiofiles`. For real production use, stream directly to object storage.
5. **`BackgroundTasks` for a multi-minute LLM pipeline** — replace with the existing Celery `run_ingestion` task. Background tasks share the request worker; large uploads can take down the API.
6. **JWT secret default is a known string** — `Settings` must refuse to start if `ENV != "development"` and `JWT_SECRET_KEY` is the default.
7. **Sort-by SQL-injection-adjacent issue** (`getattr(Model, user_input)`) — replace with an explicit allow-list dict.
8. **Socket.IO `connect` handler collision** — move chat/training handlers to dedicated namespaces (`/chat`, `/training`) or merge auth into a single connect handler.
9. **No file size / MIME validation** on upload endpoint — add `Content-Length` cap, magic-byte verification (e.g. `python-magic`), and reject anything `> N MB`.
10. **Embeddings never generated** for `KnowledgeBase` — vector search is dead code. Generate inline (`text-embedding-004`) before insert, or run a follow-up Celery job.

---

## 3. Improvements (Production-grade Optimizations)

### 3.1 Upload endpoint

- Stream upload with size cap; reject when `total > MAX_UPLOAD_BYTES`.
- Hash the file (SHA-256) and store on `TrainingJob` for idempotency / dedupe.
- Persist to **object storage** (S3/MinIO) instead of local disk — local disk dies with the pod and breaks horizontal scaling.
- Use a **temp upload path → atomic rename** to its final name once `TrainingJob` is created. Currently: insert job → write file → never confirm. If the write fails after the insert, you have an orphan job.
- Add a proper `response_model` so OpenAPI matches reality.
- Return `202 Accepted` with `Location: /api/training/status/{id}` per REST conventions for async ingest.

### 3.2 Pipeline orchestration

- Make all `training_tools` **`async def`** and let the runtime await them. Removes the `asyncio.run` foot-gun, removes per-call session/connection churn, unlocks bulk batching.
- Move `trigger_metric_recomputation` to **end-of-job** based on records actually committed, not LLM-reported counts.
- Add **circuit breaker / retry** around per-chunk LLM calls (Gemini 5xx, quota, timeout). A single chunk failure is currently silently dropped from `records_stored`.
- Track **partial completion** in DB so a failed pipeline can resume by chunk index.
- Replace the LLM-driven extractor for **structured Excel/CSV** with deterministic Python parsing. Per-chunk LLM extraction of OHLCV is wasteful, slow, expensive, and non-deterministic. Reserve the LLM for `research_text` and ambiguous tabular layouts.

### 3.3 Persistence

- Add `bulk_upsert_companies` and use it from `store_company_metadata` — currently commits per row.
- Set Postgres **`statement_timeout`** at the engine level. One bad query shouldn't be able to hold a connection forever.
- Add **DB-side defaults** (`server_default=func.now()`) and stop maintaining `updated_at` from Python.
- Wrap per-job ingestion in a **single transaction per chunk** (`async with db.begin():`) so partial chunk failures don't leak rows.

### 3.4 Auth / security

- Use `EmailStr` and add `Field(min_length=8, max_length=128)` on passwords.
- Make `decode_token` distinguish `ExpiredSignatureError` from `JWTError` and return appropriate codes.
- Prevent username-enumeration timing attacks during login.
- Rotate refresh tokens on `/refresh`; current implementation does not — long-lived static refresh tokens.
- Add **rate limiting** (`slowapi` or upstream Nginx) on `/auth/*` and `/training/upload`.
- Configure CORS without `allow_methods=["*"]` plus `allow_credentials=True` — explicit method allow-lists are safer.

### 3.5 Observability

- Replace `print` with `logging` configured via `logging.config.dictConfig`. Emit JSON logs in production; include `request_id`, `user_id`, `job_id`, `chunk_index`.
- Add `/metrics` for Prometheus (request count, ingest duration, chunk failures, DB pool stats).
- Add request-id middleware (`X-Request-ID`) and propagate it to Celery tasks.

### 3.6 API hygiene

- Add `/api/v1` prefix and version your OpenAPI.
- Move query-param state changes (admin role/activate) to JSON bodies + Pydantic.
- Stop stub endpoints (`/pipeline/status`, `/forgot-password`, `/reset-password`) — return 501 until implemented.

---

## 4. Refactored Suggestions

### 4.1 `/api/training/upload` — corrected, production-shaped

```python
# api/training.py
import asyncio
import hashlib
from pathlib import Path
from uuid import uuid4

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from core.auth.dependencies import require_admin
from core.response import success_response
from db.crud import create_training_job
from db.database import get_db
from db.models import User
from schemas.training import UploadResponse
from tasks.ingestion_task import run_ingestion

router = APIRouter(prefix="/training", tags=["training"])

ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".pdf"}
ALLOWED_MIME = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "text/csv",
    "application/pdf",
}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024   # 50 MB
CHUNK = 1024 * 1024


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def upload_and_ingest(
    file: UploadFile = File(...),
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported content-type: {file.content_type}")

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = upload_dir / f"{uuid4().hex}{ext}"

    sha = hashlib.sha256()
    written = 0
    async with aiofiles.open(tmp_path, "wb") as out:
        while chunk := await file.read(CHUNK):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                await out.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(413, "File too large")
            sha.update(chunk)
            await out.write(chunk)

    job = await create_training_job(db, {
        "file_name": file.filename,
        "file_size": written,
        "status": "queued",
        "uploaded_by": user.id,
        "file_sha256": sha.hexdigest(),
    })

    final_path = upload_dir / f"{job.id}{ext}"
    tmp_path.rename(final_path)

    run_ingestion.delay(job.id, str(final_path), file.filename)

    return success_response(
        status_code=202,
        data={
            "job_id": job.id,
            "file_name": file.filename,
            "file_size": written,
            "status": "queued",
        },
        message="File uploaded. Ingestion pipeline queued.",
    )
```

Key changes: streaming + cap, MIME check, hash for idempotency, atomic rename, Celery delegation, `202 Accepted`, explicit `response_model`.

### 4.2 Pipeline runner — fix the bugs and own job state

```python
# core/adk/training_runner.py  (excerpt)
async def run_training_pipeline(*, job_id: int, file_path: str, file_name: str) -> dict:
    if runner is None or session_service is None:
        raise RuntimeError("Training runner not initialized")

    async with async_session() as db:
        await update_training_job(db, job_id, {"status": "processing"})

    raw_pages = await asyncio.to_thread(parse_file, file_path)
    chunks = chunk_pages(raw_pages)

    results = {
        "total_chunks": len(chunks),
        "chunks_processed": 0,
        "records_stored": defaultdict(int),
        "errors": [],
    }

    if not chunks:
        async with async_session() as db:
            await update_training_job(db, job_id, {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc),
                "quality_report": {"reason": "empty_or_unparsable"},
            })
        return results

    for i, chunk in enumerate(chunks):
        try:
            stored = await _process_chunk(chunk, i, len(chunks))
            results["chunks_processed"] += 1
            for k, v in stored.items():
                results["records_stored"][k] += v
        except Exception as e:
            results["errors"].append({"chunk": i + 1, "error": str(e)})
            log.exception("chunk failed", extra={"job_id": job_id, "chunk": i})

    if any(v > 0 for v in results["records_stored"].values()):
        recompute_metrics.delay()

    async with async_session() as db:
        await update_training_job(db, job_id, {
            "status": "completed" if not results["errors"] else "completed_with_errors",
            "total_chunks": results["total_chunks"],
            "chunks_processed": results["chunks_processed"],
            "records_stored": dict(results["records_stored"]),
            "quality_report": {"errors": results["errors"]},
            "completed_at": datetime.now(timezone.utc),
        })
    return results
```

### 4.3 Storage tools — drop `asyncio.run`, return real counts

```python
# core/tools/training_tools.py  (excerpt)
async def store_prices(records_json: str) -> dict:
    records = json.loads(records_json) if isinstance(records_json, str) else records_json
    cleaned = [_clean_price(r) for r in records]
    cleaned = [r for r in cleaned if r]
    if not cleaned:
        return {"records_stored": 0, "status": "empty"}
    async with async_session() as db:
        await bulk_upsert_prices(db, cleaned)
    return {"records_stored": len(cleaned), "status": "ok"}
```

Then register them as **async** function tools so ADK awaits them inside the parent loop. No nested loops, no `RuntimeError`.

### 4.4 Sort-by hardening

```python
# db/crud.py
_METRIC_SORT_COLS = {
    "composite_score": ComputedMetric.composite_score,
    "sharpe_252d": ComputedMetric.sharpe_252d,
    "avg_esg_composite": ComputedMetric.avg_esg_composite,
    "annual_return": ComputedMetric.annual_return,
}

def _resolve_metric_sort(name: str):
    return _METRIC_SORT_COLS.get(name, ComputedMetric.composite_score)
```

### 4.5 Settings safety

```python
# config.py
from pydantic import model_validator

class Settings(BaseSettings):
    ENV: str = "development"
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production"
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_BYTES: int = 50 * 1024 * 1024
    ...

    @model_validator(mode="after")
    def _enforce_prod_safety(self):
        if self.ENV != "development" and self.JWT_SECRET_KEY.startswith("dev-secret"):
            raise RuntimeError("Refusing to start in non-dev env with default JWT secret")
        return self
```

### 4.6 Architectural separation of concerns

The current layout works (`api / core / db / tasks / drl / schemas`) but violates clean-architecture in two recurring ways:

- **API handlers reach into `core/tools/*` directly**, passing raw query params to functions that touch the DB, format response data, and handle errors. There is no service layer. Recommend introducing `services/` between `api/` and `db/crud` so route handlers stay thin (auth + validation + response shaping) and business logic is testable without HTTP.
- **`core/tools/` is doing two unrelated jobs**: ADK-callable agent tools _and_ general-purpose business helpers used by REST routes. Split into:
  - `core/agent_tools/` — only invoked by ADK agents, follows ADK signature conventions
  - `services/` — invoked by REST routes, async-native, returns Pydantic DTOs

This single split removes most of the "sync function called from async route" bugs and makes it obvious what's being mocked in tests.

---

## TL;DR

The `/api/training/upload` endpoint **does not work end-to-end** today: storage tools crash on `asyncio.run` inside the running loop, the job row is never updated, an early `return results` hits an undefined name on empty files, the embedding column for the knowledge base is always `NULL`, and a long LLM pipeline is run via `BackgroundTasks` instead of the Celery task that already exists in the repo. Around the endpoint, the broader backend has correctness issues (UTC/naive datetime mixing), a security default (JWT secret), an arbitrary-`getattr` SQL sort vector, an N+1 metric pipeline, dead vector search, missing rate limits and tests, and a Socket.IO connect-handler collision. Fix the ten items in §2 first; the rest in §3 will turn this into a production-shaped service.
