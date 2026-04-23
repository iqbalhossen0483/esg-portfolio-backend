import asyncio

from dotenv import load_dotenv
load_dotenv()

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

from config import settings
from core.logging import get_logger

log = get_logger(__name__)

celery_app = Celery(
    "esg_portfolio",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "tasks.ingestion_task",
        "tasks.pipeline_task",
        "tasks.training_task",
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


# One event loop per worker process, shared across tasks. asyncio.run() would
# create and tear down a loop per task, leaving the module-level SQLAlchemy
# engine's asyncpg pool holding waiter futures from a dead loop — the source
# of "got Future ... attached to a different loop" on the second task.
_loop: asyncio.AbstractEventLoop | None = None


def run_async(coro):
    """Run a coroutine on the worker's persistent event loop."""
    if _loop is None:
        raise RuntimeError("Worker event loop not initialized")
    return _loop.run_until_complete(coro)


@worker_process_init.connect
def _init_worker(**_):
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    from core.adk.training_runner import init_training_runner

    async def _init():
        init_training_runner()

    _loop.run_until_complete(_init())
    log.info("worker event loop initialized")


@worker_process_shutdown.connect
def _shutdown_worker(**_):
    global _loop
    if _loop is None:
        return
    try:
        from db.database import engine
        _loop.run_until_complete(engine.dispose())
    except Exception:
        log.exception("engine dispose failed on worker shutdown")
    try:
        _loop.close()
    except Exception:
        log.exception("loop close failed on worker shutdown")
    _loop = None
