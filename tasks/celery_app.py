from dotenv import load_dotenv
load_dotenv()
from celery import Celery

from config import settings

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

