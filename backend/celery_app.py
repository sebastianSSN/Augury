import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "augury",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,         # Results kept 1 hour
    task_track_started=True,     # STARTED state is reported
    worker_prefetch_multiplier=1, # One task at a time (training is CPU-heavy)
    task_acks_late=True,          # Ack after completion, not before (safer on crashes)
)
