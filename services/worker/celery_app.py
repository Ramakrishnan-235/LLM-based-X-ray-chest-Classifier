from celery import Celery
from apps.api.config import settings


celery_app = Celery(
    "cxr_insight_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["services.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # Strict backpressure: 1 job at a time per GPU worker
    task_routes={
        "services.worker.tasks.run_cv_pipeline": {"queue": "cxr_cv_local"},
        "services.worker.tasks.run_vlm_report_pipeline": {"queue": "cxr_vlm_remote"},
    },
)
