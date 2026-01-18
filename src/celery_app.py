from celery import Celery

from src.settings import settings

# Ensure worker initialisation hooks are registered
import src.worker_allocation  # noqa: F401


celery_app = Celery(
    "dianxin_auto_train",
    broker=settings.celery_broker_url_value,
    backend=settings.celery_result_backend_value,
)

celery_app.conf.update(
    task_track_started=True,
    result_expires=None,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    enable_utc=True,
    timezone="UTC",
)


__all__ = ["celery_app"]
