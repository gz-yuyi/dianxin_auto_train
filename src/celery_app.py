from celery import Celery

from src.config import get_celery_backend_url, get_celery_broker_url

# Ensure worker initialisation hooks are registered
import src.worker_allocation  # noqa: F401


celery_app = Celery(
    "dianxin_auto_train",
    broker=get_celery_broker_url(),
    backend=get_celery_backend_url(),
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
