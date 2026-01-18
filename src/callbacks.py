from urllib.parse import urljoin

import requests
from loguru import logger

from src.settings import settings


STATUS_TYPE_BY_STATUS: dict[str, int] = {
    "queued": 1,
    "training": 2,
    "completed": 3,
    "failed": 4,
    "stopped": 5,
    "cancelled": 5,
}

FAILURE_MESSAGE_REQUIRED = {4, 5}


def _post_json(url: str, payload: dict) -> None:
    timeout = settings.external_callback_timeout
    logger.debug("Sending callback to {} with payload {}", url, payload)
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except Exception as e:
        logger.warning("Failed to send callback to {}: {}", url, str(e))


def send_progress_callback(task_id: str, epoch: int, metrics: dict, callback_url: str | None) -> None:
    if callback_url is None:
        return
    payload = {
        "taskId": task_id,
        "epoch": epoch,
        "trainAccuracy": metrics["train_accuracy"],
        "trainLoss": metrics["train_loss"],
        "valAccuracy": metrics["val_accuracy"],
        "valLoss": metrics["val_loss"],
        "progressPercentage": metrics["progress_percentage"],
        "f1Score": metrics["f1_score"],
    }
    _post_json(callback_url, payload)


def send_external_epoch_callback(task_id: str, epoch: int, metrics: dict) -> None:
    base_url = settings.external_callback_base_url
    if base_url is None:
        return
    url = urljoin(base_url.rstrip("/") + "/", "api/model/train/notify/result")
    payload = {
        "trainTaskId": task_id,
        "epoch": epoch,
        "accuracy": metrics["train_accuracy"],
        "loss": metrics["train_loss"],
        "valAccuracy": metrics["val_accuracy"],
        "valLoss": metrics["val_loss"],
        "f1Score": metrics["f1_score"],
    }
    _post_json(url, payload)


def _resolve_status_callback_url() -> str | None:
    status_url = settings.external_status_callback_url_value
    if status_url:
        trimmed = status_url.rstrip("/")
        sentinel = "/api/model/train/notify/status"
        if trimmed.endswith(sentinel):
            return trimmed
        if sentinel in trimmed:
            return status_url
        return urljoin(trimmed + "/", sentinel.lstrip("/"))
    base_url = settings.external_callback_base_url
    if base_url is None:
        return None
    return urljoin(base_url.rstrip("/") + "/", "api/model/train/notify/status")


def send_external_status_callback(task_id: str, status_type: int, failure_message: str | None = None) -> None:
    target_url = _resolve_status_callback_url()
    if target_url is None:
        return
    payload = {
        "trainTaskId": task_id,
        "statusType": status_type,
    }
    if failure_message:
        payload["failureMessage"] = failure_message[:500]
    elif status_type in FAILURE_MESSAGE_REQUIRED:
        payload["failureMessage"] = ""
    _post_json(target_url, payload)


def send_external_status_change(task_id: str, status: str, failure_message: str | None = None) -> None:
    status_type = STATUS_TYPE_BY_STATUS.get(status)
    if status_type is None:
        logger.warning("Unsupported status {} for external status callback", status)
        return
    send_external_status_callback(task_id, status_type, failure_message)


__all__ = [
    "send_external_epoch_callback",
    "send_external_status_callback",
    "send_external_status_change",
    "send_progress_callback",
]
