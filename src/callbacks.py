from urllib.parse import urljoin

import requests
from loguru import logger

from src.config import (
    get_callback_timeout,
    get_external_callback_base_url,
    get_external_publish_callback_url,
)


def _post_json(url: str, payload: dict) -> None:
    timeout = get_callback_timeout()
    logger.debug("Sending callback to {} with payload {}", url, payload)
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()


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
    base_url = get_external_callback_base_url()
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


def send_external_publish_callback(task_id: str, publish_result: int, failure_message: str | None) -> None:
    publish_url = get_external_publish_callback_url()
    base_url = get_external_callback_base_url()
    if publish_url is None and base_url is None:
        return
    target_url = publish_url
    if target_url is None:
        target_url = urljoin(base_url.rstrip("/") + "/", "api/model/train/notify/publish_result")
    payload = {
        "trainTaskId": task_id,
        "publishResult": publish_result,
        "failureMessage": failure_message or "",
    }
    _post_json(target_url, payload)


__all__ = [
    "send_external_epoch_callback",
    "send_external_publish_callback",
    "send_progress_callback",
]
