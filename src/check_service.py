import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from src.config import get_data_root


DEFAULT_TIMEOUT = 30
POLL_INTERVAL = 2


def build_sample_payload(training_data_file: str) -> dict[str, Any]:
    return {
        "model_name_cn": "环境保护污染分类模型",
        "model_name_en": "environmental_pollution_classifier",
        "training_data_file": training_data_file,
        "base_model": "bert-base-chinese",
        "hyperparameters": {
            "learning_rate": 3e-5,
            "epochs": 1,
            "batch_size": 8,
            "max_sequence_length": 128,
            "random_seed": 42,
            "train_val_split": 0.2,
            "text_column": "内容合并",
            "label_column": "标签列",
            "sheet_name": None,
        },
        "callback_url": None,
    }


def resolve_training_file(filename: str | None) -> str:
    if filename is None or not filename.strip():
        default_file = "legacy/环境保护_空气污染--样例1000.xlsx"
        logger.debug("No dataset provided, using default {}", default_file)
        filename = default_file
    path = Path(filename)
    if not path.is_absolute():
        path = get_data_root() / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return str(path)


def post_json(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_json(url: str) -> dict:
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def run_service_check(base_url: str, dataset: str | None) -> None:
    logger.info("Running integration check against {}", base_url)

    training_file = resolve_training_file(dataset)
    payload = build_sample_payload(training_file)
    create_url = f"{base_url}/training/tasks"
    create_response = post_json(create_url, payload)
    task_id = create_response["task_id"]
    logger.info("Created task {} with status {}", task_id, create_response["status"])

    detail_url = f"{base_url}/training/tasks/{task_id}"
    stop_url = f"{base_url}/training/tasks/{task_id}/stop"
    delete_url = f"{base_url}/training/tasks/{task_id}"

    end_time = time.time() + DEFAULT_TIMEOUT
    last_status = create_response["status"]

    while time.time() < end_time:
        detail = get_json(detail_url)
        status = detail["status"]
        if status != last_status:
            logger.info("Task {} status changed from {} to {}", task_id, last_status, status)
            last_status = status
        if status in {"training", "completed", "failed"}:
            break
        time.sleep(POLL_INTERVAL)

    logger.info("Stopping task {}", task_id)
    post_json(stop_url, {})
    logger.info("Deleting task {}", task_id)
    response = requests.delete(delete_url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    logger.info("Integration check complete")


__all__ = ["run_service_check"]
