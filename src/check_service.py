import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from src.config import get_data_root


DEFAULT_TIMEOUT = 600
POLL_INTERVAL = 2


def build_sample_payload(training_data_file: str) -> dict[str, Any]:
    return {
        "model_name_cn": "环境保护污染分类模型",
        "model_name_en": "environmental_pollution_classifier",
        "training_data_file": training_data_file,
        "base_model": "bert-base-chinese",
        "hyperparameters": {
            "learning_rate": 3e-5,
            "epochs": 2,
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


def delete_request(url: str) -> dict:
    response = requests.delete(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    if response.content:
        return response.json()
    return {}


def run_service_check(base_url: str, dataset: str | None, cleanup: bool) -> None:
    logger.info("Running integration check against {}", base_url)

    training_file = resolve_training_file(dataset)
    payload = build_sample_payload(training_file)
    create_url = f"{base_url}/training/tasks"
    create_response = post_json(create_url, payload)
    task_id = create_response["task_id"]
    logger.info("Created task {} with status {}", task_id, create_response["status"])

    detail_url = f"{base_url}/training/tasks/{task_id}"
    delete_url = f"{base_url}/training/tasks/{task_id}"

    end_time = time.time() + DEFAULT_TIMEOUT
    last_status = create_response["status"]
    detail = create_response

    while time.time() < end_time:
        detail = get_json(detail_url)
        status = detail["status"]
        if status != last_status:
            logger.info("Task {} status changed from {} to {}", task_id, last_status, status)
            last_status = status
        progress = detail.get("progress")
        if progress is not None and progress.get("progress_percentage") is not None:
            logger.info(
                "Progress {:.1f}% (epoch {}/{})",
                progress["progress_percentage"],
                progress.get("current_epoch"),
                progress.get("total_epochs"),
            )
        if status == "completed":
            break
        if status == "failed":
            raise RuntimeError(f"Task {task_id} failed: {detail.get('error_message')}")
        time.sleep(POLL_INTERVAL)
    else:
        raise TimeoutError(f"Timed out waiting for task {task_id} to complete")

    if detail["status"] != "completed":
        raise RuntimeError(f"Task {task_id} ended in unexpected status {detail['status']}")

    artifacts = detail.get("artifacts") or {}
    model_path = artifacts.get("model_path")
    label_mapping_path = artifacts.get("label_mapping_path")

    if model_path is None or label_mapping_path is None:
        raise ValueError("Artifacts missing in task detail response")

    model_file = Path(model_path)
    label_file = Path(label_mapping_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")
    if not label_file.exists():
        raise FileNotFoundError(f"Label mapping artifact not found: {label_file}")

    logger.info("Model artifact stored at {}", model_file)
    logger.info("Label mapping stored at {}", label_file)

    if cleanup:
        logger.info("Cleaning up task {}", task_id)
        delete_request(delete_url)
        logger.info("Cleanup completed for task {}", task_id)
    else:
        logger.info("Task {} retained for further inspection", task_id)

    logger.info("Integration check complete")


__all__ = ["run_service_check"]
