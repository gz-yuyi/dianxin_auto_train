from celery import states
from loguru import logger

from src.callbacks import (
    send_external_epoch_callback,
    send_external_publish_callback,
    send_progress_callback,
)
from src.celery_app import celery_app
from src.storage import clear_stop_request, is_stop_requested, iso_now, load_task_record, update_task_record
from src.training.service import run_training_loop


@celery_app.task(bind=True, name="training.run_task")
def start_training_task(self, task_id: str) -> dict:
    record = load_task_record(task_id)
    if record is None:
        logger.error("Task record {} not found", task_id)
        return {"status": "missing"}

    request_payload = record["request_payload"]
    if record.get("status") == "stopped" or is_stop_requested(task_id):
        logger.info("Task {} marked as stopped before execution", task_id)
        return {"status": "stopped"}

    update_task_record(
        task_id,
        {
            "status": "training",
            "started_at": iso_now(),
            "progress": {
                "current_epoch": 0,
                "total_epochs": request_payload["hyperparameters"]["epochs"],
                "progress_percentage": 0.0,
                "train_accuracy": None,
                "train_loss": None,
                "val_accuracy": None,
                "val_loss": None,
            },
            "error_message": None,
        },
    )

    def progress_handler(epoch: int, metrics: dict) -> None:
        progress_data = {
            "current_epoch": epoch,
            "total_epochs": metrics["total_epochs"],
            "progress_percentage": metrics["progress_percentage"],
            "train_accuracy": metrics["train_accuracy"],
            "train_loss": metrics["train_loss"],
            "val_accuracy": metrics["val_accuracy"],
            "val_loss": metrics["val_loss"],
        }
        update_task_record(
            task_id,
            {
                "progress": progress_data,
                "status": "training",
            },
        )
        self.update_state(state="PROGRESS", meta=progress_data)
        send_progress_callback(task_id, epoch, metrics, request_payload.get("callback_url"))
        send_external_epoch_callback(task_id, epoch, metrics)

    def stop_checker() -> bool:
        return is_stop_requested(task_id)

    result = run_training_loop(
        task_id=task_id,
        request_payload=request_payload,
        progress_handler=progress_handler,
        stop_requested=stop_checker,
    )

    clear_stop_request(task_id)

    status = result["status"]
    logger.info("Task {} completed with status {}", task_id, status)

    if status == "completed":
        artifacts = {
            "model_path": result["model_path"],
            "label_mapping_path": result["label_mapping_path"],
        }
        update_task_record(
            task_id,
            {
                "status": "completed",
                "completed_at": iso_now(),
                "progress": {
                    "current_epoch": request_payload["hyperparameters"]["epochs"],
                    "total_epochs": request_payload["hyperparameters"]["epochs"],
                    "progress_percentage": 100.0,
                    "train_accuracy": None,
                    "train_loss": None,
                    "val_accuracy": None,
                    "val_loss": None,
                },
                "artifacts": artifacts,
            },
        )
        send_external_publish_callback(task_id, publish_result=1, failure_message=None)
        return {
            "status": "completed",
            "artifacts": artifacts,
            "best_val_accuracy": result.get("best_val_accuracy"),
        }

    update_task_record(
        task_id,
        {
            "status": "stopped",
            "completed_at": iso_now(),
        },
    )
    send_external_publish_callback(task_id, publish_result=0, failure_message="Training stopped by request")
    self.update_state(state=states.REVOKED)
    return {"status": "stopped"}


__all__ = ["start_training_task"]
