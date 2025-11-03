from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from src.callbacks import send_external_status_change
from src.celery_app import celery_app
from src.schemas import (
    DeleteTaskResponse,
    StopTaskResponse,
    TaskProgress,
    TrainingTaskCreateRequest,
    TrainingTaskDetail,
    TrainingTaskListItem,
    TrainingTaskListResponse,
    TrainingTaskResponse,
)
from src.storage import (
    create_task_record,
    delete_task_record,
    iso_now,
    list_task_records,
    load_task_record,
    set_stop_request,
    update_task_record,
)
from src.tasks import start_training_task


router = APIRouter(prefix="/training/tasks", tags=["training"])


def serialize_progress(progress: dict | None) -> TaskProgress | None:
    if progress is None:
        return None
    return TaskProgress(**progress)


def serialize_detail(record: dict) -> TrainingTaskDetail:
    return TrainingTaskDetail(
        task_id=record["task_id"],
        status=record["status"],
        model_name_cn=record["model_name_cn"],
        model_name_en=record["model_name_en"],
        created_at=record["created_at"],
        started_at=record.get("started_at"),
        completed_at=record.get("completed_at"),
        updated_at=record["updated_at"],
        progress=serialize_progress(record.get("progress")),
        error_message=record.get("error_message"),
        artifacts=record.get("artifacts"),
    )


def serialize_list_item(record: dict) -> TrainingTaskListItem:
    return TrainingTaskListItem(
        task_id=record["task_id"],
        status=record["status"],
        model_name_cn=record["model_name_cn"],
        model_name_en=record["model_name_en"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


@router.post("", response_model=TrainingTaskResponse)
def create_training_task(payload: TrainingTaskCreateRequest) -> TrainingTaskResponse:
    task_id = uuid4().hex
    created_at = iso_now()
    request_data = payload.model_dump(mode="json")
    record = {
        "task_id": task_id,
        "status": "queued",
        "model_name_cn": payload.model_name_cn,
        "model_name_en": payload.model_name_en,
        "training_data_file": payload.training_data_file,
        "base_model": payload.base_model,
        "callback_url": request_data.get("callback_url"),
        "created_at": created_at,
        "started_at": None,
        "completed_at": None,
        "updated_at": created_at,
        "progress": None,
        "error_message": None,
        "artifacts": None,
        "request_payload": request_data,
        "celery_task_id": None,
    }
    create_task_record(record)
    send_external_status_change(task_id, "queued")
    celery_task_id = f"training-{task_id}"
    start_training_task.apply_async(args=(task_id,), task_id=celery_task_id)
    update_task_record(task_id, {"celery_task_id": celery_task_id})

    return TrainingTaskResponse(
        task_id=task_id,
        status="queued",
        created_at=created_at,
        message="训练任务已提交，正在排队等待处理",
    )


@router.get("/{task_id}", response_model=TrainingTaskDetail)
def get_training_task(task_id: str) -> TrainingTaskDetail:
    record = load_task_record(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return serialize_detail(record)


@router.get("", response_model=TrainingTaskListResponse)
def list_training_tasks(
    status: str | None = Query(default=None, description="任务状态筛选"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> TrainingTaskListResponse:
    records = list_task_records()
    if status is not None:
        records = [record for record in records if record["status"] == status]
    total = len(records)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = records[start:end]
    tasks = [serialize_list_item(record) for record in page_items]
    return TrainingTaskListResponse(total=total, page=page, page_size=page_size, tasks=tasks)


@router.post("/{task_id}/stop", response_model=StopTaskResponse)
def stop_training_task(task_id: str) -> StopTaskResponse:
    record = load_task_record(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if record["status"] in {"completed", "failed", "stopped"}:
        return StopTaskResponse(task_id=task_id, status=record["status"], message="任务已结束")

    set_stop_request(task_id)
    celery_task_id = record.get("celery_task_id")
    if celery_task_id is not None:
        celery_app.control.revoke(celery_task_id, terminate=True)
    update_task_record(task_id, {"status": "stopped"})
    return StopTaskResponse(task_id=task_id, status="stopped", message="训练任务已停止")


@router.delete("/{task_id}", response_model=DeleteTaskResponse)
def delete_training_task(task_id: str) -> DeleteTaskResponse:
    record = load_task_record(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    celery_task_id = record.get("celery_task_id")
    if celery_task_id is not None:
        celery_app.control.revoke(celery_task_id, terminate=True)
    delete_task_record(task_id)
    return DeleteTaskResponse(task_id=task_id, message="训练任务已删除")
