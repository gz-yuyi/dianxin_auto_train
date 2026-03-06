from uuid import uuid4

from fastapi import APIRouter, HTTPException, Path, Query

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


router = APIRouter(prefix="/training/tasks", tags=["训练任务"])


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


@router.post(
    "",
    response_model=TrainingTaskResponse,
    summary="创建训练任务",
    description="提交一个新的模型训练任务，任务会进入队列等待 Worker 处理。",
    response_description="训练任务提交结果",
)
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


@router.get(
    "/{task_id}",
    response_model=TrainingTaskDetail,
    summary="获取训练任务详情",
    description="根据任务 ID 查询训练任务的详细状态、时间信息和产物信息。",
    response_description="训练任务详情",
)
def get_training_task(
    task_id: str = Path(..., title="任务 ID", description="要查询的训练任务 ID"),
) -> TrainingTaskDetail:
    record = load_task_record(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return serialize_detail(record)


@router.get(
    "",
    response_model=TrainingTaskListResponse,
    summary="查询训练任务列表",
    description="按分页方式查询训练任务列表，支持按任务状态筛选。",
    response_description="训练任务列表",
)
def list_training_tasks(
    status: str | None = Query(default=None, title="任务状态", description="按任务状态筛选"),
    page: int = Query(default=1, ge=1, title="页码", description="分页页码，从 1 开始"),
    page_size: int = Query(default=20, ge=1, le=100, title="每页数量", description="每页返回的任务数量"),
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


@router.post(
    "/{task_id}/stop",
    response_model=StopTaskResponse,
    summary="停止训练任务",
    description="停止指定训练任务；如果任务已结束，则返回当前结束状态。",
    response_description="停止训练任务结果",
)
def stop_training_task(
    task_id: str = Path(..., title="任务 ID", description="要停止的训练任务 ID"),
) -> StopTaskResponse:
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


@router.delete(
    "/{task_id}",
    response_model=DeleteTaskResponse,
    summary="删除训练任务",
    description="删除指定训练任务记录；如果任务仍在运行，会先撤销对应的 Celery 任务。",
    response_description="删除训练任务结果",
)
def delete_training_task(
    task_id: str = Path(..., title="任务 ID", description="要删除的训练任务 ID"),
) -> DeleteTaskResponse:
    record = load_task_record(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    celery_task_id = record.get("celery_task_id")
    if celery_task_id is not None:
        celery_app.control.revoke(celery_task_id, terminate=True)
    delete_task_record(task_id)
    return DeleteTaskResponse(task_id=task_id, message="训练任务已删除")
