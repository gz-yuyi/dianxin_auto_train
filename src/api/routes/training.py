"""训练任务路由"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime
from src.api.models.requests import CreateTrainingTaskRequest, TaskQueryParams
from src.api.models.responses import (
    CreateTrainingTaskResponse, 
    TrainingTaskResponse, 
    TaskListResponse,
    SimpleTaskResponse,
    ErrorResponse
)
from src.services.training import training_service
from src.core.constants import TaskStatus
from src.utils.logger import logger

router = APIRouter(prefix="/training", tags=["training"])

def task_to_response(task) -> TrainingTaskResponse:
    """将任务模型转换为响应模型"""
    return TrainingTaskResponse(
        task_id=task.task_id,
        status=task.status,
        model_name_cn=task.model_name_cn,
        model_name_en=task.model_name_en,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        progress=task.progress if task.status in [TaskStatus.TRAINING, TaskStatus.COMPLETED] else None,
        error_message=task.error_message
    )

@router.post("/tasks", response_model=CreateTrainingTaskResponse)
async def create_training_task(request: CreateTrainingTaskRequest):
    """创建训练任务"""
    logger.info(f"收到创建训练任务请求 - 模型名称: {request.model_name_cn}")
    
    try:
        # 创建任务
        task_data = request.model_dump()
        task = training_service.create_task(task_data)
        
        # 异步开始训练
        asyncio.create_task(training_service.start_training(task.task_id))
        
        return CreateTrainingTaskResponse(
            task_id=task.task_id,
            status=task.status,
            created_at=task.created_at,
            message="训练任务已提交，正在排队等待处理"
        )
        
    except ValueError as e:
        logger.error(f"创建训练任务失败 - 错误: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建训练任务时发生未知错误 - 错误: {e}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

@router.get("/tasks/{task_id}", response_model=TrainingTaskResponse)
async def get_task_status(task_id: str):
    """查询任务状态"""
    logger.info(f"查询任务状态 - 任务ID: {task_id}")
    
    task = training_service.get_task(task_id)
    if not task:
        logger.warning(f"任务不存在 - 任务ID: {task_id}")
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return task_to_response(task)

@router.get("/tasks", response_model=TaskListResponse)
async def get_task_list(
    status: Optional[str] = Query(None, description="按状态筛选"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量")
):
    """获取任务列表"""
    logger.info(f"获取任务列表 - 状态: {status}, 页码: {page}, 每页数量: {page_size}")
    
    # 获取所有任务
    all_tasks = training_service.get_all_tasks(status)
    
    # 分页
    total = len(all_tasks)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    tasks_page = all_tasks[start_idx:end_idx]
    
    # 转换为响应模型
    task_responses = [task_to_response(task) for task in tasks_page]
    
    return TaskListResponse(
        total=total,
        page=page,
        page_size=page_size,
        tasks=task_responses
    )

@router.post("/tasks/{task_id}/stop", response_model=SimpleTaskResponse)
async def stop_training_task(task_id: str):
    """停止训练任务"""
    logger.info(f"停止训练任务 - 任务ID: {task_id}")
    
    success = training_service.stop_task(task_id)
    if not success:
        logger.warning(f"停止任务失败 - 任务ID: {task_id}")
        raise HTTPException(status_code=400, detail="无法停止任务，任务可能不存在或状态不允许")
    
    task = training_service.get_task(task_id)
    return SimpleTaskResponse(
        task_id=task_id,
        status=task.status,
        message="训练任务已停止"
    )

@router.delete("/tasks/{task_id}", response_model=SimpleTaskResponse)
async def delete_training_task(task_id: str):
    """删除训练任务"""
    logger.info(f"删除训练任务 - 任务ID: {task_id}")
    
    success = training_service.delete_task(task_id)
    if not success:
        logger.warning(f"删除任务失败 - 任务ID: {task_id}")
        raise HTTPException(status_code=400, detail="无法删除任务，任务可能不存在或正在训练中")
    
    return SimpleTaskResponse(
        task_id=task_id,
        status=TaskStatus.STOPPED,
        message="训练任务已删除"
    )

# 导入asyncio用于创建异步任务
import asyncio