"""响应数据模型"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from src.core.constants import TaskStatus

class TaskProgress(BaseModel):
    """任务进度信息"""
    current_epoch: Optional[int] = Field(default=None, description="当前轮次")
    total_epochs: Optional[int] = Field(default=None, description="总轮次")
    progress_percentage: Optional[float] = Field(default=None, description="进度百分比")
    train_accuracy: Optional[float] = Field(default=None, description="训练准确率")
    train_loss: Optional[float] = Field(default=None, description="训练损失")
    val_accuracy: Optional[float] = Field(default=None, description="验证准确率")
    val_loss: Optional[float] = Field(default=None, description="验证损失")

class TrainingTaskResponse(BaseModel):
    """训练任务响应"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    model_name_cn: Optional[str] = Field(default=None, description="模型中文名称")
    model_name_en: Optional[str] = Field(default=None, description="模型英文名称")
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(default=None, description="开始时间")
    completed_at: Optional[datetime] = Field(default=None, description="完成时间")
    progress: Optional[TaskProgress] = Field(default=None, description="训练进度")
    error_message: Optional[str] = Field(default=None, description="错误信息")

class CreateTrainingTaskResponse(BaseModel):
    """创建训练任务响应"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    created_at: datetime = Field(..., description="创建时间")
    message: str = Field(..., description="响应消息")

class TaskListResponse(BaseModel):
    """任务列表响应"""
    total: int = Field(..., description="总任务数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    tasks: List[TrainingTaskResponse] = Field(..., description="任务列表")

class SimpleTaskResponse(BaseModel):
    """简单任务响应"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    message: str = Field(..., description="响应消息")

class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    detail: Optional[str] = Field(default=None, description="详细错误信息")