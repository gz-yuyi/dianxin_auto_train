"""任务数据模型"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import uuid4
from src.core.constants import TaskStatus

@dataclass
class TaskProgress:
    """任务进度"""
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    progress_percentage: Optional[float] = None
    train_accuracy: Optional[float] = None
    train_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_loss: Optional[float] = None

@dataclass
class TrainingTask:
    """训练任务模型"""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    status: TaskStatus = TaskStatus.QUEUED
    model_name_cn: Optional[str] = None
    model_name_en: Optional[str] = None
    training_data_file: Optional[str] = None
    base_model: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    callback_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "model_name_cn": self.model_name_cn,
            "model_name_en": self.model_name_en,
            "training_data_file": self.training_data_file,
            "base_model": self.base_model,
            "hyperparameters": self.hyperparameters,
            "callback_url": self.callback_url,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": {
                "current_epoch": self.progress.current_epoch,
                "total_epochs": self.progress.total_epochs,
                "progress_percentage": self.progress.progress_percentage,
                "train_accuracy": self.progress.train_accuracy,
                "train_loss": self.progress.train_loss,
                "val_accuracy": self.progress.val_accuracy,
                "val_loss": self.progress.val_loss
            },
            "error_message": self.error_message
        }
    
    def update_status(self, status: TaskStatus):
        """更新任务状态"""
        self.status = status
        if status == TaskStatus.TRAINING and not self.started_at:
            self.started_at = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED]:
            self.completed_at = datetime.now()
    
    def update_progress(self, progress_data: Dict[str, Any]):
        """更新训练进度"""
        for key, value in progress_data.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)