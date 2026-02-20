from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class LoraConfig(BaseModel):
    enabled: bool = Field(False, description="Enable LoRA fine-tuning")
    r: int = Field(8, ge=1, description="LoRA rank")
    lora_alpha: float = Field(16, gt=0, description="LoRA alpha")
    lora_dropout: float = Field(0.1, ge=0.0, lt=1.0, description="LoRA dropout")
    target_modules: list[str] | str = Field(
        default_factory=lambda: ["query", "value"],
        description="Target module names for LoRA injection, or 'all-linear'",
    )


class HyperParameters(BaseModel):
    learning_rate: float = Field(3e-5, description="Learning rate for optimizer")
    epochs: int = Field(5, ge=1, description="Number of training epochs")
    batch_size: int = Field(64, ge=1, description="Batch size for training")
    max_sequence_length: int = Field(512, ge=8, description="Maximum token length")
    precision: str = Field("fp32", description="Training precision: fp32, fp16, or bf16")
    gradient_accumulation_steps: int = Field(1, ge=1, description="Number of steps to accumulate gradients")
    early_stopping_enabled: bool = Field(False, description="Enable early stopping")
    early_stopping_patience: int = Field(3, ge=1, description="Epochs without improvement before stopping")
    early_stopping_min_delta: float = Field(0.0, ge=0.0, description="Minimum change to count as improvement")
    early_stopping_metric: str = Field(
        "val_accuracy",
        description="Metric for early stopping: val_accuracy, val_loss, f1_score",
    )
    random_seed: int = Field(1999, description="Random seed for reproducibility")
    train_val_split: float = Field(0.2, ge=0.0, lt=1.0, description="Holdout ratio (0 means no validation split)")
    text_column: str = Field(..., description="Name of text column in dataset")
    label_column: str = Field(..., description="Name of label column in dataset")
    sheet_name: str | None = Field(None, description="Excel sheet name if applicable")
    validation_sheet_name: str | None = Field(None, description="Validation Excel sheet name if applicable")
    lora: LoraConfig | None = Field(None, description="Optional LoRA configuration")


class TrainingTaskCreateRequest(BaseModel):
    model_name_cn: str
    model_name_en: str
    training_data_file: str
    validation_data_file: str | None = None
    base_model: str = "bert-base-chinese"
    hyperparameters: HyperParameters
    callback_url: HttpUrl | None = Field(None, description="Optional callback endpoint for progress notifications")


class TaskProgress(BaseModel):
    current_epoch: int | None = None
    total_epochs: int | None = None
    current_batch: int | None = None
    total_batches: int | None = None
    progress_percentage: float | None = None
    train_accuracy: float | None = None
    train_loss: float | None = None
    val_accuracy: float | None = None
    val_loss: float | None = None


class TrainingTaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: str
    message: str


class TrainingTaskDetail(BaseModel):
    task_id: str
    status: str
    model_name_cn: str
    model_name_en: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str
    progress: TaskProgress | None = None
    error_message: str | None = None
    artifacts: dict[str, Any] | None = None


class TrainingTaskListItem(BaseModel):
    task_id: str
    status: str
    model_name_cn: str
    model_name_en: str
    created_at: str
    updated_at: str


class TrainingTaskListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    tasks: list[TrainingTaskListItem]


class StopTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class DeleteTaskResponse(BaseModel):
    task_id: str
    message: str


class LoraModelLoadRequest(BaseModel):
    model_dir: str
    max_length: int = Field(512, ge=8, description="Maximum token length for inference")


class LoraModelLoadResponse(BaseModel):
    model_id: str
    status: str
    message: str


class LoraModelUnloadRequest(BaseModel):
    model_id: str


class LoraModelUnloadResponse(BaseModel):
    model_id: str
    status: str
    message: str


class LoraPredictRequest(BaseModel):
    model_id: str
    texts: list[str]
    top_n: int = Field(3, ge=1, description="Return top-N labels")
    batch_size: int = Field(16, ge=1, description="Requested batch size (best effort)")


class LoraPredictResponse(BaseModel):
    model_id: str
    labels: list[str]
    top_n: list[list[tuple[str, float]]]
    label_probabilities: list[dict[str, float]]


class ModelInfo(BaseModel):
    """模型信息"""
    model_id: str
    status: str = Field(..., description="Model status: loaded/unloaded")
    gpu_id: int | None = Field(None, description="GPU ID where model is loaded")
    uptime_seconds: float | None = Field(None, description="Uptime in seconds if loaded")


class ModelListResponse(BaseModel):
    """模型列表响应"""
    models: list[ModelInfo]
    total: int
    loaded_count: int


class ModelQueryRequest(BaseModel):
    """模型查询请求"""
    model_ids: list[str] = Field(..., min_length=1, description="List of model IDs to query")


class WorkerStatus(BaseModel):
    """Worker 状态信息"""
    worker_id: str
    device: str
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    memory_usage_percent: float


class InferenceServiceStatusResponse(BaseModel):
    """推理服务状态响应"""
    service_status: str = Field(..., description="Service status: running/starting/error")
    workers: list[WorkerStatus]
    total_workers: int
    loaded_models_count: int
    pending_requests: int


__all__ = [
    "DeleteTaskResponse",
    "HyperParameters",
    "InferenceServiceStatusResponse",
    "LoraConfig",
    "LoraModelLoadRequest",
    "LoraModelLoadResponse",
    "LoraModelUnloadRequest",
    "LoraModelUnloadResponse",
    "LoraPredictRequest",
    "LoraPredictResponse",
    "ModelInfo",
    "ModelListResponse",
    "ModelQueryRequest",
    "StopTaskResponse",
    "TaskProgress",
    "TrainingTaskCreateRequest",
    "TrainingTaskDetail",
    "TrainingTaskListItem",
    "TrainingTaskListResponse",
    "TrainingTaskResponse",
    "WorkerStatus",
]
