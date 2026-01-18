from typing import Any, Literal

from pydantic import AnyHttpUrl, BaseModel, Field, HttpUrl


class HyperParameters(BaseModel):
    learning_rate: float = Field(3e-5, description="Learning rate for optimizer")
    epochs: int = Field(5, ge=1, description="Number of training epochs")
    batch_size: int = Field(64, ge=1, description="Batch size for training")
    max_sequence_length: int = Field(512, ge=8, description="Maximum token length")
    random_seed: int = Field(1999, description="Random seed for reproducibility")
    train_val_split: float = Field(0.2, ge=0.0, lt=1.0, description="Holdout ratio (0 means no validation split)")
    text_column: str = Field(..., description="Name of text column in dataset")
    label_column: str = Field(..., description="Name of label column in dataset")
    sheet_name: str | None = Field(None, description="Excel sheet name if applicable")


class EmbeddingConfig(BaseModel):
    base_url: AnyHttpUrl = Field(..., description="OpenAI-compatible embedding base URL")
    model: str = Field(..., description="Embedding model name")
    api_key: str | None = Field(None, description="Optional API key for embedding service")
    batch_size: int = Field(64, ge=1, description="Batch size for embedding requests")
    timeout: float = Field(60.0, gt=0.0, description="Embedding request timeout (seconds)")
    max_retries: int = Field(2, ge=0, description="Embedding request retry count")
    extra_headers: dict[str, str] | None = Field(None, description="Extra headers for embedding requests")


class TrainingTaskCreateRequest(BaseModel):
    model_name_cn: str
    model_name_en: str
    training_data_file: str
    base_model: str = "bert-base-chinese"
    training_mode: Literal["bert", "setfit"] = Field("bert", description="Training mode")
    embedding: EmbeddingConfig | None = Field(
        None, description="Embedding configuration for setfit training"
    )
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


__all__ = [
    "DeleteTaskResponse",
    "EmbeddingConfig",
    "HyperParameters",
    "StopTaskResponse",
    "TaskProgress",
    "TrainingTaskCreateRequest",
    "TrainingTaskDetail",
    "TrainingTaskListItem",
    "TrainingTaskListResponse",
    "TrainingTaskResponse",
]
