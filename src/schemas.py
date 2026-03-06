from typing import Any

from pydantic import BaseModel, Field, HttpUrl


class LoraConfig(BaseModel):
    enabled: bool = Field(False, title="是否启用 LoRA", description="是否启用 LoRA 微调")
    r: int = Field(8, ge=1, title="LoRA 秩", description="LoRA 分解秩")
    lora_alpha: float = Field(16, gt=0, title="LoRA Alpha", description="LoRA 缩放系数")
    lora_dropout: float = Field(0.1, ge=0.0, lt=1.0, title="LoRA Dropout", description="LoRA Dropout 比例")
    target_modules: list[str] | str = Field(
        default_factory=lambda: ["query", "value"],
        title="目标模块",
        description="LoRA 注入的目标模块名称，或填写 'all-linear'",
    )


class HyperParameters(BaseModel):
    learning_rate: float = Field(3e-5, title="学习率", description="优化器学习率")
    epochs: int = Field(5, ge=1, title="训练轮数", description="训练总轮数")
    batch_size: int = Field(64, ge=1, title="批大小", description="训练批大小")
    max_sequence_length: int = Field(512, ge=8, title="最大序列长度", description="输入文本的最大 Token 长度")
    precision: str = Field("fp32", title="训练精度", description="训练精度，可选 fp32、fp16 或 bf16")
    gradient_accumulation_steps: int = Field(1, ge=1, title="梯度累积步数", description="梯度累积的步数")
    early_stopping_enabled: bool = Field(False, title="是否启用早停", description="是否启用 Early Stopping")
    early_stopping_patience: int = Field(3, ge=1, title="早停耐心值", description="连续多少个 epoch 无提升后停止训练")
    early_stopping_min_delta: float = Field(0.0, ge=0.0, title="早停最小变化量", description="视为指标提升的最小变化量")
    early_stopping_metric: str = Field(
        "val_accuracy",
        title="早停指标",
        description="Early Stopping 使用的指标，可选 val_accuracy、val_loss、f1_score",
    )
    random_seed: int = Field(1999, title="随机种子", description="用于复现结果的随机种子")
    train_val_split: float = Field(0.2, ge=0.0, lt=1.0, title="验证集划分比例", description="从训练集划分验证集的比例，0 表示不划分")
    text_column: str = Field(..., title="文本列名", description="数据集中存放文本内容的列名")
    label_column: str = Field(..., title="标签列名", description="数据集中存放标签的列名")
    sheet_name: str | None = Field(None, title="训练集工作表名", description="当输入文件为 Excel 时使用的工作表名称")
    validation_sheet_name: str | None = Field(None, title="验证集工作表名", description="验证集 Excel 文件使用的工作表名称")
    lora: LoraConfig | None = Field(None, title="LoRA 配置", description="可选的 LoRA 微调配置")


class TrainingTaskCreateRequest(BaseModel):
    model_name_cn: str = Field(..., title="模型中文名称", description="训练后模型的中文名称")
    model_name_en: str = Field(..., title="模型英文名称", description="训练后模型的英文名称")
    training_data_file: str = Field(..., title="训练数据文件", description="训练数据文件路径")
    validation_data_file: str | None = Field(None, title="验证数据文件", description="可选的独立验证数据文件路径")
    base_model: str = Field("bert-base-chinese", title="基础模型", description="训练使用的基础预训练模型名称或路径")
    hyperparameters: HyperParameters = Field(..., title="超参数配置", description="训练任务的超参数配置")
    callback_url: HttpUrl | None = Field(None, title="回调地址", description="可选的训练进度回调地址")


class TaskProgress(BaseModel):
    current_epoch: int | None = Field(None, title="当前轮次", description="当前训练到的 epoch")
    total_epochs: int | None = Field(None, title="总轮次", description="训练总 epoch 数")
    current_batch: int | None = Field(None, title="当前批次", description="当前训练到的 batch")
    total_batches: int | None = Field(None, title="总批次", description="总 batch 数")
    progress_percentage: float | None = Field(None, title="进度百分比", description="训练整体进度百分比")
    train_accuracy: float | None = Field(None, title="训练准确率", description="当前训练准确率")
    train_loss: float | None = Field(None, title="训练损失", description="当前训练损失")
    val_accuracy: float | None = Field(None, title="验证准确率", description="当前验证准确率")
    val_loss: float | None = Field(None, title="验证损失", description="当前验证损失")


class TrainingTaskResponse(BaseModel):
    task_id: str = Field(..., title="任务 ID", description="训练任务 ID")
    status: str = Field(..., title="任务状态", description="训练任务当前状态")
    created_at: str = Field(..., title="创建时间", description="任务创建时间")
    message: str = Field(..., title="提示信息", description="接口返回的提示信息")


class TrainingTaskDetail(BaseModel):
    task_id: str = Field(..., title="任务 ID", description="训练任务 ID")
    status: str = Field(..., title="任务状态", description="训练任务当前状态")
    model_name_cn: str = Field(..., title="模型中文名称", description="模型中文名称")
    model_name_en: str = Field(..., title="模型英文名称", description="模型英文名称")
    created_at: str = Field(..., title="创建时间", description="任务创建时间")
    started_at: str | None = Field(None, title="开始时间", description="任务开始执行时间")
    completed_at: str | None = Field(None, title="完成时间", description="任务完成时间")
    updated_at: str = Field(..., title="更新时间", description="任务最近更新时间")
    progress: TaskProgress | None = Field(None, title="进度信息", description="训练进度详情")
    error_message: str | None = Field(None, title="错误信息", description="任务失败时的错误信息")
    artifacts: dict[str, Any] | None = Field(None, title="产物信息", description="训练产物路径等信息")


class TrainingTaskListItem(BaseModel):
    task_id: str = Field(..., title="任务 ID", description="训练任务 ID")
    status: str = Field(..., title="任务状态", description="训练任务当前状态")
    model_name_cn: str = Field(..., title="模型中文名称", description="模型中文名称")
    model_name_en: str = Field(..., title="模型英文名称", description="模型英文名称")
    created_at: str = Field(..., title="创建时间", description="任务创建时间")
    updated_at: str = Field(..., title="更新时间", description="任务最近更新时间")


class TrainingTaskListResponse(BaseModel):
    total: int = Field(..., title="总数", description="符合条件的任务总数")
    page: int = Field(..., title="页码", description="当前页码")
    page_size: int = Field(..., title="每页数量", description="当前每页数量")
    tasks: list[TrainingTaskListItem] = Field(..., title="任务列表", description="当前页任务列表")


class StopTaskResponse(BaseModel):
    task_id: str = Field(..., title="任务 ID", description="训练任务 ID")
    status: str = Field(..., title="任务状态", description="停止后的任务状态")
    message: str = Field(..., title="提示信息", description="停止任务结果说明")


class DeleteTaskResponse(BaseModel):
    task_id: str = Field(..., title="任务 ID", description="训练任务 ID")
    message: str = Field(..., title="提示信息", description="删除任务结果说明")


class LoraModelLoadRequest(BaseModel):
    model_dir: str = Field(..., title="模型目录", description="待加载模型所在目录")
    max_length: int = Field(512, ge=8, title="最大长度", description="推理时允许的最大 Token 长度")


class LoraModelLoadResponse(BaseModel):
    model_id: str = Field(..., title="模型 ID", description="加载后分配的模型 ID")
    status: str = Field(..., title="状态", description="模型加载状态")
    message: str = Field(..., title="提示信息", description="模型加载结果说明")


class LoraModelUnloadRequest(BaseModel):
    model_id: str = Field(..., title="模型 ID", description="要卸载的模型 ID")


class LoraModelUnloadResponse(BaseModel):
    model_id: str = Field(..., title="模型 ID", description="已卸载的模型 ID")
    status: str = Field(..., title="状态", description="模型卸载状态")
    message: str = Field(..., title="提示信息", description="模型卸载结果说明")


class LoraPredictRequest(BaseModel):
    model_id: str = Field(..., title="模型 ID", description="用于预测的模型 ID")
    texts: list[str] = Field(..., title="文本列表", description="待预测的文本列表")
    top_n: int = Field(3, ge=1, title="Top N", description="返回概率最高的前 N 个标签")
    batch_size: int = Field(16, ge=1, title="批大小", description="请求的推理批大小，实际执行时尽力满足")


class LoraPredictResponse(BaseModel):
    model_id: str = Field(..., title="模型 ID", description="执行预测所用的模型 ID")
    labels: list[str] = Field(..., title="预测标签", description="每条文本的最终预测标签")
    top_n: list[list[tuple[str, float]]] = Field(..., title="Top N 结果", description="每条文本的前 N 个候选标签及其概率")
    label_probabilities: list[dict[str, float]] = Field(..., title="标签概率", description="每条文本对应的各标签概率分布")


class ModelInfo(BaseModel):
    """模型信息"""
    model_id: str = Field(..., title="模型 ID", description="模型 ID")
    status: str = Field(..., title="状态", description="模型状态: loaded/unloaded")
    gpu_id: int | None = Field(None, title="GPU ID", description="模型加载的 GPU ID")
    uptime_seconds: float | None = Field(None, title="运行时长（秒）", description="模型加载时长（秒）")


class ModelListResponse(BaseModel):
    """模型列表响应"""
    models: list[ModelInfo] = Field(..., title="模型列表", description="模型列表")
    total: int = Field(..., title="模型总数", description="模型总数")
    loaded_count: int = Field(..., title="已加载数量", description="已加载模型数量")


class ModelQueryRequest(BaseModel):
    """模型查询请求"""
    model_ids: list[str] = Field(..., min_length=1, title="模型 ID 列表", description="要查询的模型 ID 列表")


class WorkerStatus(BaseModel):
    """Worker 状态信息"""
    worker_id: str = Field(..., title="Worker ID", description="Worker ID")
    device: str = Field(..., title="设备类型", description="设备类型")
    total_memory_mb: float = Field(..., title="总显存（MB）", description="总显存（MB）")
    used_memory_mb: float = Field(..., title="已用显存（MB）", description="已用显存（MB）")
    free_memory_mb: float = Field(..., title="空闲显存（MB）", description="空闲显存（MB）")
    memory_usage_percent: float = Field(..., title="显存使用率（%）", description="显存使用率（%）")


class InferenceServiceStatusResponse(BaseModel):
    """推理服务状态响应"""
    service_status: str = Field(..., title="服务状态", description="服务状态: running/starting/error")
    workers: list[WorkerStatus] = Field(..., title="Worker 列表", description="Worker 列表")
    total_workers: int = Field(..., title="Worker 总数", description="Worker 总数")
    loaded_models_count: int = Field(..., title="已加载模型数量", description="已加载模型数量")
    pending_requests: int = Field(..., title="待处理请求数", description="待处理请求数")


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
