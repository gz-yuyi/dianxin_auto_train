# API接口设计文档

## 基础信息
- 基础URL: `http://localhost:8000/api/v1`
- 数据格式: JSON
- 认证方式: 暂无（后续可添加API Key认证）

## 接口列表

### 1. 提交训练任务
**POST** `/training/tasks`

**请求参数**:
```json
{
    "model_name_cn": "环境保护污染分类模型",
    "model_name_en": "environmental_pollution_classifier",
    "training_data_file": "环境保护_空气污染--样例1000.xlsx",
    "validation_data_file": "环境保护_空气污染--验证.xlsx",
    "base_model": "bert-base-chinese",
    "hyperparameters": {
        "learning_rate": 3e-5,
        "epochs": 10,
        "batch_size": 64,
        "max_sequence_length": 512,
        "early_stopping_enabled": false,
        "early_stopping_patience": 3,
        "early_stopping_min_delta": 0.0,
        "early_stopping_metric": "val_accuracy",
        "random_seed": 42,
        "train_val_split": 0.2,
        "text_column": "内容合并",
        "label_column": "标签列",
        "sheet_name": null,
        "validation_sheet_name": null,
        "lora": {
            "enabled": true,
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "target_modules": ["query", "value"]
        }
    },
    "callback_url": "http://example.com/training/callback"
}
```
> `validation_data_file` 可选，提供时将使用该文件作为验证集并忽略 `train_val_split`。
> `early_stopping_enabled` 仅在有验证集时生效；`early_stopping_metric` 支持 `val_accuracy`、`val_loss`、`f1_score`。
> `lora` 为可选配置，启用后会使用 LoRA 训练；任务完成后 `artifacts` 会额外返回 `lora_adapter_path` 与 `classifier_head_path`。

**响应**:
```json
{
    "task_id": "task_123456",
    "status": "queued",
    "created_at": "2024-01-01T12:00:00Z",
    "message": "训练任务已提交，正在排队等待处理"
}
```

### 2. 查询任务状态
**GET** `/training/tasks/{task_id}`

**响应**:
```json
{
    "task_id": "task_123456",
    "status": "training",
    "model_name_cn": "环境保护污染分类模型",
    "model_name_en": "environmental_pollution_classifier",
    "created_at": "2024-01-01T12:00:00Z",
    "started_at": "2024-01-01T12:01:00Z",
    "progress": {
        "current_epoch": 8,
        "total_epochs": 10,
        "progress_percentage": 80,
        "train_accuracy": 0.98,
        "train_loss": 0.02,
        "val_accuracy": 0.95,
        "val_loss": 0.05
    },
    "error_message": null
}
```

### 3. 获取任务列表
**GET** `/training/tasks`

**查询参数**:
- `status`: 按状态筛选 (queued, training, completed, failed, stopped)
- `page`: 页码，默认1
- `page_size`: 每页数量，默认20

**响应**:
```json
{
    "total": 100,
    "page": 1,
    "page_size": 20,
    "tasks": [
        {
            "task_id": "task_123456",
            "status": "training",
            "model_name_cn": "环境保护污染分类模型",
            "model_name_en": "environmental_pollution_classifier",
            "created_at": "2024-01-01T12:00:00Z",
            "updated_at": "2024-01-01T12:05:00Z"
        }
    ]
}
```

### 4. 停止训练任务
**POST** `/training/tasks/{task_id}/stop`

**响应**:
```json
{
    "task_id": "task_123456",
    "status": "stopped",
    "message": "训练任务已停止"
}
```

### 5. 删除训练任务
**DELETE** `/training/tasks/{task_id}`

**响应**:
```json
{
    "task_id": "task_123456",
    "message": "训练任务已删除"
}
```

## 状态说明

| 状态 | 说明 |
|------|------|
| queued | 任务已提交，等待处理 |
| training | 任务正在训练中 |
| completed | 训练完成 |
| failed | 训练失败 |
| stopped | 任务被手动停止 |
