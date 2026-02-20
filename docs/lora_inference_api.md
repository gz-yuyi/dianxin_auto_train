# LoRA 推理服务接口文档

## 基础信息
- 基础URL: `http://localhost:8000/api/v1`
- 数据格式: JSON
- 认证方式: 暂无（后续可添加API Key认证）
- 范围: 仅支持 LoRA 方式；所有模型共用同一 base model，不支持全量微调模型

## 接口列表

### 1. 加载 LoRA 模型
**POST** `/inference/models/load`

**请求参数**:
```json
{
    "model_dir": "local-fc525ca520d745daa38c0dd0fdd0c5d7",
    "max_length": 512
}
```

**参数说明**:
- `model_dir`: 模型目录名称（训练任务输出目录名）
- `max_length`: 推理最大长度（可选，默认 512）

**响应**:
```json
{
    "model_id": "local-fc525ca520d745daa38c0dd0fdd0c5d7",
    "status": "loaded",
    "message": "model loaded"
}
```

**备注**:
- 服务端会根据训练输出的命名规则动态拼装路径：
  - LoRA adapter: `<model_dir>/<model_stem>.lora`
  - 分类头: `<model_dir>/<model_stem>.head.pt`
  - 标签映射: `<model_dir>/<model_stem>.head.pt.pkl`
- `model_stem` 对应训练时的 `model_name_en`（去除 `.pt` 后缀）。
- 如果 `model_id` 已存在，可返回 `status: "loaded"`（幂等加载）或覆盖加载（实现选其一）。

### 2. 卸载 LoRA 模型
**POST** `/inference/models/unload`

**请求参数**:
```json
{
    "model_id": "yangyan_v3"
}
```

**响应**:
```json
{
    "model_id": "yangyan_v3",
    "status": "unloaded",
    "message": "model unloaded"
}
```

**备注**:
- 仅卸载 LoRA adapter 与分类头，不影响 base model 常驻。

### 3. 分类推理
**POST** `/inference/predict`

**请求参数**:
```json
{
    "model_id": "yangyan_v3",
    "texts": [
        "这是第一条文本",
        "这是第二条文本"
    ],
    "top_n": 3,
    "batch_size": 16
}
```

**响应**:
```json
{
    "model_id": "yangyan_v3",
    "labels": [
        "类别A",
        "类别B"
    ],
    "top_n": [
        [
            ["类别A", 0.91],
            ["类别C", 0.06],
            ["类别B", 0.03]
        ],
        [
            ["类别B", 0.85],
            ["类别A", 0.10],
            ["类别C", 0.05]
        ]
    ],
    "label_probabilities": [
        {
            "类别A": 0.91,
            "类别B": 0.03,
            "类别C": 0.06
        },
        {
            "类别A": 0.10,
            "类别B": 0.85,
            "类别C": 0.05
        }
    ]
}
```

**参数说明**:
- `model_id`: 已加载的 LoRA 模型标识
- `texts`: 待分类文本数组
- `top_n`: 返回前 N 个类别（可选，默认 3）
- `batch_size`: 推理批大小（可选，默认 16）

## 错误返回示例
```json
{
    "error": "model_not_loaded",
    "message": "model_id yangyan_v3 is not loaded"
}
```

### 4. 可用模型列表
**GET** `/inference/models`

**响应**:
```json
{
    "models": [
        {
            "model_id": "local-fc525ca520d745daa38c0dd0fdd0c5d7",
            "status": "loaded",
            "gpu_id": 0,
            "uptime_seconds": 3600
        },
        {
            "model_id": "yangyan_v3",
            "status": "loaded",
            "gpu_id": 1,
            "uptime_seconds": 1800
        },
        {
            "model_id": "test_model_v2",
            "status": "unloaded",
            "gpu_id": null,
            "uptime_seconds": null
        }
    ],
    "total": 3,
    "loaded_count": 2
}
```

**字段说明**:
- `models`: 模型列表
  - `model_id`: 模型标识
  - `status`: 模型状态 (`loaded`/`unloaded`)
  - `gpu_id`: 模型加载的 GPU ID（未加载时为 null）
  - `uptime_seconds`: 已运行时长（秒），未加载时为 null
- `total`: 模型总数
- `loaded_count`: 已加载模型数量

### 5. 根据模型名称列表查询模型
**POST** `/inference/models/query`

**请求参数**:
```json
{
    "model_ids": ["local-fc525ca520d745daa38c0dd0fdd0c5d7", "yangyan_v3"]
}
```

**参数说明**:
- `model_ids`: 要查询的模型 ID 列表（必填）

**响应**:
与「可用模型列表」接口格式相同，但只返回查询列表中存在的模型：
```json
{
    "models": [
        {
            "model_id": "local-fc525ca520d745daa38c0dd0fdd0c5d7",
            "status": "loaded",
            "gpu_id": 0,
            "uptime_seconds": 3600
        },
        {
            "model_id": "yangyan_v3",
            "status": "loaded",
            "gpu_id": 1,
            "uptime_seconds": 1800
        }
    ],
    "total": 2,
    "loaded_count": 2
}
```

**错误返回**:
```json
{
    "error": "invalid_request",
    "message": "model_ids must not be empty"
}
```

### 6. 推理服务状态查询
**GET** `/inference/status`

**响应**:
```json
{
    "service_status": "running",
    "workers": [
        {
            "worker_id": "cuda0-0",
            "device": "cuda:0",
            "total_memory_mb": 24576,
            "used_memory_mb": 8192,
            "free_memory_mb": 16384,
            "memory_usage_percent": 33.3
        },
        {
            "worker_id": "cuda1-0",
            "device": "cuda:1",
            "total_memory_mb": 24576,
            "used_memory_mb": 4096,
            "free_memory_mb": 20480,
            "memory_usage_percent": 16.7
        }
    ],
    "total_workers": 2,
    "loaded_models_count": 2,
    "pending_requests": 5
}
```

**字段说明**:
- `service_status`: 服务状态 (`running`/`starting`/`error`)
- `workers`: Worker 列表
  - `worker_id`: Worker 标识
  - `device`: 设备名称
  - `total_memory_mb`: 总显存（MB）
  - `used_memory_mb`: 已用显存（MB）
  - `free_memory_mb`: 剩余显存（MB）
  - `memory_usage_percent`: 显存使用率（%）
- `total_workers`: Worker 总数
- `loaded_models_count`: 已加载模型数量
- `pending_requests`: 待处理请求数

## 状态说明

| 状态 | 说明 |
|------|------|
| loaded | 模型已加载 |
| unloaded | 模型已卸载 |
| loading | 模型加载中（可选状态）|
| unloading | 模型卸载中（可选状态）|
