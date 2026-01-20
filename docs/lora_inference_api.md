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

## 状态说明

| 状态 | 说明 |
|------|------|
| loaded | 模型已加载 |
| unloaded | 模型已卸载 |
