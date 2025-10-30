# BERT训练API服务

基于BERT模型的文本分类训练API服务，支持异步训练任务管理。

## 功能特性

- 🚀 异步训练任务管理
- 📊 训练进度实时监控
- 🎯 基于BERT的文本分类
- 📁 文件上传和管理
- 🔧 可配置的超参数
- 📡 训练完成回调通知
- 📝 完整的RESTful API

## 项目结构

```
├── src/                      # 源代码目录
│   ├── api/                  # API服务相关
│   │   ├── app.py           # FastAPI应用实例
│   │   ├── routes/          # 路由模块
│   │   └── models/          # Pydantic数据模型
│   ├── core/                # 核心业务逻辑
│   │   ├── config.py        # 配置管理
│   │   └── constants.py     # 常量定义
│   ├── services/            # 业务服务层
│   │   ├── training.py      # 训练任务服务
│   │   └── storage.py       # 文件存储服务
│   ├── models/              # 数据模型
│   │   └── task.py          # 任务模型
│   ├── ml/                  # 机器学习相关
│   │   ├── trainer.py       # 训练器
│   │   ├── dataset.py       # 数据集处理
│   │   └── model.py         # 模型定义
│   ├── workers/             # Celery任务
│   │   └── tasks.py         # 异步任务定义
│   └── utils/               # 工具函数
│       └── logger.py        # 日志工具
├── data/                    # 数据目录
│   ├── uploads/            # 上传的训练数据文件
│   └── models/             # 训练好的模型文件
├── logs/                    # 日志文件
└── tests/                   # 测试代码
```

## 安装依赖

使用 `uv` 包管理器安装依赖：

```bash
uv pip install -e .
```

或者使用 `pip`：

```bash
pip install -e .
```

## 快速开始

### 1. 启动Redis服务

确保Redis服务已启动：

```bash
redis-server
```

### 2. 启动API服务

```bash
python main.py api
```

API服务将在 `http://localhost:8000` 启动，文档地址： `http://localhost:8000/api/v1/docs`

### 3. 启动Celery Worker（可选）

如果需要使用Celery进行任务队列管理：

```bash
python main.py worker
```

## API接口

### 创建训练任务

**POST** `/api/v1/training/tasks`

```json
{
    "model_name_cn": "环境保护污染分类模型",
    "model_name_en": "environmental_pollution_classifier",
    "training_data_file": "环境保护_空气污染--样例1000.xlsx",
    "base_model": "bert-base-chinese",
    "hyperparameters": {
        "learning_rate": 3e-5,
        "epochs": 10,
        "batch_size": 64,
        "max_sequence_length": 512,
        "random_seed": 42,
        "train_val_split": 0.2,
        "text_column": "内容合并",
        "label_column": "标签列",
        "sheet_name": null
    },
    "callback_url": "http://example.com/training/callback"
}
```

### 查询任务状态

**GET** `/api/v1/training/tasks/{task_id}`

### 获取任务列表

**GET** `/api/v1/training/tasks`

### 停止训练任务

**POST** `/api/v1/training/tasks/{task_id}/stop`

### 删除训练任务

**DELETE** `/api/v1/training/tasks/{task_id}`

## 配置说明

主要配置项在 `src/core/config.py` 中：

- `API_HOST`: API服务主机地址，默认 `0.0.0.0`
- `API_PORT`: API服务端口，默认 `8000`
- `CELERY_BROKER_URL`: Celery消息代理URL
- `CELERY_RESULT_BACKEND`: Celery结果后端URL
- `MAX_CONCURRENT_TASKS`: 最大并发训练任务数

## 开发原则

1. **函数式编程优先**：避免滥用面向对象，优先使用函数式编程
2. **无异常处理**：当前阶段不做异常处理，错误直接抛出以便调试
3. **统一日志**：使用loguru进行日志记录
4. **模块化设计**：功能模块划分清晰，职责单一

## 测试

运行测试：

```bash
python main.py test
```

或者使用pytest：

```bash
pytest tests/
```

## 部署

### Docker部署

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install -e .

COPY . .

EXPOSE 8000

CMD ["python", "main.py", "api"]
```

### 生产环境

在生产环境中，建议：

1. 使用Gunicorn作为WSGI服务器
2. 配置Nginx作为反向代理
3. 使用Supervisor管理进程
4. 配置日志轮转
5. 设置适当的环境变量

## 注意事项

1. 确保训练数据文件格式正确（Excel文件）
2. 文本列和标签列名称需要与实际情况匹配
3. 训练过程会消耗较多计算资源，建议在GPU环境下运行
4. 模型文件会保存在 `data/models/` 目录下

## 许可证

MIT License