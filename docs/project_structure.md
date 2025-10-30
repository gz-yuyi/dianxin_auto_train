# 项目目录结构设计

## 设计原则
1. **扁平化结构**: 层级尽量扁平，避免过深嵌套
2. **模块清晰**: 功能模块划分清晰，职责单一
3. **不过度拆分**: 避免文件过多，保持适度聚合
4. **可维护性**: 便于维护和扩展
5. **统一命令入口**: 所有命令通过main.py统一管理，使用click框架

## 目录结构

```
/Users/hanbing/yuyi/dianxin_auto_train/
├── main.py                    # 程序统一入口
├── pyproject.toml            # 项目依赖配置
├── README.md                 # 项目说明
├── .gitignore               # Git忽略文件
├── .python-version          # Python版本
├── docs/                    # 文档目录
│   ├── api_design.md        # API接口设计
│   ├── project_structure.md # 项目结构说明
│   └── deployment.md        # 部署文档
├── src/                     # 源代码目录
│   ├── __init__.py
│   ├── api/                 # API服务相关
│   │   ├── __init__.py
│   │   ├── app.py          # FastAPI应用实例
│   │   ├── routes/         # 路由模块
│   │   │   ├── __init__.py
│   │   │   ├── training.py # 训练任务相关路由
│   │   │   └── health.py   # 健康检查路由
│   │   └── models/         # Pydantic模型
│   │       ├── __init__.py
│   │       ├── requests.py # 请求数据模型
│   │       └── responses.py # 响应数据模型
│   ├── core/               # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── config.py       # 配置管理
│   │   ├── constants.py    # 常量定义
│   │   └── exceptions.py   # 自定义异常
│   ├── services/           # 业务服务层
│   │   ├── __init__.py
│   │   ├── training.py     # 训练任务服务
│   │   ├── callback.py     # 回调服务
│   │   └── storage.py      # 文件存储服务
│   ├── models/             # 数据模型
│   │   ├── __init__.py
│   │   ├── task.py         # 任务模型
│   │   └── training.py     # 训练相关模型
│   ├── ml/                 # 机器学习相关
│   │   ├── __init__.py
│   │   ├── trainer.py      # 训练器
│   │   ├── dataset.py      # 数据集处理
│   │   ├── model.py        # 模型定义
│   │   └── utils.py        # ML工具函数
│   ├── workers/            # Celery任务
│   │   ├── __init__.py
│   │   └── tasks.py        # 异步任务定义
│   └── utils/              # 工具函数
│       ├── __init__.py
│       ├── logger.py       # 日志工具
│       └── helpers.py      # 辅助函数
├── data/                   # 数据目录
│   ├── uploads/           # 上传的训练数据文件
│   └── models/            # 训练好的模型文件
├── logs/                   # 日志文件
└── tests/                  # 测试代码
    ├── __init__.py
    ├── test_api.py        # API测试
    └── test_services.py   # 服务测试
```

## 核心模块说明

### API模块 (src/api/)
- **app.py**: FastAPI应用实例，包含中间件、异常处理等
- **routes/**: 路由模块，按功能分组
- **models/**: Pydantic数据模型，定义请求和响应结构

### 核心模块 (src/core/)
- **config.py**: 配置管理，支持环境变量
- **constants.py**: 项目常量定义
- **exceptions.py**: 自定义异常类

### 服务模块 (src/services/)
- **training.py**: 训练任务的业务逻辑
- **callback.py**: 回调通知服务
- **storage.py**: 文件上传和存储管理

### 模型模块 (src/models/)
- **task.py**: 任务数据模型
- **training.py**: 训练相关数据模型

### 机器学习模块 (src/ml/)
- **trainer.py**: 训练流程封装
- **dataset.py**: 数据集处理逻辑
- **model.py**: BERT模型定义
- **utils.py**: ML相关工具函数

### 工作模块 (src/workers/)
- **tasks.py**: Celery异步任务定义

## 文件组织原则

1. **按功能聚合**: 相关功能放在同一模块中
2. **避免过度拆分**: 单个文件包含相关的一组功能
3. **清晰的命名**: 文件名清晰表达其用途
4. **统一入口**: 所有功能通过main.py统一管理，使用click命令行框架

## 依赖管理

使用`uv`作为包管理器，所有依赖在`pyproject.toml`中定义：

```toml
[project]
name = "bert-training-api"
version = "0.1.0"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "celery>=5.3.0",
    "redis>=5.0.0",
    "click>=8.1.0",
    "pandas>=2.0.0",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "scikit-learn>=1.3.0",
    "pydantic>=2.0.0",
    "python-multipart>=0.0.6",
    "aiofiles>=23.0.0",
]
```

## 启动方式

所有命令都通过main.py统一管理，使用click框架提供命令行接口：

### API服务
```bash
python main.py api
```

### Celery Worker
```bash
python main.py worker
```

### 数据库初始化
```bash
python main.py init-db
```

### 帮助信息
```bash
python main.py --help
```