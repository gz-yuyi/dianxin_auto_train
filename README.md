# BERT训练API服务

基于BERT模型的文本分类训练API服务，支持异步训练任务管理。

## Quick Start

### 1. 环境配置

```bash
# 复制示例配置文件
cp .env.example .env

# 编辑配置文件（根据需要修改）
nano .env
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 初始化数据库

```bash
uv run python main.py init-db
```

### 4. 启动服务

```bash
# 启动API服务
uv run python main.py api --reload

# 启动Celery Worker（可选）
uv run python main.py worker
```

### 5. 验证配置（可选）

```bash
# 检查配置是否正确加载
uv run python main.py check-config
```

服务将在 `http://localhost:8000` 启动。

## API文档

访问 `http://localhost:8000/api/v1/docs` 查看完整的API文档。
