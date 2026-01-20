## 电信自动训练（Dianxin Auto Train）

基于 BERT + Celery 的文本分类自动微调流水线。

### 环境要求
- Python 3.12+
- Redis（Celery broker/result backend）
- GPU 可选；未提供时自动回退到 CPU 训练

### 安装
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # 如需安装 uv
uv sync
cp .env.example .env
```
按需修改 `.env` 中的 Redis、数据路径、回调地址和输出目录等配置。`uv` 管理的虚拟环境位于 `.venv`。

### 启动服务
先启动 Redis，然后在两个终端分别执行：

```bash
# 终端 1 – Celery worker
uv run python main.py worker

# 终端 2 – FastAPI 服务
uv run python main.py api --host 0.0.0.0 --port 8000
```

如需做端到端集成校验（API + worker 已启动）：

```bash
uv run python main.py check-service --host 127.0.0.1 --port 8000
```

Worker 会根据 `GPU_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` 自动设置并发；未设置时回退到单 CPU worker。每个 worker 进程绑定单张 GPU，保证单卡只跑一个训练任务。

### 本地训练（CLI）
无需 API 或 Celery worker，可直接使用 `train` 子命令进行本地训练；Payload 与 API 一致，默认不发送回调。

```bash
echo '{"model_name_cn":"数研测试_扬言类别(全量数据-v3)","model_name_en":"test_yangyan6","training_data_file":"3_扬言_2025-12-03_06-42-31_2025-12-17_03-45-18_2025-12-17_03-46-16.xlsx","base_model":"/app/models/bert-base-chinese","hyperparameters":{"learning_rate":3.0E-5,"epochs":5,"batch_size":64,"max_sequence_length":512,"random_seed":1999,"train_val_split":0.2,"text_column":"文本内容","label_column":"标签列"},"callback_url":"http://10.196.193.98:8089/service"}' | uv run python main.py train --payload -
```

如需回调，添加 `--callback`：

```bash
uv run python main.py train --payload-file payload.json --callback
```

### LoRA（可选）
要启用 LoRA（PEFT），在 `hyperparameters` 中加入 `lora` 配置：

```json
{
  "hyperparameters": {
    "lora": {
      "enabled": true,
      "r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.1,
      "target_modules": ["query", "value"]
    }
  }
}
```

启用 LoRA 时会输出 `<model_name_en>.lora` 目录与 `<model_name_en>.head.pt` 分类头（不会保存完整 `.pt` 权重）。推理时可用 `--lora-adapter` 指定 adapter 目录，或将其与 `.head.pt` 放在同一目录下。

### 使用 Docker 运行

可使用 `Dockerfile` 与 `docker-compose.yml` 启动完整栈（FastAPI、Celery worker、Redis）。

1. `cp .env.example .env` 并按需调整；Docker 环境下建议保持 Redis URL 默认值或改为 `redis://redis:6379/...`（compose 已配置）。
2. 将训练数据放到 `./data`，会挂载到容器内 `/app/data`；模型产物默认在 `./artifacts` 下持久化。
3. （可选）`.env` 中设置 `DX_IMAGE_TAG` 指定镜像版本，默认 `latest`。
4. 启动：
   ```bash
   docker compose up -d
   ```
   若更新 `DX_IMAGE_TAG`，建议执行 `docker compose pull` 拉取最新镜像。
5. API 地址：http://localhost:8000。可用 `docker compose logs -f api` 或 `docker compose logs -f worker` 查看日志。

停止服务：`docker compose down`（加 `-v` 会删除 Redis 数据卷）。

### GPU 加速（可选）

- 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 以支持容器访问 GPU。
- 复制 `.env.example` 为 `.env` 并调整 `DX_GPU_COUNT`、`NVIDIA_VISIBLE_DEVICES`、`NVIDIA_DRIVER_CAPABILITIES`。
- 使用 GPU 覆盖文件启动：
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
  ```
  若无 GPU，可不加覆盖文件，镜像会自动回退到 CPU。

### 离线部署

无外网环境可在在线机器生成离线包：

```bash
./scripts/build_offline_bundle.sh --output ./offline_bundle
```

脚本会拉取镜像并 `docker save` 到 `images/*.tar`，下载默认预训练模型到 `models/`，复制 `docker-compose.yml`、`docker-compose.gpu.yml`、`.env.example` 和文档到输出目录，并生成 `<bundle>.tar.gz`。将离线包拷贝到目标机器后依次 `docker load -i images/<name>.tar`，复制 `.env.example` 为 `.env` 并启动 `docker compose up -d`（有 GPU 可加 `-f docker-compose.gpu.yml`）。完整流程见 `docs/offline_deployment.md`。

### 自动镜像发布

仓库内 ` .github/workflows/docker-build-push.yml` 会在 `main` 分支、任意 `v*` tag、或手动触发时构建并推送镜像到：
`crpi-lxfoqbwevmx9mc1q.cn-chengdu.personal.cr.aliyuncs.com/yuyi_tech/dianxin_auto_train`

需要在仓库 secrets 中配置 `ALIYUN_REGISTRY_USERNAME` / `ALIYUN_REGISTRY_PASSWORD`。每次构建会推送 `latest`、`sha-<git-sha>` 以及 tag 对应版本。

### API
服务 API 位于 `/api/v1/training/tasks`：
- `POST /training/tasks` – 提交训练任务（详见 `docs/training_api.md`）
- `GET /training/tasks/{task_id}` – 查询任务状态与指标
- `GET /training/tasks` – 任务列表（支持状态过滤）
- `POST /training/tasks/{task_id}/stop` – 请求停止任务
- `DELETE /training/tasks/{task_id}` – 删除任务记录

### LoRA 推理服务
提供 LoRA 专用推理 API（基座模型共享、按模型目录加载 adapter）。详见 `docs/lora_inference_api.md`。

- 模型目录：训练输出目录名（例如 `artifacts/<task_id>`）。
- 命名规则：`<model_name_en>.lora` 与 `<model_name_en>.head.pt` 位于该目录。
- 主要接口：`POST /inference/models/load`、`POST /inference/predict`、`POST /inference/models/unload`。
- Worker 设置：`INFERENCE_BASE_MODEL`、`INFERENCE_WORKERS_PER_GPU`、`INFERENCE_MAX_BATCH_SIZE`。

### 回调
每个 epoch 会向 `callback_url` 推送进度；同时可向 `.env` 配置的外部回调地址推送（`EXTERNAL_CALLBACK_BASE_URL` / `EXTERNAL_PUBLISH_CALLBACK_URL`）。详见 `docs/external_callback.md`。

### 训练模型推理（CLI）

使用 `inference.py` 进行推理（不影响 legacy 脚本）：

- 多选一分类：
  ```bash
  uv run python inference.py multi-class \
    --excel legacy/环境保护_空气污染--样例1000.xlsx \
    --text-column 内容合并 \
    --model-path artifacts/<task_id>/<model_name>.pt \
    --label-mapping artifacts/<task_id>/<model_name>.pt.pkl \
    --base-model bert-base-chinese \
    --max-length 512 \
    --top-n 3 \
    --device auto
  ```
- 单标签是否判断：
  ```bash
  uv run python inference.py single-judge \
    --excel <input>.xlsx --text-column 内容合并 --label-column 标签列 \
    --model-path artifacts/<task_id>/<model_name>.pt \
    --label-mapping artifacts/<task_id>/<model_name>.pt.pkl \
    --top-k 2 --threshold 0.4
  ```
- 多标签判断（按列名前缀匹配多个标签列，如 `label_1`, `label_2`）：
  ```bash
  uv run python inference.py multi-judge \
    --excel <input>.xlsx --text-column 内容合并 --label-prefix label_ \
    --model-path artifacts/<task_id>/<model_name>.pt \
    --label-mapping artifacts/<task_id>/<model_name>.pt.pkl \
    --top-k 2 --threshold 0.4
  ```

所有命令都支持 `--sheet` 指定 Excel sheet，`--output` 设置输出文件名，`--device` 强制使用 `cpu` / `cuda:0`。默认 `max_length=512`、`base-model=bert-base-chinese`，可按训练配置调整。
