## Dianxin Auto Train

Automatic fine-tuning pipeline for text classification models based on BERT and Celery.

### Requirements
- Python 3.12+
- Redis server (for Celery broker/result backend)
- GPU is optional; falls back to CPU training

### Installation
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv if needed
uv sync
cp .env.example .env
```
Adjust `.env` to point at Redis, data files, callback URLs, and output directories. The virtual environment managed by `uv` will live under `.venv`.

### Running the Services
Start a Redis instance, then in separate terminals:

```bash
# Terminal 1 – Celery worker
uv run python main.py worker

# Terminal 2 – FastAPI server
uv run python main.py api --host 0.0.0.0 --port 8000
```

To verify everything end-to-end, run the integration check once the API and worker are up:

```bash
uv run python main.py check-service --host 127.0.0.1 --port 8000
```

The worker automatically sets its concurrency from `GPU_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`; if these are unset it falls back to a single CPU worker. Each worker process pins itself to a single GPU so only one training task runs on any given GPU at a time.

### Local Training (CLI)
You can run training directly without the API or Celery worker by using the `train` subcommand. It accepts the same JSON payload as the API and defaults to no callbacks.

```bash
echo '{"model_name_cn":"数研测试_扬言类别(全量数据-v3)","model_name_en":"test_yangyan6","training_data_file":"3_扬言_2025-12-03_06-42-31_2025-12-17_03-45-18_2025-12-17_03-46-16.xlsx","base_model":"/app/models/bert-base-chinese","hyperparameters":{"learning_rate":3.0E-5,"epochs":5,"batch_size":64,"max_sequence_length":512,"random_seed":1999,"train_val_split":0.2,"text_column":"文本内容","label_column":"标签列"},"callback_url":"http://10.196.193.98:8089/service"}' | uv run python main.py train --payload -
```

To enable callbacks, add `--callback`:

```bash
uv run python main.py train --payload-file payload.json --callback
```

### LoRA (optional)
To use LoRA via PEFT, add a `lora` block under `hyperparameters`:

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

Training emits a `<model_name_en>.lora` adapter directory plus a `<model_name_en>.head.pt` classifier head when LoRA is enabled (full `.pt` weights are not saved). For inference, pass `--lora-adapter` or keep the adapter directory next to the `.head.pt`.

### Running with Docker

You can run the full stack (FastAPI, Celery worker, Redis) using the included `Dockerfile` and `docker-compose.yml`.

1. `cp .env.example .env` and adjust anything project-specific. When running under Docker, keep the Redis URLs at their defaults or override them to `redis://redis:6379/...` (the compose file already does this).
2. Place any training datasets under `./data` so they are mounted into the containers at `/app/data`. Model artifacts will persist under `./artifacts`.
3. (Optional) Set `DX_IMAGE_TAG` in `.env` if you need a specific published tag; it defaults to `latest`.
4. Start everything with:
   ```bash
   docker compose up -d
   ```
   Run `docker compose pull` whenever you bump `DX_IMAGE_TAG` to grab the latest pushed image.
5. The API becomes available at [http://localhost:8000](http://localhost:8000). Use `docker compose logs -f api` or `docker compose logs -f worker` to follow service logs.

Stop the stack with `docker compose down` (add `-v` to wipe the Redis volume if needed).

### GPU Acceleration (optional)

- Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host so Docker can access the GPU driver.
- Copy `.env.example` to `.env` and adjust `DX_GPU_COUNT`, `NVIDIA_VISIBLE_DEVICES`, and `NVIDIA_DRIVER_CAPABILITIES` if you need to expose a subset of GPUs.
- Start the stack with the GPU override together with the base compose file:
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
  ```
  Skip the override file if you are on a CPU-only host—the same image falls back to CPU execution automatically.

### Offline Deployment

For 运维 environments without Internet access, generate a self-contained bundle on an online machine:

```bash
./scripts/build_offline_bundle.sh --output ./offline_bundle
```

The script pulls the application + Redis images, exports them to `images/*.tar`, downloads the default pretrained model from ModelScope into `models/`, places `docker-compose.yml`, `docker-compose.gpu.yml`, and `.env.example` at the bundle root, and finally emits `<bundle>.tar.gz`. Transfer the archive to the offline host, run `docker load -i images/<name>.tar` for each image, copy `.env.example` to `.env`, and start the stack from the bundle directory with `docker compose up -d` (add `-f docker-compose.gpu.yml` if the offline host has NVIDIA GPUs). See `docs/offline_deployment.md` for the full workflow.

### Automated Image Publishing

A GitHub Actions workflow (`.github/workflows/docker-build-push.yml`) builds the Docker image and pushes it to `crpi-lxfoqbwevmx9mc1q.cn-chengdu.personal.cr.aliyuncs.com/yuyi_tech/dianxin_auto_train` on every push to `main`, any `v*` tag, or manual dispatch. Define the secrets `ALIYUN_REGISTRY_USERNAME` / `ALIYUN_REGISTRY_PASSWORD` in the repository so the workflow can log in. Each run publishes `latest` (for the default branch), a `sha-<git-sha>` tag, and any matching git tag.

### APIs
The service exposes REST endpoints under `/api/v1/training/tasks`. Key operations:
- `POST /training/tasks` – submit a training job (see `docs/training_api.md` for payload schema)
- `GET /training/tasks/{task_id}` – fetch task status and metrics
- `GET /training/tasks` – list tasks with optional status filtering
- `POST /training/tasks/{task_id}/stop` – request job cancellation
- `DELETE /training/tasks/{task_id}` – remove task metadata

### LoRA Inference Service
The service provides a LoRA-only inference API that loads adapters by model directory name and routes requests
through a shared base model. See `docs/lora_inference_api.md` for the full schema.

- Model directory: use the training output directory name (for example the `task_id` under `artifacts/`).
- Adapter + head naming: `<model_name_en>.lora` and `<model_name_en>.head.pt` under that directory.
- Key endpoints: `POST /inference/models/load`, `POST /inference/predict`, `POST /inference/models/unload`.
- Worker settings: `INFERENCE_BASE_MODEL`, `INFERENCE_WORKERS_PER_GPU`, `INFERENCE_MAX_BATCH_SIZE`.

### Callbacks
Each epoch publishes progress to any `callback_url` provided in the submission and to the external endpoints defined in `.env` (`EXTERNAL_CALLBACK_BASE_URL` / `EXTERNAL_PUBLISH_CALLBACK_URL`). See `docs/external_callback.md` for payload formats.

### Inference for Trained Models

Use the new CLI script `inference.py` (does not change the legacy scripts) to run predictions with trained checkpoints:

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
- 多标签判断（按列名前缀匹配多个标签列，例如 `label_1`, `label_2`）：
  ```bash
  uv run python inference.py multi-judge \
    --excel <input>.xlsx --text-column 内容合并 --label-prefix label_ \
    --model-path artifacts/<task_id>/<model_name>.pt \
    --label-mapping artifacts/<task_id>/<model_name>.pt.pkl \
    --top-k 2 --threshold 0.4
  ```

All commands support `--sheet` to select a specific Excel sheet, `--output` to set the output file name, and `--device` to force `cpu` / `cuda:0`. The defaults keep `max_length=512` and `base-model=bert-base-chinese`; adjust them to match how the model was trained.
