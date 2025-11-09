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

### Automated Image Publishing

A GitHub Actions workflow (`.github/workflows/docker-build-push.yml`) builds the Docker image and pushes it to `crpi-lxfoqbwevmx9mc1q.cn-chengdu.personal.cr.aliyuncs.com/yuyi_tech/dianxin_auto_train` on every push to `main`, any `v*` tag, or manual dispatch. Define the secrets `ALIYUN_REGISTRY_USERNAME` / `ALIYUN_REGISTRY_PASSWORD` in the repository so the workflow can log in. Each run publishes `latest` (for the default branch), a `sha-<git-sha>` tag, and any matching git tag.

### APIs
The service exposes REST endpoints under `/api/v1/training/tasks`. Key operations:
- `POST /training/tasks` – submit a training job (see `docs/api.md` for payload schema)
- `GET /training/tasks/{task_id}` – fetch task status and metrics
- `GET /training/tasks` – list tasks with optional status filtering
- `POST /training/tasks/{task_id}/stop` – request job cancellation
- `DELETE /training/tasks/{task_id}` – remove task metadata

### Callbacks
Each epoch publishes progress to any `callback_url` provided in the submission and to the external endpoints defined in `.env` (`EXTERNAL_CALLBACK_BASE_URL` / `EXTERNAL_PUBLISH_CALLBACK_URL`). See `docs/external_callback.md` for payload formats.
