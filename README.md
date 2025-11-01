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
uv run python main.py worker --concurrency 1

# Terminal 2 – FastAPI server
uv run python main.py api --host 0.0.0.0 --port 8000
```

To verify everything end-to-end, run the integration check once the API and worker are up:

```bash
uv run python main.py check-service --host 127.0.0.1 --port 8000
```

### APIs
The service exposes REST endpoints under `/api/v1/training/tasks`. Key operations:
- `POST /training/tasks` – submit a training job (see `docs/api.md` for payload schema)
- `GET /training/tasks/{task_id}` – fetch task status and metrics
- `GET /training/tasks` – list tasks with optional status filtering
- `POST /training/tasks/{task_id}/stop` – request job cancellation
- `DELETE /training/tasks/{task_id}` – remove task metadata

### Callbacks
Each epoch publishes progress to any `callback_url` provided in the submission and to the external endpoints defined in `.env` (`EXTERNAL_CALLBACK_BASE_URL` / `EXTERNAL_PUBLISH_CALLBACK_URL`). See `docs/external_callback.md` for payload formats.
