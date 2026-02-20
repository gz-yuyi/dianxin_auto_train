import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def get_api_host() -> str:
    return env_str("API_HOST", "0.0.0.0")


def get_api_port() -> int:
    return env_int("API_PORT", 8000)


def _build_redis_url(db: int) -> str:
    """Build Redis URL from separate configuration variables."""
    host = env_str("REDIS_HOST", "localhost")
    port = env_int("REDIS_PORT", 6379)
    password = env_str("REDIS_PASSWORD")
    username = env_str("REDIS_USERNAME")
    
    # Build authentication part
    auth = ""
    if username and password:
        auth = f"{username}:{password}@"
    elif password:
        auth = f":{password}@"
    
    return f"redis://{auth}{host}:{port}/{db}"


def get_redis_url() -> str:
    """Get main Redis URL (for storage)."""
    db = env_int("REDIS_DB_MAIN", 0)
    return _build_redis_url(db)


def get_celery_broker_url() -> str:
    """Get Celery broker URL."""
    db = env_int("REDIS_DB_BROKER", 1)
    return _build_redis_url(db)


def get_celery_backend_url() -> str:
    """Get Celery result backend URL."""
    db = env_int("REDIS_DB_BACKEND", 2)
    return _build_redis_url(db)


def get_model_output_dir() -> Path:
    path_value = env_str("MODEL_OUTPUT_DIR")
    if path_value is None:
        return project_root() / "artifacts"
    path = Path(path_value)
    if not path.is_absolute():
        return project_root() / path
    return path


def get_data_root() -> Path:
    path_value = env_str("TRAINING_DATA_ROOT")
    if path_value is None:
        return project_root()
    path = Path(path_value)
    if not path.is_absolute():
        return project_root() / path
    return path


def get_external_callback_base_url() -> str | None:
    return env_str("EXTERNAL_CALLBACK_BASE_URL")


def get_external_status_callback_url() -> str | None:
    return env_str("EXTERNAL_STATUS_CALLBACK_URL") or env_str("EXTERNAL_PUBLISH_CALLBACK_URL")


def get_external_publish_callback_url() -> str | None:
    """Deprecated alias, kept for backward compatibility."""
    return get_external_status_callback_url()


def get_callback_timeout() -> float:
    return env_float("EXTERNAL_CALLBACK_TIMEOUT", 10.0)


def get_inference_base_model() -> str:
    return env_str("INFERENCE_BASE_MODEL", "bert-base-chinese") or "bert-base-chinese"


def get_inference_workers_per_gpu() -> int:
    return env_int("INFERENCE_WORKERS_PER_GPU", 1)


def get_inference_max_batch_size() -> int:
    return env_int("INFERENCE_MAX_BATCH_SIZE", 16)


def get_inference_queue_age_weight_seconds() -> float:
    return env_float("INFERENCE_QUEUE_AGE_WEIGHT_SECONDS", 5.0)


def get_inference_unload_timeout() -> float:
    return env_float("INFERENCE_UNLOAD_TIMEOUT", 60.0)


def parse_visible_gpu_devices() -> list[str]:
    raw = env_str("GPU_VISIBLE_DEVICES")
    if raw is None:
        raw = env_str("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return []
    devices = [device.strip() for device in raw.split(",")]
    return [device for device in devices if device]


def get_worker_max_concurrency() -> int:
    visible_devices = parse_visible_gpu_devices()
    if visible_devices:
        return len(visible_devices)
    # CPU fallback
    return 1


__all__ = [
    "get_api_host",
    "get_api_port",
    "get_celery_backend_url",
    "get_celery_broker_url",
    "get_callback_timeout",
    "get_data_root",
    "get_external_callback_base_url",
    "get_external_publish_callback_url",
    "get_external_status_callback_url",
    "get_inference_base_model",
    "get_inference_max_batch_size",
    "get_inference_queue_age_weight_seconds",
    "get_inference_unload_timeout",
    "get_inference_workers_per_gpu",
    "get_model_output_dir",
    "get_redis_url",
    "parse_visible_gpu_devices",
    "get_worker_max_concurrency",
    "project_root",
]
