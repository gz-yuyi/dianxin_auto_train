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


def get_redis_url() -> str:
    return env_str("REDIS_URL", "redis://localhost:6379/0")


def get_celery_broker_url() -> str:
    return env_str("CELERY_BROKER_URL", get_redis_url())


def get_celery_backend_url() -> str:
    return env_str("CELERY_RESULT_BACKEND", get_celery_broker_url())


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


def get_external_publish_callback_url() -> str | None:
    return env_str("EXTERNAL_PUBLISH_CALLBACK_URL")


def get_callback_timeout() -> float:
    return env_float("EXTERNAL_CALLBACK_TIMEOUT", 10.0)


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
    gpu_count = 0

    if visible_devices:
        gpu_count = len(visible_devices)
    else:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
        except Exception:
            gpu_count = 0

    if gpu_count > 0:
        return gpu_count
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
    "get_model_output_dir",
    "get_redis_url",
    "parse_visible_gpu_devices",
    "get_worker_max_concurrency",
    "project_root",
]
