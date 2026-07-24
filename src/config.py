import os
from pathlib import Path
from dotenv import load_dotenv

from src.device_utils import get_visible_devices


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


def get_inference_predict_timeout() -> float:
    """预测请求等待推理结果的最长时间（秒）。超时后请求被取消并返回 504。"""
    return env_float("INFERENCE_PREDICT_TIMEOUT", 120.0)


def get_inference_max_pending_items() -> int:
    """所有模型队列中允许积压的最大推理条目数，超过后拒绝新请求（快速失败）。"""
    return env_int("INFERENCE_MAX_PENDING_ITEMS", 1000)


def get_inference_worker_watchdog_timeout() -> float:
    """Worker 单个操作允许的最长执行时间（秒），超过判定为卡死并退出进程以便容器重启。
    设置为 0 可禁用看门狗。"""
    return env_float("INFERENCE_WORKER_WATCHDOG_TIMEOUT", 600.0)


def get_inference_empty_cache_on_unload() -> bool:
    """卸载模型后是否回收显存。
    empty_cache 的驱动调用在 Ascend 上可能永久阻塞，因此默认关闭，
    且即使开启也只在与推理无关的独立线程中执行。"""
    value = (env_str("INFERENCE_EMPTY_CACHE_ON_UNLOAD", "false") or "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def get_inference_cb_failure_threshold() -> int:
    """网关熔断器：upstream 连续失败多少次后被暂时摘除。"""
    return env_int("INFERENCE_CB_FAILURE_THRESHOLD", 3)


def get_inference_cb_cooldown_seconds() -> float:
    """网关熔断器：upstream 被摘除后的冷却时间（秒），冷却结束后半开重试。"""
    return env_float("INFERENCE_CB_COOLDOWN_SECONDS", 60.0)


def get_inference_upstream_urls() -> list[str]:
    raw_value = env_str("INFERENCE_UPSTREAM_URLS", "") or ""
    urls = []
    for item in raw_value.split(","):
        url = item.strip().rstrip("/")
        if url:
            urls.append(url)
    return urls


def is_inference_gateway_enabled() -> bool:
    return bool(get_inference_upstream_urls())


def get_inference_upstream_timeout() -> float:
    return env_float("INFERENCE_UPSTREAM_TIMEOUT", 60.0)


def get_inference_lb_policy() -> str:
    policy = (env_str("INFERENCE_LB_POLICY", "round_robin") or "round_robin").lower()
    if policy not in {"round_robin", "least_pending"}:
        return "round_robin"
    return policy


def parse_visible_gpu_devices() -> list[str]:
    return get_visible_devices()


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
    "get_inference_cb_cooldown_seconds",
    "get_inference_cb_failure_threshold",
    "get_inference_empty_cache_on_unload",
    "get_inference_max_batch_size",
    "get_inference_max_pending_items",
    "get_inference_predict_timeout",
    "get_inference_queue_age_weight_seconds",
    "get_inference_unload_timeout",
    "get_inference_worker_watchdog_timeout",
    "get_inference_lb_policy",
    "get_inference_unload_timeout",
    "get_inference_upstream_timeout",
    "get_inference_upstream_urls",
    "get_inference_workers_per_gpu",
    "is_inference_gateway_enabled",
    "get_model_output_dir",
    "get_redis_url",
    "parse_visible_gpu_devices",
    "get_worker_max_concurrency",
    "project_root",
]
