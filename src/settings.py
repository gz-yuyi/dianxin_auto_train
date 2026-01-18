from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None
    model_output_dir: str | None = None
    training_data_root: str | None = None
    external_callback_base_url: str | None = None
    external_status_callback_url: str | None = None
    external_publish_callback_url: str | None = None
    external_callback_timeout: float = 10.0
    gpu_visible_devices: str | None = None
    cuda_visible_devices: str | None = None
    embedding_base_url: str | None = None
    embedding_model_name: str | None = None
    embedding_model: str | None = None
    embedding_api_key: str | None = None
    log_level: str = "INFO"

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def model_output_path(self) -> Path:
        path_value = self.model_output_dir
        if not path_value:
            return self.project_root / "artifacts"
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def data_root(self) -> Path:
        path_value = self.training_data_root
        if not path_value:
            return self.project_root
        path = Path(path_value)
        if path.is_absolute():
            return path
        return self.project_root / path

    @property
    def celery_broker_url_value(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def celery_result_backend_value(self) -> str:
        return self.celery_result_backend or self.celery_broker_url_value

    @property
    def external_status_callback_url_value(self) -> str | None:
        return self.external_status_callback_url or self.external_publish_callback_url

    @property
    def embedding_model_name_value(self) -> str | None:
        return self.embedding_model_name or self.embedding_model

    def visible_gpu_devices(self) -> list[str]:
        raw = self.gpu_visible_devices or self.cuda_visible_devices
        if not raw:
            return []
        devices = [device.strip() for device in raw.split(",")]
        return [device for device in devices if device]

    def worker_max_concurrency(self) -> int:
        visible_devices = self.visible_gpu_devices()
        if visible_devices:
            return len(visible_devices)
        return 1


settings = Settings()


__all__ = ["settings", "Settings"]
