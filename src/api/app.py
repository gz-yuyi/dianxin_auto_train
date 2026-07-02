from fastapi import FastAPI
from loguru import logger

from src.api.routes.inference import router as inference_router
from src.api.routes.training import data_router as training_data_router
from src.api.routes.training import router as training_router
from src.config import is_inference_gateway_enabled
from src.logging_utils import configure_logging


def _get_inference_manager():
    from src.inference.service import get_inference_manager

    return get_inference_manager()


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="电信自动训练服务",
        version="1.0.0",
        description="用于模型训练与推理管理的自动化服务接口",
    )
    app.include_router(training_router)
    app.include_router(training_data_router)
    app.include_router(inference_router)

    @app.on_event("startup")
    def start_inference_workers() -> None:
        if is_inference_gateway_enabled():
            logger.info("Inference gateway mode enabled; local inference workers are not started")
            return
        manager = _get_inference_manager()
        manager.start()

    @app.on_event("shutdown")
    def stop_inference_workers() -> None:
        if is_inference_gateway_enabled():
            return
        manager = _get_inference_manager()
        manager.stop()

    return app


__all__ = ["create_app"]
