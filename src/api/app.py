from fastapi import FastAPI

from src.api.routes.training import router as training_router
from src.logging_utils import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="Dianxin Auto Train",
        version="1.0.0",
        description="Automatic model training service",
    )
    app.include_router(training_router, prefix="/api/v1")
    return app


__all__ = ["create_app"]
