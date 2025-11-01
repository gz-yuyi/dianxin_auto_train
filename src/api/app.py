"""FastAPI应用实例"""

from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.api.routes import training, health
from src.utils.logger import logger
from src.core.constants import API_PREFIX


def create_app() -> FastAPI:
    """创建FastAPI应用"""

    app = FastAPI(
        title="BERT训练API",
        description="基于BERT模型的文本分类训练服务",
        version="0.1.0",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=f"{API_PREFIX}/redoc",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )

    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """记录请求日志"""
        logger.info(f"收到请求: {request.method} {request.url.path}")

        start_time = datetime.now()
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"请求完成: {request.method} {request.url.path} - 状态码: {response.status_code} - 耗时: {process_time:.3f}s"
        )

        return response

    # 异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """全局异常处理"""
        logger.error(f"未处理的异常 - 路径: {request.url.path}, 错误: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "内部服务器错误",
                "detail": str(exc),
            },
        )

    # 注册路由
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(training.router, prefix=API_PREFIX)

    # 根路径
    @app.get("/")
    async def root():
        return {
            "message": "BERT训练API服务",
            "version": "0.1.0",
            "docs": f"{API_PREFIX}/docs",
        }

    logger.info("FastAPI应用创建完成")
    return app


# 创建应用实例
app = create_app()
