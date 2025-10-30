"""健康检查路由"""
from fastapi import APIRouter
from src.utils.logger import logger

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    """健康检查"""
    logger.debug("健康检查请求")
    return {
        "status": "healthy",
        "service": "bert-training-api",
        "timestamp": "2024-01-01T00:00:00Z"  # 这里应该使用实际时间
    }

@router.get("/ready")
async def readiness_check():
    """就绪检查"""
    logger.debug("就绪检查请求")
    return {
        "status": "ready",
        "service": "bert-training-api",
        "timestamp": "2024-01-01T00:00:00Z"  # 这里应该使用实际时间
    }