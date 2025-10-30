"""配置管理"""
from pathlib import Path
from typing import Optional
from src.utils.logger import logger

class Config:
    """应用配置"""
    
    # 基础配置
    APP_NAME: str = "bert-training-api"
    DEBUG: bool = False
    
    # API配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    
    # 文件存储配置
    DATA_DIR: str = "data"
    UPLOADS_DIR: str = f"{DATA_DIR}/uploads"
    MODELS_DIR: str = f"{DATA_DIR}/models"
    
    # Celery配置
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # 训练配置
    MAX_CONCURRENT_TASKS: int = 3
    DEFAULT_BATCH_SIZE: int = 64
    DEFAULT_EPOCHS: int = 10
    DEFAULT_LEARNING_RATE: float = 3e-5
    
    def __init__(self):
        """初始化配置"""
        self._create_directories()
        logger.info(f"配置初始化完成，数据目录: {self.DATA_DIR}")
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.DATA_DIR,
            self.UPLOADS_DIR,
            self.MODELS_DIR,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.debug(f"创建目录: {directory}")

# 全局配置实例
config = Config()