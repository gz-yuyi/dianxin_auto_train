"""配置管理"""

import os
from pathlib import Path


# 在模块级别加载dotenv，确保配置加载前环境变量已设置
def _load_dotenv():
    """加载.env文件"""
    try:
        from dotenv import load_dotenv

        # 获取项目根目录的.env文件路径
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
        )
        if os.path.exists(env_path):
            load_dotenv(env_path)
            return True
        else:
            # 尝试在当前工作目录查找.env文件
            if os.path.exists(".env"):
                load_dotenv(".env")
                return True
        return False
    except ImportError:
        return False


# 加载.env文件（如果存在）
_dotenv_loaded = _load_dotenv()

# 延迟导入logger，避免循环导入
_logger = None


def _get_logger():
    """获取logger实例"""
    global _logger
    if _logger is None:
        from src.utils.logger import logger

        _logger = logger
    return _logger


class Config:
    """应用配置"""

    # 默认配置值
    _DEFAULTS = {
        # 基础配置
        "APP_NAME": "bert-training-api",
        "DEBUG": False,
        # API配置
        "API_HOST": "0.0.0.0",
        "API_PORT": 8000,
        "API_WORKERS": 1,
        # 文件存储配置
        "DATA_DIR": "data",
        "UPLOADS_DIR": None,  # 将由DATA_DIR计算得出
        "MODELS_DIR": None,  # 将由DATA_DIR计算得出
        # Celery配置
        "CELERY_BROKER_URL": "redis://localhost:6379/0",
        "CELERY_RESULT_BACKEND": "redis://localhost:6379/0",
        # 训练配置
        "MAX_CONCURRENT_TASKS": 3,
        "DEFAULT_BATCH_SIZE": 64,
        "DEFAULT_EPOCHS": 10,
        "DEFAULT_LEARNING_RATE": 3e-5,
    }

    # 环境变量映射
    _ENV_MAPPINGS = {
        # 基础配置
        "APP_NAME": "APP_NAME",
        "DEBUG": "DEBUG",
        # API配置
        "API_HOST": "API_HOST",
        "API_PORT": "API_PORT",
        "API_WORKERS": "API_WORKERS",
        # 文件存储配置
        "DATA_DIR": "DATA_DIR",
        "UPLOADS_DIR": "UPLOADS_DIR",
        "MODELS_DIR": "MODELS_DIR",
        # Celery配置
        "CELERY_BROKER_URL": "CELERY_BROKER_URL",
        "CELERY_RESULT_BACKEND": "CELERY_RESULT_BACKEND",
        # 训练配置
        "MAX_CONCURRENT_TASKS": "MAX_CONCURRENT_TASKS",
        "DEFAULT_BATCH_SIZE": "DEFAULT_BATCH_SIZE",
        "DEFAULT_EPOCHS": "DEFAULT_EPOCHS",
        "DEFAULT_LEARNING_RATE": "DEFAULT_LEARNING_RATE",
    }

    def __init__(self):
        """初始化配置"""
        self._load_from_env()
        self._create_directories()
        logger = _get_logger()
        logger.info(f"配置初始化完成，数据目录: {self.DATA_DIR}")

    def _load_from_env(self):
        """从环境变量加载配置"""
        # 记录dotenv加载状态
        logger = _get_logger()
        if _dotenv_loaded:
            logger.info("已加载 .env 文件配置")

        for attr_name, env_name in self._ENV_MAPPINGS.items():
            env_value = os.environ.get(env_name)
            if env_value is not None:
                # 获取默认值类型
                default_value = self._DEFAULTS.get(attr_name)
                if default_value is not None:
                    # 类型转换
                    converted_value = self._convert_type(env_value, default_value)
                    setattr(self, attr_name, converted_value)
                    logger.debug(f"从环境变量加载 {env_name} = {converted_value}")
                else:
                    # 对于没有默认值的字段（如UPLOADS_DIR, MODELS_DIR），直接使用字符串值
                    setattr(self, attr_name, env_value)
                    logger.debug(f"从环境变量加载 {env_name} = {env_value}")
            else:
                # 使用默认值
                default_value = self._DEFAULTS.get(attr_name)
                if default_value is not None:
                    setattr(self, attr_name, default_value)

        # 特殊处理：如果UPLOADS_DIR和MODELS_DIR没有通过环境变量设置，则根据DATA_DIR计算
        if not os.environ.get("UPLOADS_DIR"):
            self.UPLOADS_DIR = f"{self.DATA_DIR}/uploads"
        if not os.environ.get("MODELS_DIR"):
            self.MODELS_DIR = f"{self.DATA_DIR}/models"

    def _convert_type(self, value: str, default_value):
        """类型转换"""
        if isinstance(default_value, bool):
            return value.lower() in ("true", "1", "yes", "on")
        elif isinstance(default_value, int):
            return int(value)
        elif isinstance(default_value, float):
            return float(value)
        else:
            return value

    def _create_directories(self):
        """创建必要的目录"""
        directories = [self.DATA_DIR, self.UPLOADS_DIR, self.MODELS_DIR, "logs"]
        logger = _get_logger()

        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.debug(f"创建目录: {directory}")

    def get(self, key: str, default=None):
        """获取配置值"""
        return getattr(self, key, default)

    def to_dict(self):
        """返回所有配置的字典表示"""
        return {
            "APP_NAME": self.APP_NAME,
            "DEBUG": self.DEBUG,
            "API_HOST": self.API_HOST,
            "API_PORT": self.API_PORT,
            "API_WORKERS": self.API_WORKERS,
            "DATA_DIR": self.DATA_DIR,
            "UPLOADS_DIR": self.UPLOADS_DIR,
            "MODELS_DIR": self.MODELS_DIR,
            "CELERY_BROKER_URL": self.CELERY_BROKER_URL,
            "CELERY_RESULT_BACKEND": self.CELERY_RESULT_BACKEND,
            "MAX_CONCURRENT_TASKS": self.MAX_CONCURRENT_TASKS,
            "DEFAULT_BATCH_SIZE": self.DEFAULT_BATCH_SIZE,
            "DEFAULT_EPOCHS": self.DEFAULT_EPOCHS,
            "DEFAULT_LEARNING_RATE": self.DEFAULT_LEARNING_RATE,
        }


# 全局配置实例
config = Config()
