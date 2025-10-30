"""项目常量定义"""
from enum import Enum

class TaskStatus(str, Enum):
    QUEUED = "queued"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

class BaseModel(str, Enum):
    BERT_BASE_CHINESE = "bert-base-chinese"
    BERT_BASE_UNCASED = "bert-base-uncased"

# 默认超参数
DEFAULT_HYPERPARAMETERS = {
    "learning_rate": 3e-5,
    "epochs": 10,
    "batch_size": 64,
    "max_sequence_length": 512,
    "random_seed": 42,
    "train_val_split": 0.2,
    "text_column": "内容合并",
    "label_column": "标签列",
    "sheet_name": None
}

# 文件路径常量
DATA_DIR = "data"
UPLOADS_DIR = f"{DATA_DIR}/uploads"
MODELS_DIR = f"{DATA_DIR}/models"

# API常量
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# 任务常量
MAX_CONCURRENT_TASKS = 3