"""请求数据模型"""

from pydantic import BaseModel, Field
from typing import Optional
from src.core.constants import BaseModel as BaseModelEnum


class Hyperparameters(BaseModel):
    """超参数模型"""

    learning_rate: float = Field(default=3e-5, gt=0, description="学习率")
    epochs: int = Field(default=10, ge=1, description="训练轮数")
    batch_size: int = Field(default=64, ge=1, description="批次大小")
    max_sequence_length: int = Field(
        default=512, ge=1, le=512, description="最大序列长度"
    )
    random_seed: int = Field(default=42, description="随机种子")
    train_val_split: float = Field(
        default=0.2, ge=0, le=1, description="训练验证集划分比例"
    )
    text_column: str = Field(default="内容合并", description="文本列名")
    label_column: str = Field(default="标签列", description="标签列名")
    sheet_name: Optional[str] = Field(default=None, description="Excel工作表名")


class CreateTrainingTaskRequest(BaseModel):
    """创建训练任务请求"""

    model_name_cn: str = Field(
        ..., min_length=1, max_length=100, description="模型中文名称"
    )
    model_name_en: str = Field(
        ..., pattern=r"^[a-zA-Z0-9_]+$", description="模型英文名称"
    )
    training_data_file: str = Field(..., description="训练数据文件名")
    base_model: BaseModelEnum = Field(
        default=BaseModelEnum.BERT_BASE_CHINESE, description="基础模型"
    )
    hyperparameters: Hyperparameters = Field(
        default_factory=Hyperparameters, description="超参数"
    )
    callback_url: Optional[str] = Field(default=None, description="回调URL")


class TaskQueryParams(BaseModel):
    """任务查询参数"""

    status: Optional[str] = Field(default=None, description="任务状态")
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页数量")
