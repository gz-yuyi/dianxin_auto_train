"""数据集处理模块"""

import pickle
from typing import Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.utils.logger import logger


class TextClassificationDataset(Dataset):
    """文本分类数据集"""

    def __init__(
        self, texts: list, labels: list, tokenizer: BertTokenizer, max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"初始化数据集，样本数量: {len(texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 使用tokenizer进行分词
        encoding = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }, label


def load_excel_data(
    file_path: str,
    text_column: str,
    label_column: str,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """加载Excel数据"""
    logger.info(f"加载Excel文件: {file_path}")

    # 读取Excel文件
    df_text = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[text_column])
    df_label = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[label_column])

    # 重命名列
    df_text.columns = ["text"]
    df_label.columns = ["label"]

    # 合并数据
    df = pd.concat([df_text, df_label], axis=1)

    # 数据清洗
    df["text"] = df["text"].astype(str).fillna("")

    logger.info(f"数据加载完成，样本数量: {len(df)}")
    return df


def create_label_mappings(labels: pd.Series) -> Tuple[Dict[str, int], Dict[int, str]]:
    """创建标签映射"""
    unique_labels = labels.unique()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    logger.info(f"创建标签映射，标签数量: {len(label2id)}")
    return label2id, id2label


def save_label_mappings(
    label2id: Dict[str, int], id2label: Dict[int, str], save_path: str
):
    """保存标签映射"""
    with open(save_path, "wb") as f:
        pickle.dump((label2id, id2label), f)
    logger.info(f"标签映射已保存: {save_path}")


def load_label_mappings(load_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """加载标签映射"""
    with open(load_path, "rb") as f:
        label2id, id2label = pickle.load(f)
    logger.info(f"标签映射已加载: {load_path}")
    return label2id, id2label


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """划分数据集"""
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    logger.info(f"数据集划分完成 - 训练集: {len(train_df)}, 验证集: {len(val_df)}")
    return train_df, val_df


def prepare_training_data(
    file_path: str,
    text_column: str,
    label_column: str,
    sheet_name: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """准备训练数据"""
    logger.info("开始准备训练数据")

    # 加载数据
    df = load_excel_data(file_path, text_column, label_column, sheet_name)

    # 创建标签映射
    label2id, id2label = create_label_mappings(df[label_column])

    # 转换标签
    df[label_column] = df[label_column].map(label2id)

    # 划分数据集
    train_df, val_df = split_dataset(df, test_size, random_state)

    logger.info("训练数据准备完成")
    return train_df, val_df, label2id, id2label
