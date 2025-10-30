"""模型定义模块"""
import torch
import torch.nn as nn
from transformers import BertModel
from typing import Dict, Any
from src.utils.logger import logger

class BertClassifier(nn.Module):
    """BERT分类器模型"""
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.5, model_name: str = "bert-base-chinese"):
        super(BertClassifier, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        logger.info(f"初始化BERT分类器 - 类别数: {num_classes}, dropout: {dropout_rate}, 模型: {model_name}")
        
        # BERT模型
        self.bert = BertModel.from_pretrained(model_name)
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        logger.info("BERT分类器初始化完成")
    
    def forward(self, input_ids, attention_mask):
        """前向传播"""
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用[CLS]token的输出
        pooled_output = outputs.pooler_output
        
        # dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": "bert_classifier",
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "bert_hidden_size": self.bert.config.hidden_size,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

def create_model(num_classes: int, dropout_rate: float = 0.5, model_name: str = "bert-base-chinese") -> BertClassifier:
    """创建BERT分类器模型"""
    return BertClassifier(num_classes, dropout_rate, model_name)