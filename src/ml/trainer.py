"""训练器模块"""
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from src.utils.logger import logger
from src.ml.dataset import TextClassificationDataset, save_label_mappings
from src.ml.model import BertClassifier

class BertTrainer:
    """BERT训练器"""
    
    def __init__(self, model: BertClassifier, device: str = "auto"):
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        logger.info(f"训练器初始化完成 - 设备: {self.device}")
        logger.info(f"模型信息: {model.get_model_info()}")
    
    def _get_device(self, device: str) -> torch.device:
        """获取计算设备"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        torch_device = torch.device(device)
        logger.info(f"使用设备: {torch_device}")
        
        return torch_device
    
    def setup_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
        logger.info(f"随机种子已设置: {seed}")
    
    def create_data_loaders(
        self,
        train_dataset: TextClassificationDataset,
        val_dataset: TextClassificationDataset,
        batch_size: int = 64,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        logger.info(f"数据加载器创建完成 - 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # 移动数据到设备
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # 移动数据到设备
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_dataset: TextClassificationDataset,
        val_dataset: TextClassificationDataset,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 3e-5,
        save_path: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """训练模型"""
        logger.info(f"开始训练 - 轮数: {epochs}, 批次大小: {batch_size}, 学习率: {learning_rate}")
        
        # 创建数据加载器
        train_loader, val_loader = self.create_data_loaders(train_dataset, val_dataset, batch_size)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练循环
        best_val_accuracy = 0
        training_history = []
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_accuracy = self.validate(val_loader, criterion)
            
            # 记录历史
            epoch_history = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }
            training_history.append(epoch_history)
            
            # 进度回调
            if progress_callback:
                progress_callback(epoch + 1, epochs, train_accuracy, train_loss, val_accuracy, val_loss)
            
            # 日志
            logger.info(f"Epoch {epoch + 1}/{epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_val_accuracy and save_path:
                best_val_accuracy = val_accuracy
                self.save_model(save_path)
                logger.info(f"最佳模型已保存 - 验证准确率: {best_val_accuracy:.4f}")
        
        logger.info(f"训练完成 - 最佳验证准确率: {best_val_accuracy:.4f}")
        
        return {
            "training_history": training_history,
            "best_val_accuracy": best_val_accuracy,
            "total_epochs": epochs
        }
    
    def save_model(self, save_path: str):
        """保存模型"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"模型已保存: {save_path}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"模型已加载: {model_path}")

def create_trainer(num_classes: int, model_name: str = "bert-base-chinese", dropout_rate: float = 0.5, device: str = "auto") -> BertTrainer:
    """创建训练器"""
    from src.ml.model import create_model
    model = create_model(num_classes, dropout_rate, model_name)
    return BertTrainer(model, device)