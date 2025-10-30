"""训练任务服务"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import uuid4
from src.utils.logger import logger
from src.models.task import TrainingTask, TaskProgress
from src.core.constants import TaskStatus
from src.services.storage import storage_service
from src.ml.dataset import prepare_training_data, TextClassificationDataset
from src.ml.trainer import create_trainer
from transformers import BertTokenizer

class TrainingService:
    """训练任务服务"""
    
    def __init__(self):
        self.tasks: Dict[str, TrainingTask] = {}
        self.active_tasks: Dict[str, TrainingTask] = {}
        logger.info("训练任务服务初始化完成")
    
    def create_task(self, task_data: Dict[str, Any]) -> TrainingTask:
        """创建训练任务"""
        logger.info(f"创建训练任务 - 模型名称: {task_data['model_name_cn']}")
        
        # 检查训练数据文件是否存在
        if not storage_service.file_exists(task_data['training_data_file']):
            logger.error(f"训练数据文件不存在: {task_data['training_data_file']}")
            raise ValueError(f"训练数据文件不存在: {task_data['training_data_file']}")
        
        # 创建任务
        task = TrainingTask(
            model_name_cn=task_data['model_name_cn'],
            model_name_en=task_data['model_name_en'],
            training_data_file=task_data['training_data_file'],
            base_model=task_data['base_model'],
            hyperparameters=task_data['hyperparameters'],
            callback_url=task_data.get('callback_url')
        )
        
        # 保存任务
        self.tasks[task.task_id] = task
        
        logger.info(f"训练任务创建成功 - 任务ID: {task.task_id}")
        return task
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self, status: Optional[str] = None) -> List[TrainingTask]:
        """获取所有任务"""
        tasks = list(self.tasks.values())
        
        if status:
            tasks = [task for task in tasks if task.status.value == status]
        
        # 按创建时间倒序排列
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        return tasks
    
    def update_task_progress(self, task_id: str, progress_data: Dict[str, Any]):
        """更新任务进度"""
        task = self.get_task(task_id)
        if task:
            task.update_progress(progress_data)
            logger.debug(f"任务进度已更新 - 任务ID: {task_id}, 进度: {progress_data}")
    
    def stop_task(self, task_id: str) -> bool:
        """停止任务"""
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"任务不存在: {task_id}")
            return False
        
        if task.status not in [TaskStatus.QUEUED, TaskStatus.TRAINING]:
            logger.warning(f"任务状态不允许停止: {task.status}")
            return False
        
        task.update_status(TaskStatus.STOPPED)
        
        # 从活跃任务中移除
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        logger.info(f"任务已停止 - 任务ID: {task_id}")
        return True
    
    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        task = self.get_task(task_id)
        if not task:
            logger.warning(f"任务不存在: {task_id}")
            return False
        
        # 检查任务状态
        if task.status == TaskStatus.TRAINING:
            logger.warning(f"任务正在训练中，无法删除: {task_id}")
            return False
        
        # 删除任务
        del self.tasks[task_id]
        
        # 清理相关文件
        try:
            if task.training_data_file:
                storage_service.delete_file(task.training_data_file)
            
            if task.model_name_en:
                storage_service.delete_model(task.model_name_en)
        except Exception as e:
            logger.warning(f"清理任务文件时出错: {e}")
        
        logger.info(f"任务已删除 - 任务ID: {task_id}")
        return True
    
    def get_active_task_count(self) -> int:
        """获取活跃任务数量"""
        return len(self.active_tasks)
    
    def can_start_new_task(self) -> bool:
        """检查是否可以开始新任务"""
        from src.core.constants import MAX_CONCURRENT_TASKS
        return self.get_active_task_count() < MAX_CONCURRENT_TASKS
    
    async def start_training(self, task_id: str):
        """开始训练任务"""
        task = self.get_task(task_id)
        if not task:
            logger.error(f"任务不存在: {task_id}")
            return
        
        if task.status != TaskStatus.QUEUED:
            logger.warning(f"任务状态不是等待中: {task.status}")
            return
        
        if not self.can_start_new_task():
            logger.warning(f"并发任务数量已达上限，任务排队中")
            return
        
        # 更新任务状态
        task.update_status(TaskStatus.TRAINING)
        self.active_tasks[task_id] = task
        
        logger.info(f"开始训练任务 - 任务ID: {task_id}")
        
        try:
            # 执行训练
            await self._execute_training(task)
            
            # 训练完成
            task.update_status(TaskStatus.COMPLETED)
            logger.info(f"训练任务完成 - 任务ID: {task_id}")
            
        except Exception as e:
            # 训练失败
            task.update_status(TaskStatus.FAILED)
            task.error_message = str(e)
            logger.error(f"训练任务失败 - 任务ID: {task_id}, 错误: {e}")
            raise
        
        finally:
            # 从活跃任务中移除
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _execute_training(self, task: TrainingTask):
        """执行训练"""
        logger.info(f"执行训练 - 任务ID: {task.task_id}")
        
        # 获取文件路径
        file_path = storage_service.get_file_path(task.training_data_file)
        
        # 准备训练数据
        train_df, val_df, label2id, id2label = prepare_training_data(
            file_path=file_path,
            text_column=task.hyperparameters["text_column"],
            label_column=task.hyperparameters["label_column"],
            sheet_name=task.hyperparameters.get("sheet_name"),
            test_size=task.hyperparameters["train_val_split"],
            random_state=task.hyperparameters["random_seed"]
        )
        
        # 保存标签映射
        label_mapping_path = storage_service.get_model_path(f"{task.model_name_en}_labels.pkl")
        save_label_mappings(label2id, id2label, label_mapping_path)
        
        # 创建tokenizer和数据集
        tokenizer = BertTokenizer.from_pretrained(task.base_model)
        
        train_dataset = TextClassificationDataset(
            texts=train_df["text"].tolist(),
            labels=train_df[task.hyperparameters["label_column"]].tolist(),
            tokenizer=tokenizer,
            max_length=task.hyperparameters["max_sequence_length"]
        )
        
        val_dataset = TextClassificationDataset(
            texts=val_df["text"].tolist(),
            labels=val_df[task.hyperparameters["label_column"]].tolist(),
            tokenizer=tokenizer,
            max_length=task.hyperparameters["max_sequence_length"]
        )
        
        # 创建训练器
        num_classes = len(label2id)
        trainer = create_trainer(
            num_classes=num_classes,
            model_name=task.base_model,
            device="auto"
        )
        
        # 设置随机种子
        trainer.setup_seed(task.hyperparameters["random_seed"])
        
        # 定义进度回调函数
        def progress_callback(epoch, total_epochs, train_acc, train_loss, val_acc, val_loss):
            progress_data = {
                "current_epoch": epoch,
                "total_epochs": total_epochs,
                "progress_percentage": (epoch / total_epochs) * 100,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "val_accuracy": val_acc,
                "val_loss": val_loss
            }
            self.update_task_progress(task.task_id, progress_data)
        
        # 模型保存路径
        model_save_path = storage_service.get_model_path(task.model_name_en)
        
        # 开始训练
        training_result = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=task.hyperparameters["epochs"],
            batch_size=task.hyperparameters["batch_size"],
            learning_rate=task.hyperparameters["learning_rate"],
            save_path=model_save_path,
            progress_callback=progress_callback
        )
        
        logger.info(f"训练执行完成 - 任务ID: {task.task_id}, 最佳验证准确率: {training_result['best_val_accuracy']}")
        
        # 发送回调通知
        if task.callback_url:
            await self._send_callback(task, training_result)
    
    async def _send_callback(self, task: TrainingTask, training_result: Dict[str, Any]):
        """发送回调通知"""
        try:
            import aiohttp
            
            callback_data = {
                "task_id": task.task_id,
                "status": task.status.value,
                "model_name_cn": task.model_name_cn,
                "model_name_en": task.model_name_en,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "training_result": training_result
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(task.callback_url, json=callback_data) as response:
                    logger.info(f"回调通知已发送 - 任务ID: {task.task_id}, 状态码: {response.status}")
                    
        except Exception as e:
            logger.error(f"发送回调通知失败 - 任务ID: {task.task_id}, 错误: {e}")

# 全局训练服务实例
training_service = TrainingService()