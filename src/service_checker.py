"""服务检查模块"""

import os
import time
from typing import Dict, Optional

import pandas as pd
import requests

from src.core.config import config
from src.utils.logger import logger


class ServiceChecker:
    """服务检查器，用于集成测试"""

    def __init__(self, base_url: Optional[str] = None):
        """初始化服务检查器

        Args:
            base_url: API基础URL，如果为None则使用配置文件中的设置
        """
        self.base_url = base_url or f"http://{config.API_HOST}:{config.API_PORT}"
        self.test_dataset_path = "legacy/环境保护_空气污染--样例1000.xlsx"
        self.test_dataset_filename = "环境保护_空气污染--样例1000.xlsx"

    def run_integration_test(self):
        """运行集成测试"""
        logger.info("开始集成测试...")

        # 1. 检查服务健康状态
        if not self.check_health():
            logger.error("服务健康检查失败")
            return

        # 2. 准备测试数据
        test_data = self.prepare_test_data()
        if not test_data:
            logger.error("测试数据准备失败")
            return

        # 3. 创建训练任务
        task_id = self.create_training_task(test_data)
        if not task_id:
            logger.error("创建训练任务失败")
            return

        # 4. 监控任务状态
        if not self.monitor_task(task_id):
            logger.error("任务监控失败")
            return

        # 5. 验证任务结果
        if not self.verify_task_result(task_id):
            logger.error("任务结果验证失败")
            return

        logger.info("集成测试成功完成！")

    def check_health(self) -> bool:
        """检查服务健康状态"""
        logger.info("检查服务健康状态...")

        response = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"服务状态: {health_data}")
            return True
        else:
            logger.error(f"健康检查失败，状态码: {response.status_code}")
            return False

    def prepare_test_data(self) -> Optional[Dict]:
        """准备测试数据"""
        logger.info(f"准备测试数据，使用数据集: {self.test_dataset_path}")

        # 检查文件是否存在
        if not os.path.exists(self.test_dataset_path):
            logger.error(f"测试数据集不存在: {self.test_dataset_path}")
            return None

        # 复制文件到上传目录
        if not self.copy_file_to_uploads():
            logger.error("复制文件到上传目录失败")
            return None

        # 读取数据集信息
        df = pd.read_excel(
            self.test_dataset_path, sheet_name=0, nrows=10
        )  # 只读取前10行用于测试
        logger.info("数据集基本信息:")
        logger.info(f"- 行数: {len(df)}")
        logger.info(f"- 列数: {len(df.columns)}")
        logger.info(f"- 列名: {list(df.columns)}")

        # 准备训练任务数据
        from src.core.constants import BaseModel
        
        training_data = {
            "model_name_cn": "测试模型-环境保护空气污染",
            "model_name_en": "test_env_protection_air_pollution",
            "training_data_file": self.test_dataset_filename,
            "base_model": BaseModel.BERT_BASE_CHINESE,
            "hyperparameters": {
                "learning_rate": 3e-5,
                "epochs": 2,  # 减少训练轮数用于测试
                "batch_size": 16,
                "max_sequence_length": 128,
                "random_seed": 42,
                "train_val_split": 0.2,
                "text_column": "内容合并",
                "label_column": "标签列",
                "sheet_name": None,
            },
            "callback_url": None,
        }

        return training_data

    def copy_file_to_uploads(self) -> bool:
        """复制测试文件到上传目录"""
        import shutil
        from src.core.config import config
        
        uploads_dir = config.UPLOADS_DIR
        target_path = os.path.join(uploads_dir, self.test_dataset_filename)
        
        try:
            # 确保上传目录存在
            os.makedirs(uploads_dir, exist_ok=True)
            
            # 复制文件
            shutil.copy2(self.test_dataset_path, target_path)
            logger.info(f"文件已复制到上传目录: {self.test_dataset_path} -> {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"复制文件失败: {e}")
            return False

    def create_training_task(self, training_data: Dict) -> Optional[str]:
        """创建训练任务"""
        logger.info("创建训练任务...")

        response = requests.post(
            f"{self.base_url}/api/v1/training/tasks", json=training_data, timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            logger.info(f"训练任务创建成功，任务ID: {task_id}")
            return task_id
        else:
            logger.error(f"创建训练任务失败，状态码: {response.status_code}")
            logger.error(f"响应内容: {response.text}")
            return None

    def monitor_task(self, task_id: str, max_wait_time: int = 300) -> bool:
        """监控任务状态"""
        logger.info(f"开始监控任务状态，任务ID: {task_id}")

        start_time = time.time()
        check_interval = 10  # 每10秒检查一次

        while time.time() - start_time < max_wait_time:
            # 获取任务状态
            response = requests.get(
                f"{self.base_url}/api/v1/training/tasks/{task_id}", timeout=10
            )

            if response.status_code == 200:
                task_info = response.json()
                status = task_info.get("status")
                progress = task_info.get("progress", 0)

                logger.info(f"任务状态: {status}, 进度: {progress}%")

                if status == "COMPLETED":
                    logger.info("任务已完成")
                    return True
                elif status == "FAILED":
                    error_message = task_info.get("error_message", "未知错误")
                    logger.error(f"任务失败: {error_message}")
                    return False
                elif status == "STOPPED":
                    logger.warning("任务已停止")
                    return False
                else:
                    # 任务还在进行中，继续等待
                    time.sleep(check_interval)
            else:
                logger.error(f"获取任务状态失败，状态码: {response.status_code}")
                return False

        # 超时
        logger.error(f"任务监控超时，最大等待时间: {max_wait_time}秒")
        return False

    def verify_task_result(self, task_id: str) -> bool:
        """验证任务结果"""
        logger.info(f"验证任务结果，任务ID: {task_id}")

        # 获取任务详情
        response = requests.get(f"{self.base_url}/api/v1/training/tasks/{task_id}", timeout=10)

        if response.status_code == 200:
            task_info = response.json()

            # 验证任务状态
            if task_info.get("status") != "COMPLETED":
                logger.error(f"任务状态不是COMPLETED: {task_info.get('status')}")
                return False

            # 验证任务基本信息
            required_fields = [
                "task_id",
                "model_name_cn",
                "model_name_en",
                "created_at",
            ]
            for field in required_fields:
                if not task_info.get(field):
                    logger.error(f"任务缺少必要字段: {field}")
                    return False

            # 验证进度信息
            progress = task_info.get("progress")
            if progress is not None and progress < 100:
                logger.warning(f"任务进度未达到100%: {progress}%")

            logger.info("任务结果验证通过")
            logger.info("任务详情:")
            logger.info(f"- 任务ID: {task_info.get('task_id')}")
            logger.info(f"- 模型中文名: {task_info.get('model_name_cn')}")
            logger.info(f"- 模型英文名: {task_info.get('model_name_en')}")
            logger.info(f"- 状态: {task_info.get('status')}")
            logger.info(f"- 创建时间: {task_info.get('created_at')}")
            logger.info(f"- 开始时间: {task_info.get('started_at')}")
            logger.info(f"- 完成时间: {task_info.get('completed_at')}")
            logger.info(f"- 进度: {task_info.get('progress')}%")

            return True
        else:
            logger.error(f"获取任务详情失败，状态码: {response.status_code}")
            return False
