"""数据库模块"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import logger


class Database:
    """SQLite数据库管理"""

    def __init__(self, db_path: str = "data/training.db"):
        """初始化数据库

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init_db(self):
        """初始化数据库表结构"""
        logger.info(f"初始化数据库: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建训练任务表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_tasks (
                task_id TEXT PRIMARY KEY,
                model_name_cn TEXT NOT NULL,
                model_name_en TEXT NOT NULL,
                training_data_file TEXT NOT NULL,
                base_model TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,
                callback_url TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                progress REAL DEFAULT 0,
                created_at TIMESTAMP NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)

        # 创建任务进度历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                current_epoch INTEGER,
                total_epochs INTEGER,
                progress_percentage REAL,
                train_accuracy REAL,
                train_loss REAL,
                val_accuracy REAL,
                val_loss REAL,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (task_id) REFERENCES training_tasks(task_id)
            )
        """)

        conn.commit()
        conn.close()

        logger.info("数据库初始化完成")

    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使查询结果可以像字典一样访问
        return conn

    def insert_task(self, task_data: Dict[str, Any]) -> str:
        """插入训练任务

        Args:
            task_data: 任务数据

        Returns:
            任务ID
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO training_tasks (
                task_id, model_name_cn, model_name_en, training_data_file,
                base_model, hyperparameters, callback_url, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_data["task_id"],
                task_data["model_name_cn"],
                task_data["model_name_en"],
                task_data["training_data_file"],
                task_data["base_model"],
                json.dumps(task_data["hyperparameters"]),
                task_data.get("callback_url"),
                task_data["status"],
                datetime.now(),
            ),
        )

        conn.commit()
        conn.close()

        return task_data["task_id"]

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务

        Args:
            task_id: 任务ID

        Returns:
            任务数据或None
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM training_tasks WHERE task_id = ?
        """,
            (task_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            task_data = dict(row)
            # 解析JSON字段
            if task_data["hyperparameters"]:
                task_data["hyperparameters"] = json.loads(task_data["hyperparameters"])
            return task_data

        return None

    def get_all_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取所有任务

        Args:
            status: 状态筛选，如果为None则返回所有任务

        Returns:
            任务列表
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute(
                """
                SELECT * FROM training_tasks WHERE status = ? ORDER BY created_at DESC
            """,
                (status,),
            )
        else:
            cursor.execute("""
                SELECT * FROM training_tasks ORDER BY created_at DESC
            """)

        rows = cursor.fetchall()
        conn.close()

        tasks = []
        for row in rows:
            task_data = dict(row)
            # 解析JSON字段
            if task_data["hyperparameters"]:
                task_data["hyperparameters"] = json.loads(task_data["hyperparameters"])
            tasks.append(task_data)

        return tasks

    def update_task_status(
        self, task_id: str, status: str, error_message: Optional[str] = None
    ):
        """更新任务状态

        Args:
            task_id: 任务ID
            status: 新状态
            error_message: 错误信息（可选）
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        if status in ["TRAINING"]:
            cursor.execute(
                """
                UPDATE training_tasks 
                SET status = ?, started_at = ?, error_message = ?
                WHERE task_id = ?
            """,
                (status, datetime.now(), error_message, task_id),
            )
        elif status in ["COMPLETED", "FAILED", "STOPPED"]:
            cursor.execute(
                """
                UPDATE training_tasks 
                SET status = ?, completed_at = ?, error_message = ?
                WHERE task_id = ?
            """,
                (status, datetime.now(), error_message, task_id),
            )
        else:
            cursor.execute(
                """
                UPDATE training_tasks 
                SET status = ?, error_message = ?
                WHERE task_id = ?
            """,
                (status, error_message, task_id),
            )

        conn.commit()
        conn.close()

    def update_task_progress(self, task_id: str, progress: float):
        """更新任务进度

        Args:
            task_id: 任务ID
            progress: 进度百分比
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE training_tasks 
            SET progress = ?
            WHERE task_id = ?
        """,
            (progress, task_id),
        )

        conn.commit()
        conn.close()

    def insert_progress_history(self, task_id: str, progress_data: Dict[str, Any]):
        """插入进度历史记录

        Args:
            task_id: 任务ID
            progress_data: 进度数据
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO task_progress (
                task_id, current_epoch, total_epochs, progress_percentage,
                train_accuracy, train_loss, val_accuracy, val_loss, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_id,
                progress_data.get("current_epoch"),
                progress_data.get("total_epochs"),
                progress_data.get("progress_percentage"),
                progress_data.get("train_accuracy"),
                progress_data.get("train_loss"),
                progress_data.get("val_accuracy"),
                progress_data.get("val_loss"),
                datetime.now(),
            ),
        )

        conn.commit()
        conn.close()

    def delete_task(self, task_id: str) -> bool:
        """删除任务

        Args:
            task_id: 任务ID

        Returns:
            是否删除成功
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # 先删除进度历史
        cursor.execute("DELETE FROM task_progress WHERE task_id = ?", (task_id,))

        # 再删除任务
        cursor.execute("DELETE FROM training_tasks WHERE task_id = ?", (task_id,))

        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted


# 全局数据库实例
db = Database()
