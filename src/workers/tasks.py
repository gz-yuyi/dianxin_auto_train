"""Celery异步任务定义"""
from celery import Celery
from src.core.config import config
from src.utils.logger import logger

# 创建Celery应用
celery_app = Celery(
    "bert_training",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND
)

# 配置Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1小时
    task_soft_time_limit=3300,  # 55分钟
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=10
)

@celery_app.task(bind=True)
def train_model_task(self, task_id: str):
    """训练模型任务"""
    logger.info(f"Celery任务开始执行 - 任务ID: {task_id}")
    
    try:
        # 导入训练服务
        from src.services.training import training_service
        
        # 执行训练
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(training_service.start_training(task_id))
        loop.close()
        
        logger.info(f"Celery任务执行完成 - 任务ID: {task_id}")
        return {"task_id": task_id, "status": "completed"}
        
    except Exception as e:
        logger.error(f"Celery任务执行失败 - 任务ID: {task_id}, 错误: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise

def send_training_task(task_id: str):
    """发送训练任务到Celery"""
    logger.info(f"发送训练任务到Celery - 任务ID: {task_id}")
    task = train_model_task.delay(task_id)
    logger.info(f"Celery任务已发送 - 任务ID: {task_id}, Celery任务ID: {task.id}")
    return task.id