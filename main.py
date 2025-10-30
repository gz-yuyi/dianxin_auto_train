"""程序统一入口"""
import click
import uvicorn
from celery import Celery
from src.api.app import app
from src.workers.tasks import celery_app
from src.core.config import config
from src.utils.logger import logger

@click.group()
def cli():
    """BERT训练API命令行工具"""
    pass

@cli.command()
@click.option("--host", default=config.API_HOST, help="API服务主机地址")
@click.option("--port", default=config.API_PORT, help="API服务端口")
@click.option("--workers", default=config.API_WORKERS, help="工作进程数量")
@click.option("--reload", is_flag=True, help="开发模式下自动重载")
def api(host: str, port: int, workers: int, reload: bool):
    """启动API服务"""
    logger.info(f"启动API服务 - 主机: {host}, 端口: {port}, 工作进程: {workers}")
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info"
    )

@cli.command()
@click.option("--concurrency", default=2, help="并发工作进程数量")
@click.option("--loglevel", default="info", help="日志级别")
def worker(concurrency: int, loglevel: str):
    """启动Celery Worker"""
    logger.info(f"启动Celery Worker - 并发数: {concurrency}, 日志级别: {loglevel}")
    
    # 使用Celery命令启动worker
    celery_app.worker_main([
        "worker",
        "--loglevel={}".format(loglevel),
        "--concurrency={}".format(concurrency),
        "--pool=prefork"
    ])

@cli.command()
def init_db():
    """初始化数据库"""
    logger.info("初始化数据库...")
    
    # 这里可以添加数据库初始化逻辑
    # 目前使用内存存储，所以不需要实际的数据库初始化
    
    logger.info("数据库初始化完成")

@cli.command()
def test():
    """运行测试"""
    logger.info("运行测试...")
    
    # 这里可以添加测试逻辑
    # 目前只是占位符
    
    logger.info("测试完成")

if __name__ == "__main__":
    cli()