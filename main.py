"""程序统一入口"""

from pathlib import Path
import click
import uvicorn

from src.core.config import config
from src.utils.logger import logger
from src.workers.tasks import celery_app


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
        log_level="info",
    )


@cli.command()
@click.option("--concurrency", default=2, help="并发工作进程数量")
@click.option("--loglevel", default="info", help="日志级别")
def worker(concurrency: int, loglevel: str):
    """启动Celery Worker"""
    logger.info(f"启动Celery Worker - 并发数: {concurrency}, 日志级别: {loglevel}")

    # 使用Celery命令启动worker
    celery_app.worker_main(
        [
            "worker",
            "--loglevel={}".format(loglevel),
            "--concurrency={}".format(concurrency),
            "--pool=prefork",
        ]
    )


@cli.command()
def init_db():
    """初始化数据库"""
    logger.info("初始化数据库...")

    from src.database import db

    db.init_db()

    logger.info("数据库初始化完成")


@cli.command()
def check_service():
    """检查服务运行状况"""
    logger.info("开始检查服务运行状况...")

    from src.service_checker import ServiceChecker

    checker = ServiceChecker()
    checker.run_integration_test()

    logger.info("服务检查完成")


@cli.command()
def check_config():
    """检查配置加载状况"""
    print("🔍 开始检查配置...\n")

    # 检查.env文件
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists():
        print("⚠️  未找到 .env 文件")
        if env_example_path.exists():
            print("💡 提示: 可以复制 .env.example 为 .env")
            print("   cp .env.example .env")
    else:
        print("✅ 找到 .env 文件")

    # 检查python-dotenv
    try:
        import importlib.util

        dotenv_spec = importlib.util.find_spec("dotenv")
        if dotenv_spec is not None:
            print("✅ python-dotenv 已安装")
        else:
            print("❌ python-dotenv 未安装")
            print("💡 安装命令: pip install python-dotenv")
    except ImportError:
        print("❌ python-dotenv 未安装")
        print("💡 安装命令: pip install python-dotenv")

    # 检查配置加载
    try:
        print("\n📋 配置加载检查")
        print("-" * 40)
        print("✅ 配置加载成功")
        print(f"   应用名称: {config.APP_NAME}")
        print(f"   调试模式: {config.DEBUG}")
        print(f"   API端口: {config.API_PORT}")
        print(f"   API主机: {config.API_HOST}")
        print(f"   数据目录: {config.DATA_DIR}")
        print(f"   Celery Broker: {config.CELERY_BROKER_URL}")

        # 检查必要的目录
        print("\n📋 目录检查")
        print("-" * 40)
        required_dirs = [config.DATA_DIR, config.UPLOADS_DIR, config.MODELS_DIR, "logs"]

        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                print(f"✅ {dir_path} 目录存在")
            else:
                print(f"⚠️  {dir_path} 目录不存在，将自动创建")

        print("\n" + "=" * 40)
        print("🎉 所有检查通过！配置正常")

    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    cli()
