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


@cli.command()
@click.option("--model", default="all", help="要下载的模型 (bert-base-chinese, bert-base-uncased, all)")
@click.option("--force", is_flag=True, help="强制重新下载已存在的模型")
def download_model(model: str, force: bool):
    """下载训练需要的基座模型"""
    logger.info(f"开始下载模型: {model}")
    
    from transformers import BertTokenizer, AutoModel, AutoConfig
    from src.core.constants import BaseModel
    from src.core.config import config
    
    # 确保模型目录存在
    from pathlib import Path
    models_dir = Path(config.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要下载的模型列表
    models_to_download = []
    if model == "all":
        models_to_download = [BaseModel.BERT_BASE_CHINESE, BaseModel.BERT_BASE_UNCASED]
    elif model == BaseModel.BERT_BASE_CHINESE:
        models_to_download = [BaseModel.BERT_BASE_CHINESE]
    elif model == BaseModel.BERT_BASE_UNCASED:
        models_to_download = [BaseModel.BERT_BASE_UNCASED]
    else:
        logger.error(f"不支持的模型: {model}")
        logger.info(f"支持的模型: {BaseModel.BERT_BASE_CHINESE}, {BaseModel.BERT_BASE_UNCASED}, all")
        return
    
    # 下载模型
    for model_name in models_to_download:
        logger.info(f"正在下载模型: {model_name}")
        try:
            # 下载模型配置
            config_obj = AutoConfig.from_pretrained(
                model_name.value,
                cache_dir=models_dir,
                force_download=force
            )
            logger.info(f"✅ {model_name} 配置下载成功")
            
            # 下载tokenizer
            tokenizer = BertTokenizer.from_pretrained(
                model_name.value,
                cache_dir=models_dir,
                force_download=force
            )
            logger.info(f"✅ {model_name} tokenizer下载成功")
            
            # 下载模型
            model_obj = AutoModel.from_pretrained(
                model_name.value,
                cache_dir=models_dir,
                force_download=force
            )
            logger.info(f"✅ {model_name} 模型下载成功")
            
            # 显示模型信息
            logger.info(f"模型信息 - {model_name}:")
            logger.info(f"  - 隐藏层大小: {config_obj.hidden_size}")
            logger.info(f"  - 注意力头数: {config_obj.num_attention_heads}")
            logger.info(f"  - 隐藏层数: {config_obj.num_hidden_layers}")
            logger.info(f"  - 词汇表大小: {config_obj.vocab_size}")
            
        except Exception as e:
            logger.error(f"❌ 模型 {model_name} 下载失败: {e}")
    
    logger.info("模型下载完成")


if __name__ == "__main__":
    cli()
