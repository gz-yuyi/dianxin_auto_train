import click
import os
import uvicorn
from loguru import logger
from transformers import AutoTokenizer, AutoModel

from src.api.app import create_app
from src.celery_app import celery_app
from src.check_service import run_service_check
from src.config import get_api_host, get_api_port, get_worker_max_concurrency
from src.logging_utils import configure_logging


@click.group()
def cli() -> None:
    configure_logging()
    logger.info("CLI initialized")


@cli.command("api")
@click.option("--host", default=get_api_host(), show_default=True, help="API server host")
@click.option("--port", default=get_api_port(), show_default=True, type=int, help="API server port")
def start_api(host: str, port: int) -> None:
    uvicorn.run("src.api.app:create_app", factory=True, host=host, port=port, reload=False)


@cli.command("worker")
def start_worker() -> None:
    resolved = get_worker_max_concurrency()
    if resolved < 1:
        resolved = 1
    logger.info("Starting worker with concurrency {}", resolved)
    argv = ["worker", "--loglevel=info", f"--concurrency={resolved}"]
    celery_app.worker_main(argv)


@cli.command("check-service")
@click.option("--host", default=get_api_host(), show_default=True, help="API server host")
@click.option("--port", default=get_api_port(), show_default=True, type=int, help="API server port")
@click.option(
    "--dataset",
    default=None,
    help="Optional dataset path for running the integration check",
    show_default=False,
)
@click.option(
    "--cleanup/--no-cleanup",
    default=True,
    help="Delete task after verification",
    show_default=True,
)
def check_service(host: str, port: int, dataset: str | None, cleanup: bool) -> None:
    base_url = f"http://{host}:{port}/api/v1"
    run_service_check(base_url, dataset, cleanup)


@cli.command("download-model")
@click.option(
    "--model-name",
    default="google-bert/bert-base-chinese",
    help="Model name to download",
    show_default=True,
)
@click.option(
    "--output-dir",
    default="bert-base-chinese",
    help="Local directory to save the model",
    show_default=True,
)
@click.option(
    "--source",
    default="huggingface",
    type=click.Choice(["huggingface", "modelscope"]),
    help="Model download source",
    show_default=True,
)
def download_model(model_name: str, output_dir: str, source: str) -> None:
    """Download a model from Hugging Face or ModelScope to local directory."""
    logger.info(f"Downloading model {model_name} from {source} to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if source == "modelscope":
        # For ModelScope, we need to use the ModelScope hub
        from modelscope import snapshot_download
        import shutil
        
        # Create a temporary directory for download
        temp_dir = f"{output_dir}_temp"
        logger.info("Downloading from ModelScope...")
        snapshot_download(model_name, cache_dir=temp_dir)
        
        # Find the actual model directory (it will be in a nested structure)
        model_dir = None
        for root, dirs, files in os.walk(temp_dir):
            if "config.json" in files and "pytorch_model.bin" in files or "model.safetensors" in files:
                model_dir = root
                break
        
        if model_dir:
            # Move all files from the nested directory to the output directory
            for file_name in os.listdir(model_dir):
                shutil.move(os.path.join(model_dir, file_name), os.path.join(output_dir, file_name))
            
            # Remove the temporary directory
            shutil.rmtree(temp_dir)
            logger.info(f"Model successfully downloaded from ModelScope to {output_dir}")
        else:
            logger.error(f"Could not find model files in downloaded content")
    else:
        # Default: Hugging Face
        logger.info("Downloading tokenizer from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("Downloading model from Hugging Face...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(output_dir)
        
        logger.info(f"Model successfully downloaded from Hugging Face to {output_dir}")


if __name__ == "__main__":
    cli()
