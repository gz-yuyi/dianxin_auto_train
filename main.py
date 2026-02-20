import click
import json
import os
import sys
import uuid
import uvicorn
from loguru import logger
from transformers import AutoTokenizer, AutoModel

from src.api.app import create_app
from src.callbacks import send_external_epoch_callback, send_external_status_change, send_progress_callback
from src.celery_app import celery_app
from src.check_service import run_service_check
from src.config import get_api_host, get_api_port, get_worker_max_concurrency
from src.logging_utils import configure_logging
from src.schemas import TrainingTaskCreateRequest
from src.training.service import run_training_loop


def load_training_payload(payload: str | None, payload_file: str | None) -> dict:
    if payload and payload_file:
        raise click.UsageError("Provide either --payload or --payload-file, not both.")
    if payload_file:
        with open(payload_file, "r", encoding="utf-8") as handle:
            raw = handle.read()
    elif payload:
        if payload.strip() == "-":
            raw = sys.stdin.read()
        else:
            raw = payload
    else:
        if sys.stdin.isatty():
            raise click.UsageError("Provide --payload, --payload-file, or pipe JSON into stdin.")
        raw = sys.stdin.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON payload: {exc}") from exc


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


@cli.command("train")
@click.option("--payload", "-p", default=None, help="Training payload JSON string (use '-' to read stdin)")
@click.option(
    "--payload-file",
    "-f",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to JSON file containing the training payload",
)
@click.option("--task-id", default=None, help="Optional task id (defaults to a generated value)")
@click.option("--callback/--no-callback", default=False, help="Enable callback notifications")
def train(payload: str | None, payload_file: str | None, task_id: str | None, callback: bool) -> None:
    payload_data = load_training_payload(payload, payload_file)
    try:
        request = TrainingTaskCreateRequest(**payload_data)
    except Exception as exc:
        raise click.ClickException(f"Invalid training payload: {exc}") from exc

    request_payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    if request_payload.get("callback_url") is not None:
        request_payload["callback_url"] = str(request_payload["callback_url"])
    if not callback:
        request_payload["callback_url"] = None

    task_id = task_id or f"local-{uuid.uuid4().hex}"
    logger.info("Starting local training task {}", task_id)
    if callback:
        send_external_status_change(task_id, "training")

    def progress_handler(epoch: int, metrics: dict) -> None:
        if callback:
            send_progress_callback(task_id, epoch, metrics, request_payload.get("callback_url"))
            send_external_epoch_callback(task_id, epoch, metrics)
        logger.info(
            "Task {} epoch {}/{} completed with metrics: train_acc={:.3f}, val_acc={:.3f}, f1={:.3f}",
            task_id,
            epoch,
            metrics["total_epochs"],
            metrics["train_accuracy"],
            metrics["val_accuracy"],
            metrics.get("f1_score", 0),
        )

    def batch_progress_handler(epoch: int, batch: int, metrics: dict) -> None:
        logger.debug(
            "Task {} epoch {}/{}, batch {}/{} - Progress: {:.1f}%, Train Acc: {:.3f}, Train Loss: {:.3f}",
            task_id,
            epoch,
            metrics["total_epochs"],
            batch,
            metrics["total_batches"],
            metrics["batch_progress_percentage"],
            metrics["train_accuracy"],
            metrics["train_loss"],
        )

    def stop_requested() -> bool:
        return False

    try:
        result = run_training_loop(
            task_id=task_id,
            request_payload=request_payload,
            progress_handler=progress_handler,
            batch_progress_handler=batch_progress_handler,
            stop_requested=stop_requested,
        )
    except Exception as exc:
        if callback:
            send_external_status_change(task_id, "failed", str(exc))
        raise click.ClickException(f"Training failed: {exc}") from exc

    status = result["status"]
    if callback:
        send_external_status_change(task_id, status)
    logger.info("Task {} completed with status {}", task_id, status)
    if status == "completed":
        if result.get("lora_enabled"):
            logger.info(
                "Artifacts saved: lora_adapter_path={}, classifier_head_path={}, label_mapping_path={}",
                result.get("lora_adapter_path"),
                result.get("classifier_head_path"),
                result.get("label_mapping_path"),
            )
        else:
            logger.info(
                "Artifacts saved: model_path={}, label_mapping_path={}",
                result.get("model_path"),
                result.get("label_mapping_path"),
            )


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
    base_url = f"http://{host}:{port}/"
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


@cli.command("export-openapi")
@click.option(
    "--output",
    "-o",
    default="openapi.json",
    show_default=True,
    help="Output file path for OpenAPI JSON",
)
@click.option(
    "--pretty/--no-pretty",
    default=False,
    help="Pretty print JSON output",
    show_default=True,
)
def export_openapi(output: str, pretty: bool) -> None:
    """Export OpenAPI/Swagger schema to JSON file."""
    app = create_app()
    openapi_schema = app.openapi()
    
    indent = 2 if pretty else None
    with open(output, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, ensure_ascii=False, indent=indent)
    
    logger.info(f"OpenAPI schema exported to {output}")
    click.echo(f"OpenAPI schema saved to: {output}")


@cli.command("smoke-test")
@click.option(
    "--port",
    default=8001,
    show_default=True,
    type=int,
    help="API server port for smoke test (use non-default to avoid conflicts)",
)
@click.option(
    "--training-timeout",
    default=600,
    show_default=True,
    type=int,
    help="Maximum seconds to wait for training completion",
)
@click.option(
    "--epochs",
    default=2,
    show_default=True,
    type=int,
    help="Number of training epochs for smoke test",
)
def smoke_test(port: int, training_timeout: int, epochs: int) -> None:
    """Run smoke test: start services, train model, test all inference APIs."""
    import sys
    from pathlib import Path
    
    # 添加 scripts 目录到路径
    scripts_dir = Path(__file__).parent / "scripts"
    sys.path.insert(0, str(scripts_dir))
    
    # 导入并运行冒烟测试
    from smoke_test import main as smoke_test_main
    
    # 覆盖配置
    import smoke_test as st
    st.API_PORT = port
    st.BASE_URL = f"http://{st.API_HOST}:{port}"
    st.TRAINING_TIMEOUT = training_timeout
    
    # 修改训练时的 epochs
    original_create_training_task = st.create_training_task
    
    def patched_create_training_task(train_file: str, test_file: str) -> str:
        """创建训练任务（使用自定义 epochs）"""
        st.log_info(f"创建 LoRA 训练任务（epochs={epochs}）...")
        
        # 注意：测试集没有标签列，不使用单独验证集文件
        # 使用 train_val_split 从训练集划分验证集
        payload = {
            "model_name_cn": "冒烟测试模型",
            "model_name_en": "smoke_test_model",
            "training_data_file": train_file,
            "validation_data_file": None,
            "base_model": "bert-base-chinese",
            "hyperparameters": {
                "learning_rate": 5e-5,
                "epochs": epochs,
                "batch_size": 32,
                "max_sequence_length": 128,
                "precision": "fp32",
                "gradient_accumulation_steps": 1,
                "early_stopping_enabled": False,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.001,
                "early_stopping_metric": "val_accuracy",
                "random_seed": 42,
                "train_val_split": 0.2,
                "text_column": "文本内容",
                "label_column": "标签列",
                "sheet_name": None,
                "validation_sheet_name": None,
                "lora": {
                    "enabled": True,
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.1,
                    "target_modules": ["query", "value"]
                }
            },
            "callback_url": None
        }
        
        import requests
        resp = requests.post(
            f"{st.BASE_URL}/training/tasks",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        
        task_id = resp.json()["task_id"]
        st.log_info(f"训练任务已创建: {task_id}")
        return task_id
    
    st.create_training_task = patched_create_training_task
    
    # 运行测试
    exit_code = smoke_test_main()
    sys.exit(exit_code)


if __name__ == "__main__":
    cli()
