import click
import uvicorn
from loguru import logger

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


if __name__ == "__main__":
    cli()
