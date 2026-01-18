from loguru import logger

from src.settings import settings


def configure_logging() -> None:
    logger.remove()
    level = settings.log_level
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=False,
        enqueue=False,
        diagnose=False,
        backtrace=False,
    )


__all__ = ["configure_logging"]
