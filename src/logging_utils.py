from loguru import logger

from src.config import env_str


def configure_logging() -> None:
    logger.remove()
    level = env_str("LOG_LEVEL", "INFO")
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=False,
        enqueue=False,
        diagnose=False,
        backtrace=False,
    )


__all__ = ["configure_logging"]
