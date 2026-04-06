import os
from multiprocessing import current_process

from celery import signals
from loguru import logger

from src.device_utils import (
    available_device_count,
    get_auto_device,
    get_visible_devices,
    visible_device_env_var,
)
from src.logging_utils import configure_logging


def available_gpu_devices() -> list[str]:
    devices = get_visible_devices()
    if devices:
        return devices
    device = get_auto_device()
    if device.type == "cpu":
        return []
    return [str(i) for i in range(available_device_count(device.type))]


@signals.worker_process_init.connect
def assign_gpu_to_process(**kwargs) -> None:
    configure_logging()
    devices = available_gpu_devices()
    device = get_auto_device()
    if not devices:
        logger.info("Worker process {} running on CPU", current_process().pid)
        return

    process = current_process()
    index = getattr(process, "index", 0)
    accelerator_id = devices[index % len(devices)]
    os.environ["GPU_VISIBLE_DEVICES"] = accelerator_id
    env_name = visible_device_env_var(device.type)
    if env_name:
        os.environ[env_name] = accelerator_id
    logger.info("Worker process {} assigned to {} {}", process.pid, device.type.upper(), accelerator_id)


__all__ = []
