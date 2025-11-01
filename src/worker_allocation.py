import os
from multiprocessing import current_process

from celery import signals
from loguru import logger

from src.config import parse_visible_gpu_devices


def available_gpu_devices() -> list[str]:
    devices = parse_visible_gpu_devices()
    if devices:
        return devices
    try:
        import torch

        if torch.cuda.is_available():
            return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []
    return []


@signals.worker_process_init.connect
def assign_gpu_to_process(**kwargs) -> None:
    devices = available_gpu_devices()
    if not devices:
        logger.info("Worker process {} running on CPU", current_process().pid)
        return

    process = current_process()
    index = getattr(process, "index", 0)
    gpu_id = devices[index % len(devices)]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    logger.info("Worker process {} assigned to GPU {}", process.pid, gpu_id)


__all__ = []
