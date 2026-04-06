import contextlib
import os
from typing import Any

import torch


def _ensure_npu_namespace() -> None:
    if hasattr(torch, "npu"):
        return
    with contextlib.suppress(ImportError):
        import torch_npu  # noqa: F401


def _device_module(device_type: str) -> Any | None:
    if device_type == "cuda" and hasattr(torch, "cuda"):
        return torch.cuda
    if device_type == "npu":
        _ensure_npu_namespace()
        if hasattr(torch, "npu"):
            return torch.npu
    return None


class NoOpGradScaler:
    def is_enabled(self) -> bool:
        return False

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None


def is_cuda_available() -> bool:
    return bool(hasattr(torch, "cuda") and torch.cuda.is_available())


def is_npu_available() -> bool:
    _ensure_npu_namespace()
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


def is_accelerator_available() -> bool:
    return is_cuda_available() or is_npu_available()


def get_auto_device() -> torch.device:
    if is_cuda_available():
        return torch.device("cuda:0")
    if is_npu_available():
        return torch.device("npu:0")
    return torch.device("cpu")


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return get_auto_device()
    device = torch.device(device_str)
    if device.type == "npu":
        _ensure_npu_namespace()
    return device


def set_current_device(device: torch.device) -> None:
    module = _device_module(device.type)
    if module is None or device.type == "cpu":
        return
    device_index = 0 if device.index is None else device.index
    module.set_device(device_index)


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed_all(seed)
    if is_npu_available():
        torch.npu.manual_seed_all(seed)


def enable_deterministic_mode() -> None:
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True


def visible_device_env_var(device_type: str) -> str | None:
    if device_type == "cuda":
        return "CUDA_VISIBLE_DEVICES"
    if device_type == "npu":
        return "ASCEND_RT_VISIBLE_DEVICES"
    return None


def get_visible_devices() -> list[str]:
    for name in ("GPU_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
        raw = os.getenv(name)
        if raw is None:
            continue
        devices = [device.strip() for device in raw.split(",")]
        cleaned = [device for device in devices if device]
        if cleaned:
            return cleaned
    return []


def available_device_count(device_type: str) -> int:
    module = _device_module(device_type)
    if module is None:
        return 0
    return int(module.device_count())


def get_available_accelerator_devices() -> list[torch.device]:
    visible = get_visible_devices()
    if visible:
        auto_device = get_auto_device()
        if auto_device.type == "cpu":
            return []
        return [torch.device(f"{auto_device.type}:{idx}") for idx in range(len(visible))]
    if is_cuda_available():
        return [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
    if is_npu_available():
        return [torch.device(f"npu:{idx}") for idx in range(torch.npu.device_count())]
    return []


def device_supports_precision(device: torch.device, precision: str) -> bool:
    if precision == "fp32":
        return True
    if device.type == "cuda":
        return True
    if device.type == "npu":
        module = _device_module("npu")
        return bool(module is not None and hasattr(module, "amp"))
    return False


def create_grad_scaler(device: torch.device, precision: str) -> Any:
    if precision != "fp16":
        return NoOpGradScaler()
    module = _device_module(device.type)
    if module is not None and hasattr(module, "amp") and hasattr(module.amp, "GradScaler"):
        return module.amp.GradScaler()
    amp_module = getattr(torch, "amp", None)
    if amp_module is not None and hasattr(amp_module, "GradScaler"):
        try:
            return amp_module.GradScaler(device.type, enabled=True)
        except TypeError:
            return amp_module.GradScaler(enabled=True)
    return NoOpGradScaler()


def autocast_context(device: torch.device, precision: str, amp_dtype: torch.dtype | None):
    if precision not in {"fp16", "bf16"}:
        return contextlib.nullcontext()
    module = _device_module(device.type)
    if module is not None and hasattr(module, "amp") and hasattr(module.amp, "autocast"):
        return module.amp.autocast(dtype=amp_dtype)
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, dtype=amp_dtype)
    return contextlib.nullcontext()


def empty_cache(device: torch.device) -> None:
    module = _device_module(device.type)
    if module is not None and hasattr(module, "empty_cache"):
        module.empty_cache()


def get_device_memory_info(device: torch.device) -> tuple[float, float, float, float]:
    module = _device_module(device.type)
    if module is None:
        return 0.0, 0.0, 0.0, 0.0
    try:
        total_memory = module.get_device_properties(device).total_memory
        reserved = module.memory_reserved(device)
        allocated = module.memory_allocated(device)
        free = total_memory - reserved
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

    total_mb = total_memory / 1024 / 1024
    used_mb = allocated / 1024 / 1024
    free_mb = free / 1024 / 1024
    usage_percent = round((used_mb / total_mb) * 100, 2) if total_mb > 0 else 0.0
    return round(total_mb, 2), round(used_mb, 2), round(free_mb, 2), usage_percent


__all__ = [
    "autocast_context",
    "available_device_count",
    "create_grad_scaler",
    "device_supports_precision",
    "empty_cache",
    "enable_deterministic_mode",
    "get_auto_device",
    "get_available_accelerator_devices",
    "get_device_memory_info",
    "get_visible_devices",
    "is_accelerator_available",
    "is_cuda_available",
    "is_npu_available",
    "manual_seed_all",
    "resolve_device",
    "set_current_device",
    "visible_device_env_var",
]
