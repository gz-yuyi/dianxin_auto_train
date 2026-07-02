import contextlib
import hashlib
import inspect
import json
import pickle
import queue
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, PeftModel
from torch import nn
from transformers import AutoModel, AutoTokenizer

from src.config import (
    get_inference_base_model,
    get_inference_max_batch_size,
    get_inference_queue_age_weight_seconds,
    get_inference_unload_timeout,
    get_inference_workers_per_gpu,
    get_model_output_dir,
)
from src.device_utils import (
    empty_cache,
    get_available_accelerator_devices,
    get_device_memory_info,
    set_current_device,
)

# Global lock for model loading to prevent meta tensor issues with multiple GPUs
_MODEL_LOAD_LOCK = threading.Lock()

_PEFT_CONFIG_FILENAME = "adapter_config.json"
_PEFT_METADATA_KEYS = {"peft_version"}
_PEFT_COMPAT_ROOT = Path(tempfile.gettempdir()) / "dianxin_auto_train_peft_compat"
_MODEL_META_FILENAME = "model_meta.json"
DEFAULT_CLASSIFIER_POOLING_STRATEGY = "mean_cls"
DEFAULT_OUTPUT_ACTIVATION = "none"
LEGACY_CLASSIFIER_POOLING_STRATEGY = "pooler_or_mean"
LEGACY_OUTPUT_ACTIVATION = "relu"
VALID_CLASSIFIER_POOLING_STRATEGIES = {DEFAULT_CLASSIFIER_POOLING_STRATEGY, LEGACY_CLASSIFIER_POOLING_STRATEGY}
VALID_OUTPUT_ACTIVATIONS = {DEFAULT_OUTPUT_ACTIVATION, LEGACY_OUTPUT_ACTIVATION}


class InferenceModelStateError(RuntimeError):
    """Raised when loaded model artifacts are internally inconsistent."""


def _torch_load(path: Path, *, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _hash_file(path: Path, digest) -> None:
    digest.update(path.as_posix().encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")


def _build_artifact_cache_key(
    adapter_path: Path,
    head_path: Path,
    label_mapping_path: Path,
    model_meta_path: Path | None = None,
) -> str:
    digest = hashlib.sha256()
    for path in (head_path, label_mapping_path):
        _hash_file(path, digest)
    if model_meta_path is not None and model_meta_path.exists():
        _hash_file(model_meta_path, digest)
    for path in sorted(item for item in adapter_path.rglob("*") if item.is_file()):
        _hash_file(path, digest)
    return digest.hexdigest()


def _validate_label_mappings(model_id: str, label_to_id: object, id_to_label: object) -> None:
    if not isinstance(label_to_id, dict) or not isinstance(id_to_label, dict):
        raise InferenceModelStateError(f"Model {model_id} label mapping must contain two dictionaries")
    if not label_to_id or not id_to_label:
        raise InferenceModelStateError(f"Model {model_id} label mapping must not be empty")

    expected_ids = set(range(len(id_to_label)))
    actual_ids = set(id_to_label)
    if actual_ids != expected_ids:
        raise InferenceModelStateError(
            f"Model {model_id} label ids must be contiguous 0..{len(id_to_label) - 1}; "
            f"got {sorted(actual_ids)}"
        )

    inverse = {label: idx for idx, label in id_to_label.items()}
    if label_to_id != inverse:
        raise InferenceModelStateError(f"Model {model_id} label_to_id and id_to_label are inconsistent")


def _classifier_head_num_labels(state_dict: object, head_path: Path) -> int:
    if not isinstance(state_dict, dict):
        raise InferenceModelStateError(f"Classifier head {head_path} must be a state dict")

    weight = state_dict.get("weight")
    bias = state_dict.get("bias")
    if weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2:
        raise InferenceModelStateError(f"Classifier head {head_path} is missing a 2D weight tensor")

    num_labels = int(weight.shape[0])
    if bias is not None:
        if not hasattr(bias, "shape") or len(bias.shape) != 1 or int(bias.shape[0]) != num_labels:
            raise InferenceModelStateError(f"Classifier head {head_path} bias shape does not match weight shape")
    return num_labels


def _load_model_meta(model_root: Path) -> tuple[Path | None, str, str]:
    meta_path = model_root / _MODEL_META_FILENAME
    if not meta_path.exists():
        return None, LEGACY_CLASSIFIER_POOLING_STRATEGY, LEGACY_OUTPUT_ACTIVATION
    try:
        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InferenceModelStateError(f"Failed to read model metadata {meta_path}: {exc}") from exc
    pooling_strategy = str(
        raw_meta.get("classifier_pooling_strategy", DEFAULT_CLASSIFIER_POOLING_STRATEGY)
    ).strip().lower()
    output_activation = str(raw_meta.get("output_activation", DEFAULT_OUTPUT_ACTIVATION)).strip().lower()
    if pooling_strategy not in VALID_CLASSIFIER_POOLING_STRATEGIES:
        raise InferenceModelStateError(
            f"Model metadata {meta_path} has unsupported classifier_pooling_strategy: {pooling_strategy}"
        )
    if output_activation not in VALID_OUTPUT_ACTIVATIONS:
        raise InferenceModelStateError(
            f"Model metadata {meta_path} has unsupported output_activation: {output_activation}"
        )
    return meta_path, pooling_strategy, output_activation


def _is_noop_unsupported_peft_config(key: str, value: object, raw_config: dict) -> bool:
    if key in _PEFT_METADATA_KEYS:
        return True
    if value is None or value is False or value == {} or value == []:
        return True
    # PEFT 0.18 writes the default QaLoRA group size even when QaLoRA is disabled.
    if key == "qalora_group_size" and not raw_config.get("use_qalora", False):
        return True
    return False


def _resolve_peft_adapter_path(adapter_path: Path) -> Path:
    """Return a PEFT-compatible adapter path without mutating trained artifacts.

    Newer PEFT versions persist no-op config keys that older runtime builds cannot
    parse. When all unsupported keys are no-ops, create a temp adapter directory
    with a sanitized config and symlinks to the original adapter weights.
    """
    config_path = adapter_path / _PEFT_CONFIG_FILENAME
    if not config_path.exists():
        return adapter_path

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    supported_keys = set(inspect.signature(LoraConfig).parameters)
    unsupported = {key: value for key, value in raw_config.items() if key not in supported_keys}
    if not unsupported:
        return adapter_path

    unsafe = {
        key: value
        for key, value in unsupported.items()
        if not _is_noop_unsupported_peft_config(key, value, raw_config)
    }
    if unsafe:
        raise ValueError(
            f"Adapter config contains unsupported active PEFT options in {config_path}: {sorted(unsafe)}"
        )

    sanitized_config = {key: value for key, value in raw_config.items() if key in supported_keys}
    digest = hashlib.sha256(
        str(adapter_path.resolve()).encode("utf-8") + b"\0" + config_path.read_bytes()
    ).hexdigest()[:16]
    compat_path = _PEFT_COMPAT_ROOT / digest
    compat_path.mkdir(parents=True, exist_ok=True)

    sanitized_path = compat_path / _PEFT_CONFIG_FILENAME
    sanitized_text = json.dumps(sanitized_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if not sanitized_path.exists() or sanitized_path.read_text(encoding="utf-8") != sanitized_text:
        sanitized_path.write_text(sanitized_text, encoding="utf-8")

    for source in adapter_path.iterdir():
        if source.name == _PEFT_CONFIG_FILENAME:
            continue
        target = compat_path / source.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(source)
        except FileExistsError:
            continue

    logger.warning(
        "Using sanitized PEFT config for {} after ignoring no-op unsupported keys: {}",
        adapter_path,
        sorted(unsupported),
    )
    return compat_path


class InferenceRequest:
    def __init__(self, model_id: str, texts: list[str], top_n: int):
        self.model_id = model_id
        self.texts = texts
        self.top_n = top_n
        self.created_at = time.time()
        self.future: Future = Future()
        self._lock = threading.Lock()
        self._pending = len(texts)
        self._labels: list[str | None] = [None] * len(texts)
        self._top_n: list[list[tuple[str, float]] | None] = [None] * len(texts)
        self._probs: list[dict[str, float] | None] = [None] * len(texts)

    def set_item_result(
        self,
        index: int,
        label: str,
        top_n: list[tuple[str, float]],
        probs: dict[str, float],
    ) -> None:
        with self._lock:
            if self.future.done():
                return
            self._labels[index] = label
            self._top_n[index] = top_n
            self._probs[index] = probs
            self._pending -= 1
            if self._pending == 0:
                self.future.set_result(
                    {
                        "model_id": self.model_id,
                        "labels": [label for label in self._labels if label is not None],
                        "top_n": [item for item in self._top_n if item is not None],
                        "label_probabilities": [item for item in self._probs if item is not None],
                    }
                )

    def set_exception(self, exc: Exception) -> None:
        with self._lock:
            if not self.future.done():
                self.future.set_exception(exc)


@dataclass
class InferenceItem:
    request: InferenceRequest
    index: int
    text: str
    enqueued_at: float


@dataclass
class AdapterState:
    model_id: str
    adapter_path: Path
    head_path: Path
    label_mapping_path: Path
    max_length: int
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    cache_key: str
    classifier_pooling_strategy: str = LEGACY_CLASSIFIER_POOLING_STRATEGY
    output_activation: str = LEGACY_OUTPUT_ACTIVATION
    draining: bool = False
    active_batches: int = 0
    unload_event: threading.Event = field(default_factory=threading.Event)
    loaded_at: float = field(default_factory=time.time)


class InferenceTextClassifier(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.bert = encoder
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.head: nn.Linear | None = None
        self.pooling_strategy = LEGACY_CLASSIFIER_POOLING_STRATEGY
        self.output_activation = LEGACY_OUTPUT_ACTIVATION

    def set_head(self, head: nn.Linear) -> None:
        self.head = head

    def set_adapter_behavior(self, pooling_strategy: str, output_activation: str) -> None:
        if pooling_strategy not in VALID_CLASSIFIER_POOLING_STRATEGIES:
            raise InferenceModelStateError(f"Unsupported classifier_pooling_strategy: {pooling_strategy}")
        if output_activation not in VALID_OUTPUT_ACTIVATIONS:
            raise InferenceModelStateError(f"Unsupported output_activation: {output_activation}")
        self.pooling_strategy = pooling_strategy
        self.output_activation = output_activation

    def forward(self, input_ids, attention_mask):
        if self.head is None:
            raise RuntimeError("Classifier head is not set for current adapter")
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
        mean_output = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        if self.pooling_strategy == DEFAULT_CLASSIFIER_POOLING_STRATEGY:
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                cls_output = outputs.pooler_output
            else:
                cls_output = outputs.last_hidden_state[:, 0, :]
            pooled_output = 0.5 * mean_output + 0.5 * cls_output
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = mean_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.head(dropout_output)
        if self.output_activation == LEGACY_OUTPUT_ACTIVATION:
            return self.relu(linear_output)
        return linear_output


class InferenceWorker:
    def __init__(
        self,
        *,
        worker_id: str,
        device: torch.device,
        base_model: str,
        max_batch_size: int,
        manager: "LoraInferenceManager",
    ):
        self.worker_id = worker_id
        self.device = device
        self.base_model_name = base_model
        self.max_batch_size = max_batch_size
        self.manager = manager
        self.control_queue: queue.Queue[tuple[Callable[["InferenceWorker"], None], Future]] = queue.Queue()
        self.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"infer-worker-{worker_id}", daemon=True)
        self._ready = threading.Event()
        self._base_model: nn.Module | None = None
        self._peft_model: PeftModel | None = None
        self._classifier: InferenceTextClassifier | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._adapter_name_map: dict[str, str] = {}
        self._head_cache: dict[str, nn.Linear] = {}
        self._adapter_cache_keys: dict[str, str] = {}

    def start(self) -> None:
        self.thread.start()

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout=timeout)

    def submit_control(self, func: Callable[["InferenceWorker"], None]) -> Future:
        future: Future = Future()
        self.control_queue.put((func, future))
        self.manager.notify_workers()
        return future

    def stop(self) -> None:
        self.shutdown_event.set()
        self.manager.notify_workers()

    def _run(self) -> None:
        try:
            self._initialize()
        except Exception as exc:
            logger.exception("Worker {} failed to initialize: {}", self.worker_id, exc)
            self._ready.set()
            return

        self._ready.set()
        while not self.shutdown_event.is_set():
            self._drain_control_queue()
            adapter_id, items = self.manager.get_next_batch(self.max_batch_size, timeout=0.2)
            if adapter_id is None or not items:
                continue
            try:
                self._ensure_adapter_ready(adapter_id)
                self._run_batch(adapter_id, items)
            except Exception as exc:
                logger.exception("Worker {} failed on adapter {}: {}", self.worker_id, adapter_id, exc)
                for item in items:
                    item.request.set_exception(exc)
            finally:
                self.manager.finish_batch(adapter_id)

    def _initialize(self) -> None:
        # 使用全局锁确保只有一个 Worker 同时加载模型
        # 这是为了避免多 GPU 环境下出现 meta tensor 错误
        with _MODEL_LOAD_LOCK:
            if self.device.type != "cpu":
                set_current_device(self.device)
            
            # 加载基础模型到 CPU，然后转移到目标设备
            self._base_model = AutoModel.from_pretrained(self.base_model_name)
            self._base_model = self._base_model.to(self.device)
            self._base_model.eval()
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # 创建分类器并转移到目标设备
            self._classifier = InferenceTextClassifier(self._base_model)
            self._classifier = self._classifier.to(self.device)
            self._classifier.eval()
        
        logger.info("Worker {} ready on {}", self.worker_id, self.device)

    def _drain_control_queue(self) -> None:
        while True:
            try:
                func, future = self.control_queue.get_nowait()
            except queue.Empty:
                break
            try:
                func(self)
            except Exception as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)

    def _ensure_adapter_ready(self, adapter_id: str) -> None:
        model_info = self.manager.get_adapter_state(adapter_id)
        if model_info is None:
            raise RuntimeError(f"Adapter {adapter_id} not registered")
        cached_key = self._adapter_cache_keys.get(adapter_id)
        if (
            cached_key == model_info.cache_key
            and adapter_id in self._adapter_name_map
            and adapter_id in self._head_cache
        ):
            return
        if adapter_id in self._adapter_name_map or adapter_id in self._head_cache:
            self._evict_adapter(adapter_id, require_delete=True)

        if self._peft_model is None:
            self._load_first_adapter(adapter_id, model_info.adapter_path)
            if self._classifier is not None:
                self._classifier.bert = self._peft_model
        elif adapter_id not in self._adapter_name_map:
            self._load_additional_adapter(adapter_id, model_info.adapter_path)
        if adapter_id not in self._head_cache:
            head = self._load_head(model_info.head_path, len(model_info.id_to_label))
            self._head_cache[adapter_id] = head
        self._adapter_cache_keys[adapter_id] = model_info.cache_key

    def _load_first_adapter(self, adapter_id: str, adapter_path: Path) -> None:
        if self._base_model is None:
            raise RuntimeError("Base model not initialized")
        peft_adapter_path = _resolve_peft_adapter_path(adapter_path)
        signature = inspect.signature(PeftModel.from_pretrained)
        if "adapter_name" in signature.parameters:
            self._peft_model = PeftModel.from_pretrained(
                self._base_model, peft_adapter_path, adapter_name=adapter_id, is_trainable=False
            ).to(self.device)
            adapter_name = adapter_id
        else:
            self._peft_model = PeftModel.from_pretrained(self._base_model, peft_adapter_path).to(self.device)
            adapter_name = getattr(self._peft_model, "active_adapter", "default")
            if adapter_name != adapter_id and hasattr(self._peft_model, "load_adapter"):
                self._peft_model.load_adapter(peft_adapter_path, adapter_name=adapter_id, is_trainable=False)
                adapter_name = adapter_id
        self._peft_model.eval()
        self._adapter_name_map[adapter_id] = adapter_name

    def _load_additional_adapter(self, adapter_id: str, adapter_path: Path) -> None:
        if self._peft_model is None:
            raise RuntimeError("PEFT model not initialized")
        if not hasattr(self._peft_model, "load_adapter"):
            raise RuntimeError("Current PEFT version does not support loading multiple adapters")
        peft_adapter_path = _resolve_peft_adapter_path(adapter_path)
        self._peft_model.load_adapter(peft_adapter_path, adapter_name=adapter_id, is_trainable=False)
        self._peft_model.eval()
        self._adapter_name_map[adapter_id] = adapter_id

    def _load_head(self, head_path: Path, num_labels: int) -> nn.Linear:
        if self._classifier is None or self._classifier.bert is None:
            raise RuntimeError("Classifier not initialized")
        hidden = self._classifier.bert.config.hidden_size
        head = nn.Linear(hidden, num_labels)
        state_dict = _torch_load(head_path, map_location="cpu")
        actual_num_labels = _classifier_head_num_labels(state_dict, head_path)
        if actual_num_labels != num_labels:
            raise InferenceModelStateError(
                f"Classifier head {head_path} outputs {actual_num_labels} labels, "
                f"but label mapping contains {num_labels}"
            )
        head.load_state_dict(state_dict)
        head.to(self.device)
        head.eval()
        return head

    def _activate_adapter(self, adapter_id: str) -> AdapterState:
        if self._peft_model is None or self._classifier is None:
            raise RuntimeError("PEFT model not initialized")
        model_info = self.manager.get_adapter_state(adapter_id)
        if model_info is None:
            raise RuntimeError(f"Adapter {adapter_id} not registered")
        if self._adapter_cache_keys.get(adapter_id) != model_info.cache_key:
            self._ensure_adapter_ready(adapter_id)
            model_info = self.manager.get_adapter_state(adapter_id)
            if model_info is None:
                raise RuntimeError(f"Adapter {adapter_id} not registered")
        adapter_name = self._adapter_name_map.get(adapter_id)
        if adapter_name is None:
            raise RuntimeError(f"Adapter {adapter_id} not loaded in worker")
        if hasattr(self._peft_model, "set_adapter"):
            self._peft_model.set_adapter(adapter_name)
        else:
            active = getattr(self._peft_model, "active_adapter", None)
            if adapter_name != active:
                raise RuntimeError("PEFT adapter switching is not supported by this version")
        self._classifier.set_head(self._head_cache[adapter_id])
        self._classifier.set_adapter_behavior(model_info.classifier_pooling_strategy, model_info.output_activation)
        return model_info

    def _run_batch(self, adapter_id: str, items: list[InferenceItem]) -> None:
        if self._tokenizer is None or self._classifier is None:
            raise RuntimeError("Worker not initialized")
        model_info = self._activate_adapter(adapter_id)
        texts = [item.text for item in items]
        encoded = self._tokenizer(
            texts,
            padding="max_length",
            max_length=model_info.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self._classifier(input_ids, attention_mask)
            self._validate_outputs(adapter_id, outputs, len(items), len(model_info.id_to_label))
            probs = F.softmax(outputs, dim=1).float().cpu().tolist()

        id_to_label = model_info.id_to_label
        labels = [id_to_label[idx] for idx in range(len(id_to_label))]
        for item, row in zip(items, probs):
            label_probs = {label: float(prob) for label, prob in zip(labels, row)}
            sorted_items = sorted(label_probs.items(), key=lambda kv: kv[1], reverse=True)
            top_n = min(item.request.top_n, len(sorted_items))
            item.request.set_item_result(item.index, sorted_items[0][0], sorted_items[:top_n], label_probs)

    def _validate_outputs(self, adapter_id: str, outputs, batch_size: int, num_labels: int) -> None:
        shape = getattr(outputs, "shape", None)
        if shape is None or len(shape) != 2:
            raise InferenceModelStateError(f"Model {adapter_id} produced invalid output shape: {shape}")
        actual_batch_size = int(shape[0])
        actual_num_labels = int(shape[1])
        if actual_batch_size != batch_size:
            raise InferenceModelStateError(
                f"Model {adapter_id} produced batch size {actual_batch_size}, expected {batch_size}"
            )
        if actual_num_labels != num_labels:
            head = self._head_cache.get(adapter_id)
            head_out_features = getattr(head, "out_features", None)
            raise InferenceModelStateError(
                f"Model {adapter_id} produced {actual_num_labels} logits, "
                f"but label mapping contains {num_labels} labels "
                f"(worker={self.worker_id}, head_out_features={head_out_features}, "
                f"cache_key={self._adapter_cache_keys.get(adapter_id)})"
            )

    def unload_adapter(self, adapter_id: str) -> None:
        self._evict_adapter(adapter_id, require_delete=False)
        empty_cache(self.device)

    def _evict_adapter(self, adapter_id: str, *, require_delete: bool) -> None:
        adapter_name = self._adapter_name_map.get(adapter_id)
        if (
            require_delete
            and self._peft_model is not None
            and adapter_name is not None
            and not hasattr(self._peft_model, "delete_adapter")
        ):
            raise InferenceModelStateError(
                "Current PEFT version does not support reloading changed adapters without deleting the old adapter"
            )
        self._adapter_name_map.pop(adapter_id, None)
        self._head_cache.pop(adapter_id, None)
        self._adapter_cache_keys.pop(adapter_id, None)
        if self._peft_model is None or adapter_name is None:
            return
        if hasattr(self._peft_model, "delete_adapter"):
            self._peft_model.delete_adapter(adapter_name)


class LoraInferenceManager:
    def __init__(self) -> None:
        self.base_model = get_inference_base_model()
        self.max_batch_size = max(1, get_inference_max_batch_size())
        self.age_weight_seconds = max(0.1, get_inference_queue_age_weight_seconds())
        self.unload_timeout = max(1.0, get_inference_unload_timeout())
        self.workers_per_gpu = max(1, get_inference_workers_per_gpu())
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.adapter_states: dict[str, AdapterState] = {}
        self.adapter_queues: dict[str, deque[InferenceItem]] = {}
        self.workers: list[InferenceWorker] = []
        self.shutdown_event = threading.Event()

    def start(self) -> None:
        if self.workers:
            return
        devices = self._available_devices()
        if not devices:
            devices = [torch.device("cpu")]
        for device in devices:
            for worker_idx in range(self.workers_per_gpu):
                worker_id = f"{device.type}{device.index or 0}-{worker_idx}"
                worker = InferenceWorker(
                    worker_id=worker_id,
                    device=device,
                    base_model=self.base_model,
                    max_batch_size=self.max_batch_size,
                    manager=self,
                )
                self.workers.append(worker)
                worker.start()
        for worker in self.workers:
            worker.wait_ready(timeout=600)

    def stop(self) -> None:
        self.shutdown_event.set()
        for worker in self.workers:
            worker.stop()
        self.notify_workers()
        for worker in self.workers:
            worker.thread.join(timeout=5)

    def notify_workers(self) -> None:
        with self.condition:
            self.condition.notify_all()

    def load_model(self, model_dir: str, max_length: int) -> str:
        model_id = model_dir
        adapter_state = self._resolve_adapter_state(model_id, max_length)
        with self.lock:
            if model_id in self.adapter_states and self.adapter_states[model_id].draining:
                raise RuntimeError("model is unloading")
            if model_id not in self.adapter_states:
                self.adapter_states[model_id] = adapter_state
                self.adapter_queues.setdefault(model_id, deque())
            else:
                self.adapter_states[model_id] = adapter_state
        self._preload_adapter(model_id)
        return model_id

    def unload_model(self, model_id: str) -> None:
        with self.lock:
            state = self.adapter_states.get(model_id)
            if state is None:
                raise KeyError("model not loaded")
            state.draining = True
            state.unload_event.clear()
            should_unload = not self.adapter_queues.get(model_id) and state.active_batches == 0
        if should_unload:
            self._finalize_unload(model_id)
        if not state.unload_event.wait(timeout=self.unload_timeout):
            raise TimeoutError("model unload timed out")

    def enqueue(self, model_id: str, texts: list[str], top_n: int) -> Future:
        if not texts:
            raise ValueError("texts must not be empty")
        request = InferenceRequest(model_id, texts, top_n)
        now = time.time()
        items = [InferenceItem(request=request, index=i, text=text, enqueued_at=now) for i, text in enumerate(texts)]
        with self.lock:
            state = self.adapter_states.get(model_id)
            if state is None:
                raise KeyError("model not loaded")
            if state.draining:
                raise RuntimeError("model is unloading")
            queue_ref = self.adapter_queues.setdefault(model_id, deque())
            queue_ref.extend(items)
            self.condition.notify_all()
        return request.future

    def get_next_batch(self, max_batch_size: int, timeout: float) -> tuple[str | None, list[InferenceItem]]:
        with self.condition:
            end = time.time() + timeout
            while True:
                if self.shutdown_event.is_set():
                    return None, []
                adapter_id = self._select_adapter_locked()
                if adapter_id is not None:
                    items = self._pop_items_locked(adapter_id, max_batch_size)
                    if items:
                        state = self.adapter_states.get(adapter_id)
                        if state is not None:
                            state.active_batches += 1
                        return adapter_id, items
                remaining = end - time.time()
                if remaining <= 0:
                    return None, []
                self.condition.wait(timeout=remaining)

    def finish_batch(self, adapter_id: str) -> None:
        with self.lock:
            state = self.adapter_states.get(adapter_id)
            if state is None:
                return
            state.active_batches = max(0, state.active_batches - 1)
            should_unload = state.draining and not self.adapter_queues.get(adapter_id) and state.active_batches == 0
        if should_unload:
            self._finalize_unload(adapter_id)

    def get_adapter_state(self, adapter_id: str) -> AdapterState | None:
        with self.lock:
            return self.adapter_states.get(adapter_id)

    def list_models(self) -> list[dict]:
        """列出所有模型及其状态"""
        models = []
        model_output_dir = get_model_output_dir()
        
        if model_output_dir.exists():
            for model_dir in model_output_dir.iterdir():
                if model_dir.is_dir():
                    model_id = model_dir.name
                    with self.lock:
                        state = self.adapter_states.get(model_id)
                    
                    if state is not None:
                        # 模型已加载
                        uptime = time.time() - state.loaded_at
                        # 获取 GPU ID（从任意一个 worker 获取）
                        gpu_id = None
                        if self.workers:
                            gpu_id = self.workers[0].device.index if self.workers[0].device.type != "cpu" else None
                        models.append({
                            "model_id": model_id,
                            "status": "loaded",
                            "gpu_id": gpu_id,
                            "uptime_seconds": round(uptime, 2)
                        })
                    else:
                        # 模型未加载但目录存在
                        models.append({
                            "model_id": model_id,
                            "status": "unloaded",
                            "gpu_id": None,
                            "uptime_seconds": None
                        })
        
        return models

    def query_models(self, model_ids: list[str]) -> list[dict]:
        """根据模型ID列表查询模型"""
        all_models = {m["model_id"]: m for m in self.list_models()}
        result = []
        for model_id in model_ids:
            if model_id in all_models:
                result.append(all_models[model_id])
        return result

    def get_service_status(self) -> dict:
        """获取推理服务状态"""
        workers_status = []
        
        for worker in self.workers:
            device = worker.device
            worker_status = {
                "worker_id": worker.worker_id,
                "device": str(device),
                "total_memory_mb": 0.0,
                "used_memory_mb": 0.0,
                "free_memory_mb": 0.0,
                "memory_usage_percent": 0.0
            }
            
            total_mb, used_mb, free_mb, usage_percent = get_device_memory_info(device)
            worker_status["total_memory_mb"] = total_mb
            worker_status["used_memory_mb"] = used_mb
            worker_status["free_memory_mb"] = free_mb
            worker_status["memory_usage_percent"] = usage_percent
            
            workers_status.append(worker_status)
        
        # 统计待处理请求数
        pending_count = 0
        with self.lock:
            for queue_ref in self.adapter_queues.values():
                pending_count += len(queue_ref)
            loaded_count = len(self.adapter_states)
        
        return {
            "service_status": "running" if self.workers else "starting",
            "workers": workers_status,
            "total_workers": len(self.workers),
            "loaded_models_count": loaded_count,
            "pending_requests": pending_count
        }

    def _available_devices(self) -> list[torch.device]:
        return get_available_accelerator_devices()

    def _select_adapter_locked(self) -> str | None:
        now = time.time()
        best_adapter = None
        best_score = -1.0
        for adapter_id, queue_ref in self.adapter_queues.items():
            if not queue_ref:
                continue
            oldest_age = now - queue_ref[0].enqueued_at
            score = len(queue_ref) + (oldest_age / self.age_weight_seconds)
            if score > best_score:
                best_score = score
                best_adapter = adapter_id
        return best_adapter

    def _pop_items_locked(self, adapter_id: str, max_batch_size: int) -> list[InferenceItem]:
        queue_ref = self.adapter_queues.get(adapter_id)
        if not queue_ref:
            return []
        items: list[InferenceItem] = []
        while queue_ref and len(items) < max_batch_size:
            items.append(queue_ref.popleft())
        return items

    def _preload_adapter(self, model_id: str) -> None:
        if not self.workers:
            return
        futures = [worker.submit_control(lambda w, mid=model_id: w._ensure_adapter_ready(mid)) for worker in self.workers]
        for future in futures:
            future.result(timeout=300)

    def _finalize_unload(self, model_id: str) -> None:
        with self.lock:
            state = self.adapter_states.get(model_id)
        if state is None:
            return
        futures = [worker.submit_control(lambda w, mid=model_id: w.unload_adapter(mid)) for worker in self.workers]
        for future in futures:
            with contextlib.suppress(Exception):
                future.result(timeout=60)
        with self.lock:
            self.adapter_states.pop(model_id, None)
            self.adapter_queues.pop(model_id, None)
            state.unload_event.set()

    def _resolve_adapter_state(self, model_id: str, max_length: int) -> AdapterState:
        model_root = get_model_output_dir() / model_id
        if not model_root.exists():
            raise FileNotFoundError(f"Model directory not found: {model_root}")
        head_candidates = list(model_root.glob("*.head.pt"))
        if len(head_candidates) != 1:
            raise ValueError(f"Expected exactly one .head.pt file in {model_root}")
        head_path = head_candidates[0]
        model_stem = head_path.name[: -len(".head.pt")]
        adapter_path = model_root / f"{model_stem}.lora"
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
        label_mapping_path = model_root / f"{head_path.name}.pkl"
        if not label_mapping_path.exists():
            raise FileNotFoundError(f"Label mapping not found: {label_mapping_path}")
        with label_mapping_path.open("rb") as handle:
            label_to_id, id_to_label = pickle.load(handle)
        _validate_label_mappings(model_id, label_to_id, id_to_label)
        state_dict = _torch_load(head_path, map_location="cpu")
        head_num_labels = _classifier_head_num_labels(state_dict, head_path)
        if head_num_labels != len(id_to_label):
            raise InferenceModelStateError(
                f"Classifier head {head_path} outputs {head_num_labels} labels, "
                f"but label mapping contains {len(id_to_label)}"
            )
        model_meta_path, classifier_pooling_strategy, output_activation = _load_model_meta(model_root)
        cache_key = _build_artifact_cache_key(adapter_path, head_path, label_mapping_path, model_meta_path)
        return AdapterState(
            model_id=model_id,
            adapter_path=adapter_path,
            head_path=head_path,
            label_mapping_path=label_mapping_path,
            max_length=max_length,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            cache_key=cache_key,
            classifier_pooling_strategy=classifier_pooling_strategy,
            output_activation=output_activation,
        )


_INFERENCE_MANAGER: LoraInferenceManager | None = None


def get_inference_manager() -> LoraInferenceManager:
    global _INFERENCE_MANAGER
    if _INFERENCE_MANAGER is None:
        _INFERENCE_MANAGER = LoraInferenceManager()
    return _INFERENCE_MANAGER
