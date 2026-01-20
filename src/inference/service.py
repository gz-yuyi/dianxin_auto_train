import contextlib
import inspect
import pickle
import queue
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
from peft import PeftModel
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
    draining: bool = False
    active_batches: int = 0
    unload_event: threading.Event = field(default_factory=threading.Event)


class InferenceTextClassifier(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.bert = encoder
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.head: nn.Linear | None = None

    def set_head(self, head: nn.Linear) -> None:
        self.head = head

    def forward(self, input_ids, attention_mask):
        if self.head is None:
            raise RuntimeError("Classifier head is not set for current adapter")
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
            pooled_output = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.head(dropout_output)
        return self.relu(linear_output)


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
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self._base_model = AutoModel.from_pretrained(self.base_model_name).to(self.device)
        self._base_model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self._classifier = InferenceTextClassifier(self._base_model).to(self.device)
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
        if adapter_id in self._adapter_name_map and adapter_id in self._head_cache:
            return
        model_info = self.manager.get_adapter_state(adapter_id)
        if model_info is None:
            raise RuntimeError(f"Adapter {adapter_id} not registered")
        if self._peft_model is None:
            self._load_first_adapter(adapter_id, model_info.adapter_path)
            if self._classifier is not None:
                self._classifier.bert = self._peft_model
        elif adapter_id not in self._adapter_name_map:
            self._load_additional_adapter(adapter_id, model_info.adapter_path)
        if adapter_id not in self._head_cache:
            head = self._load_head(model_info.head_path, len(model_info.id_to_label))
            self._head_cache[adapter_id] = head

    def _load_first_adapter(self, adapter_id: str, adapter_path: Path) -> None:
        if self._base_model is None:
            raise RuntimeError("Base model not initialized")
        signature = inspect.signature(PeftModel.from_pretrained)
        if "adapter_name" in signature.parameters:
            self._peft_model = PeftModel.from_pretrained(
                self._base_model, adapter_path, adapter_name=adapter_id, is_trainable=False
            ).to(self.device)
            adapter_name = adapter_id
        else:
            self._peft_model = PeftModel.from_pretrained(self._base_model, adapter_path).to(self.device)
            adapter_name = getattr(self._peft_model, "active_adapter", "default")
            if adapter_name != adapter_id and hasattr(self._peft_model, "load_adapter"):
                self._peft_model.load_adapter(adapter_path, adapter_name=adapter_id, is_trainable=False)
                adapter_name = adapter_id
        self._peft_model.eval()
        self._adapter_name_map[adapter_id] = adapter_name

    def _load_additional_adapter(self, adapter_id: str, adapter_path: Path) -> None:
        if self._peft_model is None:
            raise RuntimeError("PEFT model not initialized")
        if not hasattr(self._peft_model, "load_adapter"):
            raise RuntimeError("Current PEFT version does not support loading multiple adapters")
        self._peft_model.load_adapter(adapter_path, adapter_name=adapter_id, is_trainable=False)
        self._adapter_name_map[adapter_id] = adapter_id

    def _load_head(self, head_path: Path, num_labels: int) -> nn.Linear:
        if self._classifier is None or self._classifier.bert is None:
            raise RuntimeError("Classifier not initialized")
        hidden = self._classifier.bert.config.hidden_size
        head = nn.Linear(hidden, num_labels)
        state_dict = torch.load(head_path, map_location="cpu")
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
            probs = F.softmax(outputs, dim=1).float().cpu().tolist()

        id_to_label = model_info.id_to_label
        for item, row in zip(items, probs):
            labels = [id_to_label[idx] for idx in range(len(row))]
            label_probs = {label: float(prob) for label, prob in zip(labels, row)}
            sorted_items = sorted(label_probs.items(), key=lambda kv: kv[1], reverse=True)
            top_n = min(item.request.top_n, len(sorted_items))
            item.request.set_item_result(item.index, sorted_items[0][0], sorted_items[:top_n], label_probs)

    def unload_adapter(self, adapter_id: str) -> None:
        adapter_name = self._adapter_name_map.pop(adapter_id, None)
        self._head_cache.pop(adapter_id, None)
        if self._peft_model is None or adapter_name is None:
            return
        if hasattr(self._peft_model, "delete_adapter"):
            self._peft_model.delete_adapter(adapter_name)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


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

    def _available_devices(self) -> list[torch.device]:
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
        return []

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
        return AdapterState(
            model_id=model_id,
            adapter_path=adapter_path,
            head_path=head_path,
            label_mapping_path=label_mapping_path,
            max_length=max_length,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
        )


_INFERENCE_MANAGER: LoraInferenceManager | None = None


def get_inference_manager() -> LoraInferenceManager:
    global _INFERENCE_MANAGER
    if _INFERENCE_MANAGER is None:
        _INFERENCE_MANAGER = LoraInferenceManager()
    return _INFERENCE_MANAGER
