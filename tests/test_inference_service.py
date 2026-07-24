import os
import sys
import threading
import types
import unittest
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from unittest import mock


def _install_runtime_stubs() -> None:
    torch = types.ModuleType("torch")

    class Device:
        def __init__(self, spec: str):
            raw_type, _, raw_index = spec.partition(":")
            self.type = raw_type
            self.index = int(raw_index) if raw_index else None

        def __str__(self) -> str:
            if self.index is None:
                return self.type
            return f"{self.type}:{self.index}"

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def device_count() -> int:
            return 0

    torch.device = Device
    torch.Tensor = object
    torch.dtype = object
    torch.optim = types.SimpleNamespace(Optimizer=object)
    torch.backends = types.SimpleNamespace()
    torch.cuda = Cuda()
    torch.no_grad = lambda: NoGrad()
    torch.load = lambda *args, **kwargs: {}

    nn = types.ModuleType("torch.nn")

    class Module:
        def __init__(self):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    class Linear(Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.loaded_state_dict = None

        def load_state_dict(self, state_dict):
            self.loaded_state_dict = state_dict

    nn.Module = Module
    nn.Linear = Linear

    functional = types.ModuleType("torch.nn.functional")
    functional.softmax = lambda outputs, dim: outputs

    peft = types.ModuleType("peft")

    class LoraConfig:
        def __init__(self, r=None, lora_alpha=None):
            self.r = r
            self.lora_alpha = lora_alpha

    class PeftModel(Module):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def load_adapter(self, *args, **kwargs):
            return None

        def delete_adapter(self, *args, **kwargs):
            return None

    peft.LoraConfig = LoraConfig
    peft.PeftModel = PeftModel

    transformers = types.ModuleType("transformers")
    transformers.AutoModel = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: Module())
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())

    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", nn)
    sys.modules.setdefault("torch.nn.functional", functional)
    sys.modules.setdefault("peft", peft)
    sys.modules.setdefault("transformers", transformers)


_install_runtime_stubs()

from src.inference import service


class FakeShapeTensor:
    def __init__(self, shape):
        self.shape = shape


class FakeHead:
    def __init__(self, out_features: int):
        self.out_features = out_features


class FakePeftModel:
    def __init__(self):
        self.deleted: list[str] = []

    def delete_adapter(self, adapter_name: str) -> None:
        self.deleted.append(adapter_name)


class FakeManager:
    def __init__(self, state: service.AdapterState):
        self.state = state

    def get_adapter_state(self, adapter_id: str):
        if adapter_id == self.state.model_id:
            return self.state
        return None

    def notify_workers(self) -> None:
        return None


class InferenceServiceTests(unittest.TestCase):
    def test_validate_outputs_rejects_logit_label_mismatch(self):
        worker = object.__new__(service.InferenceWorker)
        worker.worker_id = "cpu0-0"
        worker._head_cache = {"chengshi_huanwei": FakeHead(out_features=8)}
        worker._adapter_cache_keys = {"chengshi_huanwei": "old-cache-key"}

        with self.assertRaisesRegex(service.InferenceModelStateError, "produced 8 logits"):
            worker._validate_outputs(
                "chengshi_huanwei",
                FakeShapeTensor((1, 8)),
                batch_size=1,
                num_labels=7,
            )

    def test_changed_artifact_cache_key_reloads_adapter_and_head(self):
        state = service.AdapterState(
            model_id="chengshi_huanwei",
            adapter_path=Path("/tmp/chengshi_huanwei.lora"),
            head_path=Path("/tmp/chengshi_huanwei.head.pt"),
            label_mapping_path=Path("/tmp/chengshi_huanwei.head.pt.pkl"),
            max_length=512,
            label_to_id={"a": 0, "b": 1},
            id_to_label={0: "a", 1: "b"},
            cache_key="new-cache-key",
        )
        manager = FakeManager(state)
        worker = service.InferenceWorker(
            worker_id="cpu0-0",
            device=service.torch.device("cpu"),
            base_model="bert-base-chinese",
            max_batch_size=8,
            manager=manager,
        )
        peft_model = FakePeftModel()
        worker._peft_model = peft_model
        worker._classifier = types.SimpleNamespace(bert=types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=768)))
        worker._adapter_name_map = {"chengshi_huanwei": "old-adapter-name"}
        worker._head_cache = {"chengshi_huanwei": FakeHead(out_features=8)}
        worker._adapter_cache_keys = {"chengshi_huanwei": "old-cache-key"}

        loaded_adapters: list[str] = []

        def load_additional(adapter_id, adapter_path):
            loaded_adapters.append(adapter_id)
            worker._adapter_name_map[adapter_id] = adapter_id

        worker._load_additional_adapter = load_additional
        worker._load_head = lambda head_path, num_labels: FakeHead(out_features=num_labels)

        worker._ensure_adapter_ready("chengshi_huanwei")

        self.assertEqual(peft_model.deleted, ["old-adapter-name"])
        self.assertEqual(loaded_adapters, ["chengshi_huanwei"])
        self.assertEqual(worker._adapter_cache_keys["chengshi_huanwei"], "new-cache-key")
        self.assertEqual(worker._head_cache["chengshi_huanwei"].out_features, 2)

    def test_label_mapping_must_be_contiguous_and_inverse(self):
        with self.assertRaisesRegex(service.InferenceModelStateError, "contiguous"):
            service._validate_label_mappings(
                "bad_model",
                {"a": 0, "b": 2},
                {0: "a", 2: "b"},
            )

        with self.assertRaisesRegex(service.InferenceModelStateError, "inconsistent"):
            service._validate_label_mappings(
                "bad_model",
                {"a": 0, "b": 1},
                {0: "a", 1: "c"},
            )


class InferenceQueueGuardTests(unittest.TestCase):
    """针对“Worker 卡死导致服务整体无响应”事故的防护测试。"""

    def _make_manager(self, model_id: str = "m1") -> service.LoraInferenceManager:
        manager = service.LoraInferenceManager()
        state = service.AdapterState(
            model_id=model_id,
            adapter_path=Path("/tmp/m1.lora"),
            head_path=Path("/tmp/m1.head.pt"),
            label_mapping_path=Path("/tmp/m1.head.pt.pkl"),
            max_length=512,
            label_to_id={"a": 0},
            id_to_label={0: "a"},
            cache_key="k",
        )
        manager.adapter_states[model_id] = state
        manager.adapter_queues[model_id] = deque()
        return manager

    def test_enqueue_rejects_when_queue_is_full(self):
        manager = self._make_manager()
        manager.max_pending_items = 3
        manager.enqueue("m1", ["t1", "t2"], 1)
        with self.assertRaisesRegex(RuntimeError, "queue is full"):
            manager.enqueue("m1", ["t3", "t4"], 1)
        # 原队列不受影响
        self.assertEqual(len(manager.adapter_queues["m1"]), 2)

    def test_cancel_request_removes_queued_items_and_marks_cancelled(self):
        manager = self._make_manager()
        request1 = manager.enqueue("m1", ["a", "b"], 1)
        request2 = manager.enqueue("m1", ["c"], 1)

        removed = manager.cancel_request("m1", request1)

        self.assertEqual(removed, 2)
        self.assertTrue(request1.cancelled.is_set())
        self.assertFalse(request2.cancelled.is_set())
        remaining = manager.adapter_queues["m1"]
        self.assertEqual(len(remaining), 1)
        self.assertIs(remaining[0].request, request2)

    def test_finalize_unload_raises_when_worker_never_completes(self):
        manager = self._make_manager()
        manager.unload_timeout = 0.1
        stuck_future: Future = Future()  # 永不完成，模拟 Worker 卡死

        class StuckWorker:
            def submit_control(self, func):
                return stuck_future

        manager.workers = [StuckWorker()]

        with self.assertRaises(TimeoutError):
            manager._finalize_unload("m1")
        # 失败时不能静默删除状态（避免误报“卸载成功”）
        self.assertIn("m1", manager.adapter_states)

    def test_unload_adapter_defers_empty_cache_to_janitor(self):
        worker = object.__new__(service.InferenceWorker)
        worker.device = service.torch.device("cpu")
        evicted: list[str] = []
        worker._evict_adapter = lambda adapter_id, require_delete: evicted.append(adapter_id)
        submitted: list[object] = []
        original_submit = service._CACHE_JANITOR.submit
        service._CACHE_JANITOR.submit = lambda device: submitted.append(device)
        try:
            worker.unload_adapter("m1")
        finally:
            service._CACHE_JANITOR.submit = original_submit
        self.assertEqual(evicted, ["m1"])
        # empty_cache 必须交给 janitor，绝不能在 Worker 线程内同步执行
        self.assertEqual(submitted, [worker.device])

    def test_janitor_is_disabled_by_default(self):
        with mock.patch.dict(os.environ, {"INFERENCE_EMPTY_CACHE_ON_UNLOAD": "false"}):
            janitor = service._CacheJanitor()
            janitor.submit(service.torch.device("cpu"))
            self.assertIsNone(janitor._thread)
            self.assertTrue(janitor._queue.empty())

    def test_worker_skips_cancelled_items_before_batch(self):
        manager = self._make_manager()
        request = manager.enqueue("m1", ["a"], 1)
        item = manager.adapter_queues["m1"][0]
        request.cancelled.set()

        batch_calls: list[str] = []
        worker = object.__new__(service.InferenceWorker)
        worker.worker_id = "cpu0-0"
        worker.max_batch_size = 8
        worker.shutdown_event = manager.shutdown_event
        worker._ready = threading.Event()
        worker._initialize = lambda: None
        worker._ensure_adapter_ready = lambda adapter_id: batch_calls.append(adapter_id)
        worker._run_batch = lambda adapter_id, items: batch_calls.append(adapter_id)
        worker._begin_op = lambda name: None
        worker._end_op = lambda: None
        worker._drain_control_queue = lambda: None

        class OneShotManager:
            def __init__(self, real):
                self.calls = 0
                self.real = real

            def get_next_batch(self, max_batch_size, timeout):
                self.calls += 1
                if self.calls == 1:
                    return "m1", [item]
                self.real.shutdown_event.set()
                return None, []

            def finish_batch(self, adapter_id):
                self.real.finish_batch(adapter_id)

        manager.adapter_states["m1"].active_batches = 0
        worker.manager = OneShotManager(manager)
        worker._run()

        # 已取消的请求不应触发 adapter 加载/批处理
        self.assertEqual(batch_calls, [])
        self.assertFalse(request.future.done())


if __name__ == "__main__":
    unittest.main()
