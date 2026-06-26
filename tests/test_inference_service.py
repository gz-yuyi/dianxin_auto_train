import sys
import types
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
