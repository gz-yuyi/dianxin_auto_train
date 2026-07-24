import sys
import types
import unittest
from unittest import mock


def _install_runtime_stubs() -> None:
    # src.config -> src.device_utils 在 import 时需要 torch 模块存在
    torch = types.ModuleType("torch")
    torch.device = lambda spec: spec
    torch.Tensor = object
    torch.dtype = object
    torch.optim = types.SimpleNamespace(Optimizer=object)
    torch.backends = types.SimpleNamespace()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    sys.modules.setdefault("torch", torch)


_install_runtime_stubs()

from src.inference import gateway


class GatewayCircuitBreakerTests(unittest.TestCase):
    def _make_gateway(self) -> gateway.InferenceGateway:
        return gateway.InferenceGateway(
            upstream_urls=[
                "http://up0:9011",
                "http://up1:9011",
                "http://up2:9011",
            ],
            timeout=1.0,
            policy="round_robin",
        )

    def _fail(self, gw: gateway.InferenceGateway, index: int, times: int) -> None:
        for _ in range(times):
            gw._record_upstream_failure(index)

    def test_failed_upstream_is_excluded_after_threshold(self):
        gw = self._make_gateway()
        self.assertEqual(gw._cb_failure_threshold, 3)

        self._fail(gw, 0, 2)
        # 未达阈值，仍参与轮询
        self.assertIn(gw.upstreams[0], gw._healthy_upstreams())

        gw._record_upstream_failure(0)
        # 达到阈值，被摘除
        healthy = gw._healthy_upstreams()
        self.assertNotIn(gw.upstreams[0], healthy)
        self.assertEqual(len(healthy), 2)

        # 轮询选择永远不会选中被摘除的 upstream
        selected = {gw._select_round_robin_upstream().index for _ in range(10)}
        self.assertEqual(selected, {1, 2})

    def test_all_dead_falls_back_to_full_round_robin(self):
        gw = self._make_gateway()
        for index in (0, 1, 2):
            self._fail(gw, index, gw._cb_failure_threshold)
        self.assertEqual(gw._healthy_upstreams(), [])
        # 全部被熔断时仍尽力尝试，避免直接放弃服务
        selected = {gw._select_round_robin_upstream().index for _ in range(6)}
        self.assertEqual(selected, {0, 1, 2})

    def test_success_resets_failure_state(self):
        gw = self._make_gateway()
        self._fail(gw, 1, gw._cb_failure_threshold)
        self.assertNotIn(gw.upstreams[1], gw._healthy_upstreams())

        gw._record_upstream_success(1)
        self.assertIn(gw.upstreams[1], gw._healthy_upstreams())
        self.assertEqual(gw._upstream_failures.get(1, 0), 0)

    def test_dead_upstream_rejoins_after_cooldown(self):
        gw = self._make_gateway()
        with mock.patch.object(gateway, "time") as mock_time_module:
            mock_time_module.time.return_value = 1000.0
            self._fail(gw, 2, gw._cb_failure_threshold)
            self.assertNotIn(gw.upstreams[2], gw._healthy_upstreams())

            # 冷却期内仍被摘除
            mock_time_module.time.return_value = 1000.0 + gw._cb_cooldown_seconds - 1
            self.assertNotIn(gw.upstreams[2], gw._healthy_upstreams())

            # 冷却结束后半开恢复
            mock_time_module.time.return_value = 1000.0 + gw._cb_cooldown_seconds + 1
            self.assertIn(gw.upstreams[2], gw._healthy_upstreams())

    def test_predict_records_failure_and_success(self):
        gw = self._make_gateway()
        calls: list[int] = []

        def fake_request(method, upstream, path, payload=None):
            calls.append(upstream.index)
            if upstream.index == 0:
                raise gateway.InferenceGatewayError(504, {"message": "timeout"})
            return {"ok": True}

        gw._request_json = fake_request

        # 让轮询先命中 up0（失败），再命中 up1（成功）
        gw._cursor = 0
        with self.assertRaises(gateway.InferenceGatewayError):
            gw.predict({"model_id": "m", "texts": ["t"], "top_n": 1})
        self.assertEqual(gw._upstream_failures.get(0), 1)

        result = gw.predict({"model_id": "m", "texts": ["t"], "top_n": 1})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(gw._upstream_failures.get(1, 0), 0)

        # 继续让 up0 失败直至熔断，之后 predict 只发往健康 upstream
        self._fail(gw, 0, gw._cb_failure_threshold - 1)
        gw._cursor = 0
        result = gw.predict({"model_id": "m", "texts": ["t"], "top_n": 1})
        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls[-1], 1)


if __name__ == "__main__":
    unittest.main()
