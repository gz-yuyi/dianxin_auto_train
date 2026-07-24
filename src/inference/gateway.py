from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests
from loguru import logger

from src.config import (
    get_inference_cb_cooldown_seconds,
    get_inference_cb_failure_threshold,
    get_inference_lb_policy,
    get_inference_upstream_timeout,
    get_inference_upstream_urls,
)


@dataclass(frozen=True)
class InferenceUpstream:
    index: int
    base_url: str


class InferenceGatewayError(RuntimeError):
    def __init__(self, status_code: int, detail: Any):
        super().__init__(str(detail))
        self.status_code = status_code
        self.detail = detail


class InferenceGateway:
    def __init__(self, upstream_urls: list[str], timeout: float, policy: str) -> None:
        if not upstream_urls:
            raise ValueError("at least one inference upstream is required")
        self.upstreams = [
            InferenceUpstream(index=index, base_url=url.rstrip("/"))
            for index, url in enumerate(upstream_urls)
        ]
        self.timeout = max(1.0, timeout)
        self.policy = policy
        self._cursor = 0
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        # 熔断器状态：连续失败计数与摘除截止时间
        self._cb_failure_threshold = max(1, get_inference_cb_failure_threshold())
        self._cb_cooldown_seconds = max(1.0, get_inference_cb_cooldown_seconds())
        self._upstream_failures: dict[int, int] = {}
        self._upstream_dead_until: dict[int, float] = {}

    def _record_upstream_success(self, index: int) -> None:
        with self._lock:
            self._upstream_failures.pop(index, None)
            self._upstream_dead_until.pop(index, None)

    def _record_upstream_failure(self, index: int) -> None:
        with self._lock:
            failures = self._upstream_failures.get(index, 0) + 1
            self._upstream_failures[index] = failures
            if failures >= self._cb_failure_threshold:
                dead_until = time.time() + self._cb_cooldown_seconds
                if self._upstream_dead_until.get(index, 0.0) < dead_until:
                    self._upstream_dead_until[index] = dead_until
                    logger.warning(
                        "Inference upstream {} marked unhealthy for {}s after {} consecutive failures",
                        index,
                        self._cb_cooldown_seconds,
                        failures,
                    )

    def _healthy_upstreams(self) -> list[InferenceUpstream]:
        now = time.time()
        with self._lock:
            return [
                upstream
                for upstream in self.upstreams
                if self._upstream_dead_until.get(upstream.index, 0.0) <= now
            ]

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        upstream = self._select_upstream()
        try:
            result = self._request_json("post", upstream, "/inference/predict", payload)
        except InferenceGatewayError:
            self._record_upstream_failure(upstream.index)
            raise
        self._record_upstream_success(upstream.index)
        return result

    def load_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        results = self._broadcast("post", "/inference/models/load", payload)
        failures = [result for result in results if not result["ok"]]
        if failures:
            raise InferenceGatewayError(
                502,
                {
                    "message": "model load failed on one or more inference upstreams",
                    "upstreams": failures,
                },
            )

        model_ids = {result["data"].get("model_id") for result in results}
        model_ids.discard(None)
        if len(model_ids) != 1:
            raise InferenceGatewayError(
                502,
                {
                    "message": "inference upstreams returned inconsistent model ids",
                    "upstreams": results,
                },
            )
        return {
            "model_id": next(iter(model_ids)),
            "status": "loaded",
            "message": f"model loaded on {len(results)} upstreams",
        }

    def unload_model(self, payload: dict[str, Any]) -> dict[str, Any]:
        results = self._broadcast("post", "/inference/models/unload", payload)
        failures = [result for result in results if not result["ok"]]
        if failures:
            raise InferenceGatewayError(
                502,
                {
                    "message": "model unload failed on one or more inference upstreams",
                    "upstreams": failures,
                },
            )
        return {
            "model_id": payload.get("model_id"),
            "status": "unloaded",
            "message": f"model unloaded on {len(results)} upstreams",
        }

    def list_models(self) -> dict[str, Any]:
        results = self._broadcast("get", "/inference/models")
        model_groups = [
            result["data"].get("models", [])
            for result in results
            if result["ok"]
        ]
        if not model_groups:
            raise InferenceGatewayError(
                502,
                {
                    "message": "failed to query models from all inference upstreams",
                    "upstreams": results,
                },
            )

        models = self._merge_model_groups(model_groups, expected_upstreams=len(self.upstreams))
        loaded_count = sum(1 for model in models if model["status"] == "loaded")
        return {"models": models, "total": len(models), "loaded_count": loaded_count}

    def query_models(self, payload: dict[str, Any]) -> dict[str, Any]:
        results = self._broadcast("post", "/inference/models/query", payload)
        model_groups = [
            result["data"].get("models", [])
            for result in results
            if result["ok"]
        ]
        if not model_groups:
            raise InferenceGatewayError(
                502,
                {
                    "message": "failed to query models from all inference upstreams",
                    "upstreams": results,
                },
            )

        models = self._merge_model_groups(model_groups, expected_upstreams=len(self.upstreams))
        loaded_count = sum(1 for model in models if model["status"] == "loaded")
        return {"models": models, "total": len(models), "loaded_count": loaded_count}

    def get_service_status(self) -> dict[str, Any]:
        results = self._broadcast("get", "/inference/status")
        workers: list[dict[str, Any]] = []
        pending_requests = 0
        loaded_counts: list[int] = []
        service_statuses: list[str] = []
        failed_count = 0

        for result in results:
            if not result["ok"]:
                failed_count += 1
                logger.warning("Inference upstream status query failed: {}", result)
                continue

            upstream = result["upstream"]
            data = result["data"]
            for worker in data.get("workers", []):
                worker = dict(worker)
                worker["worker_id"] = f"{upstream['name']}/{worker.get('worker_id', 'worker')}"
                workers.append(worker)
            pending_requests += int(data.get("pending_requests", 0))
            loaded_counts.append(int(data.get("loaded_models_count", 0)))
            service_statuses.append(str(data.get("service_status", "unknown")))

        if not service_statuses:
            service_status = "error"
        elif failed_count or any(status != "running" for status in service_statuses):
            service_status = "degraded"
        else:
            service_status = "running"

        return {
            "service_status": service_status,
            "workers": workers,
            "total_workers": len(workers),
            "loaded_models_count": max(loaded_counts, default=0),
            "pending_requests": pending_requests,
        }

    def _select_upstream(self) -> InferenceUpstream:
        if self.policy == "least_pending":
            upstream = self._select_least_pending_upstream()
            if upstream is not None:
                return upstream
        return self._select_round_robin_upstream()

    def _select_round_robin_upstream(self) -> InferenceUpstream:
        # 优先在健康的 upstream 中轮询；全部被熔断时退化为全量轮询（尽力而为）
        candidates = self._healthy_upstreams() or self.upstreams
        with self._lock:
            upstream = candidates[self._cursor % len(candidates)]
            self._cursor += 1
        return upstream

    def _select_least_pending_upstream(self) -> InferenceUpstream | None:
        results = self._broadcast("get", "/inference/status")
        candidates = []
        for result in results:
            if not result["ok"]:
                continue
            pending = int(result["data"].get("pending_requests", 0))
            candidates.append((pending, result["upstream"]["index"]))
        if not candidates:
            return None
        _, index = min(candidates)
        return self.upstreams[index]

    def _broadcast(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(self.upstreams)) as executor:
            future_by_upstream = {
                executor.submit(self._request_json, method, upstream, path, payload): upstream
                for upstream in self.upstreams
            }
            for future in as_completed(future_by_upstream):
                upstream = future_by_upstream[future]
                try:
                    data = future.result()
                except InferenceGatewayError as exc:
                    results.append(
                        {
                            "ok": False,
                            "upstream": self._upstream_info(upstream),
                            "status_code": exc.status_code,
                            "detail": exc.detail,
                        }
                    )
                else:
                    results.append(
                        {
                            "ok": True,
                            "upstream": self._upstream_info(upstream),
                            "data": data,
                        }
                    )
        results.sort(key=lambda item: item["upstream"]["index"])
        return results

    def _request_json(
        self,
        method: str,
        upstream: InferenceUpstream,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{upstream.base_url}{path}"
        try:
            response = self._session().request(
                method=method,
                url=url,
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise InferenceGatewayError(
                504,
                {
                    "message": "inference upstream request timed out",
                    "upstream": self._upstream_info(upstream),
                    "url": url,
                },
            ) from exc
        except requests.RequestException as exc:
            raise InferenceGatewayError(
                502,
                {
                    "message": "inference upstream request failed",
                    "upstream": self._upstream_info(upstream),
                    "url": url,
                    "error": str(exc),
                },
            ) from exc

        if response.status_code >= 400:
            raise InferenceGatewayError(
                response.status_code,
                {
                    "message": "inference upstream returned an error",
                    "upstream": self._upstream_info(upstream),
                    "url": url,
                    "status_code": response.status_code,
                    "detail": self._response_detail(response),
                },
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise InferenceGatewayError(
                502,
                {
                    "message": "inference upstream returned non-json response",
                    "upstream": self._upstream_info(upstream),
                    "url": url,
                    "body": response.text[:500],
                },
            ) from exc
        if not isinstance(data, dict):
            raise InferenceGatewayError(
                502,
                {
                    "message": "inference upstream returned invalid json payload",
                    "upstream": self._upstream_info(upstream),
                    "url": url,
                },
            )
        return data

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _merge_model_groups(
        self,
        model_groups: list[list[dict[str, Any]]],
        expected_upstreams: int,
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for group in model_groups:
            for model in group:
                model_id = model.get("model_id")
                if not model_id:
                    continue
                item = merged.setdefault(
                    str(model_id),
                    {
                        "seen_count": 0,
                        "loaded_count": 0,
                        "uptimes": [],
                        "gpu_ids": [],
                    },
                )
                item["seen_count"] += 1
                if model.get("status") == "loaded":
                    item["loaded_count"] += 1
                if model.get("uptime_seconds") is not None:
                    item["uptimes"].append(float(model["uptime_seconds"]))
                if model.get("gpu_id") is not None:
                    item["gpu_ids"].append(model["gpu_id"])

        models = []
        for model_id, item in sorted(merged.items()):
            loaded_count = item["loaded_count"]
            if loaded_count == expected_upstreams:
                status = "loaded"
            elif loaded_count > 0:
                status = "partial"
            else:
                status = "unloaded"
            models.append(
                {
                    "model_id": model_id,
                    "status": status,
                    "gpu_id": item["gpu_ids"][0] if item["gpu_ids"] else None,
                    "uptime_seconds": min(item["uptimes"]) if item["uptimes"] else None,
                }
            )
        return models

    def _response_detail(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text[:500]

    def _upstream_info(self, upstream: InferenceUpstream) -> dict[str, Any]:
        return {
            "index": upstream.index,
            "name": f"upstream-{upstream.index}",
            "url": upstream.base_url,
        }


_INFERENCE_GATEWAY: InferenceGateway | None = None


def get_inference_gateway() -> InferenceGateway:
    global _INFERENCE_GATEWAY
    if _INFERENCE_GATEWAY is None:
        _INFERENCE_GATEWAY = InferenceGateway(
            upstream_urls=get_inference_upstream_urls(),
            timeout=get_inference_upstream_timeout(),
            policy=get_inference_lb_policy(),
        )
    return _INFERENCE_GATEWAY


__all__ = [
    "InferenceGateway",
    "InferenceGatewayError",
    "get_inference_gateway",
]
