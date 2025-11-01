import json
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from redis import Redis

from src.config import get_redis_url


TASK_HASH_KEY = "training:tasks:data"
TASK_INDEX_KEY = "training:tasks:index"
STOP_KEY_TEMPLATE = "training:tasks:{task_id}:stop_requested"

_redis: Redis | None = None


def redis_client() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(get_redis_url(), decode_responses=True)
    return _redis


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_task_record(record: dict[str, Any]) -> dict[str, Any]:
    data = record.copy()
    data["updated_at"] = data.get("updated_at", data["created_at"])
    payload = json.dumps(data)
    client = redis_client()
    task_id = data["task_id"]
    client.hset(TASK_HASH_KEY, task_id, payload)
    client.zadd(TASK_INDEX_KEY, {task_id: time.time()})
    return data


def load_task_record(task_id: str) -> dict[str, Any] | None:
    client = redis_client()
    payload = client.hget(TASK_HASH_KEY, task_id)
    if payload is None:
        return None
    return json.loads(payload)


def update_task_record(task_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    current = load_task_record(task_id)
    if current is None:
        return current
    merged = current | updates
    merged["updated_at"] = iso_now()
    client = redis_client()
    client.hset(TASK_HASH_KEY, task_id, json.dumps(merged))
    return merged


def delete_task_record(task_id: str) -> None:
    client = redis_client()
    client.hdel(TASK_HASH_KEY, task_id)
    client.zrem(TASK_INDEX_KEY, task_id)
    clear_stop_request(task_id)


def list_task_records() -> Sequence[dict[str, Any]]:
    client = redis_client()
    task_ids = client.zrevrange(TASK_INDEX_KEY, 0, -1)
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        record = load_task_record(task_id)
        if record is not None:
            records.append(record)
    return records


def set_stop_request(task_id: str) -> None:
    client = redis_client()
    client.set(STOP_KEY_TEMPLATE.format(task_id=task_id), "1")


def clear_stop_request(task_id: str) -> None:
    client = redis_client()
    client.delete(STOP_KEY_TEMPLATE.format(task_id=task_id))


def is_stop_requested(task_id: str) -> bool:
    client = redis_client()
    exists = client.exists(STOP_KEY_TEMPLATE.format(task_id=task_id))
    return bool(exists)


__all__ = [
    "create_task_record",
    "delete_task_record",
    "is_stop_requested",
    "iso_now",
    "list_task_records",
    "load_task_record",
    "set_stop_request",
    "update_task_record",
]
