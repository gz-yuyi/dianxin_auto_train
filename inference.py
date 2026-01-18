import datetime
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import click
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertModel, BertTokenizer


class BertClassifier(nn.Module):
    """Same architecture as the legacy training script."""

    def __init__(self, base_model: str, out_features: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, out_features)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return self.relu(linear_output)


def pick_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_label_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    import pickle

    with path.open("rb") as handle:
        label2id, id2label = pickle.load(handle)
    return label2id, id2label


def load_model(
    model_path: Path, base_model: str, num_labels: int, device: torch.device
) -> BertClassifier:
    model = BertClassifier(base_model, out_features=num_labels)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def tokenize_texts(
    texts: Sequence[str],
    tokenizer: BertTokenizer,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def predict_batch(
    texts: Sequence[str],
    model: BertClassifier,
    tokenizer: BertTokenizer,
    max_length: int,
    id_to_label: dict[int, str],
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], list[list[tuple[str, float]]], list[dict[str, float]]]:
    labels: list[str] = []
    top_all: list[list[tuple[str, float]]] = []
    label_prob_dicts: list[dict[str, float]] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        input_ids, attention_mask = tokenize_texts(
            batch_texts, tokenizer, max_length, device
        )
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

        for row in probs:
            label_prob_dict = {
                id_to_label[idx]: float(prob) for idx, prob in enumerate(row)
            }
            label_prob_dicts.append(label_prob_dict)
            sorted_items = sorted(
                label_prob_dict.items(), key=lambda item: item[1], reverse=True
            )
            top_all.append(sorted_items)
            labels.append(sorted_items[0][0])

    return labels, top_all, label_prob_dicts


def timestamped_filename(prefix: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.xlsx"


def build_embeddings_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/embeddings"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/embeddings"
    return f"{trimmed}/v1/embeddings"


def load_embedding_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_embedding_config(
    config: dict,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    batch_size: int | None,
    timeout: float | None,
    max_retries: int | None,
    extra_headers: dict[str, str] | None,
) -> dict:
    resolved = dict(config)
    if base_url:
        resolved["base_url"] = base_url
    if model:
        resolved["model"] = model
    if api_key:
        resolved["api_key"] = api_key
    if batch_size:
        resolved["batch_size"] = batch_size
    if timeout:
        resolved["timeout"] = timeout
    if max_retries is not None:
        resolved["max_retries"] = max_retries
    if extra_headers:
        resolved["extra_headers"] = extra_headers
    return resolved


def request_embedding_batch(
    session: requests.Session,
    embedding_config: dict,
    inputs: list[str],
) -> list[list[float]]:
    base_url = str(embedding_config["base_url"])
    url = build_embeddings_url(base_url)
    headers = {"Content-Type": "application/json"}
    api_key = embedding_config.get("api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    extra_headers = embedding_config.get("extra_headers") or {}
    headers.update(extra_headers)
    payload = {"model": embedding_config["model"], "input": inputs}
    timeout = embedding_config.get("timeout", 60.0)
    max_retries = embedding_config.get("max_retries", 2)
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = session.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if "data" not in data:
                raise ValueError(f"Embedding response missing 'data': {data}")
            items = data["data"]
            if not isinstance(items, list):
                raise ValueError("Embedding response 'data' is not a list")
            if items and "index" in items[0]:
                items = sorted(items, key=lambda item: item.get("index", 0))
            embeddings = [item["embedding"] for item in items]
            return embeddings
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            time.sleep(min(2**attempt, 5))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Embedding request failed without an exception")


def predict_setfit_batch(
    texts: Sequence[str],
    classifier,
    id_to_label: dict[int, str],
    embedding_config: dict,
    top_n: int,
) -> tuple[list[str], list[list[tuple[str, float]]], list[dict[str, float]]]:
    labels: list[str] = []
    top_all: list[list[tuple[str, float]]] = []
    label_prob_dicts: list[dict[str, float]] = []
    classes = classifier.classes_
    session = requests.Session()
    batch_size = int(embedding_config.get("batch_size") or 64)
    total_batches = math.ceil(len(texts) / batch_size)

    for batch_idx in range(total_batches):
        batch_texts = list(texts[batch_idx * batch_size : (batch_idx + 1) * batch_size])
        embeddings = request_embedding_batch(session, embedding_config, batch_texts)
        vectors = np.asarray(embeddings, dtype=np.float32)
        probs = classifier.predict_proba(vectors)

        for row in probs:
            label_prob_dict = {
                id_to_label[int(classes[idx])]: float(prob)
                for idx, prob in enumerate(row)
            }
            label_prob_dicts.append(label_prob_dict)
            sorted_items = sorted(
                label_prob_dict.items(), key=lambda item: item[1], reverse=True
            )
            top_items = sorted_items[:top_n] if top_n > 0 else sorted_items
            top_all.append(top_items)
            labels.append(top_items[0][0] if top_items else "")

    return labels, top_all, label_prob_dicts


@click.group()
def cli() -> None:
    """BERT 文本分类推理脚本"""


@cli.command("multi-class")
@click.option(
    "--excel",
    "excel_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="输入 Excel 文件",
)
@click.option("--sheet", "sheet_name", default=None, help="工作表名，可选")
@click.option("--text-column", required=True, help="文本列名")
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="模型权重 .pt 路径",
)
@click.option(
    "--label-mapping",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="标签映射 .pkl 路径",
)
@click.option(
    "--base-model",
    default="bert-base-chinese",
    show_default=True,
    help="预训练模型名称或本地路径",
)
@click.option(
    "--max-length", default=512, show_default=True, type=int, help="分词最大长度"
)
@click.option("--top-n", default=3, show_default=True, type=int, help="返回前 N 个标签")
@click.option(
    "--device", default="auto", show_default=True, help="推理设备，如 auto/cpu/cuda:0"
)
@click.option(
    "--batch-size", default=16, show_default=True, type=int, help="推理批大小"
)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="输出文件路径，默认为时间戳命名",
)
def multi_class(
    excel_path: str,
    sheet_name: str | None,
    text_column: str,
    model_path: str,
    label_mapping: str,
    base_model: str,
    max_length: int,
    top_n: int,
    device: str,
    batch_size: int,
    output: str | None,
) -> None:
    """多选一分类，输出 Top-N 结果。"""
    device_resolved = pick_device(device)
    label2id, id2label = load_label_mapping(Path(label_mapping))
    model = load_model(
        Path(model_path), base_model, num_labels=len(label2id), device=device_resolved
    )
    tokenizer = BertTokenizer.from_pretrained(base_model)

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df[text_column] = df[text_column].astype(str).fillna("")
    texts = df[text_column].tolist()

    labels, top_all, _ = predict_batch(
        texts,
        model,
        tokenizer,
        max_length,
        id2label,
        device_resolved,
        batch_size,
    )
    df["Prediction"] = labels

    for rank in range(top_n):
        df[f"Top{rank + 1}_Label"] = [
            items[rank][0] if len(items) > rank else None for items in top_all
        ]
        df[f"Top{rank + 1}_Probability"] = [
            items[rank][1] if len(items) > rank else None for items in top_all
        ]

    output_path = Path(output) if output else Path(timestamped_filename("multi_class"))
    df.to_excel(output_path, index=False)
    click.echo(f"保存结果到 {output_path}")


@cli.command("setfit-multi-class")
@click.option(
    "--excel",
    "excel_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="输入 Excel 文件",
)
@click.option("--sheet", "sheet_name", default=None, help="工作表名，可选")
@click.option("--text-column", required=True, help="文本列名")
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="SetFit 分类器 .pkl 路径",
)
@click.option(
    "--label-mapping",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="标签映射 .pkl 路径",
)
@click.option(
    "--embedding-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="训练输出的 embedding 配置 .json",
)
@click.option(
    "--base-url",
    default=None,
    help="Embedding base URL（优先于环境变量 EMBEDDING_BASE_URL 与配置文件）",
)
@click.option(
    "--embed-model",
    default=None,
    help="Embedding 模型名（优先于环境变量 EMBEDDING_MODEL_NAME 与配置文件）",
)
@click.option(
    "--api-key",
    default=None,
    help="Embedding API Key，默认读取环境变量 EMBEDDING_API_KEY",
)
@click.option(
    "--embedding-batch-size",
    default=None,
    type=int,
    help="Embedding batch size 覆盖配置文件",
)
@click.option(
    "--timeout",
    default=None,
    type=float,
    help="Embedding 请求超时（秒）覆盖配置文件",
)
@click.option(
    "--max-retries",
    default=2,
    show_default=True,
    type=int,
    help="Embedding 请求重试次数",
)
@click.option(
    "--extra-headers",
    default=None,
    help="额外请求头（JSON 字符串）",
)
@click.option("--top-n", default=3, show_default=True, type=int, help="返回前 N 个标签")
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="输出文件路径，默认为时间戳命名",
)
def setfit_multi_class(
    excel_path: str,
    sheet_name: str | None,
    text_column: str,
    model_path: str,
    label_mapping: str,
    embedding_config: str,
    base_url: str | None,
    embed_model: str | None,
    api_key: str | None,
    embedding_batch_size: int | None,
    timeout: float | None,
    max_retries: int,
    extra_headers: str | None,
    top_n: int,
    output: str | None,
) -> None:
    """SetFit 方式多选一分类，使用 embedding 接口推理。"""
    config = load_embedding_config(Path(embedding_config))
    env_base_url = os.getenv("EMBEDDING_BASE_URL")
    env_model = os.getenv("EMBEDDING_MODEL_NAME") or os.getenv("EMBEDDING_MODEL")
    extra_headers_dict: dict[str, str] | None = None
    if extra_headers:
        try:
            extra_headers_dict = json.loads(extra_headers)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"extra-headers 不是合法 JSON: {exc}") from exc
        if not isinstance(extra_headers_dict, dict):
            raise click.ClickException("extra-headers 必须是 JSON 对象")
    resolved_config = resolve_embedding_config(
        config,
        base_url=base_url or env_base_url,
        model=embed_model or env_model,
        api_key=api_key or os.getenv("EMBEDDING_API_KEY"),
        batch_size=embedding_batch_size,
        timeout=timeout,
        max_retries=max_retries,
        extra_headers=extra_headers_dict,
    )
    if not resolved_config.get("base_url") or not resolved_config.get("model"):
        raise click.ClickException("Embedding 配置缺少 base_url 或 model")

    label2id, id2label = load_label_mapping(Path(label_mapping))
    with Path(model_path).open("rb") as handle:
        import pickle

        classifier = pickle.load(handle)

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df[text_column] = df[text_column].astype(str).fillna("")
    texts = df[text_column].tolist()

    labels, top_all, _ = predict_setfit_batch(
        texts, classifier, id2label, resolved_config, top_n
    )
    df["Prediction"] = labels

    if top_n > 0:
        for rank in range(top_n):
            df[f"Top{rank + 1}_Label"] = [
                items[rank][0] if len(items) > rank else None for items in top_all
            ]
            df[f"Top{rank + 1}_Probability"] = [
                items[rank][1] if len(items) > rank else None for items in top_all
            ]

    output_path = Path(output) if output else Path(timestamped_filename("setfit_multi_class"))
    df.to_excel(output_path, index=False)
    click.echo(f"保存结果到 {output_path}")


@cli.command("single-judge")
@click.option(
    "--excel",
    "excel_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="输入 Excel 文件",
)
@click.option("--sheet", "sheet_name", default=None, help="工作表名，可选")
@click.option("--text-column", required=True, help="文本列名")
@click.option("--label-column", required=True, help="待判断的标签列名")
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="模型权重 .pt 路径",
)
@click.option(
    "--label-mapping",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="标签映射 .pkl 路径",
)
@click.option(
    "--base-model",
    default="bert-base-chinese",
    show_default=True,
    help="预训练模型名称或本地路径",
)
@click.option(
    "--max-length", default=512, show_default=True, type=int, help="分词最大长度"
)
@click.option(
    "--top-k", default=2, show_default=True, type=int, help="判断时查看前 K 个标签"
)
@click.option(
    "--threshold",
    default=0.4,
    show_default=True,
    type=float,
    help="前 K 标签中的概率阈值",
)
@click.option(
    "--device", default="auto", show_default=True, help="推理设备，如 auto/cpu/cuda:0"
)
@click.option(
    "--batch-size", default=16, show_default=True, type=int, help="推理批大小"
)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="输出文件路径，默认为时间戳命名",
)
def single_judge(
    excel_path: str,
    sheet_name: str | None,
    text_column: str,
    label_column: str,
    model_path: str,
    label_mapping: str,
    base_model: str,
    max_length: int,
    top_k: int,
    threshold: float,
    device: str,
    batch_size: int,
    output: str | None,
) -> None:
    """单标签是否判断。"""
    device_resolved = pick_device(device)
    label2id, id2label = load_label_mapping(Path(label_mapping))
    model = load_model(
        Path(model_path), base_model, num_labels=len(label2id), device=device_resolved
    )
    tokenizer = BertTokenizer.from_pretrained(base_model)

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df[text_column] = df[text_column].astype(str).fillna("")
    texts = df[text_column].tolist()
    topics = df[label_column].astype(str).tolist()

    _, top_all, label_prob_dicts = predict_batch(
        texts,
        model,
        tokenizer,
        max_length,
        id2label,
        device_resolved,
        batch_size,
    )

    def judge(prob_dict: dict[str, float], topic: str) -> str:
        top_items = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)[
            :top_k
        ]
        for label, prob in top_items:
            if label == topic and prob >= threshold:
                return "属于"
        return "不属于"

    results = [
        judge(prob_dict, topic) for prob_dict, topic in zip(label_prob_dicts, topics)
    ]
    df["Prediction"] = results

    output_path = Path(output) if output else Path(timestamped_filename("single_judge"))
    df.to_excel(output_path, index=False)
    click.echo(f"保存结果到 {output_path}")


@cli.command("multi-judge")
@click.option(
    "--excel",
    "excel_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="输入 Excel 文件",
)
@click.option("--sheet", "sheet_name", default=None, help="工作表名，可选")
@click.option("--text-column", required=True, help="文本列名")
@click.option(
    "--label-prefix", required=True, help="多标签列的前缀，匹配所有以此前缀开头的列"
)
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="模型权重 .pt 路径",
)
@click.option(
    "--label-mapping",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="标签映射 .pkl 路径",
)
@click.option(
    "--base-model",
    default="bert-base-chinese",
    show_default=True,
    help="预训练模型名称或本地路径",
)
@click.option(
    "--max-length", default=512, show_default=True, type=int, help="分词最大长度"
)
@click.option(
    "--top-k", default=2, show_default=True, type=int, help="判断时查看前 K 个标签"
)
@click.option(
    "--threshold",
    default=0.4,
    show_default=True,
    type=float,
    help="前 K 标签中的概率阈值",
)
@click.option(
    "--device", default="auto", show_default=True, help="推理设备，如 auto/cpu/cuda:0"
)
@click.option(
    "--batch-size", default=16, show_default=True, type=int, help="推理批大小"
)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="输出文件路径，默认为时间戳命名",
)
def multi_judge(
    excel_path: str,
    sheet_name: str | None,
    text_column: str,
    label_prefix: str,
    model_path: str,
    label_mapping: str,
    base_model: str,
    max_length: int,
    top_k: int,
    threshold: float,
    device: str,
    batch_size: int,
    output: str | None,
) -> None:
    """多标签是否判断。"""
    device_resolved = pick_device(device)
    label2id, id2label = load_label_mapping(Path(label_mapping))
    model = load_model(
        Path(model_path), base_model, num_labels=len(label2id), device=device_resolved
    )
    tokenizer = BertTokenizer.from_pretrained(base_model)

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df[text_column] = df[text_column].astype(str).fillna("")
    label_columns = [col for col in df.columns if col.startswith(label_prefix)]
    if not label_columns:
        raise click.ClickException(f"未找到以 {label_prefix} 开头的标签列")

    texts = df[text_column].tolist()
    topics_per_row: list[list[str]] = [
        [str(df[col].iloc[i]) for col in label_columns] for i in range(len(df))
    ]

    _, _, label_prob_dicts = predict_batch(
        texts,
        model,
        tokenizer,
        max_length,
        id2label,
        device_resolved,
        batch_size,
    )

    def judge(prob_dict: dict[str, float], topics: Iterable[str]) -> list[str]:
        sorted_items = sorted(
            prob_dict.items(), key=lambda item: item[1], reverse=True
        )[:top_k]
        top_map = {label: prob for label, prob in sorted_items}
        return [
            "属于" if top_map.get(topic, 0.0) >= threshold else "不属于"
            for topic in topics
        ]

    results_matrix = [
        judge(prob_dict, topics)
        for prob_dict, topics in zip(label_prob_dicts, topics_per_row)
    ]

    output_df = pd.DataFrame()
    output_df[text_column] = texts
    for idx, col in enumerate(label_columns):
        output_df[col] = [row[idx] for row in topics_per_row]
        output_df[f"result{idx}"] = [row[idx] for row in results_matrix]

    output_path = Path(output) if output else Path(timestamped_filename("multi_judge"))
    output_df.to_excel(output_path, index=False)
    click.echo(f"保存结果到 {output_path}")


if __name__ == "__main__":
    cli()
