import datetime
from pathlib import Path
from typing import Iterable, Sequence

import click
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from peft import PeftModel
from transformers import BertModel, BertTokenizer


class BertClassifier(nn.Module):
    """Same architecture as the legacy training script."""

    def __init__(self, base_model: str, out_features: int, bert_model: nn.Module | None = None):
        super().__init__()
        self.bert = bert_model if bert_model is not None else BertModel.from_pretrained(base_model)
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


def resolve_model_stem(model_path: Path) -> str:
    stem = model_path.stem
    if stem.endswith(".head"):
        stem = stem[: -len(".head")]
    return stem


def resolve_lora_adapter_path(model_path: Path, lora_adapter_path: str | None) -> Path | None:
    if lora_adapter_path:
        return Path(lora_adapter_path)
    candidate = model_path.with_name(f"{resolve_model_stem(model_path)}.lora")
    if candidate.exists():
        return candidate
    return None


def load_model(
    model_path: Path,
    base_model: str,
    num_labels: int,
    device: torch.device,
    lora_adapter_path: str | None = None,
) -> BertClassifier:
    resolved_adapter_path = resolve_lora_adapter_path(model_path, lora_adapter_path)
    if resolved_adapter_path is not None:
        base = BertModel.from_pretrained(base_model)
        bert = PeftModel.from_pretrained(base, resolved_adapter_path)
        model = BertClassifier(base_model, out_features=num_labels, bert_model=bert)
        head_state = torch.load(model_path, map_location=device)
        model.linear.load_state_dict(head_state)
    else:
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
    help="模型权重 .pt 路径（LoRA 时为 .head.pt）",
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
    "--lora-adapter",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="LoRA adapter 目录（可选，不填则自动查找同名 .lora 目录）",
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
    lora_adapter: str | None,
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
        Path(model_path),
        base_model,
        num_labels=len(label2id),
        device=device_resolved,
        lora_adapter_path=lora_adapter,
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
    help="模型权重 .pt 路径（LoRA 时为 .head.pt）",
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
    "--lora-adapter",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="LoRA adapter 目录（可选，不填则自动查找同名 .lora 目录）",
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
    lora_adapter: str | None,
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
        Path(model_path),
        base_model,
        num_labels=len(label2id),
        device=device_resolved,
        lora_adapter_path=lora_adapter,
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
    help="模型权重 .pt 路径（LoRA 时为 .head.pt）",
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
    "--lora-adapter",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="LoRA adapter 目录（可选，不填则自动查找同名 .lora 目录）",
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
    lora_adapter: str | None,
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
        Path(model_path),
        base_model,
        num_labels=len(label2id),
        device=device_resolved,
        lora_adapter_path=lora_adapter,
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
