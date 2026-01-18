import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests
import torch
from loguru import logger
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

from src.settings import settings


class TextClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, text_column: str, label_column: str, max_length: int):
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            for text in dataframe[text_column]
        ]
        self.labels = dataframe[label_column].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


class BertClassifier(nn.Module):
    def __init__(self, base_model: str, output_dim: int):
        super().__init__()
        self.bert = BertModel.from_pretrained(base_model)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return self.relu(linear_output)


def resolve_dataset_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return settings.data_root / path


def build_embeddings_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/embeddings"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/embeddings"
    return f"{trimmed}/v1/embeddings"


def sanitize_embedding_config(embedding_config: dict) -> dict[str, object]:
    return {
        "base_url": str(embedding_config.get("base_url")),
        "model": embedding_config.get("model"),
        "batch_size": embedding_config.get("batch_size"),
        "timeout": embedding_config.get("timeout"),
        "max_retries": embedding_config.get("max_retries"),
        "parallelism": embedding_config.get("parallelism"),
    }


def save_embedding_metadata(path: Path, embedding_config: dict) -> None:
    safe_config = sanitize_embedding_config(embedding_config)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_config, handle, ensure_ascii=True, indent=2)


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
            time.sleep(min(2 ** attempt, 5))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Embedding request failed without an exception")


def collect_embeddings(
    texts: list[str],
    embedding_config: dict,
    stop_requested: Callable[[], bool],
) -> np.ndarray | None:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    batch_size = int(embedding_config.get("batch_size") or 64)
    total_batches = math.ceil(len(texts) / batch_size)
    embeddings: list[list[float]] = []
    log_interval = max(1, total_batches // 10)
    parallelism = int(embedding_config.get("parallelism") or 1)
    parallelism = max(1, parallelism)

    if parallelism <= 1:
        session = requests.Session()
        for batch_idx in range(total_batches):
            if stop_requested():
                return None
            batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_embeddings = request_embedding_batch(session, embedding_config, batch_texts)
            if len(batch_embeddings) != len(batch_texts):
                raise ValueError("Embedding response size mismatch")
            embeddings.extend(batch_embeddings)
            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
                logger.info(
                    "Embedding progress: {}/{} batches ({:.1f}%)",
                    batch_idx + 1,
                    total_batches,
                    (batch_idx + 1) * 100.0 / total_batches,
                )
        return np.asarray(embeddings, dtype=np.float32)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch_batch(batch_texts: list[str]) -> list[list[float]]:
        with requests.Session() as thread_session:
            return request_embedding_batch(thread_session, embedding_config, batch_texts)

    batch_specs: list[tuple[int, list[str]]] = []
    for batch_idx in range(total_batches):
        batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_specs.append((batch_idx, batch_texts))

    embeddings_by_index: list[list[list[float]] | None] = [None] * total_batches
    completed = 0

    executor = ThreadPoolExecutor(max_workers=parallelism)
    futures = {
        executor.submit(fetch_batch, batch_texts): (batch_idx, len(batch_texts))
        for batch_idx, batch_texts in batch_specs
    }
    try:
        for future in as_completed(futures):
            if stop_requested():
                executor.shutdown(wait=False, cancel_futures=True)
                return None
            batch_idx, expected_size = futures[future]
            batch_embeddings = future.result()
            if len(batch_embeddings) != expected_size:
                raise ValueError("Embedding response size mismatch")
            embeddings_by_index[batch_idx] = batch_embeddings
            completed += 1
            if completed % log_interval == 0 or completed == total_batches:
                logger.info(
                    "Embedding progress: {}/{} batches ({:.1f}%)",
                    completed,
                    total_batches,
                    completed * 100.0 / total_batches,
                )
    finally:
        executor.shutdown(wait=True, cancel_futures=False)

    for batch_embeddings in embeddings_by_index:
        if batch_embeddings is None:
            raise RuntimeError("Missing embedding batch during parallel fetch")
        embeddings.extend(batch_embeddings)

    return np.asarray(embeddings, dtype=np.float32)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def select_device(task_id: str) -> torch.device:
    if torch.cuda.is_available():
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible:
            # When CUDA_VISIBLE_DEVICES is set per-process we always use local index 0
            torch.cuda.set_device(0)
            return torch.device("cuda:0")
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    return torch.device("cpu")


def prepare_dataframe(
    dataset_path: Path,
    sheet_name: str | None,
    text_column: str,
    label_column: str,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    if sheet_name is None:
        dataframe = pd.read_excel(dataset_path)
    else:
        dataframe = pd.read_excel(dataset_path, sheet_name=sheet_name)

    dataframe[text_column] = dataframe[text_column].astype(str).fillna("")
    labels = dataframe[label_column].astype(str)
    dataframe[label_column] = labels
    # Align with legacy/Bert_Train_Sample_V1.py: keep appearance order (no sorting).
    unique_labels = list(labels.unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    dataframe[label_column] = dataframe[label_column].map(label_to_id)
    return dataframe, label_to_id, id_to_label


def build_dataloaders(
    dataframe: pd.DataFrame,
    tokenizer: BertTokenizer,
    hp: dict,
    text_column: str,
    label_column: str,
) -> tuple[DataLoader, DataLoader, int]:
    # Align with legacy/Bert_Train_Sample_V1.py: dataset split uses a fixed random_state=42
    # (independent from training random seed).
    if hp["train_val_split"] == 0:
        train_df = dataframe
        dev_df = dataframe.iloc[0:0]
    else:
        train_df, holdout_df = train_test_split(
            dataframe,
            test_size=hp["train_val_split"],
            stratify=None,
            random_state=42,
        )
        dev_df, _ = train_test_split(
            holdout_df,
            test_size=0.5,
            stratify=None,
            random_state=42,
        )

    train_dataset = TextClassificationDataset(train_df, tokenizer, text_column, label_column, hp["max_sequence_length"])
    dev_dataset = TextClassificationDataset(dev_df, tokenizer, text_column, label_column, hp["max_sequence_length"])
    train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=hp["batch_size"])
    return train_loader, dev_loader, len(dev_dataset)


def split_dataframe(dataframe: pd.DataFrame, hp: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, holdout_df = train_test_split(
        dataframe,
        test_size=hp["train_val_split"],
        stratify=None,
        random_state=42,
    )
    dev_df, _ = train_test_split(
        holdout_df,
        test_size=0.5,
        stratify=None,
        random_state=42,
    )
    return train_df, dev_df


def save_label_mappings(mapping_path: Path, label_to_id: dict[str, int], id_to_label: dict[int, str]) -> None:
    with mapping_path.open("wb") as handle:
        pickle.dump((label_to_id, id_to_label), handle)


def resolve_embedding_config(embedding_config: dict | None) -> dict:
    resolved = dict(embedding_config or {})
    if not resolved.get("base_url"):
        resolved["base_url"] = settings.embedding_base_url
    if not resolved.get("model"):
        resolved["model"] = settings.embedding_model_name_value
    if not resolved.get("api_key"):
        resolved["api_key"] = settings.embedding_api_key
    return resolved


def resolve_setfit_artifacts(output_dir: Path, model_name_en: str) -> tuple[Path, Path, Path]:
    base_name = model_name_en
    if base_name.endswith(".pkl"):
        base_name = base_name[:-4]
    model_path = output_dir / f"{base_name}.pkl"
    label_mapping_path = output_dir / f"{base_name}.labels.pkl"
    embedding_config_path = output_dir / f"{base_name}.embedding.json"
    return model_path, label_mapping_path, embedding_config_path


def run_setfit_training(
    *,
    task_id: str,
    request_payload: dict,
    progress_handler: Callable[[int, dict[str, float]], None],
    batch_progress_handler: Callable[[int, int, dict[str, float]], None],
    stop_requested: Callable[[], bool],
) -> dict:
    hp = request_payload["hyperparameters"]
    callback_url = request_payload.get("callback_url")
    embedding_config = resolve_embedding_config(request_payload.get("embedding"))
    if not embedding_config.get("base_url") or not embedding_config.get("model"):
        raise ValueError(
            "Embedding config requires base_url and model (payload or "
            "EMBEDDING_BASE_URL/EMBEDDING_MODEL_NAME environment variables)"
        )

    logger.info("Starting setfit training task {}", task_id)

    dataset_path = resolve_dataset_path(request_payload["training_data_file"])
    dataframe, label_to_id, id_to_label = prepare_dataframe(
        dataset_path=dataset_path,
        sheet_name=hp.get("sheet_name"),
        text_column=hp["text_column"],
        label_column=hp["label_column"],
    )
    setup_seed(hp["random_seed"])
    train_df, dev_df = split_dataframe(dataframe, hp)

    output_dir = settings.model_output_path / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path, label_mapping_path, embedding_config_path = resolve_setfit_artifacts(
        output_dir, request_payload["model_name_en"]
    )
    save_label_mappings(label_mapping_path, label_to_id, id_to_label)
    save_embedding_metadata(embedding_config_path, embedding_config)

    train_texts = train_df[hp["text_column"]].tolist()
    dev_texts = dev_df[hp["text_column"]].tolist()
    train_labels = train_df[hp["label_column"]].to_numpy()
    dev_labels = dev_df[hp["label_column"]].to_numpy()

    if len(train_labels) == 0 or len(dev_labels) == 0:
        raise ValueError("Training and validation splits must be non-empty")

    logger.info("Fetching embeddings for {} training samples", len(train_texts))
    train_vectors = collect_embeddings(train_texts, embedding_config, stop_requested)
    if train_vectors is None:
        logger.info("Stop requested for task {} during embedding fetch", task_id)
        return {
            "status": "stopped",
            "model_path": str(model_path),
            "label_mapping_path": str(label_mapping_path),
            "embedding_config_path": str(embedding_config_path),
        }
    logger.info("Fetching embeddings for {} validation samples", len(dev_texts))
    dev_vectors = collect_embeddings(dev_texts, embedding_config, stop_requested)
    if dev_vectors is None:
        logger.info("Stop requested for task {} during embedding fetch", task_id)
        return {
            "status": "stopped",
            "model_path": str(model_path),
            "label_mapping_path": str(label_mapping_path),
            "embedding_config_path": str(embedding_config_path),
        }

    classes = np.arange(len(label_to_id))
    classifier = SGDClassifier(
        loss="log_loss",
        learning_rate="optimal",
        random_state=hp["random_seed"],
        max_iter=1,
        tol=None,
    )

    best_val_accuracy = 0.0
    best_state: bytes | None = None
    total_epochs = hp["epochs"]
    batch_size = hp["batch_size"]
    rng = np.random.default_rng(hp["random_seed"])

    for epoch_index in range(total_epochs):
        if stop_requested():
            logger.info("Stop requested for task {}", task_id)
            return {
                "status": "stopped",
                "model_path": str(model_path),
                "label_mapping_path": str(label_mapping_path),
                "embedding_config_path": str(embedding_config_path),
            }

        indices = rng.permutation(len(train_vectors))
        total_batches = math.ceil(len(indices) / batch_size)

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_indices = indices[start:end]
            batch_vectors = train_vectors[batch_indices]
            batch_labels = train_labels[batch_indices]

            if epoch_index == 0 and batch_idx == 0:
                classifier.partial_fit(batch_vectors, batch_labels, classes=classes)
            else:
                classifier.partial_fit(batch_vectors, batch_labels)

            batch_predictions = classifier.predict(batch_vectors)
            batch_accuracy = float((batch_predictions == batch_labels).mean()) if len(batch_labels) else 0.0
            if len(batch_labels):
                batch_probs = classifier.predict_proba(batch_vectors)
                batch_loss = float(log_loss(batch_labels, batch_probs, labels=classes))
            else:
                batch_loss = 0.0

            batch_progress = float((batch_idx + 1) * 100.0 / total_batches) if total_batches else 100.0
            batch_metrics = {
                "epoch": epoch_index + 1,
                "total_epochs": total_epochs,
                "batch": batch_idx + 1,
                "total_batches": total_batches,
                "batch_progress_percentage": batch_progress,
                "train_accuracy": batch_accuracy,
                "train_loss": batch_loss,
                "val_accuracy": None,
                "val_loss": None,
                "callback_url": callback_url,
            }
            batch_progress_handler(epoch_index + 1, batch_idx + 1, batch_metrics)

            if stop_requested():
                logger.info("Stop requested for task {}", task_id)
                return {
                    "status": "stopped",
                    "model_path": str(model_path),
                    "label_mapping_path": str(label_mapping_path),
                    "embedding_config_path": str(embedding_config_path),
                }

        train_predictions = classifier.predict(train_vectors)
        train_accuracy = float(accuracy_score(train_labels, train_predictions))
        train_probs = classifier.predict_proba(train_vectors)
        train_loss = float(log_loss(train_labels, train_probs, labels=classes))

        val_predictions = classifier.predict(dev_vectors)
        val_accuracy = float(accuracy_score(dev_labels, val_predictions))
        val_probs = classifier.predict_proba(dev_vectors)
        val_loss = float(log_loss(dev_labels, val_probs, labels=classes))
        f1 = f1_score(dev_labels, val_predictions, average="macro", zero_division=0)

        metrics = {
            "epoch": epoch_index + 1,
            "total_epochs": total_epochs,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "f1_score": f1,
            "progress_percentage": 100.0,
            "callback_url": callback_url,
        }
        logger.info(
            "Task {} epoch {} metrics: train_acc={:.3f}, val_acc={:.3f}, f1={:.3f}",
            task_id,
            epoch_index + 1,
            train_accuracy,
            val_accuracy,
            f1,
        )
        progress_handler(epoch_index + 1, metrics)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = pickle.dumps(classifier)

    if best_state is None:
        with model_path.open("wb") as handle:
            pickle.dump(classifier, handle)
    else:
        model_path.write_bytes(best_state)

    return {
        "status": "completed",
        "model_path": str(model_path),
        "label_mapping_path": str(label_mapping_path),
        "embedding_config_path": str(embedding_config_path),
        "best_val_accuracy": best_val_accuracy,
        "total_epochs": total_epochs,
    }


def run_training_loop(
    *,
    task_id: str,
    request_payload: dict,
    progress_handler: Callable[[int, dict[str, float]], None],
    batch_progress_handler: Callable[[int, int, dict[str, float]], None],
    stop_requested: Callable[[], bool],
) -> dict:
    training_mode = request_payload.get("training_mode", "bert")
    if training_mode == "setfit":
        return run_setfit_training(
            task_id=task_id,
            request_payload=request_payload,
            progress_handler=progress_handler,
            batch_progress_handler=batch_progress_handler,
            stop_requested=stop_requested,
        )
    if training_mode != "bert":
        raise ValueError(f"Unsupported training mode: {training_mode}")

    hp = request_payload["hyperparameters"]
    callback_url = request_payload.get("callback_url")

    logger.info("Starting training task {}", task_id)

    dataset_path = resolve_dataset_path(request_payload["training_data_file"])
    dataframe, label_to_id, id_to_label = prepare_dataframe(
        dataset_path=dataset_path,
        sheet_name=hp.get("sheet_name"),
        text_column=hp["text_column"],
        label_column=hp["label_column"],
    )
    setup_seed(hp["random_seed"])
    tokenizer = BertTokenizer.from_pretrained(request_payload["base_model"])
    train_loader, dev_loader, dev_size = build_dataloaders(
        dataframe,
        tokenizer,
        hp,
        hp["text_column"],
        hp["label_column"],
    )

    output_dir = settings.model_output_path / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    # Artifacts naming convention:
    # - model weights: <name>.pt
    # - label mappings: <name>.pt.pkl
    model_name_en = request_payload["model_name_en"]
    model_filename = model_name_en if model_name_en.endswith(".pt") else f"{model_name_en}.pt"
    best_model_path = output_dir / model_filename
    label_mapping_path = output_dir / f"{model_filename}.pkl"
    save_label_mappings(label_mapping_path, label_to_id, id_to_label)

    model = BertClassifier(request_payload["base_model"], output_dim=len(label_to_id))
    device = select_device(task_id)
    logger.info("Task {} using device {}", task_id, device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=hp["learning_rate"])

    best_val_accuracy = 0.0

    total_epochs = hp["epochs"]
    has_validation = dev_size > 0

    for epoch_index in range(total_epochs):
        if stop_requested():
            logger.info("Stop requested for task {}", task_id)
            return {
                "status": "stopped",
                "model_path": str(best_model_path),
                "label_mapping_path": str(label_mapping_path),
            }

        model.train()
        total_acc_train = 0.0
        total_loss_train = 0.0
        sample_count_train = 0
        total_batches = len(train_loader)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            input_ids = inputs["input_ids"].squeeze(1).to(device)
            attention_mask = inputs["attention_mask"].squeeze(1).to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predictions = outputs.argmax(dim=1)
            total_acc_train += (predictions == labels).sum().item()
            total_loss_train += loss.item() * labels.size(0)
            sample_count_train += labels.size(0)
            
            # Calculate batch progress within current epoch
            batch_progress = float((batch_idx + 1) * 100.0 / total_batches)
            batch_metrics = {
                "epoch": epoch_index + 1,
                "total_epochs": total_epochs,
                "batch": batch_idx + 1,
                "total_batches": total_batches,
                "batch_progress_percentage": batch_progress,
                "train_accuracy": total_acc_train / sample_count_train if sample_count_train else 0.0,
                "train_loss": total_loss_train / sample_count_train if sample_count_train else 0.0,
                "val_accuracy": None,
                "val_loss": None,
                "callback_url": callback_url,
            }
            batch_progress_handler(epoch_index + 1, batch_idx + 1, batch_metrics)
            
            # Check for stop request after each batch
            if stop_requested():
                logger.info("Stop requested for task {}", task_id)
                return {
                    "status": "stopped",
                    "model_path": str(best_model_path),
                    "label_mapping_path": str(label_mapping_path),
                }

        model.eval()
        total_acc_val = 0.0
        total_loss_val = 0.0
        predictions_list: list[int] = []
        references_list: list[int] = []

        with torch.no_grad():
            for inputs, labels in dev_loader:
                input_ids = inputs["input_ids"].squeeze(1).to(device)
                attention_mask = inputs["attention_mask"].squeeze(1).to(device)
                labels = labels.to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                predictions = outputs.argmax(dim=1)

                total_acc_val += (predictions == labels).sum().item()
                total_loss_val += loss.item() * labels.size(0)
                predictions_list.extend(predictions.cpu().tolist())
                references_list.extend(labels.cpu().tolist())

        train_accuracy = total_acc_train / sample_count_train if sample_count_train else 0.0
        train_loss = total_loss_train / sample_count_train if sample_count_train else 0.0
        val_accuracy = total_acc_val / dev_size if dev_size else 0.0
        val_loss = total_loss_val / dev_size if dev_size else 0.0
        f1 = f1_score(references_list, predictions_list, average="macro", zero_division=0) if references_list else 0.0

        metrics = {
            "epoch": epoch_index + 1,
            "total_epochs": total_epochs,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "f1_score": f1,
            "progress_percentage": 100.0,  # Epoch is complete, so progress within epoch is 100%
            "callback_url": callback_url,
        }
        logger.info(
            "Task {} epoch {} metrics: train_acc={:.3f}, val_acc={:.3f}, f1={:.3f}",
            task_id,
            epoch_index + 1,
            train_accuracy,
            val_accuracy,
            f1,
        )
        progress_handler(epoch_index + 1, metrics)

        if has_validation:
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), best_model_path)

    if not has_validation:
        torch.save(model.state_dict(), best_model_path)

    return {
        "status": "completed",
        "model_path": str(best_model_path),
        "label_mapping_path": str(label_mapping_path),
        "best_val_accuracy": best_val_accuracy,
        "total_epochs": total_epochs,
    }


__all__ = ["run_training_loop"]
