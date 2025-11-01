import pickle
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

from src.config import get_data_root, get_model_output_dir


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
    return get_data_root() / path


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
    unique_labels = sorted(labels.unique())
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
    train_df, holdout_df = train_test_split(
        dataframe,
        test_size=hp["train_val_split"],
        stratify=None,
        random_state=hp["random_seed"],
    )
    dev_df, _ = train_test_split(
        holdout_df,
        test_size=0.5,
        stratify=None,
        random_state=hp["random_seed"],
    )

    train_dataset = TextClassificationDataset(train_df, tokenizer, text_column, label_column, hp["max_sequence_length"])
    dev_dataset = TextClassificationDataset(dev_df, tokenizer, text_column, label_column, hp["max_sequence_length"])
    train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=hp["batch_size"])
    return train_loader, dev_loader, len(dev_dataset)


def save_label_mappings(mapping_path: Path, label_to_id: dict[str, int], id_to_label: dict[int, str]) -> None:
    with mapping_path.open("wb") as handle:
        pickle.dump((label_to_id, id_to_label), handle)


def run_training_loop(
    *,
    task_id: str,
    request_payload: dict,
    progress_handler: Callable[[int, dict[str, float]], None],
    stop_requested: Callable[[], bool],
) -> dict:
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

    output_dir = get_model_output_dir() / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    label_mapping_path = output_dir / "label_mappings.pkl"
    save_label_mappings(label_mapping_path, label_to_id, id_to_label)

    model = BertClassifier(request_payload["base_model"], output_dim=len(label_to_id))
    device = select_device(task_id)
    logger.info("Task {} using device {}", task_id, device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=hp["learning_rate"])

    best_val_accuracy = 0.0
    best_model_path = output_dir / f"{request_payload['model_name_en']}.pt"

    total_epochs = hp["epochs"]

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

        for inputs, labels in train_loader:
            input_ids = inputs["input_ids"].squeeze(1).to(device)
            attention_mask = inputs["attention_mask"].squeeze(1).to(device)
            labels = torch.tensor(labels, device=device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predictions = outputs.argmax(dim=1)
            total_acc_train += (predictions == labels).sum().item()
            total_loss_train += loss.item() * labels.size(0)
            sample_count_train += labels.size(0)

        model.eval()
        total_acc_val = 0.0
        total_loss_val = 0.0
        predictions_list: list[int] = []
        references_list: list[int] = []

        with torch.no_grad():
            for inputs, labels in dev_loader:
                input_ids = inputs["input_ids"].squeeze(1).to(device)
                attention_mask = inputs["attention_mask"].squeeze(1).to(device)
                labels_tensor = torch.tensor(labels, device=device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels_tensor)
                predictions = outputs.argmax(dim=1)

                total_acc_val += (predictions == labels_tensor).sum().item()
                total_loss_val += loss.item() * labels_tensor.size(0)
                predictions_list.extend(predictions.cpu().tolist())
                references_list.extend(labels)

        train_accuracy = total_acc_train / sample_count_train if sample_count_train else 0.0
        train_loss = total_loss_train / sample_count_train if sample_count_train else 0.0
        val_accuracy = total_acc_val / dev_size if dev_size else 0.0
        val_loss = total_loss_val / dev_size if dev_size else 0.0
        f1 = f1_score(references_list, predictions_list, average="macro", zero_division=0)

        metrics = {
            "epoch": epoch_index + 1,
            "total_epochs": total_epochs,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss,
            "f1_score": f1,
            "progress_percentage": float((epoch_index + 1) * 100.0 / total_epochs),
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
            torch.save(model.state_dict(), best_model_path)

    return {
        "status": "completed",
        "model_path": str(best_model_path),
        "label_mapping_path": str(label_mapping_path),
        "best_val_accuracy": best_val_accuracy,
        "total_epochs": total_epochs,
    }


__all__ = ["run_training_loop"]
