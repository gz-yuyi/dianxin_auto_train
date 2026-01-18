import contextlib
import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.config import get_data_root, get_model_output_dir


class TextClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, text_column: str, label_column: str, max_length: int):
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


class TextClassifier(nn.Module):
    def __init__(
        self,
        base_model: str,
        output_dim: int,
        lora_config: dict | None = None,
        base_model_instance: nn.Module | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.use_lora = lora_config is not None
        if base_model_instance is not None:
            self.bert = base_model_instance
        else:
            if torch_dtype is None:
                self.bert = AutoModel.from_pretrained(base_model)
            else:
                self.bert = AutoModel.from_pretrained(base_model, torch_dtype=torch_dtype)
        if lora_config is not None:
            self.bert = get_peft_model(self.bert, LoraConfig(**lora_config))
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.bert.config.hidden_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
            pooled_output = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return self.relu(linear_output)


def resolve_dataset_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return get_data_root() / path


def normalize_lora_config(hp: dict) -> dict | None:
    raw = hp.get("lora")
    if not raw or not raw.get("enabled", False):
        return None
    target_modules = raw.get("target_modules") or ["query", "value"]
    return {
        "r": int(raw.get("r", 8)),
        "lora_alpha": float(raw.get("lora_alpha", 16)),
        "lora_dropout": float(raw.get("lora_dropout", 0.1)),
        "target_modules": list(target_modules),
        "bias": "none",
    }


def configure_lora_trainables(model: TextClassifier) -> None:
    for name, param in model.bert.named_parameters():
        param.requires_grad = "lora_" in name
    for param in model.linear.parameters():
        param.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


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
    tokenizer: AutoTokenizer,
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


def save_label_mappings(mapping_path: Path, label_to_id: dict[str, int], id_to_label: dict[int, str]) -> None:
    with mapping_path.open("wb") as handle:
        pickle.dump((label_to_id, id_to_label), handle)


def run_training_loop(
    *,
    task_id: str,
    request_payload: dict,
    progress_handler: Callable[[int, dict[str, float]], None],
    batch_progress_handler: Callable[[int, int, dict[str, float]], None],
    stop_requested: Callable[[], bool],
) -> dict:
    hp = request_payload["hyperparameters"]
    callback_url = request_payload.get("callback_url")
    lora_config = normalize_lora_config(hp)

    logger.info("Starting training task {}", task_id)

    dataset_path = resolve_dataset_path(request_payload["training_data_file"])
    dataframe, label_to_id, id_to_label = prepare_dataframe(
        dataset_path=dataset_path,
        sheet_name=hp.get("sheet_name"),
        text_column=hp["text_column"],
        label_column=hp["label_column"],
    )
    setup_seed(hp["random_seed"])
    tokenizer = AutoTokenizer.from_pretrained(request_payload["base_model"])
    train_loader, dev_loader, dev_size = build_dataloaders(
        dataframe,
        tokenizer,
        hp,
        hp["text_column"],
        hp["label_column"],
    )

    output_dir = get_model_output_dir() / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    # Artifacts naming convention:
    # - full model weights: <name>.pt (full fine-tune only)
    # - LoRA adapter: <name>.lora (directory) + classifier head: <name>.head.pt
    # - label mappings: <weights>.pkl
    model_name_en = request_payload["model_name_en"]
    model_stem = model_name_en[:-3] if model_name_en.endswith(".pt") else model_name_en
    model_filename = f"{model_stem}.pt"
    best_model_path = output_dir / model_filename
    lora_adapter_path = None
    classifier_head_path = None
    if lora_config is None:
        label_mapping_path = output_dir / f"{model_filename}.pkl"
    else:
        lora_adapter_path = output_dir / f"{model_stem}.lora"
        classifier_head_path = output_dir / f"{model_stem}.head.pt"
        label_mapping_path = output_dir / f"{classifier_head_path.name}.pkl"
    save_label_mappings(label_mapping_path, label_to_id, id_to_label)

    device = select_device(task_id)
    logger.info("Task {} using device {}", task_id, device)

    precision = str(hp.get("precision", "fp32")).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision '{precision}', expected fp32, fp16, or bf16.")
    if device.type != "cuda" and precision != "fp32":
        logger.warning("Precision {} requested but CUDA unavailable; falling back to fp32.", precision)
        precision = "fp32"

    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None

    def save_model_artifacts() -> None:
        if lora_config is None:
            torch.save(model.state_dict(), best_model_path)
            return
        if lora_adapter_path is None or classifier_head_path is None:
            raise RuntimeError("LoRA paths were not initialized")
        lora_adapter_path.mkdir(parents=True, exist_ok=True)
        model.bert.save_pretrained(lora_adapter_path)
        torch.save(model.linear.state_dict(), classifier_head_path)

    model = TextClassifier(
        request_payload["base_model"],
        output_dim=len(label_to_id),
        lora_config=lora_config,
        torch_dtype=torch_dtype,
    )
    model = model.to(device)
    if lora_config is not None:
        configure_lora_trainables(model)
        trainable, total = count_trainable_parameters(model)
        logger.info("Task {} LoRA enabled: trainable params {}/{}", task_id, trainable, total)
    criterion = nn.CrossEntropyLoss().to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = Adam(trainable_params, lr=hp["learning_rate"])

    grad_accum_steps = int(hp.get("gradient_accumulation_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1.")

    use_amp = device.type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch_dtype if use_amp else None
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision == "fp16")

    def autocast_context():
        if not use_amp:
            return contextlib.nullcontext()
        return torch.cuda.amp.autocast(dtype=amp_dtype)

    best_val_accuracy = 0.0

    total_epochs = hp["epochs"]
    has_validation = dev_size > 0

    def build_result(status: str) -> dict:
        result = {
            "status": status,
            "label_mapping_path": str(label_mapping_path),
            "lora_enabled": lora_config is not None,
        }
        if lora_config is None:
            result["model_path"] = str(best_model_path)
        else:
            result["lora_adapter_path"] = str(lora_adapter_path)
            result["classifier_head_path"] = str(classifier_head_path)
        return result

    for epoch_index in range(total_epochs):
        if stop_requested():
            logger.info("Stop requested for task {}", task_id)
            return build_result("stopped")

        model.train()
        total_acc_train = 0.0
        total_loss_train = 0.0
        sample_count_train = 0
        total_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            input_ids = inputs["input_ids"].squeeze(1).to(device)
            attention_mask = inputs["attention_mask"].squeeze(1).to(device)
            labels = labels.to(device)
            with autocast_context():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            loss_value = loss.item()
            scaled_loss = loss / grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            predictions = outputs.argmax(dim=1)
            total_acc_train += (predictions == labels).sum().item()
            total_loss_train += loss_value * labels.size(0)
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
                return build_result("stopped")

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
                with autocast_context():
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
                save_model_artifacts()

    if not has_validation:
        save_model_artifacts()

    result = build_result("completed")
    result["best_val_accuracy"] = best_val_accuracy
    result["total_epochs"] = total_epochs
    return result


__all__ = ["run_training_loop"]
