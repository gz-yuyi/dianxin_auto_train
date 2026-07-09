import json
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
from src.device_utils import (
    autocast_context,
    create_grad_scaler,
    device_supports_precision,
    enable_deterministic_mode,
    get_auto_device,
    manual_seed_all,
    set_current_device,
    visible_device_env_var,
)


DEFAULT_LORA_TARGET_MODULES = ["query", "key", "value", "dense"]
DEFAULT_CLASSIFIER_POOLING_STRATEGY = "mean_cls"
DEFAULT_OUTPUT_ACTIVATION = "none"
LEGACY_CLASSIFIER_POOLING_STRATEGY = "pooler_or_mean"
LEGACY_OUTPUT_ACTIVATION = "relu"
VALID_CLASSIFIER_POOLING_STRATEGIES = {DEFAULT_CLASSIFIER_POOLING_STRATEGY, LEGACY_CLASSIFIER_POOLING_STRATEGY}
VALID_OUTPUT_ACTIVATIONS = {DEFAULT_OUTPUT_ACTIVATION, LEGACY_OUTPUT_ACTIVATION}


def normalize_model_stem(model_name_en: str) -> str:
    model_stem = model_name_en[:-3] if model_name_en.endswith(".pt") else model_name_en
    if not model_stem or model_stem in {".", ".."} or "/" in model_stem or "\\" in model_stem:
        raise ValueError("model_name_en must be a plain model name without path separators")
    return model_stem


class TextClassificationDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, text_column: str, label_column: str, max_length: int):
        self.texts = [
            tokenizer(
                str(text),
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
        pooling_strategy: str = DEFAULT_CLASSIFIER_POOLING_STRATEGY,
        output_activation: str = DEFAULT_OUTPUT_ACTIVATION,
    ):
        super().__init__()
        if pooling_strategy not in VALID_CLASSIFIER_POOLING_STRATEGIES:
            raise ValueError(f"Unsupported pooling_strategy: {pooling_strategy}")
        if output_activation not in VALID_OUTPUT_ACTIVATIONS:
            raise ValueError(f"Unsupported output_activation: {output_activation}")
        self.pooling_strategy = pooling_strategy
        self.output_activation = output_activation
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
        mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
        mean_output = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        if self.pooling_strategy == "mean_cls":
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                cls_output = outputs.pooler_output
            else:
                cls_output = outputs.last_hidden_state[:, 0, :]
            pooled_output = 0.5 * mean_output + 0.5 * cls_output
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = mean_output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        if self.output_activation == "relu":
            return self.relu(linear_output)
        return linear_output


def resolve_dataset_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    return get_data_root() / path


def normalize_lora_config(hp: dict) -> dict | None:
    raw = hp.get("lora")
    # LoRA is the online-publishable training mode. Treat missing/null LoRA
    # config as the optimized default; callers can still disable it explicitly
    # with {"lora": {"enabled": false}}.
    if raw is None:
        raw = {"enabled": True}
    elif hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    elif not isinstance(raw, dict):
        raw = dict(raw)
    elif not raw:
        raw = {"enabled": True}
    if not raw.get("enabled", False):
        return None
    target_modules = raw.get("target_modules") or DEFAULT_LORA_TARGET_MODULES
    if isinstance(target_modules, str):
        target_modules_value: list[str] | str = target_modules
    else:
        target_modules_value = list(target_modules)
    return {
        "r": int(raw.get("r", 16)),
        "lora_alpha": float(raw.get("lora_alpha", 32)),
        "lora_dropout": float(raw.get("lora_dropout", 0.1)),
        "target_modules": target_modules_value,
        "bias": "none",
    }


def configure_lora_trainables(model: TextClassifier) -> None:
    for name, param in model.bert.named_parameters():
        param.requires_grad = any(token in name.lower() for token in ("lora_", "modules_to_save"))
    for param in model.linear.parameters():
        param.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def setup_seed(seed: int) -> None:
    manual_seed_all(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    enable_deterministic_mode()


def select_device(task_id: str) -> torch.device:
    del task_id
    device = get_auto_device()
    if device.type != "cpu":
        visible = (
            os.getenv("CUDA_VISIBLE_DEVICES")
            or os.getenv("ASCEND_RT_VISIBLE_DEVICES")
            or os.getenv("GPU_VISIBLE_DEVICES")
        )
        if visible:
            # When a single visible device is configured per-process we always use local index 0.
            env_name = visible_device_env_var(device.type)
            if env_name and os.getenv(env_name) is None:
                os.environ[env_name] = visible
            device = torch.device(f"{device.type}:0")
        set_current_device(device)
    return device


def prepare_dataframe(
    dataset_path: Path,
    sheet_name: str | None,
    text_column: str,
    label_column: str,
    label_to_id: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    if sheet_name is None:
        dataframe = pd.read_excel(dataset_path)
    else:
        dataframe = pd.read_excel(dataset_path, sheet_name=sheet_name)

    dataframe[text_column] = dataframe[text_column].astype(str).fillna("")
    labels = dataframe[label_column].astype(str)
    dataframe[label_column] = labels
    if label_to_id is None:
        # Keep mapping stable across repeated trainings with the same label set.
        unique_labels = sorted(list(labels.unique()))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    else:
        unknown = sorted(set(labels.unique()) - set(label_to_id.keys()))
        if unknown:
            raise ValueError(f"Validation labels not in training set: {unknown}")
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    dataframe[label_column] = dataframe[label_column].map(label_to_id)
    return dataframe, label_to_id, id_to_label


def build_dataloaders(
    dataframe: pd.DataFrame,
    tokenizer: AutoTokenizer,
    hp: dict,
    text_column: str,
    label_column: str,
    validation_dataframe: pd.DataFrame | None = None,
) -> tuple[DataLoader, DataLoader, int]:
    # Dataset split uses a fixed random_state=42 (independent from training random seed).
    if validation_dataframe is not None:
        train_df = dataframe
        dev_df = validation_dataframe
    elif hp["train_val_split"] == 0:
        train_df = dataframe
        dev_df = dataframe.iloc[0:0]
    else:
        use_stratify = bool(hp.get("stratified_split", True))
        try:
            train_df, holdout_df = train_test_split(
                dataframe,
                test_size=hp["train_val_split"],
                stratify=dataframe[label_column] if use_stratify else None,
                random_state=42,
            )
            dev_df, _ = train_test_split(
                holdout_df,
                test_size=0.5,
                stratify=holdout_df[label_column] if use_stratify else None,
                random_state=42,
            )
        except ValueError as exc:
            if not use_stratify:
                raise
            logger.warning("Stratified train/validation split failed ({}); falling back to unstratified split", exc)
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


def augment_with_label_anchor_samples(
    dataframe: pd.DataFrame,
    label_to_id: dict[str, int],
    hp: dict,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    if not bool(hp.get("anchor_samples_enabled", True)):
        return dataframe
    repeat = int(hp.get("anchor_repeat", 15))
    if repeat <= 0 or not label_to_id:
        return dataframe
    anchor_df = pd.DataFrame(
        {
            text_column: list(label_to_id.keys()),
            label_column: list(label_to_id.values()),
        }
    )
    anchor_df = pd.concat([anchor_df] * repeat, ignore_index=True)
    logger.info(
        "Adding {} label anchor samples ({} labels x repeat {})",
        len(anchor_df),
        len(label_to_id),
        repeat,
    )
    return pd.concat([dataframe, anchor_df], ignore_index=True)


def normalize_classifier_pooling_strategy(hp: dict) -> str:
    value = str(hp.get("classifier_pooling_strategy", DEFAULT_CLASSIFIER_POOLING_STRATEGY)).strip().lower()
    if value not in VALID_CLASSIFIER_POOLING_STRATEGIES:
        logger.warning(
            "Unsupported classifier_pooling_strategy {}; falling back to {}",
            value,
            DEFAULT_CLASSIFIER_POOLING_STRATEGY,
        )
        return DEFAULT_CLASSIFIER_POOLING_STRATEGY
    return value


def normalize_output_activation(hp: dict) -> str:
    value = str(hp.get("output_activation", DEFAULT_OUTPUT_ACTIVATION)).strip().lower()
    if value not in VALID_OUTPUT_ACTIVATIONS:
        logger.warning(
            "Unsupported output_activation {}; falling back to {}",
            value,
            DEFAULT_OUTPUT_ACTIVATION,
        )
        return DEFAULT_OUTPUT_ACTIVATION
    return value


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
    classifier_pooling_strategy = normalize_classifier_pooling_strategy(hp)
    output_activation = normalize_output_activation(hp)

    logger.info("Starting training task {}", task_id)

    dataset_path = resolve_dataset_path(request_payload["training_data_file"])
    dataframe, label_to_id, id_to_label = prepare_dataframe(
        dataset_path=dataset_path,
        sheet_name=hp.get("sheet_name"),
        text_column=hp["text_column"],
        label_column=hp["label_column"],
    )
    dataframe = augment_with_label_anchor_samples(
        dataframe,
        label_to_id,
        hp,
        hp["text_column"],
        hp["label_column"],
    )
    validation_dataframe = None
    validation_file = request_payload.get("validation_data_file")
    if validation_file:
        validation_path = resolve_dataset_path(validation_file)
        validation_dataframe, _, _ = prepare_dataframe(
            dataset_path=validation_path,
            sheet_name=hp.get("validation_sheet_name"),
            text_column=hp["text_column"],
            label_column=hp["label_column"],
            label_to_id=label_to_id,
        )
    setup_seed(hp["random_seed"])
    tokenizer = AutoTokenizer.from_pretrained(request_payload["base_model"])
    train_loader, dev_loader, dev_size = build_dataloaders(
        dataframe,
        tokenizer,
        hp,
        hp["text_column"],
        hp["label_column"],
        validation_dataframe=validation_dataframe,
    )

    # Artifacts are stored under the model English name so inference publish/list
    # use business-visible model IDs instead of training task/algorithm IDs.
    model_name_en = request_payload["model_name_en"]
    model_stem = normalize_model_stem(model_name_en)
    output_dir = get_model_output_dir() / model_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    # Artifacts naming convention:
    # - output directory: <model_stem> (model_name_en without optional .pt suffix)
    # - full model weights: <name>.pt (full fine-tune only)
    # - LoRA adapter: <name>.lora (directory) + classifier head: <name>.head.pt
    # - label mappings: <weights>.pkl
    model_filename = f"{model_stem}.pt"
    best_model_path = output_dir / model_filename
    model_meta_path = output_dir / "model_meta.json"
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
    if not device_supports_precision(device, precision):
        logger.warning("Precision {} requested but {} does not support it; falling back to fp32.", precision, device)
        precision = "fp32"

    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None

    def save_model_meta() -> None:
        payload = {
            "format_version": 2,
            "model_id": model_stem,
            "model_name_en": model_name_en,
            "lora_enabled": lora_config is not None,
            "classifier_pooling_strategy": classifier_pooling_strategy,
            "output_activation": output_activation,
            "label_mapping_path": str(label_mapping_path),
        }
        model_meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def save_model_artifacts() -> None:
        if lora_config is None:
            torch.save(model.state_dict(), best_model_path)
            save_model_meta()
            return
        if lora_adapter_path is None or classifier_head_path is None:
            raise RuntimeError("LoRA paths were not initialized")
        lora_adapter_path.mkdir(parents=True, exist_ok=True)
        model.bert.save_pretrained(lora_adapter_path)
        torch.save(model.linear.state_dict(), classifier_head_path)
        save_model_meta()

    model = TextClassifier(
        request_payload["base_model"],
        output_dim=len(label_to_id),
        lora_config=lora_config,
        torch_dtype=torch_dtype,
        pooling_strategy=classifier_pooling_strategy,
        output_activation=output_activation,
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

    use_amp = precision in {"fp16", "bf16"} and device_supports_precision(device, precision)
    amp_dtype = torch_dtype if use_amp else None
    scaler = create_grad_scaler(device, precision if use_amp else "fp32")

    best_val_accuracy = 0.0
    best_model_metric: float | None = None
    early_stop_enabled = bool(hp.get("early_stopping_enabled", False))
    early_stop_patience = int(hp.get("early_stopping_patience", 3))
    early_stop_min_delta = float(hp.get("early_stopping_min_delta", 0.0))
    early_stop_metric = str(hp.get("early_stopping_metric", "val_accuracy")).lower()
    if early_stop_metric not in {"val_accuracy", "val_loss", "f1_score"}:
        raise ValueError(
            "early_stopping_metric must be one of: val_accuracy, val_loss, f1_score"
        )
    epochs_without_improvement = 0

    total_epochs = hp["epochs"]
    has_validation = dev_size > 0
    if early_stop_enabled and not has_validation:
        logger.warning("Early stopping enabled but no validation data; disabling.")
        early_stop_enabled = False

    def build_result(status: str) -> dict:
        result = {
            "status": status,
            "model_id": model_stem,
            "model_dir": str(output_dir),
            "label_mapping_path": str(label_mapping_path),
            "model_meta_path": str(model_meta_path),
            "lora_enabled": lora_config is not None,
        }
        if lora_config is None:
            result["model_path"] = str(best_model_path)
        else:
            result["lora_adapter_path"] = str(lora_adapter_path)
            result["classifier_head_path"] = str(classifier_head_path)
        return result

    early_stopped = False
    epochs_ran = 0
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
            with autocast_context(device, precision if use_amp else "fp32", amp_dtype):
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
            
            # Calculate overall progress across all epochs for polling clients.
            batch_progress = float(((epoch_index * total_batches) + batch_idx + 1) * 100.0 / (total_epochs * total_batches))
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
                with autocast_context(device, precision if use_amp else "fp32", amp_dtype):
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
        epochs_ran = epoch_index + 1

        if has_validation:
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

            if early_stop_metric == "val_accuracy":
                current_metric = val_accuracy
                better = best_model_metric is None or current_metric > best_model_metric + early_stop_min_delta
            elif early_stop_metric == "val_loss":
                current_metric = val_loss
                better = best_model_metric is None or current_metric < best_model_metric - early_stop_min_delta
            else:
                current_metric = f1
                better = best_model_metric is None or current_metric > best_model_metric + early_stop_min_delta

            if better:
                best_model_metric = current_metric
                epochs_without_improvement = 0
                save_model_artifacts()
            else:
                epochs_without_improvement += 1
                if early_stop_enabled and epochs_without_improvement >= early_stop_patience:
                    logger.info(
                        "Early stopping at epoch {} (metric {} did not improve for {} epochs)",
                        epoch_index + 1,
                        early_stop_metric,
                        epochs_without_improvement,
                    )
                    early_stopped = True
                    break

    if not has_validation:
        save_model_artifacts()

    result = build_result("completed")
    result["best_val_accuracy"] = best_val_accuracy
    result["total_epochs"] = total_epochs
    result["epochs_completed"] = epochs_ran
    result["early_stopped"] = early_stopped
    return result


__all__ = ["normalize_model_stem", "run_training_loop"]
