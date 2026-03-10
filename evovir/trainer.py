"""
GPU-optimised training loop for ViralClassifier.

Supports binary (BCEWithLogitsLoss) and multiclass (CrossEntropyLoss).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, random_split

from evovir.dataset import EmbeddingDataset
from evovir.model import ViralClassifier

_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class Trainer:
    def __init__(
        self,
        model: ViralClassifier,
        dataset: EmbeddingDataset,
        task: str = "binary",
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 10,
        output_dir: str | Path = "outputs",
        device: str = "cuda",
        seed: int = 42,
        precision: str = "bf16",
        multi_gpu: bool = True,
        compile_model: bool = False,
        num_workers: int = 4,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.task = task
        self.is_binary = (task == "binary")

        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        n_gpus = torch.cuda.device_count()
        print(f"[Trainer] Device: {self.device}  |  GPUs: {n_gpus}  |  Task: {task}")

        # Mixed precision
        self.amp_dtype = _DTYPE_MAP.get(precision, torch.bfloat16)
        self.use_amp = torch.cuda.is_available() and precision in ("fp16", "bf16")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and precision == "fp16")

        # Splits
        torch.manual_seed(seed)
        n = len(dataset)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)
        n_train = n - n_val - n_test
        train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )
        self.train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

        # Loss
        class_weights = dataset.class_weights.to(self.device)
        if self.is_binary:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1:2])
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Model
        model = model.to(self.device)
        if multi_gpu and n_gpus > 1:
            model = nn.DataParallel(model)
            print(f"[Trainer] DataParallel across {n_gpus} GPUs.")
        if compile_model:
            model = torch.compile(model)
            print("[Trainer] torch.compile enabled.")
        self.model = model

        # Optimiser + scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": []
        }

    def train(self, epochs: int = 50) -> None:
        best_auroc = 0.0
        no_improve = 0
        best_path = self.output_dir / "best_model.pt"

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._evaluate(self.val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auroc"].append(val_metrics["auroc"])
            self.history["val_auprc"].append(val_metrics["auprc"])

            mem = _gpu_mem_str()
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_auroc={val_metrics['auroc']:.4f} | "
                f"val_auprc={val_metrics['auprc']:.4f}"
                + (f" | GPU {mem}" if mem else "")
            )

            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                no_improve = 0
                state = _unwrap(self.model).state_dict()
                torch.save(state, best_path)
                print(f"  New best AUROC={best_auroc:.4f} — saved.")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")
        self._save_history()


    def evaluate_test(self) -> Dict[str, float]:
        best_path = self.output_dir / "best_model.pt"
        if best_path.exists():
            _unwrap(self.model).load_state_dict(
                torch.load(best_path, map_location=self.device)
            )
        metrics = self._evaluate(self.test_loader)
        print(
            f"\nTest | loss={metrics['loss']:.4f} | "
            f"auroc={metrics['auroc']:.4f} | auprc={metrics['auprc']:.4f} | "
            f"acc={metrics['accuracy']:.4f}"
        )
        return metrics


    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for embeddings, labels in self.train_loader:
            embeddings = embeddings.to(self.device, non_blocking=True)
            labels = self._prepare_labels(labels)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda" if self.device.type == "cuda" else "cpu",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(embeddings)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * len(labels)

        return total_loss / len(self.train_loader.dataset)

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        with torch.inference_mode():
            for embeddings, labels in loader:
                embeddings = embeddings.to(self.device, non_blocking=True)
                labels = self._prepare_labels(labels)

                with torch.autocast(
                    device_type="cuda" if self.device.type == "cuda" else "cpu",
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    logits = self.model(embeddings)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item() * len(labels)
                all_logits.append(logits.float().cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        return self._compute_metrics(all_logits, all_labels, total_loss / len(loader.dataset))

    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Binary needs float labels; multiclass needs long."""
        if self.is_binary:
            return labels.float().to(self.device, non_blocking=True)
        return labels.long().to(self.device, non_blocking=True)

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor, loss: float) -> Dict[str, float]:
        labels_np = labels.numpy()

        if self.is_binary:
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)
            auroc = roc_auc_score(labels_np, probs)
            auprc = average_precision_score(labels_np, probs)
        else:
            probs = torch.softmax(logits, dim=-1).numpy()
            preds = probs.argmax(axis=1)
            # macro-averaged one-vs-rest AUROC
            auroc = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
            # per-class AUPRC averaged
            n_classes = probs.shape[1]
            auprcs = []
            for c in range(n_classes):
                binary_labels = (labels_np == c).astype(int)
                if binary_labels.sum() > 0:
                    auprcs.append(average_precision_score(binary_labels, probs[:, c]))
            auprc = float(np.mean(auprcs)) if auprcs else 0.0

        return {
            "loss": loss,
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": float((preds == labels_np).mean()),
        }

    def _save_history(self) -> None:
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)



def _unwrap(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def _gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f"{alloc:.1f}/{reserved:.1f}GB"
