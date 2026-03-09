"""
Training loop for the ViralClassifier head.

Uses pre-extracted embeddings (EmbeddingDataset) so Evo 2 does not need to
be loaded during training, making this runnable on a smaller GPU or CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, random_split

from evovir.dataset import EmbeddingDataset
from evovir.model import ViralClassifier


class Trainer:
    """
    Args:
        model: ``ViralClassifier`` instance.
        dataset: Full ``EmbeddingDataset`` (train + val + test will be split from this).
        val_fraction: Fraction of data reserved for validation.
        test_fraction: Fraction of data reserved for testing.
        batch_size: Training batch size.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        patience: Early-stopping patience (epochs without val-AUROC improvement).
        output_dir: Directory where checkpoints and metrics are saved.
        device: Torch device string.
        seed: Random seed for splits.
    """

    def __init__(
        self,
        model: ViralClassifier,
        dataset: EmbeddingDataset,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        batch_size: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        patience: int = 10,
        output_dir: str | Path = "outputs",
        device: str = "cuda",
        seed: int = 42,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.patience = patience

        torch.manual_seed(seed)
        n = len(dataset)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)
        n_train = n - n_val - n_test
        self.train_ds, self.val_ds, self.test_ds = random_split(
            dataset, [n_train, n_val, n_test]
        )

        self.train_loader = DataLoader(
            self.train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        pos_weight = dataset.class_weights.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.model.to(self.device)
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": []}

    # ------------------------------------------------------------------

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

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_auroc={val_metrics['auroc']:.4f} | "
                f"val_auprc={val_metrics['auprc']:.4f}"
            )

            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                no_improve = 0
                torch.save(self.model.state_dict(), best_path)
                print(f"  ✓ New best AUROC={best_auroc:.4f} — model saved.")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping after {epoch} epochs (no improvement for {self.patience} epochs).")
                    break

        print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")
        self._save_history()

    # ------------------------------------------------------------------

    def evaluate_test(self) -> Dict[str, float]:
        best_path = self.output_dir / "best_model.pt"
        if best_path.exists():
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        metrics = self._evaluate(self.test_loader)
        print(
            f"\nTest set | loss={metrics['loss']:.4f} | "
            f"auroc={metrics['auroc']:.4f} | auprc={metrics['auprc']:.4f} | "
            f"acc={metrics['accuracy']:.4f}"
        )
        return metrics

    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for embeddings, labels in self.train_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(embeddings)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(labels)
        return total_loss / len(self.train_loader.dataset)

    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        with torch.no_grad():
            for embeddings, labels in loader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(embeddings)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * len(labels)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        probs = 1 / (1 + np.exp(-all_logits))
        preds = (probs >= 0.5).astype(int)

        return {
            "loss": total_loss / len(loader.dataset),
            "auroc": roc_auc_score(all_labels, probs),
            "auprc": average_precision_score(all_labels, probs),
            "accuracy": (preds == all_labels).mean(),
        }

    def _save_history(self) -> None:
        import json
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
