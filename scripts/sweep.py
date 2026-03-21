"""
sweep.py — Hyperparameter sweep over cached embeddings.

Runs a grid of classifier configs on a pre-extracted embedding file,
logs results to a CSV, and saves the best model.

Usage:
    python scripts/sweep.py --embeddings outputs/embeddings/embeddings.h5 \
                            --output-dir outputs/sweep
"""

import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from evovir.dataset import EmbeddingDataset
from evovir.model import ViralClassifier
from evovir.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


GRID = {
    "head_type":     ["mlp", "linear"],
    "hidden_dim":    [256, 512, 1024],
    "dropout":       [0.1, 0.3, 0.5],
    "learning_rate": [1e-3, 1e-4, 5e-5],
    "weight_decay":  [1e-4, 1e-3],
}


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--embeddings", default="outputs/embeddings/embeddings.h5")
    parser.add_argument("--output-dir", default="outputs/sweep")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Base config (for non-swept params)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "sweep_results.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build grid (skip hidden_dim for linear head)
    combos = []
    for ht, hd, do, lr, wd in itertools.product(
        GRID["head_type"], GRID["hidden_dim"], GRID["dropout"],
        GRID["learning_rate"], GRID["weight_decay"],
    ):
        if ht == "linear" and hd != GRID["hidden_dim"][0]:
            continue  # hidden_dim irrelevant for linear
        combos.append({"head_type": ht, "hidden_dim": hd, "dropout": do,
                        "learning_rate": lr, "weight_decay": wd})

    print(f"Sweep: {len(combos)} configurations")
    print(f"Embeddings: {args.embeddings}")
    print(f"Results: {results_csv}\n")

    best_auroc = 0.0
    best_name = ""

    header_written = results_csv.exists()
    csv_file = open(results_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "run", "head_type", "hidden_dim", "dropout", "learning_rate",
        "weight_decay", "val_auroc", "val_auprc", "test_auroc",
        "test_auprc", "test_acc",
    ])
    if not header_written:
        writer.writeheader()

    for i, combo in enumerate(combos, 1):
        run_name = (f"{combo['head_type']}_h{combo['hidden_dim']}"
                    f"_do{combo['dropout']}_lr{combo['learning_rate']}"
                    f"_wd{combo['weight_decay']}")
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Run {i}/{len(combos)}: {run_name}")
        print(f"{'='*60}")

        set_seed(args.seed)
        dataset = EmbeddingDataset(args.embeddings)

        model = ViralClassifier(
            embedding_dim=base_cfg["embedding_dim"],
            num_classes=base_cfg.get("num_classes", 2),
            head_type=combo["head_type"],
            hidden_dim=combo["hidden_dim"],
            dropout=combo["dropout"],
        )

        trainer = Trainer(
            model=model,
            dataset=dataset,
            task=base_cfg.get("task", "binary"),
            val_fraction=base_cfg["val_fraction"],
            test_fraction=base_cfg["test_fraction"],
            batch_size=base_cfg["batch_size"],
            learning_rate=combo["learning_rate"],
            weight_decay=combo["weight_decay"],
            patience=base_cfg["patience"],
            output_dir=str(run_dir),
            device=device,
            seed=args.seed,
            precision=base_cfg["precision"],
            multi_gpu=base_cfg["multi_gpu"],
            compile_model=base_cfg["compile_model"],
            num_workers=base_cfg["num_workers"],
        )

        trainer.train(epochs=args.epochs)
        test_metrics = trainer.evaluate_test()

        val_auroc = max(trainer.history["val_auroc"]) if trainer.history["val_auroc"] else 0
        val_auprc = max(trainer.history["val_auprc"]) if trainer.history["val_auprc"] else 0

        row = {
            "run": run_name,
            "head_type": combo["head_type"],
            "hidden_dim": combo["hidden_dim"],
            "dropout": combo["dropout"],
            "learning_rate": combo["learning_rate"],
            "weight_decay": combo["weight_decay"],
            "val_auroc": f"{val_auroc:.4f}",
            "val_auprc": f"{val_auprc:.4f}",
            "test_auroc": f"{test_metrics['auroc']:.4f}",
            "test_auprc": f"{test_metrics['auprc']:.4f}",
            "test_acc": f"{test_metrics['accuracy']:.4f}",
        }
        writer.writerow(row)
        csv_file.flush()

        if test_metrics["auroc"] > best_auroc:
            best_auroc = test_metrics["auroc"]
            best_name = run_name

        print(f"  -> test AUROC={test_metrics['auroc']:.4f} | "
              f"best so far: {best_name} ({best_auroc:.4f})")

    csv_file.close()
    print(f"\n{'='*60}")
    print(f"Sweep complete. Best: {best_name} (test AUROC={best_auroc:.4f})")
    print(f"Results: {results_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
