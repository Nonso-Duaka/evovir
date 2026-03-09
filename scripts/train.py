"""
Train the ViralClassifier head on pre-extracted embeddings.

Usage
-----
    python scripts/train.py --config configs/default.yaml

Outputs
-------
    outputs/best_model.pt
    outputs/training_history.json
"""

import argparse
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


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    emb_path = Path(cfg["embeddings_dir"]) / "embeddings.h5"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {emb_path}. "
            "Run scripts/extract_embeddings.py first."
        )

    dataset = EmbeddingDataset(emb_path)

    model = ViralClassifier(
        embedding_dim=cfg["embedding_dim"],
        head_type=cfg["head_type"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    )

    trainer = Trainer(
        model=model,
        dataset=dataset,
        val_fraction=cfg["val_fraction"],
        test_fraction=cfg["test_fraction"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience"],
        output_dir=cfg["output_dir"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=cfg["seed"],
    )

    trainer.train(epochs=cfg["epochs"])
    trainer.evaluate_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
