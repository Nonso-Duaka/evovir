"""
Train the ViralClassifier head.

Automatically extracts embeddings if they don't exist yet, then trains
and evaluates the model — all in one command.

Usage
-----
    evovir-train --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml \
        --metadata data/metadata_binary_vv.csv \
        --embeddings-path outputs/embeddings/metadata_binary_vv.h5 \
        --output-dir outputs/metadata_binary_vv
"""

import argparse
import random
from pathlib import Path
from typing import Optional

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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_metadata_path(cfg: dict, metadata_override: Optional[str] = None) -> Path:
    if metadata_override:
        return Path(metadata_override)
    if "metadata_path" in cfg:
        return Path(cfg["metadata_path"])
    return Path(cfg["data_dir"]) / "metadata.csv"


def resolve_embeddings_path(cfg: dict, metadata_path: Path, embeddings_override: Optional[str] = None) -> Path:
    if embeddings_override:
        return Path(embeddings_override)
    return Path(cfg["embeddings_dir"]) / f"{metadata_path.stem}.h5"


def resolve_output_dir(cfg: dict, metadata_path: Path, output_override: Optional[str] = None) -> Path:
    if output_override:
        return Path(output_override)

    output_dir = Path(cfg["output_dir"])
    if metadata_path.name == "metadata.csv":
        return output_dir
    return output_dir / metadata_path.stem


def extract_if_needed(cfg: dict, metadata_path: Path, emb_path: Path) -> None:
    if emb_path.exists():
        print(f"[train] Embeddings found at {emb_path}, skipping extraction.")
        return

    print("[train] Embeddings not found — extracting now...")
    from evovir.dataset import ViralDataset
    from evovir.embeddings import EmbeddingExtractor

    data_dir = Path(cfg["data_dir"])
    emb_dir = emb_path.parent
    emb_dir.mkdir(parents=True, exist_ok=True)

    ds = ViralDataset(
        metadata_path=metadata_path,
        fasta_dir=data_dir / "fasta",
        min_len=cfg["min_seq_len"],
        max_len=cfg["max_genome_len"],
        ambiguous_threshold=cfg["ambiguous_base_threshold"],
    )

    extractor = EmbeddingExtractor(
        model_name=cfg["model_name"],
        layer_name=cfg["layer_name"],
        max_seq_len=cfg["max_seq_len"],
        window_stride=cfg["window_stride"],
        extraction_batch_size=cfg["extraction_batch_size"],
        precision=cfg["precision"],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    extractor.save_to_hdf5(
        sequences=ds.sequences,
        labels=ds.labels,
        accessions=ds.accessions,
        out_path=emb_path,
    )
    print(f"[train] Embeddings saved → {emb_path}")


def main(
    config_path: str = "configs/default.yaml",
    metadata_path_override: Optional[str] = None,
    embeddings_path_override: Optional[str] = None,
    output_dir_override: Optional[str] = None,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    metadata_path = resolve_metadata_path(cfg, metadata_path_override)
    emb_path = resolve_embeddings_path(cfg, metadata_path, embeddings_path_override)
    output_dir = resolve_output_dir(cfg, metadata_path, output_dir_override)

    print(f"[train] Metadata: {metadata_path}")
    print(f"[train] Embeddings: {emb_path}")
    print(f"[train] Output dir: {output_dir}")

    set_seed(cfg["seed"])
    extract_if_needed(cfg, metadata_path, emb_path)

    task = cfg.get("task", "binary")
    dataset = EmbeddingDataset(emb_path)
    num_classes = cfg.get("num_classes", dataset.num_classes)

    model = ViralClassifier(
        embedding_dim=cfg["embedding_dim"],
        num_classes=num_classes,
        head_type=cfg["head_type"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    )

    trainer = Trainer(
        model=model,
        dataset=dataset,
        task=task,
        val_fraction=cfg["val_fraction"],
        test_fraction=cfg["test_fraction"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        patience=cfg["patience"],
        output_dir=output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=cfg["seed"],
        precision=cfg["precision"],
        multi_gpu=cfg["multi_gpu"],
        compile_model=cfg["compile_model"],
        num_workers=cfg["num_workers"],
    )

    trainer.train(epochs=cfg["epochs"])
    trainer.evaluate_test()


def cli():
    parser = argparse.ArgumentParser(description="EvoVir: train classifier")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--metadata", default=None, help="Metadata CSV to use instead of data/metadata.csv")
    parser.add_argument("--embeddings-path", default=None, help="Optional explicit HDF5 cache path")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for this run")
    args = parser.parse_args()
    main(args.config, args.metadata, args.embeddings_path, args.output_dir)


if __name__ == "__main__":
    cli()
