"""
Extract Evo 2 embeddings for all sequences in data/metadata.csv and save to HDF5.

Usage
-----
    python scripts/extract_embeddings.py --config configs/default.yaml

Outputs
-------
    outputs/embeddings/train.h5   (or a single all.h5 if no split requested)

Run this once on a GPU machine, then use train.py / evaluate.py which only
need the HDF5 files — no GPU required for those steps if you copy the files.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from evovir.dataset import ViralDataset
from evovir.embeddings import EmbeddingExtractor


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    emb_dir = Path(cfg["embeddings_dir"])
    emb_dir.mkdir(parents=True, exist_ok=True)

    # Load raw sequences
    ds = ViralDataset(
        metadata_path=data_dir / "metadata.csv",
        fasta_dir=data_dir / "fasta",
        min_len=cfg["min_seq_len"],
        max_len=cfg["max_genome_len"],
        ambiguous_threshold=cfg["ambiguous_base_threshold"],
    )

    # Extract embeddings
    extractor = EmbeddingExtractor(
        model_name=cfg["model_name"],
        layer_name=cfg["layer_name"],
        max_seq_len=cfg["max_seq_len"],
        window_stride=cfg["window_stride"],
        device="cuda:0",
    )

    out_path = emb_dir / "embeddings.h5"
    extractor.save_to_hdf5(
        sequences=ds.sequences,
        labels=ds.labels,
        accessions=ds.accessions,
        out_path=out_path,
    )
    print(f"\nDone. Embeddings saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
