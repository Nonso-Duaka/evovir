"""
Extract Evo 2 embeddings for all sequences and save to HDF5.

Usage
-----
    python scripts/extract_embeddings.py --config configs/default.yaml [--device cuda:0]

GPU notes
---------
- extraction_batch_size in config controls how many windows are processed per
  forward pass. Start at 8 and increase until you hit ~90% VRAM utilisation.
- Precision is set by `precision` in config (bf16 recommended for A100/H100).
- OOM is handled automatically by halving the batch and retrying.
"""

import argparse
from pathlib import Path

import torch
import yaml

from evovir.dataset import ViralDataset
from evovir.embeddings import EmbeddingExtractor, _log_gpu_memory


def main(config_path: str, device: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. Embedding extraction will be very slow on CPU.")

    _log_gpu_memory("start")

    data_dir = Path(cfg["data_dir"])
    emb_dir = Path(cfg["embeddings_dir"])
    emb_dir.mkdir(parents=True, exist_ok=True)

    ds = ViralDataset(
        metadata_path=data_dir / "metadata.csv",
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
        device=device,
    )

    out_path = emb_dir / "embeddings.h5"
    extractor.save_to_hdf5(
        sequences=ds.sequences,
        labels=ds.labels,
        accessions=ds.accessions,
        out_path=out_path,
    )

    _log_gpu_memory("done")
    print(f"\nEmbeddings saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda:0", help="Primary CUDA device.")
    args = parser.parse_args()
    main(args.config, args.device)
