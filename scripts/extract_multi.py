"""
extract_multi.py — Extract embeddings for multiple Evo2 layers and seq_len settings.

Saves each combination to a separate HDF5 so the sweep script can
compare embedding quality without re-extracting.

Usage:
    python scripts/extract_multi.py --config configs/default.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml

from evovir.dataset import ViralDataset
from evovir.embeddings import EmbeddingExtractor, _log_gpu_memory


LAYER_CONFIGS = [
    # (layer_name, description)
    ("blocks.14.mlp.l3", "mid-network (layer 14)"),
    ("blocks.21.mlp.l3", "upper-mid (layer 21)"),
    ("blocks.28.mlp.l3", "near-final (layer 28) — default"),
]

SEQ_LEN_CONFIGS = [
    # (max_seq_len, window_stride)
    (2000, 1000),
    (4000, 2000),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    base_emb_dir = Path(cfg["embeddings_dir"]).parent  # outputs/

    ds = ViralDataset(
        metadata_path=data_dir / "metadata.csv",
        fasta_dir=data_dir / "fasta",
        min_len=cfg["min_seq_len"],
        max_len=cfg["max_genome_len"],
        ambiguous_threshold=cfg["ambiguous_base_threshold"],
    )
    print(f"\nLoaded {len(ds)} sequences for embedding extraction.\n")

    for layer_name, layer_desc in LAYER_CONFIGS:
        for max_seq_len, window_stride in SEQ_LEN_CONFIGS:
            tag = f"{layer_name.replace('.', '_')}_seq{max_seq_len}"
            emb_dir = base_emb_dir / "embeddings" / tag
            out_path = emb_dir / "embeddings.h5"

            if out_path.exists():
                print(f"\n[skip] {tag} — already exists at {out_path}")
                continue

            print(f"\n{'='*60}")
            print(f"Extracting: {tag}")
            print(f"  Layer: {layer_name} ({layer_desc})")
            print(f"  max_seq_len={max_seq_len}, stride={window_stride}")
            print(f"{'='*60}")

            emb_dir.mkdir(parents=True, exist_ok=True)

            extractor = EmbeddingExtractor(
                model_name=cfg["model_name"],
                layer_name=layer_name,
                max_seq_len=max_seq_len,
                window_stride=window_stride,
                extraction_batch_size=cfg["extraction_batch_size"],
                precision=cfg["precision"],
                device=args.device,
            )

            extractor.save_to_hdf5(
                sequences=ds.sequences,
                labels=ds.labels,
                accessions=ds.accessions,
                out_path=out_path,
            )

            _log_gpu_memory(f"after {tag}")
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All embedding extractions complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
