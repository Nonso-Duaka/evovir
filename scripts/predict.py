"""
Run inference on new FASTA sequences with a trained model.

Usage
-----
    evovir-predict --fasta my_seqs.fa --config configs/default.yaml
    python scripts/predict.py --fasta my_seqs.fa --config configs/default.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from Bio import SeqIO

from evovir.embeddings import EmbeddingExtractor
from evovir.model import ViralClassifier


def main(fasta_path: str, config_path: str = "configs/default.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    task = cfg.get("task", "binary")
    is_binary = (task == "binary")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        print(f"No sequences found in {fasta_path}")
        return

    sequences = [str(r.seq).upper().replace("U", "T") for r in records]
    accessions = [r.id for r in records]
    print(f"[predict] Loaded {len(sequences)} sequences from {fasta_path}")

    extractor = EmbeddingExtractor(
        model_name=cfg["model_name"],
        layer_name=cfg["layer_name"],
        max_seq_len=cfg["max_seq_len"],
        window_stride=cfg["window_stride"],
        extraction_batch_size=cfg["extraction_batch_size"],
        precision=cfg["precision"],
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    embeddings_np = extractor.extract_batch(sequences)
    embeddings = torch.from_numpy(embeddings_np).to(device)

    num_classes = cfg.get("num_classes", 2)
    model = ViralClassifier(
        embedding_dim=cfg["embedding_dim"],
        num_classes=num_classes,
        head_type=cfg["head_type"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    )
    ckpt = Path(cfg["output_dir"]) / "best_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        probs = model.predict_proba(embeddings).cpu().numpy()

    if is_binary:
        df = pd.DataFrame({"accession": accessions, "prob_vertebrate": probs})
        df["prediction"] = (df["prob_vertebrate"] >= 0.5).astype(int)
    else:
        df = pd.DataFrame({"accession": accessions})
        for c in range(probs.shape[1]):
            df[f"prob_class_{c}"] = probs[:, c]
        df["prediction"] = probs.argmax(axis=1)

    print(df.to_string(index=False))
    out_csv = Path(cfg["output_dir"]) / "predictions.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nPredictions saved → {out_csv}")


def cli():
    parser = argparse.ArgumentParser(description="EvoVir: predict on new sequences")
    parser.add_argument("--fasta", required=True, help="Input FASTA file")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.fasta, args.config)


if __name__ == "__main__":
    cli()
