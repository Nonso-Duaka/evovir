"""
Evaluate a trained ViralClassifier.

Usage
-----
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --config configs/default.yaml --fasta my_viruses.fa
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from evovir.dataset import EmbeddingDataset
from evovir.model import ViralClassifier


def load_model(cfg: dict, device: torch.device) -> ViralClassifier:
    task = cfg.get("task", "binary")
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
    return model


def evaluate_hdf5(cfg: dict, device: torch.device) -> None:
    task = cfg.get("task", "binary")
    is_binary = (task == "binary")

    emb_path = Path(cfg["embeddings_dir"]) / "embeddings.h5"
    dataset = EmbeddingDataset(emb_path)
    model = load_model(cfg, device)

    embeddings = dataset.embeddings.to(device)
    labels = dataset.labels.numpy()

    with torch.no_grad():
        probs = model.predict_proba(embeddings).cpu().numpy()

    out_dir = Path(cfg["output_dir"])

    if is_binary:
        preds = (probs >= 0.5).astype(int)
        auroc = roc_auc_score(labels, probs)
        auprc = average_precision_score(labels, probs)

        print("\n-- Classification Report --")
        print(classification_report(labels, preds, target_names=["other", "vertebrate_viral"]))
        print(f"AUROC : {auroc:.4f}")
        print(f"AUPRC : {auprc:.4f}")

        _plot_roc(labels, probs, auroc, out_dir / "roc_curve.png")
        _plot_pr(labels, probs, auprc, out_dir / "pr_curve.png")
        _plot_confusion(labels, preds, ["other", "vertebrate_viral"], out_dir / "confusion_matrix.png")

        results = {"auroc": auroc, "auprc": auprc, "n_samples": int(len(labels))}
    else:
        preds = probs.argmax(axis=1)
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        n_classes = probs.shape[1]
        class_names = [f"class_{i}" for i in range(n_classes)]

        print("\n-- Classification Report --")
        print(classification_report(labels, preds, target_names=class_names))
        print(f"Macro AUROC : {auroc:.4f}")

        _plot_confusion(labels, preds, class_names, out_dir / "confusion_matrix.png")

        results = {"macro_auroc": auroc, "accuracy": float((preds == labels).mean()), "n_samples": int(len(labels))}

    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir}/")


def predict_fasta(cfg: dict, fasta_path: str, device: torch.device) -> None:
    from Bio import SeqIO
    from evovir.embeddings import EmbeddingExtractor

    task = cfg.get("task", "binary")
    is_binary = (task == "binary")

    records = list(SeqIO.parse(fasta_path, "fasta"))
    sequences = [str(r.seq).upper().replace("U", "T") for r in records]
    accessions = [r.id for r in records]

    extractor = EmbeddingExtractor(
        model_name=cfg["model_name"],
        layer_name=cfg["layer_name"],
        max_seq_len=cfg["max_seq_len"],
        window_stride=cfg["window_stride"],
        device="cuda:0",
    )
    embeddings_np = extractor.extract_batch(sequences)
    embeddings = torch.from_numpy(embeddings_np).to(device)

    model = load_model(cfg, device)
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
    print(f"\nPredictions saved -> {out_csv}")


# -- Plotting --

def _plot_roc(labels, probs, auroc, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_pr(labels, probs, auprc, path):
    prec, rec, _ = precision_recall_curve(labels, probs)
    plt.figure()
    plt.plot(rec, prec, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_confusion(labels, preds, class_names, path):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fasta", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.fasta:
        predict_fasta(cfg, args.fasta, device)
    else:
        evaluate_hdf5(cfg, device)
