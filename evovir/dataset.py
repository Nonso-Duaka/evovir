"""
Dataset classes for EvoVir.

ViralDataset    – loads raw sequences from FASTA + metadata CSV.
EmbeddingDataset – loads pre-extracted embeddings (HDF5) for classifier training.

Supports binary and multiclass. Labels are provided manually in metadata.csv.
  binary:     label column is 0 or 1
  multiclass: label column is 0, 1, 2, ...
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset


AMBIGUOUS_RE = re.compile(r"[^ACGTacgt]")


def _ambiguous_fraction(seq: str) -> float:
    return len(AMBIGUOUS_RE.findall(seq)) / max(len(seq), 1)


class ViralDataset(Dataset):
    """
    Loads sequences from FASTA files referenced in metadata.csv.

    Expected CSV columns: accession, label, fasta_file

    Args:
        metadata_path: Path to metadata CSV.
        fasta_dir: Root directory containing FASTA files.
        task: "binary" or "multiclass".
        min_len / max_len / ambiguous_threshold: QC filters.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        fasta_dir: Optional[str | Path] = None,
        task: str = "binary",
        min_len: int = 500,
        max_len: int = 300_000,
        ambiguous_threshold: float = 0.05,
    ) -> None:
        self.meta = pd.read_csv(metadata_path)
        self.fasta_dir = Path(fasta_dir) if fasta_dir else None
        self.task = task
        self.min_len = min_len
        self.max_len = max_len
        self.ambiguous_threshold = ambiguous_threshold

        self.sequences: List[str] = []
        self.labels: List[int] = []
        self.accessions: List[str] = []

        self._load()

    def _load(self) -> None:
        skipped = 0
        for _, row in self.meta.iterrows():
            seq = self._get_sequence(row)
            if seq is None:
                skipped += 1
                continue
            seq = seq.upper().replace("U", "T")
            if not self._passes_filters(seq):
                skipped += 1
                continue
            self.sequences.append(seq)
            self.labels.append(int(row["label"]))
            self.accessions.append(str(row["accession"]))

        if skipped:
            print(f"[ViralDataset] Skipped {skipped} sequences (failed QC filters).")
        counts = Counter(self.labels)
        parts = ", ".join(f"label {k}: {v}" for k, v in sorted(counts.items()))
        print(f"[ViralDataset] {len(self.labels)} sequences ({parts})")

    def _get_sequence(self, row: pd.Series) -> Optional[str]:
        if "sequence" in row and pd.notna(row["sequence"]):
            return str(row["sequence"])

        fasta_value = Path(str(row["fasta_file"]))
        candidate_paths = [fasta_value]
        if self.fasta_dir is not None and not fasta_value.is_absolute():
            candidate_paths.append(self.fasta_dir / fasta_value)

        fasta_path = next((p for p in candidate_paths if p.exists()), None)
        if fasta_path is None:
            return None

        records = list(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            return None
        return "".join(str(r.seq) for r in records)

    def _passes_filters(self, seq: str) -> bool:
        if len(seq) < self.min_len:
            return False
        if len(seq) > self.max_len:
            return False
        if _ambiguous_fraction(seq) > self.ambiguous_threshold:
            return False
        return True

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.sequences[idx], self.labels[idx]

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights. Shape (num_classes,)."""
        counts = Counter(self.labels)
        n = len(self.labels)
        weights = [n / (len(counts) * counts[c]) for c in sorted(counts.keys())]
        return torch.tensor(weights, dtype=torch.float32)


class EmbeddingDataset(Dataset):
    """
    Loads pre-extracted embeddings from an HDF5 file.

    HDF5 layout::
        /embeddings   – float32, shape (N, D)
        /labels       – int64,   shape (N,)
        /accessions   – bytes,   shape (N,)
    """

    def __init__(self, hdf5_path: str | Path) -> None:
        self.path = Path(hdf5_path)
        with h5py.File(self.path, "r") as f:
            self.embeddings = torch.from_numpy(f["embeddings"][:].astype(np.float32))
            self.labels = torch.from_numpy(f["labels"][:].astype(np.int64))
            self.accessions = [a.decode() for a in f["accessions"][:]]

        counts = Counter(self.labels.numpy().tolist())
        parts = ", ".join(f"label {k}: {v}" for k, v in sorted(counts.items()))
        print(f"[EmbeddingDataset] {len(self.labels)} samples ({parts})")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]

    @property
    def num_classes(self) -> int:
        return len(torch.unique(self.labels))

    @property
    def class_weights(self) -> torch.Tensor:
        counts = Counter(self.labels.numpy().tolist())
        n = len(self.labels)
        weights = [n / (len(counts) * counts[c]) for c in sorted(counts.keys())]
        return torch.tensor(weights, dtype=torch.float32)
