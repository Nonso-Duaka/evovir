"""
Dataset classes for EvoVir.

ViralDataset  – loads raw sequences from FASTA + labels CSV.
EmbeddingDataset – loads pre-extracted embeddings (HDF5) for classifier training.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from torch.utils.data import Dataset


# Characters that are unambiguous DNA bases
_VALID_BASES = re.compile(r"^[ACGTacgt]+$")

AMBIGUOUS_RE = re.compile(r"[^ACGTacgt]")


def _ambiguous_fraction(seq: str) -> float:
    return len(AMBIGUOUS_RE.findall(seq)) / max(len(seq), 1)


class ViralDataset(Dataset):
    """
    Loads viral genome sequences from disk.

    Expected layout::

        data/
          metadata.csv        # columns: accession, label (0/1), fasta_file (optional)
          fasta/
            vertebrate/       # one .fa per sequence, or one multi-FASTA
            non_vertebrate/

    If metadata.csv has a ``sequence`` column the sequences are read inline;
    otherwise they are read from fasta files.

    Args:
        metadata_path: Path to metadata CSV.
        fasta_dir: Root directory containing fasta sub-dirs (if sequences not inline).
        min_len: Discard sequences shorter than this.
        max_len: Discard sequences longer than this.
        ambiguous_threshold: Discard sequences whose fraction of non-ACGT bases
            exceeds this value.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        fasta_dir: Optional[str | Path] = None,
        min_len: int = 500,
        max_len: int = 300_000,
        ambiguous_threshold: float = 0.05,
    ) -> None:
        self.meta = pd.read_csv(metadata_path)
        self.fasta_dir = Path(fasta_dir) if fasta_dir else None
        self.min_len = min_len
        self.max_len = max_len
        self.ambiguous_threshold = ambiguous_threshold

        self.sequences: List[str] = []
        self.labels: List[int] = []
        self.accessions: List[str] = []

        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        skipped = 0
        for _, row in self.meta.iterrows():
            seq = self._get_sequence(row)
            if seq is None:
                skipped += 1
                continue
            seq = seq.upper().replace("U", "T")  # RNA → DNA
            if not self._passes_filters(seq):
                skipped += 1
                continue
            self.sequences.append(seq)
            self.labels.append(int(row["label"]))
            self.accessions.append(str(row["accession"]))

        if skipped:
            print(f"[ViralDataset] Skipped {skipped} sequences (failed QC filters).")
        print(
            f"[ViralDataset] Loaded {len(self.sequences)} sequences "
            f"({sum(self.labels)} vertebrate, "
            f"{len(self.labels) - sum(self.labels)} non-vertebrate)."
        )

    def _get_sequence(self, row: pd.Series) -> Optional[str]:
        if "sequence" in row and pd.notna(row["sequence"]):
            return str(row["sequence"])
        if self.fasta_dir is None:
            return None
        fasta_path = self.fasta_dir / str(row["fasta_file"])
        if not fasta_path.exists():
            return None
        records = list(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            return None
        # If multi-record (segmented genome) concatenate
        return "".join(str(r.seq) for r in records)

    def _passes_filters(self, seq: str) -> bool:
        if len(seq) < self.min_len:
            return False
        if len(seq) > self.max_len:
            return False
        if _ambiguous_fraction(seq) > self.ambiguous_threshold:
            return False
        return True

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.sequences[idx], self.labels[idx]

    # ------------------------------------------------------------------
    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for BCEWithLogitsLoss pos_weight."""
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)


class EmbeddingDataset(Dataset):
    """
    Loads pre-extracted embeddings from an HDF5 file.

    HDF5 layout (written by ``EmbeddingExtractor.save_to_hdf5``)::

        /embeddings   – float32, shape (N, D)
        /labels       – int8,    shape (N,)
        /accessions   – bytes,   shape (N,)
    """

    def __init__(self, hdf5_path: str | Path) -> None:
        self.path = Path(hdf5_path)
        with h5py.File(self.path, "r") as f:
            self.embeddings = torch.from_numpy(f["embeddings"][:].astype(np.float32))
            self.labels = torch.from_numpy(f["labels"][:].astype(np.float32))
            self.accessions = [a.decode() for a in f["accessions"][:]]

        n_pos = int(self.labels.sum().item())
        n_neg = len(self.labels) - n_pos
        print(
            f"[EmbeddingDataset] {len(self.labels)} samples "
            f"({n_pos} vertebrate, {n_neg} non-vertebrate) from {self.path.name}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]

    @property
    def class_weights(self) -> torch.Tensor:
        n_pos = int(self.labels.sum().item())
        n_neg = len(self.labels) - n_pos
        return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
