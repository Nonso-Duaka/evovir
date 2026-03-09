"""
Evo 2 embedding extraction for viral sequences.

Handles:
- Windowing of sequences that exceed the model's context length
- Batch processing
- Saving / loading embeddings to/from HDF5
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm


class EmbeddingExtractor:
    """
    Wraps an Evo2 model and extracts mean-pooled intermediate-layer embeddings.

    Args:
        model_name: Evo 2 checkpoint name (e.g. ``"evo2_7b_base"``).
        layer_name: Name of the layer to hook (e.g. ``"blocks.28.mlp.l3"``).
        max_seq_len: Maximum number of tokens per forward pass (8000 for 7b_base).
        window_stride: Stride for overlapping windows on long sequences.
        device: CUDA device string (e.g. ``"cuda:0"``).
        local_path: Optional local path to model weights (skips HuggingFace download).
    """

    def __init__(
        self,
        model_name: str = "evo2_7b_base",
        layer_name: str = "blocks.28.mlp.l3",
        max_seq_len: int = 8_000,
        window_stride: int = 4_000,
        device: str = "cuda:0",
        local_path: Optional[str] = None,
    ) -> None:
        from evo2 import Evo2  # imported here so CPU-only environments can still import evovir

        self.layer_name = layer_name
        self.max_seq_len = max_seq_len
        self.window_stride = window_stride
        self.device = device

        print(f"[EmbeddingExtractor] Loading {model_name}…")
        self.model = Evo2(model_name, local_path=local_path)
        self.tokenizer = self.model.tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        sequences: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for a list of sequences.

        Returns:
            Float32 array of shape ``(N, embedding_dim)``.
        """
        embeddings = []
        iterator = tqdm(sequences, desc="Extracting embeddings") if show_progress else sequences
        for seq in iterator:
            emb = self._extract_one(seq)
            embeddings.append(emb)
        return np.stack(embeddings)

    def save_to_hdf5(
        self,
        sequences: List[str],
        labels: List[int],
        accessions: List[str],
        out_path: str | Path,
        show_progress: bool = True,
    ) -> None:
        """
        Extract embeddings and write them to an HDF5 file.

        This is the recommended workflow: run once, then train/evaluate
        directly from the HDF5 file without reloading Evo 2.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings = self.extract_batch(sequences, show_progress=show_progress)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("embeddings", data=embeddings.astype(np.float32), compression="gzip")
            f.create_dataset("labels", data=np.array(labels, dtype=np.int8))
            f.create_dataset(
                "accessions",
                data=np.array([a.encode() for a in accessions]),
            )

        print(f"[EmbeddingExtractor] Saved {len(embeddings)} embeddings → {out_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_one(self, seq: str) -> np.ndarray:
        """
        Extract a single sequence embedding.

        Long sequences are split into overlapping windows; the per-window
        mean-pooled embeddings are averaged to produce a single vector.
        """
        windows = self._make_windows(seq)
        window_embeddings = [self._forward_window(w) for w in windows]
        return np.mean(window_embeddings, axis=0)

    def _make_windows(self, seq: str) -> List[str]:
        if len(seq) <= self.max_seq_len:
            return [seq]
        windows = []
        for start in range(0, len(seq) - self.max_seq_len + 1, self.window_stride):
            windows.append(seq[start : start + self.max_seq_len])
        # Always include a window ending at the sequence tail
        if windows[-1] != seq[-self.max_seq_len :]:
            windows.append(seq[-self.max_seq_len :])
        return windows

    def _forward_window(self, window: str) -> np.ndarray:
        """Tokenise, forward pass, mean-pool one window → 1-D numpy array."""
        from evo2.scoring import prepare_batch

        input_ids, seq_lengths = prepare_batch(
            [window], self.tokenizer, device=self.device
        )
        with torch.no_grad():
            _, emb_dict = self.model.forward(
                input_ids,
                return_embeddings=True,
                layer_names=[self.layer_name],
            )

        emb = emb_dict[self.layer_name]          # (1, seq_len, hidden_dim)
        emb = emb[0, : seq_lengths[0]]           # trim padding
        return emb.mean(dim=0).float().cpu().numpy()
