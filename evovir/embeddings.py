"""
GPU-optimised Evo 2 embedding extraction for viral sequences.

Key design choices
------------------
- All windows from ALL sequences are batched together into GPU-sized chunks,
  maximising CUDA utilisation instead of running one window at a time.
- Mixed-precision inference (bf16/fp16) via torch.autocast.
- torch.cuda.empty_cache() after each batch to avoid fragmentation on long runs.
- OOM guard: automatically halves batch_size and retries on CUDA OOM.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class EmbeddingExtractor:
    """
    Args:
        model_name: Evo 2 checkpoint (e.g. ``"evo2_7b_base"``).
        layer_name: Layer to hook (e.g. ``"blocks.28.mlp.l3"``).
        max_seq_len: Context window size.
        window_stride: Stride for overlapping windows on long sequences.
        extraction_batch_size: Number of windows per GPU forward pass.
            Increase for higher throughput; decrease on OOM.
        precision: ``"fp32"``, ``"fp16"``, or ``"bf16"``.
        device: Primary CUDA device (e.g. ``"cuda:0"``).
        local_path: Optional local path to weights (skips HuggingFace).
    """

    def __init__(
        self,
        model_name: str = "evo2_7b_base",
        layer_name: str = "blocks.28.mlp.l3",
        max_seq_len: int = 8_000,
        window_stride: int = 4_000,
        extraction_batch_size: int = 8,
        precision: str = "bf16",
        device: str = "cuda:0",
        local_path: Optional[str] = None,
    ) -> None:
        from evo2 import Evo2

        self.layer_name = layer_name
        self.max_seq_len = max_seq_len
        self.window_stride = window_stride
        self.extraction_batch_size = extraction_batch_size
        self.device = device
        self.amp_dtype = _DTYPE_MAP.get(precision, torch.bfloat16)
        self.use_amp = precision in ("fp16", "bf16")

        print(f"[EmbeddingExtractor] Loading {model_name} on {device} (precision={precision})…")
        self.evo2 = Evo2(model_name, local_path=local_path)
        self.tokenizer = self.evo2.tokenizer
        _log_gpu_memory("after model load")



    def extract_batch(
        self,
        sequences: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract mean-pooled embeddings for a list of sequences.

        All windows from all sequences are collected first, then processed
        in GPU batches of ``extraction_batch_size``, and finally aggregated
        back to one vector per sequence.

        Returns:
            float32 array of shape ``(N, embedding_dim)``.
        """
        # Step 1: build flat window list, track which slice belongs to each seq
        all_windows: List[str] = []
        seq_slices: List[Tuple[int, int]] = []
        for seq in sequences:
            windows = self._make_windows(seq)
            start = len(all_windows)
            all_windows.extend(windows)
            seq_slices.append((start, len(all_windows)))

        # Step 2: run all windows through the GPU in batches
        all_window_embs = self._forward_batched(all_windows, show_progress)

        # Step 3: mean-pool windows back per sequence
        seq_embeddings = [
            all_window_embs[s:e].mean(axis=0) for s, e in seq_slices
        ]
        return np.stack(seq_embeddings)

    def save_to_hdf5(
        self,
        sequences: List[str],
        labels: List[int],
        accessions: List[str],
        out_path: str | Path,
        show_progress: bool = True,
    ) -> None:
        """Extract embeddings and write to HDF5 (run once, train many times)."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings = self.extract_batch(sequences, show_progress=show_progress)

        with h5py.File(out_path, "w") as f:
            f.create_dataset("embeddings", data=embeddings.astype(np.float32), compression="gzip")
            f.create_dataset("labels", data=np.array(labels, dtype=np.int8))
            f.create_dataset("accessions", data=np.array([a.encode() for a in accessions]))

        print(f"[EmbeddingExtractor] Saved {len(embeddings)} embeddings → {out_path}")



    def _make_windows(self, seq: str) -> List[str]:
        if len(seq) <= self.max_seq_len:
            return [seq]
        windows = []
        for start in range(0, len(seq) - self.max_seq_len + 1, self.window_stride):
            windows.append(seq[start : start + self.max_seq_len])
        if windows[-1] != seq[-self.max_seq_len :]:
            windows.append(seq[-self.max_seq_len :])
        return windows

    def _forward_batched(self, windows: List[str], show_progress: bool) -> np.ndarray:
        """
        Run all windows through the model in GPU batches.
        Returns float32 array of shape ``(num_windows, hidden_dim)``.
        """
        from evo2.scoring import prepare_batch

        all_embs: List[np.ndarray] = []
        batch_size = self.extraction_batch_size

        pbar = tqdm(
            range(0, len(windows), batch_size),
            desc="Extracting embeddings",
            disable=not show_progress,
        )

        for i in pbar:
            batch = windows[i : i + batch_size]
            embs = self._forward_one_batch(batch, prepare_batch, batch_size)
            all_embs.extend(embs)
            if show_progress:
                pbar.set_postfix(_gpu_mem_str())
            torch.cuda.empty_cache()

        return np.stack(all_embs)

    def _forward_one_batch(self, batch, prepare_batch_fn, original_batch_size):
        """Forward pass with OOM guard — halves batch size and retries."""
        from evo2.scoring import prepare_batch

        while True:
            try:
                input_ids, seq_lengths = prepare_batch(
                    batch, self.tokenizer, device=self.device
                )
                with torch.inference_mode():
                    with torch.autocast(
                        device_type="cuda",
                        dtype=self.amp_dtype,
                        enabled=self.use_amp,
                    ):
                        _, emb_dict = self.evo2.forward(
                            input_ids,
                            return_embeddings=True,
                            layer_names=[self.layer_name],
                        )

                emb_tensor = emb_dict[self.layer_name]  # (B, seq_len, hidden_dim)
                result = []
                for b, seq_len in enumerate(seq_lengths):
                    vec = emb_tensor[b, :seq_len].float().mean(dim=0).cpu().numpy()
                    result.append(vec)
                return result

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise RuntimeError(
                        "CUDA OOM on a single window. "
                        "Reduce max_seq_len or use a GPU with more VRAM."
                    )
                mid = len(batch) // 2
                print(f"\n[OOM] Splitting batch {len(batch)} → {mid} + {len(batch)-mid}")
                left = self._forward_one_batch(batch[:mid], prepare_batch_fn, mid)
                right = self._forward_one_batch(batch[mid:], prepare_batch_fn, len(batch) - mid)
                return left + right




def _gpu_mem_str() -> dict:
    if not torch.cuda.is_available():
        return {}
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return {"GPU": f"{alloc:.1f}/{reserved:.1f}GB"}


def _log_gpu_memory(tag: str = "") -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[GPU mem{' ' + tag if tag else ''}] allocated={alloc:.2f}GB  reserved={reserved:.2f}GB")
