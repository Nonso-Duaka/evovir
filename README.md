# EvoVir

Viral host-range classifier built on [Evo 2](https://github.com/ArcInstitute/evo2) DNA language model embeddings. Supports binary and multiclass classification.

## How it works

```
Viral genome (DNA/RNA)
        |
        v
  Evo 2 backbone (frozen)          <- evo2_7b_base
        |  intermediate layer embeddings  [seq_len x 4096]
        v
  Mean-pool over positions + windows  [4096]
        |
        v
  Classification head (MLP or Linear)
        |
        v
  Binary: P(vertebrate-infecting)
  Multiclass: P(class_0), P(class_1), ...
```

The backbone is always frozen. Only the lightweight head is trained.

## Installation

```bash
# 1. Create conda env
conda create -n evo2 && conda activate evo2

# 2. Install Evo 2 dependencies
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2

# 3. Install EvoVir
pip install -e .
```

## Quick start

### 1. Prepare data (manual)

Create `data/metadata.csv` with columns: `accession`, `label`, `fasta_file`.
Put FASTA files in `data/fasta/`.

Binary labels: `0` / `1`. Multiclass labels: `0`, `1`, `2`, ...

ViraLM accession lists are included in `data/accessions/` for reference.

### 2. Train

```bash
evovir-train --config configs/default.yaml
```

This automatically extracts Evo 2 embeddings (if not cached), trains the head, and evaluates on a held-out test set. Outputs: `outputs/best_model.pt`, `outputs/training_history.json`.

### 3. Predict on new sequences

```bash
evovir-predict --fasta my_seqs.fa --config configs/default.yaml
```

Extracts embeddings, runs the trained model, prints predictions, and saves `outputs/predictions.csv`.

### Advanced: run steps individually

```bash
python scripts/extract_embeddings.py --config configs/default.yaml   # extract only
python scripts/train.py --config configs/default.yaml                # train only (needs embeddings)
python scripts/evaluate.py --config configs/default.yaml             # evaluate on dataset
```

## Configuration

All settings live in `configs/default.yaml`.

| Key | Default | Description |
|-----|---------|-------------|
| `task` | `binary` | `binary` or `multiclass` |
| `num_classes` | `2` | Number of classes (ignored for binary) |
| `model_name` | `evo2_7b_base` | Evo 2 checkpoint (8K context) |
| `layer_name` | `blocks.28.mlp.l3` | Layer to hook for embeddings |
| `max_seq_len` | `8000` | Context window; longer sequences are windowed |
| `precision` | `bf16` | `fp32`, `fp16`, or `bf16` |
| `extraction_batch_size` | `8` | Windows per forward pass |
| `head_type` | `mlp` | `linear` or `mlp` |

## Project structure

```
evovir/
├── configs/default.yaml        # all hyperparameters and settings
├── data/
│   └── accessions/             # ViraLM accession lists
├── evovir/
│   ├── dataset.py              # ViralDataset + EmbeddingDataset
│   ├── embeddings.py           # Evo 2 embedding extraction
│   ├── model.py                # LinearHead / MLPHead / ViralClassifier
│   └── trainer.py              # Training loop with AMP, early stopping, metrics
├── scripts/
│   ├── extract_embeddings.py   # Run Evo 2, save embeddings to HDF5
│   ├── train.py                # Extract + train + evaluate (evovir-train)
│   ├── predict.py              # Inference on new FASTA (evovir-predict)
│   └── evaluate.py             # Detailed evaluation on dataset
└── pyproject.toml
```
