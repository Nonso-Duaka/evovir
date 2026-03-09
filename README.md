# EvoVir

Binary classifier that predicts whether a virus infects **vertebrates** or **non-vertebrates**, built on top of [Evo 2](https://github.com/ArcInstitute/evo2) DNA language model embeddings.

## How it works

```
Viral genome (DNA/RNA)
        │
        ▼
  Evo 2 backbone (frozen)          ← evo2_7b_base recommended
        │  intermediate layer embeddings  [seq_len × 4096]
        ▼
  Mean-pool over positions + windows  [4096]
        │
        ▼
  Classification head (MLP or Linear)
        │
        ▼
  P(vertebrate-infecting)
```

The backbone is **always frozen** — only the lightweight head is trained.
This makes training fast and requires no GPU after embedding extraction.

---

## Installation

```bash
# 1. Create conda env
conda create -n evo2 && conda activate evo2

# 2. Install Evo 2 dependencies (requires NVIDIA GPU + CUDA drivers)
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge transformer-engine-torch=2.3.0
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2

# 3. Install EvoVir
pip install -e .
```

---

## Workflow

### Step 1 — Download data

Set your email in `configs/default.yaml` (`ncbi_email` field), then:

```bash
# Dry run: just print how many sequences would be downloaded
python scripts/download_data.py --config configs/default.yaml --dry-run

# Full download (may take 30–60 min depending on max_per_class)
python scripts/download_data.py --config configs/default.yaml
```

Produces:
```
data/
  metadata.csv
  fasta/
    vertebrate/
    non_vertebrate/
```

### Step 2 — Extract embeddings  *(requires GPU)*

```bash
python scripts/extract_embeddings.py --config configs/default.yaml
```

Produces `outputs/embeddings/embeddings.h5`.
Copy this file to wherever you plan to train — no GPU needed after this step.

### Step 3 — Train the classifier  *(CPU or GPU)*

```bash
python scripts/train.py --config configs/default.yaml
```

Produces `outputs/best_model.pt` and `outputs/training_history.json`.

### Step 4 — Evaluate

```bash
# Evaluate on the full dataset (uses saved split indices)
python scripts/evaluate.py --config configs/default.yaml

# Predict on your own FASTA  (requires GPU for embedding extraction)
python scripts/evaluate.py --config configs/default.yaml --fasta my_seqs.fa
```

Produces ROC/PR curves and a predictions CSV in `outputs/`.

---

## Configuration

All settings live in `configs/default.yaml`.  Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `model_name` | `evo2_7b_base` | Evo 2 checkpoint (8K context, no FP8/Hopper required) |
| `layer_name` | `blocks.28.mlp.l3` | Layer to hook for embeddings |
| `max_seq_len` | `8000` | Context window; sequences are windowed if longer |
| `max_per_class` | `5000` | Max sequences to download per class |
| `head_type` | `mlp` | `"linear"` or `"mlp"` |
| `ncbi_email` | *(set this)* | Required by NCBI Entrez |

---

## Project structure

```
evovir/
├── configs/default.yaml        # all hyperparameters and data settings
├── evovir/
│   ├── dataset.py              # ViralDataset (raw FASTA) + EmbeddingDataset (HDF5)
│   ├── embeddings.py           # Evo2 embedding extraction with windowing
│   ├── model.py                # LinearHead / MLPHead / ViralClassifier
│   └── trainer.py              # Training loop, early stopping, metrics
├── scripts/
│   ├── download_data.py        # NCBI Entrez download
│   ├── extract_embeddings.py   # GPU: run Evo 2, save embeddings to HDF5
│   ├── train.py                # Train classification head
│   └── evaluate.py             # Evaluate + predict on new FASTA
└── data/                       # Downloaded sequences (gitignored)
```

---

## Hardware requirements

| Step | Hardware |
|------|----------|
| Download data | CPU only |
| Extract embeddings | NVIDIA GPU (any modern GPU for `evo2_7b_base`) |
| Train head | CPU or GPU |
| Inference | GPU recommended; CPU feasible for small batches |

The recommended starting model (`evo2_7b_base`) does **not** require an H100 or FP8 support.
