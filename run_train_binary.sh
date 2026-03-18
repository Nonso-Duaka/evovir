#!/bin/bash
#SBATCH --job-name=evovir_train_binary
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --output=/net/galaxy/home/koes/nduaka1/evovir/logs/train_binary_%j.log

set -eo pipefail

mkdir -p logs

echo "=============================================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=============================================================="

# --- Environment setup --- #
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate evo2

cd ~/evovir

# Step 1: Build binary metadata (vertebrate=1, non_vertebrate+unknown=0)
echo ""
echo "=============================================================="
echo "Step 1: Building binary metadata.csv..."
echo "=============================================================="
python -u scripts/build_metadata.py --binary --output data/metadata.csv

# Step 2: Extract Evo2 embeddings + train classifier (auto-extracts if needed)
echo ""
echo "=============================================================="
echo "Step 2: Training binary classifier..."
echo "=============================================================="
python -u scripts/train.py --config configs/default.yaml

echo ""
echo "=============================================================="
echo "Training completed at: $(date)"
echo "=============================================================="
