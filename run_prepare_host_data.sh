#!/bin/bash
#SBATCH --job-name=OmniVir_PrepHost
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/net/galaxy/home/koes/nduaka1/evovir/logs/PrepHost_%j.log

set -euo pipefail

mkdir -p logs
mkdir -p data/processed_host

echo "=============================================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=============================================================="

# --- Environment setup --- #
module load cuda
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate omnivir

cd ~/evovir

echo "Running OmniVir host-class data preparation"
echo "  vertebrate:     ~/evovir/data/fasta/virus_vertebrate (2984)"
echo "  non_vertebrate: ~/evovir/data/fasta/virus_non_vertebrate (7399)"
echo "  unknown_host:   ~/evovir/data/fasta/virus_unknown_host (3473)"
echo "=============================================================="

python scripts/prepare_host_data.py \
    --vertebrate     data/fasta/virus_vertebrate \
    --non_vertebrate data/fasta/virus_non_vertebrate \
    --unknown_host   data/fasta/virus_unknown_host \
    --out            data/processed_host \
    --samples        500 \
    --seed           42

echo ""
echo "=============================================================="
echo "OmniVir host-class data preparation completed at: $(date)"
echo "=============================================================="
