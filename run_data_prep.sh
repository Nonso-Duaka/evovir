#!/bin/bash
#SBATCH --job-name=evovir_data_prep
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500M
#SBATCH --output=/net/galaxy/home/koes/nduaka1/evovir/logs/data_prep_%j.log

set -eo pipefail

mkdir -p /net/galaxy/home/koes/nduaka1/evovir/logs

echo "=============================================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=============================================================="

# --- Environment setup --- #
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate evo2

cd /net/galaxy/home/koes/nduaka1/evovir

EMAIL="nonsoduaka821@gmail.com"

# Step 1: Split virus accessions by host (vertebrate vs non-vertebrate)
echo ""
echo "=============================================================="
echo "Step 1: Splitting virus accessions by host taxonomy..."
echo "=============================================================="
python -u scripts/split_virus_hosts.py --email "$EMAIL"

# Step 2: Download FASTA files
echo ""
echo "=============================================================="
echo "Step 2: Downloading FASTA files from NCBI..."
echo "=============================================================="
python -u scripts/download_fasta.py --email "$EMAIL" --all

# Step 3: Build metadata CSVs for all 3 training modes
echo ""
echo "=============================================================="
echo "Step 3a: Building metadata_binary_vv.csv (vv vs non-vv only)..."
echo "=============================================================="
python -u scripts/build_metadata.py --mode binary_vv

echo ""
echo "=============================================================="
echo "Step 3b: Building metadata_binary_vv_all.csv (vv vs all others)..."
echo "=============================================================="
python -u scripts/build_metadata.py --mode binary_vv_all

echo ""
echo "=============================================================="
echo "Step 3c: Building metadata_multiclass.csv (vv, non-vv, other)..."
echo "=============================================================="
python -u scripts/build_metadata.py --mode multiclass

echo ""
echo "=============================================================="
echo "Data preparation completed at: $(date)"
echo "=============================================================="
