#!/bin/bash
#SBATCH --job-name=DL_Unknown_Host
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/net/galaxy/home/koes/nduaka1/evovir/logs/DL_Unknown_Host_%j.log

set -euo pipefail

mkdir -p logs

echo "=============================================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "=============================================================="

# --- Environment setup --- #
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"
conda activate omnivir

cd ~/evovir

echo "Downloading 3473 virus_unknown_host FASTA sequences..."
echo "=============================================================="

python scripts/download_fasta.py \
    --email nonsoduaka821@gmail.com \
    --accessions data/accessions/virus_unknown_host.txt \
    --outdir data/fasta/virus_unknown_host \
    --batch-size 100

echo ""
echo "=============================================================="
echo "Download completed at: $(date)"
echo "Downloaded files: $(ls data/fasta/virus_unknown_host/*.fa 2>/dev/null | wc -l) / 3473"
echo "=============================================================="
