#!/bin/bash
#SBATCH --job-name=setup_evo2_env
#SBATCH --partition=dept_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/net/galaxy/home/koes/nduaka1/evovir/logs/setup_evo2_env_%j.log

set -eo pipefail

mkdir -p /net/galaxy/home/koes/nduaka1/evovir/logs

echo "=============================================================="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=============================================================="

# --- Environment setup --- #
module load cuda
eval "$(/opt/anaconda3/bin/conda shell.bash hook)"

# Remove existing env if present
conda env remove -n evo2 -y 2>/dev/null || true

# Create fresh conda env
echo "Creating conda environment..."
conda create -n evo2 python=3.12 -y

conda activate evo2

# Step 1: Install CUDA compiler tools
echo "Installing cuda-nvcc and cuda-cudart-dev..."
conda install -c nvidia cuda-nvcc cuda-cudart-dev -y

# Step 2: Install Transformer Engine
echo "Installing transformer-engine-torch..."
conda install -c conda-forge transformer-engine-torch=2.3.0 -y

# Step 3: Install Flash Attention
echo "Installing flash-attn..."
pip install psutil numpy
export CUDA_HOME=$CONDA_PREFIX
export TMPDIR=$HOME/tmp_flash_attn && mkdir -p $TMPDIR
pip install flash-attn==2.8.0.post2 --no-build-isolation
rm -rf $TMPDIR

# Step 4: Install Evo 2
echo "Installing evo2..."
pip install evo2

# Step 5: Install EvoVir and its dependencies
echo "Installing evovir..."
cd /net/galaxy/home/koes/nduaka1/evovir
pip install -e ".[dev]"

# Verify installation
echo ""
echo "=============================================================="
echo "Verifying installation..."
echo "=============================================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import evo2; print('evo2 imported successfully')"
python -c "import evovir; print('evovir imported successfully')"

echo ""
echo "=============================================================="
echo "Environment setup completed at: $(date)"
echo "=============================================================="
