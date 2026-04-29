#!/usr/bin/env bash
#SBATCH --job-name=tars_train_direct
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=gpu
#SBATCH --qos=class_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/06b_train_direct_%j.out
#SBATCH --error=logs/06b_train_direct_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"

module purge
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate tars

export PYTHONPATH="${REPO_ROOT}/src"
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"

cd "${REPO_ROOT}"
mkdir -p logs

echo "[$(date)] Starting direct training..."
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader | head -1

python src/training/train_direct.py

echo "[$(date)] Training complete. Checkpoint: runs/qwen3_mini_direct/"
