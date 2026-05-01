#!/usr/bin/env bash
#SBATCH --job-name=tars_grpo
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=gpu
#SBATCH --qos=class_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/08_grpo_%j.out
#SBATCH --error=logs/08_grpo_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"

module purge
module load mamba/latest
module load gcc/15.2.0  # Needed for VAL during reward computation
eval "$(conda shell.bash hook)"
conda activate tars

export PYTHONPATH="${REPO_ROOT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"

cd "${REPO_ROOT}"
mkdir -p logs

echo "[$(date)] Starting GRPO training with VAL reward..."
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader | head -1

python src/training/grpo_trainer.py

echo "[$(date)] GRPO training complete. Checkpoint: runs/qwen3_grpo/"
