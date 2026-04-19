#!/usr/bin/env bash
#SBATCH --job-name=tars_train_mini
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/06_train_mini_%j.out
#SBATCH --error=logs/06_train_mini_%j.err

# Mini full fine-tuning run for a usable pilot checkpoint.
# Fewer steps than full (controlled by LLaMAFactory config).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module purge
module load mamba/latest
source activate tars

export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

echo "[$(date)] Starting mini full-FT training..."
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# lora_debug gives 20-step LoRA — use full for real mini baseline
MODE=full bash scripts/train_phase1.sh

CKPT_DIR=$(ls -td runs/sft_qwen3_*/ 2>/dev/null | head -1)
echo "[$(date)] Mini training complete. Checkpoint: ${CKPT_DIR}"
mkdir -p runs
echo "${CKPT_DIR}" > runs/latest_checkpoint.txt
