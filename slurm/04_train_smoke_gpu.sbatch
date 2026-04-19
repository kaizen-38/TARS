#!/usr/bin/env bash
#SBATCH --job-name=tars_train_smoke
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/04_train_smoke_%j.out
#SBATCH --error=logs/04_train_smoke_%j.err

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module purge
module load mamba/latest
source activate tars

# Keep model downloads and HF cache on scratch, not home quota.
export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

echo "[$(date)] Starting smoke LoRA training..."
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

MODE=lora_debug bash scripts/train_phase1.sh

echo "[$(date)] Smoke training complete."
echo "Checkpoint:"
ls -lh runs/sft_qwen3_*/  2>/dev/null | tail -5 || echo "(no checkpoint found)"
