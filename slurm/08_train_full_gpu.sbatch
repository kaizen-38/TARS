#!/usr/bin/env bash
#SBATCH --job-name=tars_train_full
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/08_train_full_%j.out
#SBATCH --error=logs/08_train_full_%j.err

# Full fine-tuning on all pilot instances. 3 epochs per locked config.
# Paper-facing checkpoint — do not skip to here before smoke and mini succeed.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"

module purge
module load mamba/latest
source /packages/apps/mamba/latest/etc/profile.d/conda.sh
conda activate tars

export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

echo "[$(date)] Starting full fine-tuning..."
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

MODE=full bash scripts/train_phase1.sh

CKPT_DIR=$(ls -td runs/sft_qwen3_*/ 2>/dev/null | head -1)
echo "[$(date)] Training complete. Checkpoint: ${CKPT_DIR}"

# Write checkpoint path for downstream eval job
echo "${CKPT_DIR}" > runs/latest_checkpoint.txt
