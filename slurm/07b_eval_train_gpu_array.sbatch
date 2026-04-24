#!/usr/bin/env bash
#SBATCH --job-name=tars_eval_train
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-11          # one task per domain (12 total)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/07b_eval_train_%A_%a.out
#SBATCH --error=logs/07b_eval_train_%A_%a.err

# CHECKPOINT must be set in the environment by submit_phase1.sh.
# Each task evaluates one domain across all 3 representations.
# 25 instances × 3 reps × ~30s/eval = ~37 min per task (well within 3h).

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
MANIFEST="${REPO_ROOT}/manifests/eval_train.tsv"

module purge
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate thicket311

export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

CHECKPOINT="${CHECKPOINT:-$(cat runs/latest_checkpoint.txt 2>/dev/null | tr -d '[:space:]')}"
CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3-1.7B}"

LINE=$(awk "NR==$(( SLURM_ARRAY_TASK_ID + 2 ))" "$MANIFEST")
DOMAIN=$(echo "$LINE" | cut -f1)
SPLIT=$(echo  "$LINE" | cut -f2)

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} domain=${DOMAIN} split=${SPLIT} checkpoint=${CHECKPOINT}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python src/inference/run_greedy_eval.py \
    --checkpoint-path  "${CHECKPOINT}" \
    --split            "${SPLIT}" \
    --domain           "${DOMAIN}" \
    --output-dir       "runs/eval_pilot_train" \
    --run-id           "train_${DOMAIN}"

echo "[$(date)] done: ${DOMAIN}"
