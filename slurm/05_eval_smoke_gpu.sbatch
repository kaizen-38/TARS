#!/usr/bin/env bash
#SBATCH --job-name=tars_eval_smoke
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/05_eval_smoke_%j.out
#SBATCH --error=logs/05_eval_smoke_%j.err

# CHECKPOINT env var must be set by submit_phase1.sh after job 04 completes.
# Example: CHECKPOINT=runs/sft_qwen3_phase1_lora_debug

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"

module purge
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate thicket311

export HF_HOME="${SCRATCH}/hf_cache"
export TRANSFORMERS_CACHE="${SCRATCH}/hf_cache"
export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3-1.7B}"
echo "[$(date)] Smoke eval: checkpoint=${CHECKPOINT}"

# Evaluate on a small sample: one domain from each split
PYTHONPATH="${REPO_ROOT}/src" python src/inference/run_greedy_eval.py \
    --checkpoint-path "${CHECKPOINT}" \
    --split train \
    --output-dir "runs/eval_smoke"

echo "[$(date)] Smoke eval complete."
PYTHONPATH="${REPO_ROOT}/src" python src/eval/aggregate_results.py runs/eval_smoke/run_log.jsonl || true
