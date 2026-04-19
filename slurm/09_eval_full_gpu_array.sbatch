#!/usr/bin/env bash
#SBATCH --job-name=tars_eval_full
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-11          # 12 domains; evaluate train split for in-domain, heldout for ood
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/09_eval_full_%A_%a.out
#SBATCH --error=logs/09_eval_full_%A_%a.err

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

# Read checkpoint written by job 08
CHECKPOINT="${CHECKPOINT:-$(cat runs/latest_checkpoint.txt 2>/dev/null || echo Qwen/Qwen3-1.7B)}"

# Evaluate all domains: first 8 are train (array 0-7), last 4 are heldout (array 8-11)
ALL_DOMAINS=(blocksworld gripper ferry delivery childsnack floortile rovers spanner miconic sokoban transport satellite)
DOMAIN="${ALL_DOMAINS[$SLURM_ARRAY_TASK_ID]}"
SPLIT=$( (( SLURM_ARRAY_TASK_ID < 8 )) && echo "train" || echo "heldout" )

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} domain=${DOMAIN} split=${SPLIT} checkpoint=${CHECKPOINT}"

python src/inference/run_greedy_eval.py \
    --checkpoint-path "${CHECKPOINT}" \
    --split           "${SPLIT}" \
    --domain          "${DOMAIN}" \
    --output-dir      "runs/eval_full" \
    --run-id          "full_${DOMAIN}"

echo "[$(date)] done: ${DOMAIN}"
