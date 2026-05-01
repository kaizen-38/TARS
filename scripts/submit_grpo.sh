#!/usr/bin/env bash
# submit_grpo.sh — Run GRPO training using VAL validation as reward.
# Requires: SFT checkpoint at runs/qwen3_mini_direct/
#
# Usage:
#   bash scripts/submit_grpo.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

if ! python3 -c "import typer" &>/dev/null; then
    module load mamba/latest
    eval "$(conda shell.bash hook)"
    conda activate tars
fi

WRAP_INIT="source ${REPO_ROOT}/scripts/sol_init.sh && cd ${REPO_ROOT} && export PYTHONPATH=${REPO_ROOT}/src"

LOG="${REPO_ROOT}/logs/submitted_jobs.txt"
echo "=== GRPO submission $(date) ===" | tee -a "$LOG"

submit() {
    local desc="$1"; shift
    local jid
    jid=$(sbatch "$@" | awk '{print $NF}')
    echo "${desc}: ${jid}" | tee -a "$LOG" >&2
    echo "$jid"
}

# Check if SFT checkpoint exists
if [ ! -d "${REPO_ROOT}/runs/qwen3_mini_direct" ]; then
    echo "ERROR: SFT checkpoint not found at runs/qwen3_mini_direct/"
    echo "Run: bash scripts/submit_full.sh first"
    exit 1
fi

echo "GRPO Training Pipeline"
echo "======================"
echo "Base model: runs/qwen3_mini_direct/"
echo "Reward: VAL validation (0=invalid, 1=valid, 10=goal)"
echo "Expected: 0% → 5-15% validity if successful"
echo ""

# Install TRL if not already installed
echo "Checking TRL installation..."
if ! python3 -c "import trl" &>/dev/null; then
    echo "Installing TRL..."
    pip install trl
fi

# Job 08: GRPO training
JID_GRPO=$(submit "08_train_grpo" slurm/08_train_grpo_gpu.sbatch)

# Update checkpoint pointer
JID_CHECKPOINT=$(submit "08b_update_checkpoint" \
    --dependency=afterok:${JID_GRPO} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=1G --time=00:01:00 \
    --output=logs/08b_checkpoint_%j.out \
    --wrap="echo 'runs/qwen3_grpo' > ${REPO_ROOT}/runs/latest_checkpoint.txt")

# Job 07: eval (4 heldout domains)
JID_EVAL=$(submit "07_eval_grpo" \
    --dependency=afterok:${JID_CHECKPOINT} \
    --array=0-3 \
    slurm/07_eval_pilot_gpu_array.sbatch)

# Job 10: aggregate results
JID_AGG=$(submit "10_aggregate_grpo" \
    --dependency=afterok:${JID_EVAL} \
    slurm/10_aggregate_results_cpu.sbatch)

echo ""
echo "GRPO pipeline submitted. Last job: ${JID_AGG}"
echo "Expected timeline: ~12-18 hours"
echo ""
echo "Monitor: squeue -u \$USER | grep tars"
echo "Check progress: tail -f logs/08_grpo_${JID_GRPO}.out"
