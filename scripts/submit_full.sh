#!/usr/bin/env bash
# submit_full.sh — submit the full baseline pipeline (100 inst/domain).
# Assumes smoke and pilot have already completed.
#
# This generates ~800 train + ~400 heldout = ~1200 total plans
# → ~3600 training examples after 3 representations
#
# Usage:
#   bash scripts/submit_full.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

if ! python3 -c "import typer" &>/dev/null; then
    module load mamba/latest
    eval "$(conda shell.bash hook)"
    conda activate tars
fi
export PYTHONPATH="${REPO_ROOT}/src"

WRAP_INIT="source ${REPO_ROOT}/scripts/sol_init.sh && cd ${REPO_ROOT} && export PYTHONPATH=${REPO_ROOT}/src"

N_DOMAINS=12
FULL_N=$(( N_DOMAINS * 100 - 1 ))   # 100 × 12 = 1200, array 0-1199

LOG="${REPO_ROOT}/logs/submitted_jobs.txt"
echo "=== Full baseline submission $(date) ===" | tee -a "$LOG"

submit() {
    local desc="$1"; shift
    local jid
    jid=$(sbatch "$@" | awk '{print $NF}')
    echo "${desc}: ${jid}" | tee -a "$LOG" >&2
    echo "$jid"
}

# Generate full manifests
echo "Generating full manifests..."
python3 scripts/gen_manifests.py generate --mode full

# Job 00: generate full instances (100 per domain)
JID_GEN=$(submit "00_generate_full" \
    --export=ALL,MODE=full \
    --array=0-$(( N_DOMAINS - 1 )) \
    slurm/00_generate_instances_array.sbatch)

# Build solve manifest
JID_SOLVE_MANIFEST=$(submit "00b_solve_manifest_full" \
    --dependency=afterok:${JID_GEN} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=logs/00b_manifest_full_%j.out \
    --wrap="${WRAP_INIT} && python3 scripts/gen_manifests.py solve")

# Job 01: solve instances (FD), array 0-1199
JID_SOLVE=$(submit "01_teacher_plans_full" \
    --dependency=afterok:${JID_SOLVE_MANIFEST} \
    --array=0-${FULL_N} \
    slurm/01_teacher_plans_array.sbatch)

# Build validate manifest
JID_VAL_MANIFEST=$(submit "01b_val_manifest_full" \
    --dependency=afterok:${JID_SOLVE} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=logs/01b_manifest_full_%j.out \
    --wrap="${WRAP_INIT} && python3 scripts/gen_manifests.py validate")

# Job 02: validate plans (VAL), array 0-1199
JID_VAL=$(submit "02_validate_full" \
    --dependency=afterok:${JID_VAL_MANIFEST} \
    --array=0-${FULL_N} \
    slurm/02_validate_teacher_array.sbatch)

# Job 02b: build tuples
JID_TUPLES=$(submit "02b_build_tuples_full" \
    --dependency=afterok:${JID_VAL} \
    slurm/02b_build_tuples_cpu.sbatch)

# Job 03: build dataset
JID_DATASET=$(submit "03_build_dataset_full" \
    --dependency=afterok:${JID_TUPLES} \
    slurm/03_build_dataset.sbatch)

# Job 06b: direct training (bypasses LLaMAFactory issues)
JID_TRAIN=$(submit "06b_train_direct_full" \
    --dependency=afterok:${JID_DATASET} \
    slurm/06b_train_direct_gpu.sbatch)

# Update checkpoint pointer
JID_CHECKPOINT=$(submit "06c_update_checkpoint" \
    --dependency=afterok:${JID_TRAIN} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=1G --time=00:01:00 \
    --output=logs/06c_checkpoint_%j.out \
    --wrap="echo 'runs/qwen3_mini_direct' > ${REPO_ROOT}/runs/latest_checkpoint.txt")

# Job 07: eval full (4 heldout domains)
JID_EVAL=$(submit "07_eval_full" \
    --dependency=afterok:${JID_CHECKPOINT} \
    --array=0-3 \
    slurm/07_eval_pilot_gpu_array.sbatch)

# Job 10: aggregate results
JID_AGG=$(submit "10_aggregate_full" \
    --dependency=afterok:${JID_EVAL} \
    slurm/10_aggregate_results_cpu.sbatch)

echo ""
echo "Full baseline pipeline submitted. Last job: ${JID_AGG}"
echo "Expected timeline: 6-12 hours"
echo "Monitor: squeue -u \$USER | grep tars"
echo ""
echo "Dataset will grow from 514 to ~2400-3000 examples"
echo "This is the proper full baseline for Phase 1."
