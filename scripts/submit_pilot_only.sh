#!/usr/bin/env bash
# submit_pilot_only.sh — submit ONLY the pilot pipeline (Part 2).
# Assumes smoke has already completed and data/generated/ has smoke instances.
#
# Usage:
#   bash scripts/submit_pilot_only.sh

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
PILOT_N=$(( N_DOMAINS * 25 - 1 ))   # 25 × 12 = 300, array 0-299

LOG="${REPO_ROOT}/logs/submitted_jobs.txt"
echo "=== Pilot-only submission $(date) ===" | tee -a "$LOG"

submit() {
    local desc="$1"; shift
    local jid
    jid=$(sbatch "$@" | awk '{print $NF}')
    echo "${desc}: ${jid}" | tee -a "$LOG" >&2
    echo "$jid"
}

# Generate pilot manifests
echo "Generating pilot manifests..."
python3 scripts/gen_manifests.py generate --mode pilot

# Job 00: generate pilot instances
JID_GEN=$(submit "00_generate_pilot" \
    --export=ALL,MODE=pilot \
    --array=0-$(( N_DOMAINS - 1 )) \
    slurm/00_generate_instances_array.sbatch)

# Build solve manifest
JID_SOLVE_MANIFEST=$(submit "00b_solve_manifest_pilot" \
    --dependency=afterok:${JID_GEN} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=logs/00b_manifest_pilot_%j.out \
    --wrap="${WRAP_INIT} && python3 scripts/gen_manifests.py solve")

# Job 01: solve instances (FD), array 0-299
JID_SOLVE=$(submit "01_teacher_plans_pilot" \
    --dependency=afterok:${JID_SOLVE_MANIFEST} \
    --array=0-${PILOT_N} \
    slurm/01_teacher_plans_array.sbatch)

# Build validate manifest
JID_VAL_MANIFEST=$(submit "01b_val_manifest_pilot" \
    --dependency=afterok:${JID_SOLVE} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=logs/01b_manifest_pilot_%j.out \
    --wrap="${WRAP_INIT} && python3 scripts/gen_manifests.py validate")

# Job 02: validate plans (VAL), array 0-299
JID_VAL=$(submit "02_validate_pilot" \
    --dependency=afterok:${JID_VAL_MANIFEST} \
    --array=0-${PILOT_N} \
    slurm/02_validate_teacher_array.sbatch)

# Job 02b: build tuples
JID_TUPLES=$(submit "02b_build_tuples_pilot" \
    --dependency=afterok:${JID_VAL} \
    slurm/02b_build_tuples_cpu.sbatch)

# Job 03: build dataset
JID_DATASET=$(submit "03_build_dataset_pilot" \
    --dependency=afterok:${JID_TUPLES} \
    slurm/03_build_dataset.sbatch)

# Job 06: mini train (LoRA, 4h)
JID_TRAIN=$(submit "06_train_mini" \
    --dependency=afterok:${JID_DATASET} \
    slurm/06_train_mini_gpu.sbatch)

# Job 07: eval pilot (4 heldout domains)
JID_EVAL=$(submit "07_eval_pilot" \
    --dependency=afterok:${JID_TRAIN} \
    --array=0-3 \
    slurm/07_eval_pilot_gpu_array.sbatch)

# Job 10: aggregate results
JID_AGG=$(submit "10_aggregate_pilot" \
    --dependency=afterok:${JID_EVAL} \
    slurm/10_aggregate_results_cpu.sbatch)

echo ""
echo "Pilot pipeline submitted. Last job: ${JID_AGG}"
echo "Monitor: squeue -u \$USER | grep tars"
