#!/usr/bin/env bash
# submit_phase1.sh — submit the full Phase 1 pipeline on Sol with proper dependencies.
#
# Usage:
#   bash scripts/submit_phase1.sh            # smoke only (fast sanity check)
#   bash scripts/submit_phase1.sh --pilot    # smoke → pilot → mini train → pilot eval
#   bash scripts/submit_phase1.sh --full     # smoke → pilot → full train → full eval
#
# All jobs write logs to logs/ and outputs to data/ and runs/.
# Job IDs are printed and saved to logs/submitted_jobs.txt.
#
# Prerequisites:
#   1. bash scripts/bootstrap_sol.sh    (run once, on login node)
#   2. python3 scripts/gen_manifests.py generate --mode smoke
#   3. python3 scripts/gen_manifests.py eval --split heldout
#   4. python3 scripts/gen_manifests.py eval --split train
#
# Sol account: class_cse574spring2026
# Shared data:  /data/courses/class_cse574spring2026_subbarao

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

# Activate tars environment if not already active
if ! python3 -c "import typer" &>/dev/null; then
    module load mamba/latest
    eval "$(conda shell.bash hook)"
    conda activate tars
fi
export PYTHONPATH="${REPO_ROOT}/src"

# Pre-flight: verify submodules and built tools exist
PREFLIGHT_OK=true
if [ ! -f "${REPO_ROOT}/third_party/pddl-generators/blocksworld/blocksworld" ]; then
    echo "ERROR: pddl-generators not built. Run: bash scripts/setup_third_party.sh && make build-tools" >&2
    PREFLIGHT_OK=false
fi
if [ ! -f "${REPO_ROOT}/third_party/downward/fast-downward.py" ]; then
    echo "ERROR: Fast Downward not found. Run: bash scripts/setup_third_party.sh && make build-tools" >&2
    PREFLIGHT_OK=false
fi
if [ ! -f "${REPO_ROOT}/third_party/VAL/build/bin/Validate" ] && \
   [ ! -f "${REPO_ROOT}/third_party/VAL/bin/Validate" ]; then
    echo "ERROR: VAL not built. Run: bash scripts/setup_third_party.sh && make build-tools" >&2
    PREFLIGHT_OK=false
fi
if ! $PREFLIGHT_OK; then
    echo "Fix the above errors then rerun submit_phase1.sh." >&2
    exit 1
fi

RUN_PILOT=false
RUN_FULL=false
for arg in "$@"; do
    case "$arg" in
        --pilot) RUN_PILOT=true ;;
        --full)  RUN_FULL=true; RUN_PILOT=true ;;
    esac
done

LOG="${REPO_ROOT}/logs/submitted_jobs.txt"
echo "=== Phase 1 submission $(date) ===" | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Helper: submit and record job ID
# ---------------------------------------------------------------------------
submit() {
    local desc="$1"; shift
    local jid
    jid=$(sbatch "$@" | awk '{print $NF}')
    echo "${desc}: ${jid}" | tee -a "$LOG" >&2
    echo "$jid"
}

# ---------------------------------------------------------------------------
# PART 1 — Smoke pipeline (always runs)
# ---------------------------------------------------------------------------
echo ""
echo "--- Part 1: Smoke pipeline ---"

# Generate smoke manifests
echo "Generating smoke manifests..."
python3 scripts/gen_manifests.py generate --mode smoke
python3 scripts/gen_manifests.py eval --split heldout
python3 scripts/gen_manifests.py eval --split train

N_DOMAINS=12
SMOKE_N=$(( N_DOMAINS * 5 - 1 ))   # 5 instances × 12 domains = 60, array 0-59

# Job 00: generate smoke instances (array, 12 tasks)
JID_GEN=$(submit "00_generate_smoke" \
    --export=ALL,MODE=smoke \
    --array=0-$(( N_DOMAINS - 1 )) \
    slurm/00_generate_instances_array.sbatch)

# After generate: build solve manifest
JID_SOLVE_MANIFEST=$(submit "00b_solve_manifest" \
    --dependency=afterok:${JID_GEN} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=logs/00b_manifest_%j.out \
    --wrap="cd ${REPO_ROOT} && module load mamba/latest && eval \"\$(conda shell.bash hook)\" && conda activate tars && \
            export PYTHONPATH=${REPO_ROOT}/src && \
            python3 scripts/gen_manifests.py solve && \
            python3 scripts/gen_manifests.py validate")

# Job 01: solve instances (FD), smoke array 0-59
JID_SOLVE=$(submit "01_teacher_plans_smoke" \
    --dependency=afterok:${JID_SOLVE_MANIFEST} \
    --array=0-${SMOKE_N} \
    slurm/01_teacher_plans_array.sbatch)

# After solve: build validate manifest
JID_VAL_MANIFEST=$(submit "01b_val_manifest" \
    --dependency=afterok:${JID_SOLVE} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=logs/01b_manifest_%j.out \
    --wrap="cd ${REPO_ROOT} && module load mamba/latest && eval \"\$(conda shell.bash hook)\" && conda activate tars && \
            export PYTHONPATH=${REPO_ROOT}/src && \
            python3 scripts/gen_manifests.py validate")

# Job 02: validate plans (VAL), smoke array 0-59
JID_VAL=$(submit "02_validate_smoke" \
    --dependency=afterok:${JID_VAL_MANIFEST} \
    --array=0-${SMOKE_N} \
    slurm/02_validate_teacher_array.sbatch)

# Job 03: build dataset
JID_DATASET=$(submit "03_build_dataset" \
    --dependency=afterok:${JID_VAL} \
    slurm/03_build_dataset.sbatch)

# Job 04: smoke LoRA train
JID_TRAIN_SMOKE=$(submit "04_train_smoke" \
    --dependency=afterok:${JID_DATASET} \
    slurm/04_train_smoke_gpu.sbatch)

# Job 05: smoke eval (2 domains, sanity check)
JID_EVAL_SMOKE=$(submit "05_eval_smoke" \
    --dependency=afterok:${JID_TRAIN_SMOKE} \
    slurm/05_eval_smoke_gpu.sbatch)

echo ""
echo "Smoke pipeline submitted. Monitor with: squeue -u $USER"
echo "Last job in smoke chain: ${JID_EVAL_SMOKE}"

if ! $RUN_PILOT; then
    echo ""
    echo "Smoke-only mode. To continue to pilot run:"
    echo "  bash scripts/submit_phase1.sh --pilot"
    exit 0
fi

# ---------------------------------------------------------------------------
# PART 2 — Pilot pipeline (25 instances/domain)
# ---------------------------------------------------------------------------
echo ""
echo "--- Part 2: Pilot pipeline (depends on smoke dataset) ---"

PILOT_N=$(( N_DOMAINS * 25 - 1 ))   # 25 × 12 = 300, array 0-299

python3 scripts/gen_manifests.py generate --mode pilot

JID_GEN_PILOT=$(submit "00_generate_pilot" \
    --dependency=afterok:${JID_EVAL_SMOKE} \
    --export=ALL,MODE=pilot \
    --array=0-$(( N_DOMAINS - 1 )) \
    slurm/00_generate_instances_array.sbatch)

JID_SOLVE_MANIFEST_P=$(submit "00b_solve_manifest_pilot" \
    --dependency=afterok:${JID_GEN_PILOT} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=logs/00b_manifest_pilot_%j.out \
    --wrap="cd ${REPO_ROOT} && module load mamba/latest && eval \"\$(conda shell.bash hook)\" && conda activate tars && \
            export PYTHONPATH=${REPO_ROOT}/src && \
            python3 scripts/gen_manifests.py solve")

JID_SOLVE_PILOT=$(submit "01_teacher_plans_pilot" \
    --dependency=afterok:${JID_SOLVE_MANIFEST_P} \
    --array=0-${PILOT_N} \
    slurm/01_teacher_plans_array.sbatch)

JID_VAL_MANIFEST_P=$(submit "01b_val_manifest_pilot" \
    --dependency=afterok:${JID_SOLVE_PILOT} \
    --partition=public --qos=class --account=class_cse574spring2026 \
    --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=logs/01b_manifest_pilot_%j.out \
    --wrap="cd ${REPO_ROOT} && module load mamba/latest && eval \"\$(conda shell.bash hook)\" && conda activate tars && \
            export PYTHONPATH=${REPO_ROOT}/src && \
            python3 scripts/gen_manifests.py validate")

JID_VAL_PILOT=$(submit "02_validate_pilot" \
    --dependency=afterok:${JID_VAL_MANIFEST_P} \
    --array=0-${PILOT_N} \
    slurm/02_validate_teacher_array.sbatch)

JID_DATASET_PILOT=$(submit "03_build_dataset_pilot" \
    --dependency=afterok:${JID_VAL_PILOT} \
    slurm/03_build_dataset.sbatch)

JID_TRAIN_MINI=$(submit "06_train_mini" \
    --dependency=afterok:${JID_DATASET_PILOT} \
    slurm/06_train_mini_gpu.sbatch)

JID_EVAL_PILOT=$(submit "07_eval_pilot" \
    --dependency=afterok:${JID_TRAIN_MINI} \
    --array=0-3 \
    slurm/07_eval_pilot_gpu_array.sbatch)

JID_AGG_PILOT=$(submit "10_aggregate_pilot" \
    --dependency=afterok:${JID_EVAL_PILOT} \
    slurm/10_aggregate_results_cpu.sbatch)

echo "Pilot pipeline submitted. Last job: ${JID_AGG_PILOT}"

if ! $RUN_FULL; then
    echo ""
    echo "Pilot-only mode. To continue to full run:"
    echo "  bash scripts/submit_phase1.sh --full"
    exit 0
fi

# ---------------------------------------------------------------------------
# PART 3 — Full baseline (paper-facing checkpoint)
# ---------------------------------------------------------------------------
echo ""
echo "--- Part 3: Full baseline (depends on pilot dataset) ---"

JID_TRAIN_FULL=$(submit "08_train_full" \
    --dependency=afterok:${JID_AGG_PILOT} \
    slurm/08_train_full_gpu.sbatch)

JID_EVAL_FULL=$(submit "09_eval_full" \
    --dependency=afterok:${JID_TRAIN_FULL} \
    --array=0-11 \
    slurm/09_eval_full_gpu_array.sbatch)

JID_AGG_FULL=$(submit "10_aggregate_full" \
    --dependency=afterok:${JID_EVAL_FULL} \
    slurm/10_aggregate_results_cpu.sbatch)

echo "Full baseline submitted. Last job: ${JID_AGG_FULL}"
echo ""
echo "All job IDs saved to ${LOG}"
