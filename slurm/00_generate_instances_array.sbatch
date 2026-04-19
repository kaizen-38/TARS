#!/usr/bin/env bash
#SBATCH --job-name=tars_generate
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --array=0-11          # one task per domain (12 total: 8 train + 4 heldout)
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/00_generate_%A_%a.out
#SBATCH --error=logs/00_generate_%A_%a.err

# MODE env var controls how many instances: smoke (5) or pilot (25)
# Submit via submit_phase1.sh or:
#   MODE=smoke sbatch slurm/00_generate_instances_array.sbatch

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
MANIFEST="${REPO_ROOT}/manifests/generate_${MODE:-smoke}.tsv"

module purge
module load mamba/latest
source /packages/apps/mamba/latest/etc/profile.d/conda.sh
conda activate tars

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

# Skip header line; SLURM_ARRAY_TASK_ID is 0-indexed -> line N+2
LINE=$(awk "NR==$(( SLURM_ARRAY_TASK_ID + 2 ))" "$MANIFEST")
DOMAIN=$(echo "$LINE" | cut -f1)
SPLIT=$(echo  "$LINE" | cut -f2)
N=$(echo      "$LINE" | cut -f3)
SEED=$(echo   "$LINE" | cut -f4)

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} domain=${DOMAIN} split=${SPLIT} n=${N} seed=${SEED}"

python -m cli generate-one \
    "${DOMAIN}" \
    --split "${SPLIT}" \
    --n-instances "${N}" \
    --seed "${SEED}" \
    --output-dir data/generated/instances

echo "[$(date)] done: ${DOMAIN} — $(find data/generated/instances/${DOMAIN} -name '*_meta.json' | wc -l) instances"
