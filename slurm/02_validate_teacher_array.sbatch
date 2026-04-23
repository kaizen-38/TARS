#!/usr/bin/env bash
#SBATCH --job-name=tars_validate
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --array=0-59          # smoke: update to match solve_manifest row count - 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/02_validate_%A_%a.out
#SBATCH --error=logs/02_validate_%A_%a.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
MANIFEST="${REPO_ROOT}/manifests/validate_manifest.tsv"

module purge
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate thicket311

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

LINE=$(awk "NR==$(( SLURM_ARRAY_TASK_ID + 2 ))" "$MANIFEST")
INSTANCE_ID=$(echo  "$LINE" | cut -f1)
DOMAIN_FILE=$(echo  "$LINE" | cut -f2)
PROBLEM_FILE=$(echo "$LINE" | cut -f3)
PLAN_FILE=$(echo    "$LINE" | cut -f4)

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} instance=${INSTANCE_ID}"

python - <<EOF
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from pathlib import Path
from generation.validate_with_val import validate_plan

result = validate_plan(
    problem_id="${INSTANCE_ID}",
    domain_file=Path("${REPO_ROOT}/${DOMAIN_FILE}"),
    problem_file=Path("${REPO_ROOT}/${PROBLEM_FILE}"),
    plan_file=Path("${PLAN_FILE}"),
    output_dir=Path("${REPO_ROOT}/data/generated/plans"),
    timeout=30,
)
print(f"valid={result.parsed_validity} goal={result.parsed_goal_reached} time={result.wall_clock_sec:.1f}s")
if result.parsed_validity is False:
    print(f"INVALID plan for ${INSTANCE_ID}", file=sys.stderr)
EOF

echo "[$(date)] done: ${INSTANCE_ID}"
