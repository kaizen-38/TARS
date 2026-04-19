#!/usr/bin/env bash
#SBATCH --job-name=tars_solve
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --array=0-59          # smoke: 5*12=60 rows; update to 0-299 for pilot (25*12)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/01_solve_%A_%a.out
#SBATCH --error=logs/01_solve_%A_%a.err

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${REPO_ROOT}/manifests/solve_manifest.tsv"

module purge
module load mamba/latest
source activate tars

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

LINE=$(awk "NR==$(( SLURM_ARRAY_TASK_ID + 2 ))" "$MANIFEST")
INSTANCE_ID=$(echo "$LINE" | cut -f1)
DOMAIN_FILE=$(echo "$LINE" | cut -f2)
PROBLEM_FILE=$(echo "$LINE" | cut -f3)

echo "[$(date)] task=${SLURM_ARRAY_TASK_ID} instance=${INSTANCE_ID}"

python - <<EOF
import sys
sys.path.insert(0, "${REPO_ROOT}/src")
from pathlib import Path
from generation.solve_with_fd import solve_instance

result = solve_instance(
    problem_id="${INSTANCE_ID}",
    domain_file=Path("${REPO_ROOT}/${DOMAIN_FILE}"),
    problem_file=Path("${REPO_ROOT}/${PROBLEM_FILE}"),
    output_dir=Path("${REPO_ROOT}/data/generated/plans"),
    backend="fd",
    timeout=120,
)
print(f"success={result.success} actions={len(result.action_sequence)} time={result.wall_clock_sec:.1f}s")
if not result.success:
    print(f"error={result.error}", file=sys.stderr)
    sys.exit(1)
EOF

echo "[$(date)] done: ${INSTANCE_ID}"
