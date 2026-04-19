#!/usr/bin/env bash
#SBATCH --job-name=tars_aggregate
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/10_aggregate_%j.out
#SBATCH --error=logs/10_aggregate_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"

module purge
module load mamba/latest
source /packages/apps/mamba/latest/etc/profile.d/conda.sh
conda activate tars

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

echo "[$(date)] Aggregating Phase 1 results..."

# Aggregate each eval run that has a run_log
for LOG in runs/eval_*/run_log.jsonl; do
    [[ -f "$LOG" ]] || continue
    OUTDIR=$(dirname "$LOG")
    echo "==> ${LOG}"
    python src/eval/aggregate_results.py "${LOG}" || echo "WARNING: aggregation failed for ${LOG}"
done

echo ""
echo "=== Phase 1 summary ==="
for SUMMARY in runs/eval_*/metrics_summary.json; do
    [[ -f "$SUMMARY" ]] || continue
    echo "--- $(dirname "$SUMMARY") ---"
    python -c "
import json, sys
d = json.load(open('$SUMMARY'))
for k,v in d.items():
    if isinstance(v, float):
        print(f'  {k}: {v:.3f}')
    else:
        print(f'  {k}: {v}')
"
done

echo "[$(date)] Aggregation complete."
