#!/usr/bin/env bash
#SBATCH --job-name=tars_dataset
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/03_dataset_%j.out
#SBATCH --error=logs/03_dataset_%j.err

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module purge
module load mamba/latest
source activate tars

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

echo "[$(date)] Building SFT datasets..."

python -m cli build-dataset --split train
python -m cli build-dataset --split heldout

echo "[$(date)] Dataset build complete."
echo "Row counts:"
for f in data/datasets/alpaca/*.jsonl; do
    echo "  $(wc -l < "$f") rows in $f"
done
