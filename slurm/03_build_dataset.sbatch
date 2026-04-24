#!/usr/bin/env bash
#SBATCH --job-name=tars_dataset
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/03_dataset_%j.out
#SBATCH --error=logs/03_dataset_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"

module purge
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate thicket311

export PYTHONPATH="${REPO_ROOT}/src"
cd "${REPO_ROOT}"
mkdir -p logs

echo "[$(date)] Building SFT datasets..."

python -m src.cli build-dataset --split train
python -m src.cli build-dataset --split heldout

echo "[$(date)] Dataset build complete."
echo "Row counts:"
for f in data/datasets/alpaca/*.jsonl; do
    echo "  $(wc -l < "$f") rows in $f"
done
