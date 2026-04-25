#!/usr/bin/env bash
#SBATCH --job-name=tars_tuples
#SBATCH --account=class_cse574spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/02b_tuples_%j.out
#SBATCH --error=logs/02b_tuples_%j.err
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
module purge
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate thicket311
export PYTHONPATH="${REPO_ROOT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
mkdir -p logs
echo "[$(date)] Building tuples..."
python -m src.cli build-tuples
echo "[$(date)] Tuple build complete."
ls data/generated/tuples_standard/ | wc -l
