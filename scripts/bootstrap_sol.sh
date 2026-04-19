#!/usr/bin/env bash
# Run once on a Sol login node to set up the project environment.
# Usage: bash scripts/bootstrap_sol.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="tars"

echo "==> Loading mamba..."
module purge
module load mamba/latest

echo "==> Creating conda environment '${ENV_NAME}' with Python 3.11..."
mamba create -n "${ENV_NAME}" python=3.11 -y

echo "==> Activating environment..."
source activate "${ENV_NAME}"

echo "==> Upgrading pip and setuptools..."
pip install --upgrade "pip>=24" "setuptools>=68" wheel --quiet

echo "==> Installing project in editable mode..."
cd "$REPO_ROOT"
pip install -e ".[dev]" --quiet

echo "==> Initialising git submodules..."
bash scripts/setup_third_party.sh

echo "==> Building Fast Downward and VAL..."
bash scripts/build_generators.sh
bash scripts/build_val.sh

echo ""
echo "Bootstrap complete. To activate in future sessions:"
echo "  module load mamba/latest && source activate ${ENV_NAME}"
