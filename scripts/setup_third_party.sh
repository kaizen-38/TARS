#!/usr/bin/env bash
# Initialize and update all git submodules under third_party/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Initializing git submodules..."
git submodule update --init --recursive

echo "==> Checking submodule status..."
git submodule status

# Install LLaMA-Factory as an editable package
if [ -d "third_party/LLaMAFactory" ]; then
    echo "==> Installing LLaMA-Factory..."
    pip install -e third_party/LLaMAFactory --quiet
else
    echo "WARNING: third_party/LLaMAFactory not found. Run: git submodule update --init"
fi

echo "==> Done."
