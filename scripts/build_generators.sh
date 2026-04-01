#!/usr/bin/env bash
# Build and/or install AI-Planning/pddl-generators.
# Most generators are Python scripts; this script installs any dependencies.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_DIR="$REPO_ROOT/third_party/pddl-generators"

if [ ! -d "$GEN_DIR" ]; then
    echo "ERROR: $GEN_DIR not found."
    echo "Run: git submodule update --init third_party/pddl-generators"
    exit 1
fi

echo "==> Setting up pddl-generators at $GEN_DIR"

# Install Python requirements if present
if [ -f "$GEN_DIR/requirements.txt" ]; then
    pip install -r "$GEN_DIR/requirements.txt" --quiet
fi

# Build Fast Downward (aibasel/downward) — required for teacher planner
FD_DIR="$REPO_ROOT/third_party/downward"
if [ -d "$FD_DIR" ]; then
    echo "==> Building Fast Downward..."
    cd "$FD_DIR"
    python build.py release 2>&1 | tail -20
    echo "==> Fast Downward built."
else
    echo "WARNING: $FD_DIR not found. Teacher planner will be unavailable."
fi

echo "==> Done."
