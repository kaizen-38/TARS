#!/usr/bin/env bash
# Generate 5 instances for each domain (smoke test).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Generating smoke-test instances (5 per domain)..."
PYTHONPATH="$REPO_ROOT/src" python -m cli generate-smoke \
    --seed 42 \
    --splits-config configs/splits/phase1_v1.yaml \
    --output-dir data/generated/instances

echo "==> Done. Check data/generated/instances/ for output."
