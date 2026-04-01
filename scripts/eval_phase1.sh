#!/usr/bin/env bash
# Run Phase 1 greedy evaluation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT="${CHECKPOINT:-Qwen/Qwen3-1.7B}"
SPLIT="${SPLIT:-heldout}"
SEED="${SEED:-42}"

echo "==> Running greedy evaluation: checkpoint=$CHECKPOINT split=$SPLIT"
PYTHONPATH="$REPO_ROOT/src" python src/inference/run_greedy_eval.py \
    --checkpoint-path "$CHECKPOINT" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --eval-config-path configs/eval/greedy_eval.yaml \
    --output-dir runs/eval_results

echo "==> Aggregating results..."
PYTHONPATH="$REPO_ROOT/src" python src/eval/aggregate_results.py \
    runs/eval_results/run_log.jsonl

echo "==> Evaluation complete. See runs/eval_results/"
