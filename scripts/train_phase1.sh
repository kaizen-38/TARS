#!/usr/bin/env bash
# Launch Phase 1 SFT training.
# Default mode: lora_debug (smoke test). Pass MODE=full for full fine-tuning.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-lora_debug}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"

echo "==> Launching SFT training: mode=$MODE model=$MODEL"
PYTHONPATH="$REPO_ROOT/src" python src/training/launch_sft.py \
    --mode "$MODE" \
    --model "$MODEL"

echo "==> Training complete."
