# thicket_phase1 Makefile
# All targets run from the repo root.

PYTHON     := python
PYTHONPATH := $(shell pwd)/src
SHELL      := /bin/bash

export PYTHONPATH

.PHONY: help setup build-tools generate-smoke solve-smoke build-dataset \
        train-phase1 eval-phase1 test lint format check

help:
	@echo "thicket_phase1 — Phase 1 PDDL planning baseline"
	@echo ""
	@echo "  make setup           Install Python deps and init submodules"
	@echo "  make build-tools     Build Fast Downward and VAL binaries"
	@echo "  make generate-smoke  Generate 5 instances per domain (smoke test)"
	@echo "  make solve-smoke     Solve smoke instances with FD and validate with VAL"
	@echo "  make build-dataset   Build Alpaca JSONL datasets + dataset_info.json"
	@echo "  make train-phase1    Launch LoRA smoke-test SFT (set MODE=full for full FT)"
	@echo "  make eval-phase1     Run greedy evaluation (set CHECKPOINT=<path>)"
	@echo "  make test            Run all pytest tests"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Run ruff formatter"

# ---------------------------------------------------------------------------
setup:
	@echo "==> Installing Python dependencies..."
	$(PYTHON) -m pip install -e ".[dev]" --quiet
	@echo "==> Initializing git submodules..."
	bash scripts/setup_third_party.sh
	@echo "==> Setup complete."

# ---------------------------------------------------------------------------
build-tools:
	@echo "==> Building Fast Downward and pddl-generators..."
	bash scripts/build_generators.sh
	@echo "==> Building VAL..."
	bash scripts/build_val.sh
	@echo "==> Tools built."

# ---------------------------------------------------------------------------
generate-smoke:
	@echo "==> Generating smoke-test instances (5 per domain)..."
	bash scripts/generate_smoke_data.sh

# ---------------------------------------------------------------------------
solve-smoke:
	@echo "==> Solving smoke instances with Fast Downward and validating with VAL..."
	$(PYTHON) src/cli.py solve-smoke

# ---------------------------------------------------------------------------
build-dataset:
	@echo "==> Building SFT datasets..."
	$(PYTHON) src/cli.py build-dataset
	@echo "==> Datasets written to data/datasets/alpaca/"

# ---------------------------------------------------------------------------
train-phase1:
	@echo "==> Launching SFT (MODE=$(or $(MODE),lora_debug))..."
	MODE=$(or $(MODE),lora_debug) bash scripts/train_phase1.sh

# ---------------------------------------------------------------------------
eval-phase1:
	@echo "==> Running greedy evaluation (CHECKPOINT=$(or $(CHECKPOINT),Qwen/Qwen3-1.7B))..."
	CHECKPOINT=$(or $(CHECKPOINT),Qwen/Qwen3-1.7B) bash scripts/eval_phase1.sh

# ---------------------------------------------------------------------------
test:
	@echo "==> Running tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short

# ---------------------------------------------------------------------------
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

check: lint test
