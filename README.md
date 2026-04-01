# thicket_phase1

**Phase 1: Reproduce the 1.7B LLM Planning Baseline**

Goal: lock splits/budgets/protocol, reproduce Qwen3-1.7B planning results across
standard, anonymized, and compact PDDL representations, and build a trustworthy
data + validation pipeline.

> Regime switching is NOT implemented in Phase 1.

---

## Repository Layout

```
thicket_phase1/
  configs/
    splits/phase1_v1.yaml          # Locked domain splits, budgets, smoke/pilot counts
    models/qwen3_1p7b.yaml         # Primary model config
    models/qwen2p5_1p5b_debug.yaml # Cheap debug model config
    train/sft_qwen3_phase1_full.yaml    # Locked full fine-tuning config
    train/sft_qwen3_phase1_lora_debug.yaml  # LoRA smoke-test config
    eval/greedy_eval.yaml          # Greedy decoding eval (primary Phase 1 metric)
    eval/verify_sampling.yaml      # Sampling eval (pass@k, secondary)
  data/
    generated/instances/           # PDDL domain+problem files (generated)
    generated/plans/               # Solved plans + VAL results
    generated/tuples_*/            # (domain, problem, plan) tuples per representation
    datasets/alpaca/               # Alpaca JSONL datasets for LLaMA-Factory
    datasets/dataset_info.json     # LLaMA-Factory dataset registry
  src/
    cli.py                         # Unified CLI entry point
    generation/                    # Instance generation, FD solving, VAL validation
    pddl_ops/                      # PDDL parsing, anonymization, compact serialization
    dataset/                       # SFT dataset builder, dedup, stats
    training/                      # LLaMA-Factory config writer + launch wrapper
    inference/                     # Plan generation + greedy eval runner
    eval/                          # Metrics, aggregation, manifest
    utils/                         # IO, logging, seeds
  scripts/                         # Shell scripts backing Makefile targets
  tests/                           # Pytest tests
  third_party/                     # Git submodules
  runs/                            # Training + eval outputs (gitignored)
```

---

## Third-Party Dependencies (Submodules)

| Submodule | Path | Role |
|---|---|---|
| [hiyouga/LLaMAFactory](https://github.com/hiyouga/LLaMAFactory) | `third_party/LLaMAFactory` | SFT training framework |
| [AI-Planning/pddl-generators](https://github.com/AI-Planning/pddl-generators) | `third_party/pddl-generators` | Domain instance generation |
| [KCL-Planning/VAL](https://github.com/KCL-Planning/VAL) | `third_party/VAL` | PDDL plan validation (`Validate` binary) |
| [aibasel/downward](https://github.com/aibasel/downward) | `third_party/downward` | Fast Downward teacher planner |
| [harshakokel/PlanBench](https://github.com/harshakokel/PlanBench) | `third_party/PlanBench` | **Reference only** — do not import into `src/` |
| [ValerioBelcamino/PDDL_encoder](https://github.com/ValerioBelcamino/PDDL_encoder) | `third_party/PDDL_encoder` | **Reference only** — do not import into `src/` |

> **PlanBench** and **PDDL_encoder** are included as reference material only.
> No code from these repos is imported or copied into `src/`.

---

## Setup

**Requirements:** Python 3.10+, `git`, `cmake`, `make`, C++ compiler.

```bash
# 1. Clone with submodules
git clone --recurse-submodules <your-repo-url>
cd thicket_phase1

# 2. Install Python deps + init submodules
make setup

# 3. Build Fast Downward and VAL
make build-tools
```

Or step by step:

```bash
pip install -e ".[dev]"
git submodule update --init --recursive
pip install -e third_party/LLaMAFactory
bash scripts/build_generators.sh    # builds Fast Downward
bash scripts/build_val.sh           # builds VAL Validate binary
```

---

## End-to-End Smoke Test

Run this after `make setup` and `make build-tools`:

```bash
# Step 1: Generate 5 instances per domain
make generate-smoke
# Expected: data/generated/instances/<domain>/train/ and /heldout/ dirs
# Each with *_domain.pddl, *_problem.pddl, *_meta.json

# Step 2: Solve with Fast Downward + validate with VAL
make solve-smoke
# Expected: data/generated/plans/<domain>/*.raw_plan.txt
#           data/generated/plans/<domain>/*.plan.pddl
#           data/generated/tuples_standard/*_tuple.json

# Step 3: Build SFT datasets
make build-dataset
# Expected: data/datasets/alpaca/phase1_{standard,anonymized,compact}.jsonl
#           data/datasets/dataset_info.json

# Step 4: Smoke SFT run (LoRA, 20 steps)
make train-phase1
# Expected: runs/sft_qwen3_phase1_lora_debug/

# Step 5: Greedy evaluation
make eval-phase1 CHECKPOINT=runs/sft_qwen3_phase1_lora_debug
# Expected: runs/eval_results/run_log.jsonl
#           runs/eval_results/metrics_summary.json

# Step 6: Run tests
make test
```

---

## Expected File Outputs

After `make generate-smoke`:
```
data/generated/instances/
  blocksworld/train/
    blocksworld_train_0000_42_domain.pddl
    blocksworld_train_0000_42_problem.pddl
    blocksworld_train_0000_42_meta.json
    ... (5 instances)
  ferry/heldout/
    ... (5 instances)
  miconic/heldout/
    ... (5 instances)
```

After `make build-dataset`:
```
data/datasets/
  alpaca/
    phase1_standard.jsonl
    phase1_anonymized.jsonl
    phase1_compact.jsonl
  dataset_info.json
```

After `make eval-phase1`:
```
runs/eval_results/
  run_log.jsonl           # One JSONL row per (instance, representation)
  metrics_summary.json    # Aggregated validity/goal rates by domain + representation
  raw/                    # Raw model outputs
  val_results/            # Per-instance VAL result JSONs
```

---

## Run Log Schema

Every evaluation appends rows with these fields to `run_log.jsonl`:

| Field | Description |
|---|---|
| `run_id` | Unique run identifier |
| `timestamp` | ISO 8601 UTC |
| `git_commit` | Short commit hash |
| `seed` | Global random seed |
| `domain` | PDDL domain name |
| `problem_id` | Instance identifier |
| `representation` | `standard` / `anonymized` / `compact` |
| `split` | `train` / `heldout` |
| `model_name` | HF model identifier |
| `checkpoint_path` | Fine-tuned checkpoint path |
| `planner_backend` | Teacher planner used (e.g. `fd`) |
| `valid_plan` | VAL verdict (bool or null) |
| `goal_reached` | VAL goal check (bool or null) |
| `max_new_tokens` | Decoding budget |
| `generated_tokens` | Actual tokens generated |
| `generation_time_sec` | Wall-clock generation time |
| `val_time_sec` | VAL validation time |
| `total_time_sec` | Total per-instance time |
| `error_type` | Exception class if error, else null |

---

## Key Design Decisions

1. **`enable_thinking=False` always** — set at the `apply_chat_template` call in
   `src/inference/generate_plan.py`. This is a hard Phase 1 requirement.

2. **Full fine-tuning by default** — `finetuning_type: full` in the locked config.
   LoRA config exists only for smoke testing.

3. **VAL is the single source of truth** — no heuristic plan checking.
   All validity claims trace to the `Validate` binary.

4. **No code copied from PlanBench** — PlanBench is a reference submodule.
   Prompt formats and domain lists were independently specified.

5. **Planner backend is abstracted** — `src/generation/solve_with_fd.py` has a
   `PlannerBackend` ABC. Fast Downward is the only registered backend in Phase 1.

---

## Running Full Fine-Tuning

```bash
make train-phase1 MODE=full
```

This writes the config to `runs/sft_qwen3_phase1_full/train_config.yaml` and
calls `llamafactory-cli train`. Requires a GPU with ~40GB VRAM (or use
gradient checkpointing + small batch on smaller GPUs).
