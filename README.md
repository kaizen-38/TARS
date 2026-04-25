# TARS — Thicket-Aware Regime Switching for Verifier-Grounded LLM Planning

**Phase 1: Reproduce the 1.7B LLM Planning Baseline**

Team: Rhythm Arya (rarya124), Mohak Rathod (mrathod4), Abhiram Menon (amenon28)
Course: CSE-574, ASU

> Phase 1 goal: lock splits/budgets/protocol, reproduce Qwen3-1.7B planning results
> across standard, anonymized, and compact PDDL representations, and build a
> trustworthy data + validation pipeline on ASU Sol.
> Regime switching is NOT implemented in Phase 1.

---

## Frozen Splits

**Train (8):** blocksworld, gripper, ferry, delivery, childsnack, floortile, rovers, spanner

**Held-out (4):** miconic, sokoban, transport, satellite

**Representations:** standard, anonymized, compact

---

## Repository Layout
```
TARS/
configs/
  splits/phase1_v1.yaml               # Locked domain splits and budgets
  train/sft_qwen3_phase1_full.yaml    # Full fine-tuning config
  train/sft_qwen3_phase1_lora_debug.yaml  # LoRA smoke config
  eval/greedy_eval.yaml               # Greedy eval config (max_new_tokens=2048)
data/
  generated/instances/                # PDDL domain+problem files
  generated/plans/                    # Solved plans + VAL results
  generated/tuples_standard/          # (domain, problem, plan) tuples
  datasets/alpaca/                    # Alpaca JSONL datasets (split-suffixed)
  static/delivery/                    # Pre-generated delivery instances
src/
  cli.py                              # Unified CLI: generate-one, build-tuples, build-dataset, train, eval
  generation/                         # Instance generation, FD solving, VAL validation
  pddl_ops/                           # Anonymization, compact serialization, compact decoding
  dataset/                            # SFT dataset builder
  training/                           # LLaMAFactory config + launch
  inference/                          # Plan generation + greedy eval
  eval/                               # Metrics aggregation
slurm/                                # Sbatch scripts (00-10)
scripts/
  submit_phase1.sh                    # Full pipeline submission
  gen_manifests.py                    # TSV manifest generator
third_party/                          # Git submodules (LLaMAFactory, FD, VAL, pddl-generators)
runs/                                 # Training + eval outputs (gitignored)
```

---

## Sol Setup (One-time)

```bash
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate thicket311
export PYTHONPATH=$HOME/TARS/src
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH}/hf_cache"
```

`LD_LIBRARY_PATH` is required for VAL and Fast Downward (GLIBCXX). All sbatch scripts set this automatically via `$CONDA_PREFIX`.

---

## Running the Pipeline on Sol

### Quick start: smoke + pilot
```bash
cd ~/TARS
git checkout mohak/sol-fd-fix
git pull origin mohak/sol-fd-fix

bash scripts/submit_phase1.sh --pilot
```

### After pilot completes: evaluate train domains
```bash
JID=$(sbatch --array=0-7 slurm/07b_eval_train_gpu_array.sbatch | awk '{print $NF}')
echo "train eval: $JID"
```

### Aggregate results
```bash
PYTHONPATH=src python3 src/eval/aggregate_results.py runs/eval_pilot_train/run_log.jsonl
PYTHONPATH=src python3 src/eval/aggregate_results.py runs/eval_pilot/run_log.jsonl
```

### Pipeline stages
| Job | What | Depends on |
|-----|------|-----------|
| 00_generate | Generate instances (array 12) | -- |
| 00b | Build solve manifest | 00 |
| 01_teacher_plans | FD solve (array 60/300) | 00b |
| 01b | Build validate manifest | 01 |
| 02_validate | VAL validate (array 60/300) | 01b |
| 02b_build_tuples | Build tuple JSONs | 02 |
| 03_build_dataset | Build Alpaca JSONL | 02b |
| 04/06_train | LoRA/mini train | 03 |
| 05/07_eval | Greedy eval heldout | 04/06 |
| 07b_eval_train | Greedy eval train | manual |
| 10_aggregate | Aggregate results | 07 |

---

## Dataset Files
```
After build-dataset, files are split-suffixed:
data/datasets/alpaca/
  phase1_standard_train.jsonl
  phase1_anonymized_train.jsonl
  phase1_compact_train.jsonl
  phase1_standard_heldout.jsonl
  phase1_anonymized_heldout.jsonl
  phase1_compact_heldout.jsonl
  dataset_info.json                # LLaMAFactory registry (merges, no overwrite)
```

---

## Third-Party Dependencies

| Submodule | Role |
|-----------|------|
| `third_party/LLaMAFactory` | SFT training |
| `third_party/pddl-generators` | Domain instance generation |
| `third_party/VAL` | Plan validation |
| `third_party/downward` | Fast Downward teacher planner |
| `third_party/PlanBench` | Reference only |

---

## Key Design Decisions

1. **`enable_thinking=False` always** -- hard Phase 1 requirement
2. **VAL is single source of truth** -- no heuristic plan checking
3. **LD_LIBRARY_PATH via $CONDA_PREFIX** -- portable across all team members
4. **Delivery uses static instances** -- `uv` not available on Sol
5. **Split-suffixed datasets** -- prevents overwrite on second split build
6. **Empty plans rejected before VAL** -- plans with 0 parsed actions are `valid=False`
7. **File-locked JSONL writes** -- safe for concurrent Slurm array tasks
8. **Compact action parser rejects prose** -- prevents English text inflating compact validity
