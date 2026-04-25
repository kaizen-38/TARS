# TARS — Thicket-Aware Regime Switching for Verifier-Grounded LLM Planning

**Phase 1: Reproduce the 1.7B LLM Planning Baseline**

Team: Rhythm Arya (rarya124), Mohak Rathod (mrathod4), Abhiram Menon (amenon28)
Course: CSE-574, ASU | Branch: `mohak/sol-fd-fix`

> Phase 1 goal: lock splits/budgets/protocol, reproduce Qwen3-1.7B planning results
> across standard, anonymized, and compact PDDL representations, and build a
> trustworthy data + validation pipeline on ASU Sol.
> Regime switching is NOT implemented in Phase 1.

---

## Phase 1 Baseline Results (Pilot Run)

| Split | Validity | Goal Rate | Instances |
|-------|----------|-----------|-----------|
| Train (8 domains) | 4.2% | 0% | 25/domain |
| Heldout (4 domains) | 1.4% | 0% | 25/domain |
| Generalization gap | 2.8% | 0% | — |

By representation (train):
| Representation | Validity |
|----------------|----------|
| Standard | 1.0% |
| Anonymized | 1.0% |
| Compact | 10.5% |

Model: Qwen3-1.7B + mini LoRA (~79 training tuples)

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
eval/greedy_eval.yaml               # Greedy eval config (max_new_tokens=512)
data/
generated/instances/                # PDDL domain+problem files
generated/plans/                    # Solved plans + VAL results
generated/tuples_standard/          # (domain, problem, plan) tuples
datasets/alpaca/                    # Alpaca JSONL datasets (split-suffixed)
static/delivery/                    # Pre-generated delivery instances
src/
cli.py                              # Unified CLI: generate-one, build-tuples, build-dataset, train, eval
generation/                         # Instance generation, FD solving, VAL validation
pddl_ops/                           # Anonymization, compact serialization
dataset/                            # SFT dataset builder
training/                           # LLaMAFactory config + launch
inference/                          # Plan generation + greedy eval
eval/                               # Metrics aggregation
slurm/                                # Sbatch scripts (00-10)
scripts/
submit_phase1.sh                    # Full pipeline submission
gen_manifests.py                    # TSV manifest generator
docs/
domain_generator_notes.md           # Generator commands and known issues
third_party/                          # Git submodules
runs/                                 # Training + eval outputs (gitignored)
```
---

## Sol Setup (One-time)

```bash
# Load environment
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate thicket311
export PYTHONPATH=$HOME/TARS/src
export LD_LIBRARY_PATH=/home/mrathod4/.conda/envs/thicket311/lib:${LD_LIBRARY_PATH:-}
export HF_HOME="${SCRATCH}/hf_cache"
```

**Important:** VAL and Fast Downward require `LD_LIBRARY_PATH` set as above.
This is injected automatically in all sbatch scripts.

---

## Running the Pipeline on Sol

### Smoke + Pilot (recommended)
```bash
cd ~/TARS
bash scripts/submit_phase1.sh --pilot
```

After pilot completes, manually submit train eval:
```bash
JID=$(sbatch --array=0-7 slurm/07b_eval_train_gpu_array.sbatch | awk '{print $NF}')
echo "train eval: $JID"
```

Aggregate train results:
```bash
PYTHONPATH=src python3 src/eval/aggregate_results.py runs/eval_pilot_train/run_log.jsonl
```

### Pipeline stages
| Job | What | Depends on |
|-----|------|-----------|
| 00_generate | Generate instances (array 12) | — |
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

## Known Issues and Fixes Applied

| Issue | Fix |
|-------|-----|
| FD writes `output.sas` to cwd causing parallel collisions | `cwd=tmpdir` in `solve_with_fd.py` |
| VAL/FD `GLIBCXX_3.4.32` not found | `LD_LIBRARY_PATH` injected in all sbatch and subprocess calls |
| Domain paths wrong for non-blocksworld domains | Fixed in `generate_instances.py` — only blocksworld uses `4ops/domain.pddl` |
| Dataset builder overwrites train when building heldout | Output files now split-suffixed: `phase1_standard_train.jsonl` |
| `build-tuples` not in sbatch chain | `02b_build_tuples_cpu.sbatch` added between 02 and 03 |
| `python -m cli` not found | Fixed to `python -m src.cli` in sbatch scripts |
| VAL `goal_reached` always True | Fixed regex in `validate_with_val.py` |
| `--domain` filter missing from eval script | Added to `run_greedy_eval.py` |
| Train domains never evaluated in pilot | `07b_eval_train_gpu_array.sbatch` added |

---

## Dataset Files
```
After `build-dataset`, files are split-suffixed:
data/datasets/alpaca/
phase1_standard_train.jsonl      # ~79 rows (pilot)
phase1_anonymized_train.jsonl
phase1_compact_train.jsonl
phase1_standard_heldout.jsonl    # ~36 rows (pilot)
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
| `third_party/PlanBench` | Reference only — no code imported |

---

## Key Design Decisions

1. **`enable_thinking=False` always** — hard Phase 1 requirement
2. **VAL is single source of truth** — no heuristic plan checking
3. **LD_LIBRARY_PATH workaround** — VAL and FD built against newer libstdc++ than system provides; conda env lib injected at runtime
4. **Delivery uses static instances** — `uv` not available on Sol; pre-generated in `data/static/delivery/`
5. **Split-suffixed datasets** — `phase1_standard_train.jsonl` not `phase1_standard.jsonl` to prevent overwrite on second split build
