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

**IMPORTANT:** Clone to `/scratch/<username>/TARS` not `~/TARS` — home directories are quota-limited and will fill up with generated data.

### 1. Clone and setup environment
```bash
cd /scratch/$USER
git clone https://github.com/kaizen-38/TARS.git
cd TARS
git checkout mohak/sol-fd-fix

# Create conda environment
module load mamba/latest
eval "$(conda shell.bash hook)"
conda create -n tars python=3.11 -y
conda activate tars

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

### 2. Initialize submodules
```bash
# Clone all third-party dependencies
git submodule update --init --recursive

# Install LLaMAFactory
pip install -e third_party/LLaMAFactory
```

### 3. Build Fast Downward
```bash
module load gcc/15.2.0
cd third_party/downward
python build.py -j2
cd ../..
```

### 4. Build VAL
```bash
module load gcc/15.2.0
cd third_party/VAL
mkdir -p build && cd build
cmake .. -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_C_COMPILER=$(which gcc)
make -j2
cd ../../..
```

### 5. Build pddl-generators
```bash
cd third_party/pddl-generators

# Compile C/C++ generators
for d in blocksworld gripper ferry miconic rovers satellite; do
  echo "Building $d..."
  cd $d && make -j2 && cd ..
done

# Sokoban is in a subdirectory
cd sokoban/random
make -j2
cd ../../..
```

### 6. Verify everything works
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
third_party/downward/fast-downward.py --help | head -3
third_party/VAL/build/bin/Validate 2>&1 | head -3
third_party/pddl-generators/blocksworld/blocksworld 4 3 42 | head -3
```

### 7. Set environment variables
Add to `~/.bashrc`:
```bash
export PYTHONPATH=/scratch/$USER/TARS/src
export HF_HOME="${SCRATCH}/hf_cache"
```

Then reload: `source ~/.bashrc`

---

## Running the Pipeline on Sol

### Smoke (fast sanity check, ~2 GPU-hours)
```bash
cd /scratch/$USER/TARS
conda activate tars
bash scripts/submit_phase1.sh
```

### Pilot (25 inst/domain, ~16 GPU-hours)
```bash
cd /scratch/$USER/TARS
conda activate tars

# If smoke already completed, run pilot-only
bash scripts/submit_pilot_only.sh

# Or re-run full chain (smoke + pilot)
bash scripts/submit_phase1.sh --pilot
```

### Monitor jobs
```bash
squeue -u $USER | grep tars
```

### Check results after completion
```bash
# Pilot results (heldout domains)
cat logs/10_aggregate_pilot_*.out

# If you ran train eval separately
cat logs/10_aggregate_*.out | grep -A20 "Domain breakdown"
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

## Troubleshooting

### Home directory full
If you get "No space left on device" errors:
```bash
# Clean up history spam
rm -rf ~/.hpc_history
unset HISTFILE

# Check space
du -sh ~/TARS
df -h ~

# Move to scratch if needed
mv ~/TARS /scratch/$USER/TARS
ln -s /scratch/$USER/TARS ~/TARS
```

### Jobs stuck in `Priority`
This is normal — cluster is busy. Jobs will start when capacity frees up.

### Jobs failed with `DependencyNeverSatisfied`
An upstream job failed. Check logs:
```bash
ls -lt logs/ | head -20
cat logs/<failed_job>_*.err
```

### VAL not found
Binary is at `third_party/VAL/build/bin/Validate` (not `third_party/VAL/build/Validate`). The code checks both locations.

---

## Key Design Decisions

1. **`enable_thinking=False` always** — hard Phase 1 requirement
2. **VAL is single source of truth** — no heuristic plan checking
3. **LD_LIBRARY_PATH via $CONDA_PREFIX** — portable across all team members
4. **Delivery uses static instances** — `uv` not available on Sol
5. **Split-suffixed datasets** — prevents overwrite on second split build
6. **Empty plans rejected before VAL** — plans with 0 parsed actions are `valid=False`
7. **File-locked JSONL writes** — safe for concurrent Slurm array tasks
8. **Compact action parser rejects prose** — prevents English text inflating compact validity
9. **Clone to `/scratch/` not `~/`** — home directories quota out with generated data
