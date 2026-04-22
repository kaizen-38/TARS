# Handoff to Rhythm — Phase 1 Pipeline

## Pipeline Commands (run in order from ~/TARS)
conda activate thicket
make generate-smoke   # Step 1: generate instances
make solve-smoke      # Step 2: solve with FD + validate with VAL
make build-dataset    # Step 3: build Alpaca JSONL datasets
make test             # Step 4: verify 65/65 tests pass
## To Scale to Full Run
In configs/splits/phase1_v1.yaml change:
  smoke_test.instances_per_domain: 5  →  pilot.instances_per_domain: 25
Then run the same make targets.

## Key Fixes Made (relevant for Sol setup)
1. Fast Downward writes plan to sas_plan file in CWD — pipeline reads from there
2. Search config must not be split on spaces: use --search "lazy_greedy(...)"
3. VAL binary is at third_party/VAL/bin/Validate (capital V)
4. Delivery domain uses pre-generated static instances in data/static/delivery/train/ — no uv required on Sol


## Environment
- conda env: thicket (Python 3.10)
- All paths relative to ~/TARS/
- third_party/ submodules must be cloned and built (make build-tools)

## Verified Locally
- make generate-smoke ✅ (60 instances, all 12 domains)
- make solve-smoke ✅ (all valid=True goal=True)
- make build-dataset ✅ (39/38/39 rows)
- make test ✅ (65/65 passed)
