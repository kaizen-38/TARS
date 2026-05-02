# CSE-574 LLM Planning Pivot Instructions

This repo is a verifier-grounded LLM planning project.

The original SFT/GRPO/thicket direction failed because Qwen3-1.7B produced 0% VAL-valid plans after SFT. Do not revive SFT, GRPO, TRL, LoRA training, or adapter-space search unless explicitly requested.

Current implementation target:

1. VAL-grounded failure diagnostics.
2. Fast Downward SAS parser and simulator.
3. Legality-constrained action search.
4. Prefix-salvage repair.
5. Budgeted portfolio selector.

VAL is the final correctness authority. Fast Downward SAS simulation is used only for cheap legality checks, prefix analysis, and search.

Always preserve train/heldout domain separation.

Main commands:

```bash
python -m planning_pivot.cli audit
python -m planning_pivot.cli build-index
python -m planning_pivot.cli cache-sas
python -m planning_pivot.cli diagnose
python -m planning_pivot.cli constrained-search
python -m planning_pivot.cli repair
python -m planning_pivot.cli portfolio
python -m planning_pivot.cli report
pytest -q
```

Do not use shell=True in subprocess calls unless there is no alternative.

Do not hardcode absolute paths. Use config files, environment variables, and discovery helpers.

Every experiment writes CSV/JSON artifacts under results/.

Every failure should produce a row, not a crash.

Preferred research framing:

"When small-LLM SFT collapses as autonomous PDDL plan generation, verifier-grounded structure can still produce useful planning research through failure cartography, legality-constrained search, repair, and mode selection."
