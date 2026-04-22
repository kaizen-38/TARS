# Handoff to Ram — Phase 1 Dataset

## Dataset Location
data/datasets/alpaca/

## Files
- phase1_standard.jsonl   — 39 rows (standard PDDL actions with parens)
- phase1_anonymized.jsonl — 38 rows (all identifiers replaced with sym0, sym1, ...)
- phase1_compact.jsonl    — 39 rows (actions without parens, one per line)
- dataset_info.json       — LLaMAFactory dataset registry

## Format (Alpaca JSONL)
Each row has: instruction, input, output, system
- input: "DOMAIN:\n<pddl domain>\n\nPROBLEM:\n<pddl problem>"
- output: plan actions (format varies by representation)

## Representation Examples (blocksworld instance)
Standard output:  "(unstack b3 b6)\n(putdown b3)\n..."
Anonymized output: "(sym14 sym18 sym21)\n(sym11 sym18)\n..."
Compact output:   "unstack b3 b6\nputdown b3\n..."

## Splits
- Train domains (8): blocksworld, gripper, ferry, delivery, childsnack, floortile, rovers, spanner
- Held-out domains (4): miconic, sokoban, transport, satellite
- All 12 domains represented in training set (held-out used for eval only)

## Validation
All instances externally verified by VAL (valid=True, goal=True).
Deduplication applied — a small number of near-duplicate instances removed.

## Tuple files (raw, if needed)
data/generated/tuples_standard/*_tuple.json
Each tuple contains: domain_text, problem_text, plan_actions,
anon_domain_text, anon_problem_text, anon_plan_actions, compact_plan


 