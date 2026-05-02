# Verifier-Grounded LLM Planning: From SFT Failure to Structural Recovery

## 1. Original Hypothesis and Failed SFT Assumption

We hypothesized that supervised fine-tuning of Qwen3-1.7B on teacher-generated PDDL plans
would produce a capable autonomous planner. Training loss decreased normally, but
VAL-verified plan validity remained at 0%.

## 2. Diagnostic Evidence

Low training loss did not imply VAL validity. See `results/diagnostics/` for full breakdown.

## 3. Failure Taxonomy

Failures categorized by representation, model, and domain. See figures in `results/figures/`.

## 4. Legality-Constrained Search

Using Fast Downward SAS representation to enumerate only legal actions, we evaluated
beam search with multiple rankers. See `results/constrained_search/`.

## 5. Prefix-Salvage Repair

Invalid LLM-generated plans are truncated at the first illegal action, then completed
using constrained search. See `results/repair/`.

## 6. Portfolio Selector

A lightweight classifier predicts which planning mode (ranker + beam width) to use
per instance, based on structural features. See `results/portfolio/`.

## 7. Heldout-Domain Results

Results on heldout domains (miconic, sokoban, transport, satellite) test generalization.

## 8. Limitations

- SAS simulator does not support axioms.
- Repair only helps when plans have nonzero executable prefixes.
- Portfolio selector trained on small feature set.
- No comparison against state-of-the-art LLM planners.
