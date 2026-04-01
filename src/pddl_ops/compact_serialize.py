"""Compact plan serialization.

Converts a PDDL plan to a compact, line-based action form for use as
LLM training targets.  The compact form:
  - Removes parentheses
  - Removes timestamps and cost annotations
  - Lowercases everything
  - One action per line

Example:
  Standard:   (move-block a b)
  Compact:    move-block a b

The compact form is ONLY for the training/inference string representation.
`decode_compact_plan.py` reconstructs the standard PDDL form for VAL.
"""
from __future__ import annotations

from pathlib import Path

from pddl_ops.parse_utils import ParsedPlan, ParsedAction


def plan_to_compact(plan: ParsedPlan) -> str:
    """Convert a ParsedPlan to compact string (one action per line)."""
    lines = []
    for action in plan.actions:
        line = action.name
        if action.args:
            line += " " + " ".join(action.args)
        lines.append(line)
    return "\n".join(lines)


def actions_to_compact(action_strings: list[str]) -> str:
    """Convert a list of PDDL action strings to compact form.

    Strips outer parens, timestamps, and cost annotations.

    Args:
        action_strings: e.g. ["(move a b)", "(pick-up c)"]

    Returns:
        Compact multiline string.
    """
    lines = []
    for action in action_strings:
        compact = _compact_one(action)
        if compact:
            lines.append(compact)
    return "\n".join(lines)


def _compact_one(action: str) -> str:
    """Convert a single action string to compact form."""
    line = action.strip()

    # Strip trailing comments ("; ...")
    if ";" in line:
        line = line[: line.index(";")].strip()

    # Remove timed prefix "0.000: "
    if ":" in line:
        colon_idx = line.index(":")
        try:
            float(line[:colon_idx])
            line = line[colon_idx + 1:].strip()
        except ValueError:
            pass

    # Remove trailing cost "[N.NNN]"
    if line.endswith("]") and "[" in line:
        line = line[: line.rfind("[")].strip()

    # Remove outer parens
    if line.startswith("(") and line.endswith(")"):
        line = line[1:-1].strip()

    return line.lower()


def save_compact_plan(compact_text: str, path: Path) -> None:
    """Write a compact plan string to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(compact_text + "\n", encoding="utf-8")
