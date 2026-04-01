"""Canonical normalization of PDDL strings.

Used by all transforms to ensure consistent comparison, deduplication,
and hashing across the pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path


def canonicalize_action(action: str) -> str:
    """Normalize a single action string to a canonical form.

    - Strip whitespace
    - Lowercase
    - Collapse internal whitespace
    - Ensure wrapped in parens
    - Remove trailing cost annotations
    """
    s = action.strip()

    # Remove cost annotation "[N.NNN]"
    if s.endswith("]") and "[" in s:
        s = s[: s.rfind("[")].strip()

    # Remove timed prefix "0.000: "
    if ":" in s and s.index(":") < s.find("(") if "(" in s else True:
        colon_idx = s.index(":")
        try:
            float(s[:colon_idx])
            s = s[colon_idx + 1:].strip()
        except ValueError:
            pass

    s = s.lower()

    # Ensure wrapped in parens
    if not s.startswith("("):
        s = "(" + s
    if not s.endswith(")"):
        s = s + ")"

    # Collapse whitespace inside
    s = re.sub(r"\s+", " ", s)
    return s


def canonicalize_plan(actions: list[str]) -> list[str]:
    """Canonicalize a list of action strings."""
    return [canonicalize_action(a) for a in actions if a.strip()]


def plan_text_to_canonical(text: str) -> list[str]:
    """Parse and canonicalize a plan from raw text."""
    actions = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        actions.append(canonicalize_action(line))
    return actions


def plans_are_equal(plan_a: list[str], plan_b: list[str]) -> bool:
    """Compare two plans up to canonical normalization."""
    return canonicalize_plan(plan_a) == canonicalize_plan(plan_b)
