"""Deterministic anonymization of PDDL domain/problem/plan triples.

Renames:
  - action names
  - predicate names
  - object identifiers (including type names)
  - variable names (?x -> ?v0, etc.)

while preserving PDDL keywords (define, domain, problem, :action, etc.)
and maintaining consistent mappings across the domain, problem, and plan.

The reversible mapping is saved as a JSON file alongside the anonymized files.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from utils.logging import get_logger

logger = get_logger(__name__)

# PDDL keywords that must NOT be renamed
_PDDL_KEYWORDS: frozenset[str] = frozenset(
    {
        "define", "domain", "problem", "requirements", "types", "predicates",
        "functions", "constants", "objects", "init", "goal", "metric",
        "action", "durative-action", "derived", "parameters", "precondition",
        "effect", "duration", "condition",
        ":domain", ":problem", ":requirements", ":types", ":predicates",
        ":functions", ":constants", ":objects", ":init", ":goal", ":metric",
        ":action", ":durative-action", ":derived", ":parameters",
        ":precondition", ":effect", ":duration", ":condition",
        "and", "or", "not", "forall", "exists", "when", "imply",
        "increase", "decrease", "assign", "at", "over", "all",
        "start", "end", "either",
        "=", "-", "+", "*", "/",
        # requirement flags
        ":strips", ":typing", ":equality", ":negative-preconditions",
        ":disjunctive-preconditions", ":existential-preconditions",
        ":universal-preconditions", ":conditional-effects",
        ":numeric-fluents", ":durative-actions", ":timed-initial-literals",
        ":preferences", ":constraints", ":action-costs",
        # common type names kept
        "object",
    }
)


class AnonymizationMapping:
    """Bidirectional mapping between original and anonymized names."""

    def __init__(self, seed: str = "") -> None:
        self._seed = seed
        self._orig_to_anon: dict[str, str] = {}
        self._anon_to_orig: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def _next_name(self, prefix: str) -> str:
        n = self._counters.get(prefix, 0)
        self._counters[prefix] = n + 1
        return f"{prefix}{n}"

    def get_or_create(self, original: str, prefix: str) -> str:
        """Return anonymized name, creating one deterministically if needed."""
        if original in _PDDL_KEYWORDS:
            return original
        if original not in self._orig_to_anon:
            anon = self._next_name(prefix)
            self._orig_to_anon[original] = anon
            self._anon_to_orig[anon] = original
        return self._orig_to_anon[original]

    def reverse(self, anon: str) -> str:
        """Return original name for an anonymized name."""
        return self._anon_to_orig.get(anon, anon)

    def to_dict(self) -> dict:
        return {
            "seed": self._seed,
            "orig_to_anon": dict(self._orig_to_anon),
            "anon_to_orig": dict(self._anon_to_orig),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnonymizationMapping":
        m = cls(seed=d.get("seed", ""))
        m._orig_to_anon = d["orig_to_anon"]
        m._anon_to_orig = d["anon_to_orig"]
        return m


# ---------------------------------------------------------------------------
# Regex-based token-level substitution
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Split PDDL text into tokens preserving whitespace/punctuation."""
    return re.split(r"(\s+|[();?])", text)


def _anonymize_text(
    text: str,
    mapping: AnonymizationMapping,
    *,
    is_variable_context: bool = False,
) -> str:
    """Apply anonymization mapping to a raw PDDL string."""
    tokens = _tokenize(text)
    result: list[str] = []
    for tok in tokens:
        if not tok or tok in (" ", "\t", "\n", "\r", "(", ")", ";"):
            result.append(tok)
            continue

        lower = tok.lower().strip()

        # Skip PDDL keywords
        if lower in _PDDL_KEYWORDS:
            result.append(tok)
            continue

        # Variables: ?varname
        if lower.startswith("?"):
            anon = mapping.get_or_create(lower, "?v")
            result.append(anon)
            continue

        # Skip numeric literals
        if _is_numeric(lower):
            result.append(tok)
            continue

        # Named tokens: actions, predicates, objects, types
        prefix = _choose_prefix(lower, mapping)
        anon = mapping.get_or_create(lower, prefix)
        result.append(anon)

    return "".join(result)


def _choose_prefix(name: str, mapping: AnonymizationMapping) -> str:
    """Heuristically choose a name prefix based on context."""
    # We don't have full parse context here, so use a single namespace
    # which still provides valid anonymization.
    return "sym"


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def anonymize_triple(
    domain_text: str,
    problem_text: str,
    plan_actions: list[str],
    instance_id: str,
) -> tuple[str, str, list[str], AnonymizationMapping]:
    """Anonymize a (domain, problem, plan) triple.

    Args:
        domain_text: Raw PDDL domain string.
        problem_text: Raw PDDL problem string.
        plan_actions: List of action strings like "(move a b)".
        instance_id: Used to seed the deterministic mapping.

    Returns:
        (anon_domain, anon_problem, anon_plan_actions, mapping)
    """
    mapping = AnonymizationMapping(seed=instance_id)

    anon_domain = _anonymize_text(domain_text, mapping)
    anon_problem = _anonymize_text(problem_text, mapping)
    anon_plan = [_anonymize_text(a, mapping) for a in plan_actions]

    return anon_domain, anon_problem, anon_plan, mapping


def save_anonymized_triple(
    anon_domain: str,
    anon_problem: str,
    anon_plan: list[str],
    mapping: AnonymizationMapping,
    output_dir: Path,
    instance_id: str,
) -> dict[str, Path]:
    """Write anonymized files and mapping JSON to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_path = output_dir / f"{instance_id}_anon_domain.pddl"
    problem_path = output_dir / f"{instance_id}_anon_problem.pddl"
    plan_path = output_dir / f"{instance_id}_anon_plan.pddl"
    mapping_path = output_dir / f"{instance_id}_anon_mapping.json"

    domain_path.write_text(anon_domain, encoding="utf-8")
    problem_path.write_text(anon_problem, encoding="utf-8")
    plan_path.write_text("\n".join(anon_plan) + "\n", encoding="utf-8")
    mapping_path.write_text(
        json.dumps(mapping.to_dict(), indent=2), encoding="utf-8"
    )

    logger.debug("Saved anonymized triple to %s", output_dir)
    return {
        "domain": domain_path,
        "problem": problem_path,
        "plan": plan_path,
        "mapping": mapping_path,
    }


def reverse_anonymize_plan(
    anon_plan_actions: list[str],
    mapping: AnonymizationMapping,
) -> list[str]:
    """Reverse anonymization on plan action strings using the mapping."""
    result = []
    for action in anon_plan_actions:
        tokens = _tokenize(action)
        restored = []
        for tok in tokens:
            if tok in ("(", ")", " ", ";", "\t", "\n"):
                restored.append(tok)
                continue
            lower = tok.lower().strip()
            if lower in _PDDL_KEYWORDS or _is_numeric(lower) or not tok.strip():
                restored.append(tok)
                continue
            orig = mapping.reverse(lower)
            restored.append(orig)
        result.append("".join(restored))
    return result
