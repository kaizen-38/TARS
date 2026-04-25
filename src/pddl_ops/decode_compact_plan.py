"""Decode a compact plan back to validator-compatible standard PDDL.

The compact form (one action per line, no parens/timestamps) is
deterministically reconstructed to the standard form:
  (action-name arg1 arg2 ...)

This is used before passing model output to VAL for validation.
"""
from __future__ import annotations

from pathlib import Path

from pddl_ops.parse_utils import ParsedAction, ParsedPlan, plan_to_file
from utils.logging import get_logger

logger = get_logger(__name__)


def decode_compact_plan(compact_text: str) -> ParsedPlan:
    """Parse compact plan text into a ParsedPlan.

    Args:
        compact_text: One action per line, e.g.
            move-block a b
            pick-up c

    Returns:
        ParsedPlan that can be written to standard PDDL with plan_to_file().
    """
    actions: list[ParsedAction] = []
    for raw_line in compact_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        tokens = line.lower().split()
        if not tokens:
            continue
        actions.append(ParsedAction(name=tokens[0], args=tokens[1:], timestamp=None))
    return ParsedPlan(actions=actions)


def decode_compact_plan_file(compact_path: Path) -> ParsedPlan:
    """Read a compact plan file and decode it."""
    text = compact_path.read_text(encoding="utf-8")
    return decode_compact_plan(text)


def compact_to_standard_file(compact_path: Path, output_path: Path) -> ParsedPlan:
    """Read compact plan, write standard PDDL plan, return ParsedPlan.

    The output file is suitable for passing to VAL's Validate binary.
    """
    plan = decode_compact_plan_file(compact_path)
    plan_to_file(plan, output_path, timed=False)
    logger.debug(
        "Decoded compact plan %s -> %s (%d actions)",
        compact_path,
        output_path,
        len(plan.actions),
    )
    return plan


def extract_compact_plan_from_text(model_output: str) -> str:
    """Extract the plan portion from raw model output.

    Handles common model output patterns:
    - Plans wrapped in ```...``` code blocks
    - Plans following "Plan:" or "Solution:" markers
    - Bare action lists

    Returns the cleaned compact plan text (may be empty if nothing found).
    """
    text = model_output.strip()

    # Try code block extraction
    if "```" in text:
        blocks = text.split("```")
        # blocks[1], [3], ... are inside code fences
        for i in range(1, len(blocks), 2):
            block = blocks[i].strip()
            # Remove language identifier if present
            lines = block.splitlines()
            if lines and not lines[0].strip().startswith("(") and not _looks_like_action(lines[0]):
                block = "\n".join(lines[1:]).strip()
            if block:
                return block

    # Try "Plan:" or "Solution:" marker
    for marker in ("plan:", "solution:", "answer:"):
        lower = text.lower()
        idx = lower.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()

    # Fall back to extracting lines that look like actions
    action_lines = [
        line.strip()
        for line in text.splitlines()
        if _looks_like_action(line.strip())
    ]
    return "\n".join(action_lines)


_PROSE_WORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "for", "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most", "other",
    "some", "such", "no", "only", "own", "same", "than", "too", "very",
    "just", "because", "as", "until", "while", "of", "at", "by", "about",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "what", "which", "who", "whom", "if", "it", "i",
    "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "step", "plan", "solution",
    "first", "second", "third", "next", "finally", "note", "output",
})


def _looks_like_action(line: str) -> bool:
    """Heuristic: does this line look like a compact PDDL action?"""
    if not line:
        return False
    if line.startswith("("):
        return True
    tokens = line.split()
    if not tokens:
        return False
    first = tokens[0].lower().rstrip(":.,;")
    if not first:
        return False
    if first in _PROSE_WORDS:
        return False
    if not all(c.isalnum() or c in "-_" for c in first):
        return False
    # Actions must contain a hyphen or have at least one argument
    if "-" in first or len(tokens) >= 2:
        return True
    return False
