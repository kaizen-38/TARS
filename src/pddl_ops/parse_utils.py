"""PDDL parsing utilities using the `pddl` Python package.

All transforms in thicket_phase1 operate on these canonical parsed
representations rather than on raw strings, ensuring consistent handling
of whitespace, casing, and quoting.

Depends on: pddl>=0.4.0  (pip install pddl)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Canonical in-memory representation of a plan
# ---------------------------------------------------------------------------

@dataclass
class ParsedAction:
    """A single action application in a plan."""
    name: str                    # action name (lower-cased)
    args: list[str]              # object names (lower-cased)
    timestamp: Optional[float]   # None for sequential plans

    def to_pddl(self) -> str:
        """Render as standard PDDL action string: (name arg1 arg2 ...)"""
        parts = [self.name] + self.args
        return "(" + " ".join(parts) + ")"

    def to_timed_pddl(self) -> str:
        """Render as timed plan line: N.000: (name arg1 ...) [1.000]"""
        ts = self.timestamp if self.timestamp is not None else 0.0
        return f"{ts:.3f}: {self.to_pddl()} [1.000]"


@dataclass
class ParsedPlan:
    """A sequence of parsed actions."""
    actions: list[ParsedAction]

    def to_pddl_lines(self) -> list[str]:
        return [a.to_pddl() for a in self.actions]

    def to_timed_pddl_lines(self) -> list[str]:
        return [a.to_timed_pddl() for a in self.actions]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_domain(path: Path):
    """Parse a PDDL domain file using the `pddl` package.

    Returns a pddl.Domain object (or equivalent).
    """
    try:
        import pddl  # type: ignore
        return pddl.parse_domain(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to parse domain {path}: {exc}") from exc


def parse_problem(path: Path):
    """Parse a PDDL problem file using the `pddl` package."""
    try:
        import pddl  # type: ignore
        return pddl.parse_problem(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to parse problem {path}: {exc}") from exc


def parse_plan(path: Path) -> ParsedPlan:
    """Parse a plan file into a ParsedPlan.

    Supports:
    - Timed plans:    0.000: (move a b) [1.000]
    - Sequential:     (move a b)
    - FD style:       (move a b) ; cost = ...
    """
    actions: list[ParsedAction] = []
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        action = _parse_plan_line(line)
        if action is not None:
            actions.append(action)
    return ParsedPlan(actions=actions)


def parse_plan_from_text(text: str) -> ParsedPlan:
    """Parse plan from a string (e.g. model output)."""
    actions: list[ParsedAction] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        action = _parse_plan_line(line)
        if action is not None:
            actions.append(action)
    return ParsedPlan(actions=actions)


def _parse_plan_line(line: str) -> Optional[ParsedAction]:
    """Parse a single plan line into a ParsedAction."""
    # Strip trailing comments
    if ";" in line:
        line = line[: line.index(";")].strip()

    timestamp: Optional[float] = None

    # Timed plan format: "0.000: (action ...) [1.000]"
    if ":" in line and "(" in line:
        colon_idx = line.index(":")
        try:
            timestamp = float(line[:colon_idx].strip())
            line = line[colon_idx + 1 :].strip()
        except ValueError:
            pass

    # Remove trailing cost annotation "[N.NNN]"
    if line.endswith("]") and "[" in line:
        line = line[: line.rfind("[")].strip()

    # Extract (action arg1 arg2 ...)
    if not (line.startswith("(") and line.endswith(")")):
        return None

    inner = line[1:-1].strip().lower()
    tokens = inner.split()
    if not tokens:
        return None

    return ParsedAction(name=tokens[0], args=tokens[1:], timestamp=timestamp)


def plan_to_file(plan: ParsedPlan, path: Path, *, timed: bool = False) -> None:
    """Write a ParsedPlan to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = plan.to_timed_pddl_lines() if timed else plan.to_pddl_lines()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
