from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ACTION_RE = re.compile(r"^\s*\(?\s*([A-Za-z0-9_\-]+)(.*?)\)?\s*(?:;.*)?$")
_STRIP_PREFIX_RE = re.compile(r"^(\d+[\.\)\]:]\s*|[-*]\s+|step\s*\d*[:\s]*)", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```.*$", re.MULTILINE)
_PLAN_HEADER_RE = re.compile(r"^(plan|actions|solution|steps)\s*:", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class PlanAction:
    name: str
    args: tuple[str, ...]
    raw: str


@dataclass(frozen=True)
class ExtractedPlan:
    actions: list[PlanAction]
    raw_text: str
    normalized_text: str
    surface_status: str
    notes: list[str] = field(default_factory=list)


def normalize_action_line(line: str) -> str:
    line = line.strip()
    line = _STRIP_PREFIX_RE.sub("", line).strip()
    line = re.sub(r";.*$", "", line).strip()
    line = line.strip("()")
    line = line.strip()
    if not line:
        return ""
    parts = line.lower().split()
    return "(" + " ".join(parts) + ")"


def extract_plan_text(raw_output: str) -> str:
    fences = list(_FENCE_RE.finditer(raw_output))
    if len(fences) >= 2:
        start = fences[0].end()
        end = fences[1].start()
        return raw_output[start:end].strip()

    m = _PLAN_HEADER_RE.search(raw_output)
    if m:
        return raw_output[m.end():].strip()

    return raw_output.strip()


def parse_plan_actions(plan_text: str) -> ExtractedPlan:
    notes: list[str] = []

    if not plan_text or not plan_text.strip():
        return ExtractedPlan(
            actions=[], raw_text=plan_text or "", normalized_text="",
            surface_status="empty_plan", notes=["Empty or whitespace-only input"],
        )

    open_parens = plan_text.count("(")
    close_parens = plan_text.count(")")
    if abs(open_parens - close_parens) > max(open_parens, close_parens) * 0.5 and open_parens > 0:
        notes.append(f"Unbalanced parentheses: {open_parens} open vs {close_parens} close")

    actions: list[PlanAction] = []
    normalized_lines: list[str] = []
    non_action_lines = 0

    for raw_line in plan_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("#"):
            continue

        has_parens = "(" in stripped

        norm = normalize_action_line(stripped)
        if not norm or norm == "()":
            non_action_lines += 1
            continue

        inner = norm[1:-1]
        parts = inner.split()
        if not parts:
            non_action_lines += 1
            continue

        action_name = parts[0]
        if not re.match(r"^[a-z0-9_\-]+$", action_name):
            non_action_lines += 1
            continue

        if not has_parens and len(parts) < 2:
            non_action_lines += 1
            continue

        if not has_parens and len(action_name) > 30:
            non_action_lines += 1
            continue

        args = tuple(parts[1:])

        if not has_parens and any(not re.match(r"^[a-z0-9_\-]+$", a) for a in args):
            non_action_lines += 1
            continue
        actions.append(PlanAction(name=action_name, args=args, raw=stripped))
        normalized_lines.append(norm)

    normalized_text = "\n".join(normalized_lines)

    if not actions and non_action_lines > 0:
        status = "non_pddl_text"
    elif not actions:
        status = "empty_plan"
    elif "Unbalanced parentheses" in " ".join(notes):
        status = "unbalanced_parentheses"
        notes.append(f"Parsed {len(actions)} actions despite parenthesis imbalance")
    else:
        status = "parsed_actions"

    return ExtractedPlan(
        actions=actions,
        raw_text=plan_text,
        normalized_text=normalized_text,
        surface_status=status,
        notes=notes,
    )


def write_plan_file(actions: list[PlanAction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"({a.name} {' '.join(a.args)})" if a.args else f"({a.name})" for a in actions]
    path.write_text("\n".join(lines) + "\n")


def read_plan_file(path: Path) -> ExtractedPlan:
    text = path.read_text()
    return parse_plan_actions(text)
