from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from planning_pivot.shell import run_cmd


@dataclass(frozen=True)
class ValidationResult:
    domain_path: Path
    problem_path: Path
    plan_path: Path
    valid: bool
    returncode: int
    stdout: str
    stderr: str
    seconds: float
    failure_type: str
    first_bad_step: int | None = None
    goal_not_satisfied: bool | None = None


_VALID_RE = re.compile(r"Plan\s+valid", re.IGNORECASE)
_INVALID_RE = re.compile(r"Plan\s+(invalid|not\s+valid)", re.IGNORECASE)
_GOAL_NOT_RE = re.compile(r"Goal\s+(not\s+satisfied|not\s+reached)", re.IGNORECASE)
_GOAL_OK_RE = re.compile(r"Goal\s+(satisfied|reached)", re.IGNORECASE)
_STEP_RE = re.compile(r"(?:step|action)\s*(\d+)", re.IGNORECASE)
_SYNTAX_RE = re.compile(r"syntax\s+error|parse\s+error|unexpected\s+token", re.IGNORECASE)
_PRECOND_RE = re.compile(
    r"precondition|pre-condition|not\s+applicable|cannot\s+apply", re.IGNORECASE
)
_TYPE_RE = re.compile(r"type\s+error|type\s+mismatch|wrong\s+type", re.IGNORECASE)
_ARITY_RE = re.compile(r"wrong\s+number|arity|wrong\s+arity|incorrect.*param", re.IGNORECASE)
_OBJECT_RE = re.compile(r"unknown\s+object|undefined\s+object|object\s+not\s+found", re.IGNORECASE)
_MUTEX_RE = re.compile(r"mutex|invariant", re.IGNORECASE)
_INVALID_ACTION_RE = re.compile(
    r"unknown\s+action|undefined\s+action|invalid\s+action|action\s+not\s+found", re.IGNORECASE
)


def classify_val_failure(
    stdout: str, stderr: str, returncode: int
) -> tuple[str, int | None, bool | None]:
    combined = stdout + "\n" + stderr

    if _VALID_RE.search(combined) and not _INVALID_RE.search(combined):
        goal_sat = not bool(_GOAL_NOT_RE.search(combined))
        return "valid", None, goal_sat

    if returncode == -1:
        return "timeout", None, None

    if returncode < -1 or returncode > 128:
        return "validator_crash", None, None

    first_bad: int | None = None
    m = _STEP_RE.search(combined)
    if m:
        first_bad = int(m.group(1))

    goal_not_sat: bool | None = None
    if _GOAL_NOT_RE.search(combined):
        goal_not_sat = True
    elif _GOAL_OK_RE.search(combined):
        goal_not_sat = False

    if _SYNTAX_RE.search(combined):
        return "syntax_error", first_bad, goal_not_sat
    if _INVALID_ACTION_RE.search(combined):
        return "invalid_action", first_bad, goal_not_sat
    if _ARITY_RE.search(combined):
        return "wrong_arity", first_bad, goal_not_sat
    if _OBJECT_RE.search(combined):
        return "object_error", first_bad, goal_not_sat
    if _TYPE_RE.search(combined):
        return "type_error", first_bad, goal_not_sat
    if _PRECOND_RE.search(combined):
        return "precondition_failure", first_bad, goal_not_sat
    if _MUTEX_RE.search(combined):
        return "mutex_or_invariant_failure", first_bad, goal_not_sat
    if goal_not_sat is True and first_bad is None:
        return "goal_not_satisfied", None, True

    return "unknown_invalid", first_bad, goal_not_sat


def validate_plan(
    domain_path: Path,
    problem_path: Path,
    plan_path: Path,
    val_bin: Path,
    timeout: int = 30,
) -> ValidationResult:
    cmd = [str(val_bin), str(domain_path), str(problem_path), str(plan_path)]
    result = run_cmd(cmd, timeout=timeout)

    failure_type, first_bad, goal_not_sat = classify_val_failure(
        result.stdout, result.stderr, result.returncode
    )
    valid = failure_type == "valid"

    return ValidationResult(
        domain_path=domain_path,
        problem_path=problem_path,
        plan_path=plan_path,
        valid=valid,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        seconds=result.seconds,
        failure_type=failure_type,
        first_bad_step=first_bad,
        goal_not_satisfied=goal_not_sat,
    )
