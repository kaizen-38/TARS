"""Robust wrapper around KCL-Planning/VAL (the Validate binary).

Returns a structured VALResult with:
  success, exit_code, wall_clock_sec, stdout, stderr,
  parsed_validity, parsed_goal_reached
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from utils.logging import get_logger
from utils.io import ensure_dir, dump_json

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VAL_ROOT = _REPO_ROOT / "third_party" / "VAL"


@dataclass
class VALResult:
    problem_id: str
    domain_file: str
    problem_file: str
    plan_file: str
    success: bool               # True if VAL ran without error
    exit_code: int
    wall_clock_sec: float
    stdout: str
    stderr: str
    parsed_validity: Optional[bool]     # True = plan is valid
    parsed_goal_reached: Optional[bool] # True = goal is satisfied
    error: Optional[str] = None


# Patterns in VAL's output
_VALID_RE = re.compile(r"Plan\s+valid", re.IGNORECASE)
_INVALID_RE = re.compile(r"Plan\s+(invalid|not valid)", re.IGNORECASE)
_GOAL_RE = re.compile(r"Goal\s+(not satisfied|not reached|reached|satisfied)", re.IGNORECASE)


def _find_val_binary() -> Path:
    """Locate the Validate binary built from KCL-Planning/VAL."""
    candidates = [
        _VAL_ROOT / "build" / "bin" / "Validate",
        _VAL_ROOT / "bin" / "Validate",
        _VAL_ROOT / "Validate",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError(
        f"VAL 'Validate' binary not found under {_VAL_ROOT}. "
        "Run `make build-tools` (scripts/build_val.sh)."
    )


def _parse_val_output(stdout: str) -> tuple[Optional[bool], Optional[bool]]:
    """Parse VAL output for validity and goal-reached flags."""
    validity: Optional[bool] = None
    goal_reached: Optional[bool] = None

    if _VALID_RE.search(stdout):
        validity = True
    elif _INVALID_RE.search(stdout):
        validity = False

    m = _GOAL_RE.search(stdout)
    if m:
        matched = m.group(0).lower()
        goal_reached = "not reached" not in matched and "not satisfied" not in matched

    return validity, goal_reached


def validate_plan(
    problem_id: str,
    domain_file: Path,
    problem_file: Path,
    plan_file: Path,
    output_dir: Optional[Path] = None,
    timeout: int = 30,
) -> VALResult:
    """Run VAL's Validate binary on a plan.

    Args:
        problem_id: Used to name the output metadata file.
        domain_file: PDDL domain file.
        problem_file: PDDL problem file.
        plan_file: Plan file (one action per line, PDDL format).
        output_dir: If provided, write VALResult JSON here.
        timeout: Timeout in seconds.

    Returns:
        VALResult with all fields populated.
    """
    try:
        val_bin = _find_val_binary()
    except RuntimeError as exc:
        return VALResult(
            problem_id=problem_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            plan_file=str(plan_file),
            success=False,
            exit_code=-3,
            wall_clock_sec=0.0,
            stdout="",
            stderr="",
            parsed_validity=None,
            parsed_goal_reached=None,
            error=str(exc),
        )

    cmd = [str(val_bin), str(domain_file), str(problem_file), str(plan_file)]
    logger.debug("VAL: %s", " ".join(cmd))

    t0 = time.perf_counter()
    try:
        val_env = os.environ.copy()
        conda_lib = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")
        val_env["LD_LIBRARY_PATH"] = conda_lib + ":" + val_env.get("LD_LIBRARY_PATH", "")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=val_env)
        elapsed = time.perf_counter() - t0

        validity, goal_reached = _parse_val_output(result.stdout)
        val_result = VALResult(
            problem_id=problem_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            plan_file=str(plan_file),
            success=result.returncode == 0,
            exit_code=result.returncode,
            wall_clock_sec=elapsed,
            stdout=result.stdout,
            stderr=result.stderr,
            parsed_validity=validity,
            parsed_goal_reached=goal_reached,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        val_result = VALResult(
            problem_id=problem_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            plan_file=str(plan_file),
            success=False,
            exit_code=-1,
            wall_clock_sec=elapsed,
            stdout="",
            stderr="",
            parsed_validity=None,
            parsed_goal_reached=None,
            error=f"VAL timeout after {timeout}s",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        val_result = VALResult(
            problem_id=problem_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            plan_file=str(plan_file),
            success=False,
            exit_code=-2,
            wall_clock_sec=elapsed,
            stdout="",
            stderr="",
            parsed_validity=None,
            parsed_goal_reached=None,
            error=str(exc),
        )

    if output_dir is not None:
        ensure_dir(output_dir)
        dump_json(asdict(val_result), output_dir / f"{problem_id}.val_result.json")

    log_fn = logger.info if val_result.parsed_validity else logger.warning
    log_fn(
        "VAL %s: valid=%s goal=%s time=%.2fs",
        problem_id,
        val_result.parsed_validity,
        val_result.parsed_goal_reached,
        val_result.wall_clock_sec,
    )
    if val_result.parsed_validity is None and val_result.success:
        logger.warning(
            "VAL %s: could not parse validity from stdout (exit=%d). "
            "stdout=%r stderr=%r",
            problem_id,
            val_result.exit_code,
            val_result.stdout[:500],
            val_result.stderr[:500],
        )
    return val_result
