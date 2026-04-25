"""Structured logging utilities for thicket_phase1.

Every run appends JSONL rows to a run log so that results are
reproducible and attributable to exact git commits, seeds, and configs.
"""
from __future__ import annotations

import fcntl
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import subprocess


def _get_git_commit() -> str:
    """Return the current short git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured to write human-readable lines to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class RunLogger:
    """Appends structured JSONL rows to a run log file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._git_commit = _get_git_commit()

    def log(self, row: dict[str, Any]) -> None:
        """Append a single JSONL row to the log file (atomic with file lock)."""
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        row.setdefault("git_commit", self._git_commit)
        line = json.dumps(row, default=str) + "\n"
        with self.log_path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def log_run_result(
        self,
        *,
        run_id: str,
        seed: int,
        domain: str,
        problem_id: str,
        representation: str,
        split: str,
        model_name: str,
        checkpoint_path: str | None,
        planner_backend: str,
        valid_plan: bool | None,
        goal_reached: bool | None,
        max_new_tokens: int,
        generated_tokens: int | None,
        generation_time_sec: float | None,
        val_time_sec: float | None,
        total_time_sec: float | None,
        error_type: str | None,
        num_actions: int = 0,
    ) -> None:
        """Log the canonical Phase 1 result row."""
        self.log(
            {
                "run_id": run_id,
                "seed": seed,
                "domain": domain,
                "problem_id": problem_id,
                "representation": representation,
                "split": split,
                "model_name": model_name,
                "checkpoint_path": checkpoint_path,
                "planner_backend": planner_backend,
                "valid_plan": valid_plan,
                "goal_reached": goal_reached,
                "num_actions": num_actions,
                "max_new_tokens": max_new_tokens,
                "generated_tokens": generated_tokens,
                "generation_time_sec": generation_time_sec,
                "val_time_sec": val_time_sec,
                "total_time_sec": total_time_sec,
                "error_type": error_type,
            }
        )
