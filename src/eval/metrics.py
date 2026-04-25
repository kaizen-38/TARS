"""Evaluation metrics for Phase 1."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.io import iter_jsonl
from utils.logging import get_logger

logger = get_logger(__name__)


def compute_validity_rate(rows: list[dict[str, Any]]) -> float:
    """Fraction of rows where valid_plan is True."""
    if not rows:
        return 0.0
    valid = sum(1 for r in rows if r.get("valid_plan") is True)
    return valid / len(rows)


def compute_goal_rate(rows: list[dict[str, Any]]) -> float:
    """Fraction of rows where goal_reached is True."""
    if not rows:
        return 0.0
    reached = sum(1 for r in rows if r.get("goal_reached") is True)
    return reached / len(rows)


def compute_empty_plan_rate(rows: list[dict[str, Any]]) -> float:
    """Fraction of rows where num_actions is 0."""
    if not rows:
        return 0.0
    empty = sum(1 for r in rows if r.get("num_actions", 0) == 0)
    return empty / len(rows)


def compute_avg_actions(rows: list[dict[str, Any]]) -> float:
    """Average number of actions across non-empty plans."""
    non_empty = [r.get("num_actions", 0) for r in rows if r.get("num_actions", 0) > 0]
    if not non_empty:
        return 0.0
    return sum(non_empty) / len(non_empty)


def breakdown_by_field(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float]]:
    """Break down validity/goal rates by a categorical field."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = str(row.get(field, "unknown"))
        groups.setdefault(key, []).append(row)

    result: dict[str, dict[str, float]] = {}
    for key, group_rows in sorted(groups.items()):
        result[key] = {
            "count": len(group_rows),
            "validity_rate": compute_validity_rate(group_rows),
            "goal_rate": compute_goal_rate(group_rows),
            "empty_plan_rate": compute_empty_plan_rate(group_rows),
            "avg_actions": compute_avg_actions(group_rows),
        }
    return result


def compute_all_metrics(log_path: Path) -> dict[str, Any]:
    """Compute full metrics breakdown from a run log JSONL file."""
    rows = list(iter_jsonl(log_path))
    if not rows:
        return {"error": "No rows found"}

    return {
        "total": len(rows),
        "overall": {
            "validity_rate": compute_validity_rate(rows),
            "goal_rate": compute_goal_rate(rows),
            "empty_plan_rate": compute_empty_plan_rate(rows),
            "avg_actions": compute_avg_actions(rows),
        },
        "by_domain": breakdown_by_field(rows, "domain"),
        "by_representation": breakdown_by_field(rows, "representation"),
        "by_split": breakdown_by_field(rows, "split"),
        "error_breakdown": _count_errors(rows),
    }


def _count_errors(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        err = row.get("error_type")
        if err:
            counts[err] = counts.get(err, 0) + 1
    return counts
