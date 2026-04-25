"""Aggregate evaluation results and produce summary tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from eval.metrics import compute_all_metrics
from utils.io import dump_json
from utils.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer(add_completion=False)


@app.command()
def main(
    log_path: Path = typer.Argument(..., help="Path to run_log.jsonl"),
    output_path: Optional[Path] = typer.Option(None, help="Write summary JSON here"),
) -> None:
    """Aggregate a run log into summary metrics."""
    metrics = compute_all_metrics(log_path)

    if output_path is None:
        output_path = log_path.parent / "metrics_summary.json"

    dump_json(metrics, output_path)
    logger.info("Metrics summary written to %s", output_path)

    # Print human-readable summary
    print("\n=== Phase 1 Evaluation Summary ===")
    overall = metrics.get("overall", {})
    print(f"Total instances: {metrics.get('total', 0)}")
    print(f"Validity rate:   {overall.get('validity_rate', 0):.3f}")
    print(f"Goal rate:       {overall.get('goal_rate', 0):.3f}")
    print(f"Empty plan rate: {overall.get('empty_plan_rate', 0):.3f}")
    print(f"Avg actions:     {overall.get('avg_actions', 0):.1f}")

    print("\n-- By Representation --")
    for rep, stats in metrics.get("by_representation", {}).items():
        print(
            f"  {rep:12s}: validity={stats['validity_rate']:.3f} "
            f"goal={stats['goal_rate']:.3f} "
            f"empty={stats.get('empty_plan_rate', 0):.3f} "
            f"avg_act={stats.get('avg_actions', 0):.1f} "
            f"n={stats['count']}"
        )

    print("\n-- By Domain --")
    for domain, stats in metrics.get("by_domain", {}).items():
        print(
            f"  {domain:15s}: validity={stats['validity_rate']:.3f} "
            f"goal={stats['goal_rate']:.3f} "
            f"empty={stats.get('empty_plan_rate', 0):.3f} "
            f"avg_act={stats.get('avg_actions', 0):.1f} "
            f"n={stats['count']}"
        )


if __name__ == "__main__":
    app()
