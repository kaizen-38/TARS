"""Dataset statistics utilities."""
from __future__ import annotations

from pathlib import Path

from utils.io import iter_jsonl
from utils.logging import get_logger

logger = get_logger(__name__)


def compute_stats(jsonl_path: Path) -> dict:
    """Compute basic statistics over an Alpaca JSONL dataset."""
    rows = list(iter_jsonl(jsonl_path))
    if not rows:
        return {"count": 0}

    output_lengths = [len(r.get("output", "").split()) for r in rows]
    input_lengths = [len(r.get("input", "").split()) for r in rows]

    return {
        "count": len(rows),
        "output_words_mean": sum(output_lengths) / len(output_lengths),
        "output_words_min": min(output_lengths),
        "output_words_max": max(output_lengths),
        "input_words_mean": sum(input_lengths) / len(input_lengths),
        "input_words_min": min(input_lengths),
        "input_words_max": max(input_lengths),
    }


def print_dataset_stats(alpaca_dir: Path) -> None:
    """Print stats for all datasets in alpaca_dir."""
    for jsonl_file in sorted(alpaca_dir.glob("*.jsonl")):
        stats = compute_stats(jsonl_file)
        logger.info(
            "%s: count=%d out_words=%.1f±(%.0f–%.0f)",
            jsonl_file.name,
            stats.get("count", 0),
            stats.get("output_words_mean", 0),
            stats.get("output_words_min", 0),
            stats.get("output_words_max", 0),
        )
