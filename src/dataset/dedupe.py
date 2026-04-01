"""Deduplication utilities for the SFT dataset.

Deduplicates on canonical plan hash (domain + problem + plan fingerprint)
to avoid training on near-identical examples from the same generator seed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

from utils.io import iter_jsonl, write_jsonl
from utils.logging import get_logger

logger = get_logger(__name__)


def _fingerprint(row: dict) -> str:
    """Hash (instruction, input, output) for dedup purposes."""
    key = row.get("input", "") + "|||" + row.get("output", "")
    return hashlib.sha256(key.encode()).hexdigest()


def deduplicate_jsonl(input_path: Path, output_path: Path) -> int:
    """Deduplicate a JSONL file in-place; returns removed count."""
    seen: set[str] = set()
    kept: list[dict] = []
    total = 0
    for row in iter_jsonl(input_path):
        total += 1
        fp = _fingerprint(row)
        if fp not in seen:
            seen.add(fp)
            kept.append(row)

    removed = total - len(kept)
    write_jsonl(kept, output_path)
    logger.info(
        "Dedupe %s: %d total -> %d kept (%d removed)",
        input_path.name, total, len(kept), removed,
    )
    return removed


def deduplicate_all(alpaca_dir: Path) -> dict[str, int]:
    """Run deduplication on all JSONL files in alpaca_dir."""
    results: dict[str, int] = {}
    for jsonl_file in sorted(alpaca_dir.glob("*.jsonl")):
        removed = deduplicate_jsonl(jsonl_file, jsonl_file)
        results[jsonl_file.name] = removed
    return results
