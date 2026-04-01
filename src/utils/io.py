"""I/O helpers using pathlib throughout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return a plain dict."""
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    """Write a dict to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)


def load_json(path: Path) -> Any:
    """Load JSON from a file."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(data: Any, path: Path, *, indent: int = 2) -> None:
    """Write data to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str, ensure_ascii=False)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield each line of a JSONL file as a dict."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    """Append a single row to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write (overwrite) a list of rows to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if needed; return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
