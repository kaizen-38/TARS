"""Evaluation manifest: tracks which instances have been evaluated."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.io import load_json, dump_json
from utils.logging import get_logger

logger = get_logger(__name__)


class EvalManifest:
    """Tracks evaluation status per (instance_id, representation).

    Allows resuming interrupted evaluation runs.
    """

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self._data: dict[str, bool] = {}
        if manifest_path.exists():
            self._data = load_json(manifest_path)

    def _key(self, instance_id: str, representation: str) -> str:
        return f"{instance_id}::{representation}"

    def mark_done(self, instance_id: str, representation: str) -> None:
        self._data[self._key(instance_id, representation)] = True
        dump_json(self._data, self.manifest_path)

    def is_done(self, instance_id: str, representation: str) -> bool:
        return self._data.get(self._key(instance_id, representation), False)

    def count(self) -> int:
        return sum(1 for v in self._data.values() if v)
