"""Build Alpaca-format SFT datasets for LLaMA-Factory.

Creates three datasets:
  data/datasets/alpaca/phase1_standard.jsonl
  data/datasets/alpaca/phase1_anonymized.jsonl
  data/datasets/alpaca/phase1_compact.jsonl

And the combined dataset_info.json that LLaMA-Factory uses to discover datasets.

Each JSONL row has the Alpaca fields:
  instruction, input, output, system
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.logging import get_logger
from utils.io import ensure_dir, load_json, write_jsonl, dump_json, iter_jsonl

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"
_ALPACA_DIR = _DATA_DIR / "datasets" / "alpaca"
_DATASET_INFO_PATH = _DATA_DIR / "datasets" / "dataset_info.json"

SYSTEM_PROMPT = (
    "You are an expert AI planning assistant. "
    "Given a PDDL domain and problem, produce a valid sequential plan. "
    "Output ONLY the plan actions, one per line, with no additional explanation."
)


# ---------------------------------------------------------------------------
# Prompt builders per representation
# ---------------------------------------------------------------------------

def _build_standard_example(
    domain_text: str,
    problem_text: str,
    plan_actions: list[str],
) -> dict[str, str]:
    instruction = (
        "Solve the following PDDL planning problem. "
        "Return a valid plan as a sequence of PDDL actions, one per line."
    )
    input_text = f"DOMAIN:\n{domain_text.strip()}\n\nPROBLEM:\n{problem_text.strip()}"
    output_text = "\n".join(plan_actions)
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "system": SYSTEM_PROMPT,
    }


def _build_anonymized_example(
    anon_domain_text: str,
    anon_problem_text: str,
    anon_plan_actions: list[str],
) -> dict[str, str]:
    instruction = (
        "Solve the following anonymized PDDL planning problem. "
        "All identifiers have been renamed. "
        "Return a valid plan as a sequence of PDDL actions, one per line."
    )
    input_text = (
        f"DOMAIN:\n{anon_domain_text.strip()}\n\nPROBLEM:\n{anon_problem_text.strip()}"
    )
    output_text = "\n".join(anon_plan_actions)
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "system": SYSTEM_PROMPT,
    }


def _build_compact_example(
    domain_text: str,
    problem_text: str,
    compact_plan: str,
) -> dict[str, str]:
    instruction = (
        "Solve the following PDDL planning problem. "
        "Return a valid plan in compact format: one action per line, "
        "without parentheses."
    )
    input_text = f"DOMAIN:\n{domain_text.strip()}\n\nPROBLEM:\n{problem_text.strip()}"
    return {
        "instruction": instruction,
        "input": input_text,
        "output": compact_plan.strip(),
        "system": SYSTEM_PROMPT,
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class SFTDatasetBuilder:
    """Builds and writes SFT datasets from solved instances."""

    def __init__(
        self,
        instances_dir: Path = _DATA_DIR / "generated" / "instances",
        plans_dir: Path = _DATA_DIR / "generated" / "plans",
        tuples_standard_dir: Path = _DATA_DIR / "generated" / "tuples_standard",
        tuples_anon_dir: Path = _DATA_DIR / "generated" / "tuples_anonymized",
        tuples_compact_dir: Path = _DATA_DIR / "generated" / "tuples_compact",
        output_dir: Path = _ALPACA_DIR,
    ) -> None:
        self.instances_dir = instances_dir
        self.plans_dir = plans_dir
        self.tuples_standard_dir = tuples_standard_dir
        self.tuples_anon_dir = tuples_anon_dir
        self.tuples_compact_dir = tuples_compact_dir
        self.output_dir = output_dir

    def build_all(self, split: str = "train") -> dict[str, int]:
        """Build all three datasets for a given split.

        Returns a dict mapping dataset name to row count.
        """
        ensure_dir(self.output_dir)

        standard_rows: list[dict] = []
        anon_rows: list[dict] = []
        compact_rows: list[dict] = []

        # Iterate over all solved instance tuples
        for tuple_file in sorted(self.tuples_standard_dir.glob(f"*_{split}_*.json")):
            try:
                row = self._process_tuple(tuple_file, split)
                if row is None:
                    continue
                standard_rows.append(row["standard"])
                anon_rows.append(row["anonymized"])
                compact_rows.append(row["compact"])
            except Exception as exc:
                logger.warning("Skipping %s: %s", tuple_file, exc)

        counts: dict[str, int] = {}
        for name, rows in [
            (f"phase1_standard_{split}", standard_rows),
            (f"phase1_anonymized_{split}", anon_rows),
            (f"phase1_compact_{split}", compact_rows),
        ]:
            out_path = self.output_dir / f"{name}.jsonl"
            write_jsonl(rows, out_path)
            counts[name] = len(rows)
            logger.info("Wrote %d rows to %s", len(rows), out_path)

        self._write_dataset_info(counts)
        return counts

    def _process_tuple(self, tuple_file: Path, split: str) -> dict | None:
        """Load a tuple JSON and build all three representation rows."""
        data = load_json(tuple_file)

        domain_text = data.get("domain_text", "")
        problem_text = data.get("problem_text", "")
        plan_actions = data.get("plan_actions", [])
        anon_domain = data.get("anon_domain_text", "")
        anon_problem = data.get("anon_problem_text", "")
        anon_plan = data.get("anon_plan_actions", [])
        compact_plan = data.get("compact_plan", "")

        if not domain_text or not problem_text or not plan_actions:
            return None

        return {
            "standard": _build_standard_example(domain_text, problem_text, plan_actions),
            "anonymized": _build_anonymized_example(anon_domain, anon_problem, anon_plan),
            "compact": _build_compact_example(domain_text, problem_text, compact_plan),
        }

    def _write_dataset_info(self, counts: dict[str, int]) -> None:
        """Write dataset_info.json for LLaMA-Factory, merging with existing entries."""
        existing: dict[str, Any] = {}
        if _DATASET_INFO_PATH.exists():
            existing = json.loads(_DATASET_INFO_PATH.read_text())
        for name in counts:
            existing[name] = {
                "file_name": f"alpaca/{name}.jsonl",
                "file_sha1": None,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                    "system": "system",
                },
            }
        dump_json(existing, _DATASET_INFO_PATH)
        logger.info("Wrote dataset_info.json with %d datasets", len(existing))


def build_tuple_json(
    instance_id: str,
    domain_text: str,
    problem_text: str,
    plan_actions: list[str],
    anon_domain_text: str,
    anon_problem_text: str,
    anon_plan_actions: list[str],
    compact_plan: str,
    output_dir: Path,
) -> Path:
    """Write a tuple JSON file consumed by SFTDatasetBuilder."""
    ensure_dir(output_dir)
    data = {
        "instance_id": instance_id,
        "domain_text": domain_text,
        "problem_text": problem_text,
        "plan_actions": plan_actions,
        "anon_domain_text": anon_domain_text,
        "anon_problem_text": anon_problem_text,
        "anon_plan_actions": anon_plan_actions,
        "compact_plan": compact_plan,
    }
    path = output_dir / f"{instance_id}_tuple.json"
    dump_json(data, path)
    return path
