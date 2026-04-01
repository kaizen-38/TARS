"""Dataset builder smoke tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from dataset.build_sft_dataset import (
    _build_standard_example,
    _build_anonymized_example,
    _build_compact_example,
    build_tuple_json,
    SFTDatasetBuilder,
    SYSTEM_PROMPT,
)
from dataset.dedupe import _fingerprint, deduplicate_jsonl
from dataset.stats import compute_stats


DOMAIN_TEXT = "(define (domain test) (:predicates (on ?x ?y)))"
PROBLEM_TEXT = "(define (problem p1) (:domain test) (:objects a b) (:goal (on a b)))"
PLAN_ACTIONS = ["(move a b)", "(place a b)"]
COMPACT_PLAN = "move a b\nplace a b"


class TestExampleBuilders:
    def test_standard_example_has_required_fields(self):
        ex = _build_standard_example(DOMAIN_TEXT, PROBLEM_TEXT, PLAN_ACTIONS)
        assert all(k in ex for k in ("instruction", "input", "output", "system"))
        assert ex["system"] == SYSTEM_PROMPT
        assert "(move a b)" in ex["output"]
        assert "DOMAIN:" in ex["input"]

    def test_anonymized_example_has_required_fields(self):
        ex = _build_anonymized_example(DOMAIN_TEXT, PROBLEM_TEXT, PLAN_ACTIONS)
        assert all(k in ex for k in ("instruction", "input", "output", "system"))
        assert "anonymized" in ex["instruction"].lower()

    def test_compact_example_output_has_no_parens(self):
        ex = _build_compact_example(DOMAIN_TEXT, PROBLEM_TEXT, COMPACT_PLAN)
        assert "(" not in ex["output"]
        assert ")" not in ex["output"]

    def test_system_prompt_consistent(self):
        for builder in [
            _build_standard_example(DOMAIN_TEXT, PROBLEM_TEXT, PLAN_ACTIONS),
            _build_anonymized_example(DOMAIN_TEXT, PROBLEM_TEXT, PLAN_ACTIONS),
            _build_compact_example(DOMAIN_TEXT, PROBLEM_TEXT, COMPACT_PLAN),
        ]:
            assert builder["system"] == SYSTEM_PROMPT


class TestBuildTupleJson:
    def test_creates_file(self, tmp_path):
        path = build_tuple_json(
            instance_id="test_inst_001",
            domain_text=DOMAIN_TEXT,
            problem_text=PROBLEM_TEXT,
            plan_actions=PLAN_ACTIONS,
            anon_domain_text=DOMAIN_TEXT,
            anon_problem_text=PROBLEM_TEXT,
            anon_plan_actions=PLAN_ACTIONS,
            compact_plan=COMPACT_PLAN,
            output_dir=tmp_path,
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["instance_id"] == "test_inst_001"
        assert data["plan_actions"] == PLAN_ACTIONS

    def test_all_fields_present(self, tmp_path):
        path = build_tuple_json(
            instance_id="check_fields",
            domain_text=DOMAIN_TEXT,
            problem_text=PROBLEM_TEXT,
            plan_actions=PLAN_ACTIONS,
            anon_domain_text="anon_domain",
            anon_problem_text="anon_problem",
            anon_plan_actions=["(sym0 sym1 sym2)"],
            compact_plan="sym0 sym1 sym2",
            output_dir=tmp_path,
        )
        data = json.loads(path.read_text())
        for key in (
            "instance_id", "domain_text", "problem_text", "plan_actions",
            "anon_domain_text", "anon_problem_text", "anon_plan_actions", "compact_plan"
        ):
            assert key in data


class TestSFTDatasetBuilderSmoke:
    def test_build_all_creates_files(self, tmp_path):
        # Prepare a minimal tuple file
        tuples_dir = tmp_path / "tuples_standard"
        tuples_dir.mkdir()
        output_dir = tmp_path / "alpaca"

        # Write a fake tuple JSON
        tuple_data = {
            "instance_id": "blocksworld_train_0000_42",
            "domain_text": DOMAIN_TEXT,
            "problem_text": PROBLEM_TEXT,
            "plan_actions": PLAN_ACTIONS,
            "anon_domain_text": DOMAIN_TEXT,
            "anon_problem_text": PROBLEM_TEXT,
            "anon_plan_actions": PLAN_ACTIONS,
            "compact_plan": COMPACT_PLAN,
        }
        (tuples_dir / "blocksworld_train_0000_42_tuple.json").write_text(
            json.dumps(tuple_data)
        )

        builder = SFTDatasetBuilder(
            tuples_standard_dir=tuples_dir,
            output_dir=output_dir,
        )
        counts = builder.build_all(split="train")

        assert counts["phase1_standard"] == 1
        assert (output_dir / "phase1_standard.jsonl").exists()
        assert (output_dir / "phase1_anonymized.jsonl").exists()
        assert (output_dir / "phase1_compact.jsonl").exists()


class TestDedupe:
    def test_removes_duplicates(self, tmp_path):
        from utils.io import write_jsonl
        rows = [
            {"instruction": "A", "input": "X", "output": "Y"},
            {"instruction": "A", "input": "X", "output": "Y"},  # duplicate
            {"instruction": "A", "input": "X2", "output": "Y2"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(rows, path)

        removed = deduplicate_jsonl(path, path)
        assert removed == 1

        kept = list(json.loads(l) for l in path.read_text().splitlines() if l)
        assert len(kept) == 2


class TestComputeStats:
    def test_stats_on_empty(self, tmp_path):
        from utils.io import write_jsonl
        path = tmp_path / "empty.jsonl"
        write_jsonl([], path)
        stats = compute_stats(path)
        assert stats["count"] == 0

    def test_stats_on_rows(self, tmp_path):
        from utils.io import write_jsonl
        rows = [
            {"output": "move a b\npick-up c", "input": "problem text"},
            {"output": "stack a b", "input": "problem text 2"},
        ]
        path = tmp_path / "test.jsonl"
        write_jsonl(rows, path)
        stats = compute_stats(path)
        assert stats["count"] == 2
        assert stats["output_words_min"] >= 1
