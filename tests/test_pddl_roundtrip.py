"""Round-trip tests: standard -> compact -> decoded standard plan."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pddl_ops.parse_utils import ParsedAction, ParsedPlan, parse_plan_from_text, plan_to_file
from pddl_ops.compact_serialize import plan_to_compact, actions_to_compact
from pddl_ops.decode_compact_plan import decode_compact_plan, extract_compact_plan_from_text
from pddl_ops.canonicalize import canonicalize_plan, plans_are_equal


SAMPLE_ACTIONS = [
    "(move-block a b)",
    "(pick-up c)",
    "(put-down d table)",
    "(stack e f)",
]

SAMPLE_TIMED_ACTIONS = [
    "0.000: (move-block a b) [1.000]",
    "1.000: (pick-up c) [1.000]",
    "2.000: (put-down d table) [1.000]",
]


class TestCompactRoundtrip:
    def test_compact_then_decode_preserves_actions(self):
        compact = actions_to_compact(SAMPLE_ACTIONS)
        decoded = decode_compact_plan(compact)
        decoded_strs = [a.to_pddl() for a in decoded.actions]
        assert plans_are_equal(SAMPLE_ACTIONS, decoded_strs)

    def test_timed_actions_to_compact(self):
        compact = actions_to_compact(SAMPLE_TIMED_ACTIONS)
        lines = compact.strip().splitlines()
        assert len(lines) == len(SAMPLE_TIMED_ACTIONS)
        # No parens in compact form
        for line in lines:
            assert "(" not in line
            assert ")" not in line
        # No timestamps
        for line in lines:
            assert ":" not in line

    def test_compact_to_standard_file(self, tmp_path):
        compact = actions_to_compact(SAMPLE_ACTIONS)
        decoded = decode_compact_plan(compact)
        plan_file = tmp_path / "plan.pddl"
        plan_to_file(decoded, plan_file, timed=False)
        content = plan_file.read_text()
        for action in ["(move-block a b)", "(pick-up c)"]:
            assert action in content

    def test_empty_plan(self):
        compact = actions_to_compact([])
        decoded = decode_compact_plan(compact)
        assert decoded.actions == []

    def test_single_action(self):
        actions = ["(move a b)"]
        compact = actions_to_compact(actions)
        decoded = decode_compact_plan(compact)
        assert len(decoded.actions) == 1
        assert decoded.actions[0].name == "move"
        assert decoded.actions[0].args == ["a", "b"]


class TestParsePlanFromText:
    def test_parse_standard_pddl(self):
        text = "(move a b)\n(pick-up c)\n(put-down d table)\n"
        plan = parse_plan_from_text(text)
        assert len(plan.actions) == 3
        assert plan.actions[0].name == "move"

    def test_parse_timed_pddl(self):
        text = "\n".join(SAMPLE_TIMED_ACTIONS)
        plan = parse_plan_from_text(text)
        assert len(plan.actions) == len(SAMPLE_TIMED_ACTIONS)
        assert plan.actions[0].name == "move-block"

    def test_skips_comments(self):
        text = "; This is a plan\n(move a b)\n; done\n(pick-up c)\n"
        plan = parse_plan_from_text(text)
        assert len(plan.actions) == 2

    def test_empty_text(self):
        plan = parse_plan_from_text("")
        assert plan.actions == []


class TestExtractCompactPlan:
    def test_extracts_from_code_block(self):
        model_output = (
            "Here is the solution:\n"
            "```\n"
            "move-block a b\n"
            "pick-up c\n"
            "```"
        )
        extracted = extract_compact_plan_from_text(model_output)
        assert "move-block a b" in extracted
        assert "pick-up c" in extracted

    def test_extracts_bare_actions(self):
        model_output = "move-block a b\npick-up c\nput-down d table"
        extracted = extract_compact_plan_from_text(model_output)
        assert "move-block" in extracted


class TestCanonicalize:
    def test_canonical_action(self):
        from pddl_ops.canonicalize import canonicalize_action
        assert canonicalize_action("  (MOVE-BLOCK  A  B)  ") == "(move-block a b)"

    def test_strips_cost_annotation(self):
        from pddl_ops.canonicalize import canonicalize_action
        assert canonicalize_action("(move a b) [1.000]") == "(move a b)"

    def test_plans_are_equal(self):
        plan_a = ["(move a b)", "(pick-up c)"]
        plan_b = ["  (MOVE   A  B) ", "(PICK-UP C)"]
        assert plans_are_equal(plan_a, plan_b)
