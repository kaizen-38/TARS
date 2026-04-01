"""Tests for compact plan serialization and decoding."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pddl_ops.compact_serialize import (
    plan_to_compact,
    actions_to_compact,
    _compact_one,
    save_compact_plan,
)
from pddl_ops.decode_compact_plan import (
    decode_compact_plan,
    compact_to_standard_file,
    extract_compact_plan_from_text,
    _looks_like_action,
)
from pddl_ops.parse_utils import ParsedAction, ParsedPlan


class TestCompactOne:
    def test_standard_pddl(self):
        assert _compact_one("(move a b)") == "move a b"

    def test_timed_pddl(self):
        assert _compact_one("0.000: (move a b) [1.000]") == "move a b"

    def test_uppercase_lowercased(self):
        assert _compact_one("(MOVE A B)") == "move a b"

    def test_fd_style_with_cost(self):
        assert _compact_one("(move a b) ; cost = 1") == "move a b"

    def test_empty_string(self):
        result = _compact_one("")
        assert result == ""

    def test_no_args(self):
        assert _compact_one("(handempty)") == "handempty"


class TestActionsToCompact:
    def test_multiple_actions(self):
        actions = ["(move a b)", "(pick-up c)", "(put-down d table)"]
        compact = actions_to_compact(actions)
        lines = compact.strip().splitlines()
        assert len(lines) == 3
        assert lines[0] == "move a b"
        assert lines[1] == "pick-up c"
        assert lines[2] == "put-down d table"

    def test_empty_list(self):
        assert actions_to_compact([]) == ""


class TestPlanToCompact:
    def test_from_parsed_plan(self):
        plan = ParsedPlan(actions=[
            ParsedAction(name="move", args=["a", "b"], timestamp=None),
            ParsedAction(name="pick-up", args=["c"], timestamp=None),
        ])
        compact = plan_to_compact(plan)
        assert compact == "move a b\npick-up c"

    def test_empty_plan(self):
        plan = ParsedPlan(actions=[])
        assert plan_to_compact(plan) == ""


class TestDecodeCompactPlan:
    def test_decode_basic(self):
        compact = "move a b\npick-up c\nput-down d table"
        plan = decode_compact_plan(compact)
        assert len(plan.actions) == 3
        assert plan.actions[0].name == "move"
        assert plan.actions[0].args == ["a", "b"]
        assert plan.actions[2].args == ["d", "table"]

    def test_skips_blank_lines(self):
        compact = "move a b\n\npick-up c\n"
        plan = decode_compact_plan(compact)
        assert len(plan.actions) == 2

    def test_skips_comment_lines(self):
        compact = "# This is a plan\nmove a b\n; comment\npick-up c"
        plan = decode_compact_plan(compact)
        assert len(plan.actions) == 2

    def test_timestamps_are_none(self):
        compact = "move a b\npick-up c"
        plan = decode_compact_plan(compact)
        for action in plan.actions:
            assert action.timestamp is None


class TestCompactToStandardFile:
    def test_writes_standard_pddl(self, tmp_path):
        compact_path = tmp_path / "plan_compact.txt"
        compact_path.write_text("move a b\npick-up c\n", encoding="utf-8")
        output_path = tmp_path / "plan_standard.pddl"

        plan = compact_to_standard_file(compact_path, output_path)
        assert output_path.exists()
        content = output_path.read_text()
        assert "(move a b)" in content
        assert "(pick-up c)" in content


class TestSaveCompactPlan:
    def test_saves_with_newline(self, tmp_path):
        path = tmp_path / "plan.compact.txt"
        save_compact_plan("move a b\npick-up c", path)
        content = path.read_text()
        assert content.endswith("\n")
        assert "move a b" in content


class TestLooksLikeAction:
    def test_compact_action_looks_like_action(self):
        assert _looks_like_action("move a b") is True
        assert _looks_like_action("pick-up c") is True

    def test_standard_pddl_looks_like_action(self):
        assert _looks_like_action("(move a b)") is True

    def test_empty_string(self):
        assert _looks_like_action("") is False

    def test_keyword_line(self):
        assert _looks_like_action(":domain blocksworld") is False


class TestExtractCompactPlan:
    def test_code_block(self):
        text = "Plan:\n```\nmove a b\npick-up c\n```"
        result = extract_compact_plan_from_text(text)
        assert "move a b" in result

    def test_bare_actions(self):
        text = "move a b\npick-up c"
        result = extract_compact_plan_from_text(text)
        assert "move a b" in result

    def test_plan_marker(self):
        text = "Reasoning...\nPlan:\nmove a b\npick-up c"
        result = extract_compact_plan_from_text(text)
        assert "move a b" in result
