"""Smoke test for the VAL wrapper.

Tests the VALResult data structure and output parsing logic
without requiring the Validate binary to be built.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generation.validate_with_val import (
    VALResult,
    _parse_val_output,
    validate_plan,
)


class TestParseValOutput:
    def test_valid_plan_detected(self):
        stdout = "Checking plan...\nPlan valid!\nGoal reached.\n"
        validity, goal = _parse_val_output(stdout)
        assert validity is True
        assert goal is True

    def test_invalid_plan_detected(self):
        stdout = "Checking plan...\nPlan invalid: action precondition not satisfied.\n"
        validity, goal = _parse_val_output(stdout)
        assert validity is False

    def test_goal_not_reached(self):
        stdout = "Plan valid!\nGoal not reached.\n"
        validity, goal = _parse_val_output(stdout)
        assert validity is True
        assert goal is False

    def test_empty_output(self):
        validity, goal = _parse_val_output("")
        assert validity is None
        assert goal is None


class TestValidatePlanNoVAL:
    """Test validate_plan() when the VAL binary is not present."""

    def test_returns_error_result_when_binary_missing(self, tmp_path):
        domain_file = tmp_path / "domain.pddl"
        problem_file = tmp_path / "problem.pddl"
        plan_file = tmp_path / "plan.pddl"
        for f in (domain_file, problem_file, plan_file):
            f.write_text("; placeholder", encoding="utf-8")

        with patch("generation.validate_with_val._find_val_binary") as mock_find:
            mock_find.side_effect = RuntimeError("VAL not found")
            result = validate_plan(
                problem_id="test001",
                domain_file=domain_file,
                problem_file=problem_file,
                plan_file=plan_file,
            )

        assert isinstance(result, VALResult)
        assert result.success is False
        assert result.exit_code == -3
        assert result.parsed_validity is None
        assert "VAL not found" in (result.error or "")

    def test_timeout_handled_gracefully(self, tmp_path):
        import subprocess

        domain_file = tmp_path / "domain.pddl"
        problem_file = tmp_path / "problem.pddl"
        plan_file = tmp_path / "plan.pddl"
        for f in (domain_file, problem_file, plan_file):
            f.write_text("; placeholder", encoding="utf-8")

        mock_binary = tmp_path / "Validate"
        mock_binary.write_text("#!/bin/sh\nsleep 100\n")
        mock_binary.chmod(0o755)

        with patch("generation.validate_with_val._find_val_binary", return_value=mock_binary):
            result = validate_plan(
                problem_id="timeout_test",
                domain_file=domain_file,
                problem_file=problem_file,
                plan_file=plan_file,
                timeout=1,
            )

        assert result.success is False
        assert result.exit_code == -1
        assert result.error is not None
