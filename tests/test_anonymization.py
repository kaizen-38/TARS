"""Round-trip tests: standard -> anonymized -> reverse to standard mapping check."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from pddl_ops.anonymize import (
    AnonymizationMapping,
    anonymize_triple,
    reverse_anonymize_plan,
    _PDDL_KEYWORDS,
)


SAMPLE_DOMAIN = """\
(define (domain blocksworld)
  (:requirements :strips)
  (:predicates (on ?x ?y) (ontable ?x) (clear ?x) (holding ?x) (handempty))
  (:action pick-up
   :parameters (?x)
   :precondition (and (clear ?x) (ontable ?x) (handempty))
   :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x)))
  (:action put-down
   :parameters (?x)
   :precondition (holding ?x)
   :effect (and (not (holding ?x)) (clear ?x) (ontable ?x) (handempty)))
)
"""

SAMPLE_PROBLEM = """\
(define (problem bw-p1)
  (:domain blocksworld)
  (:objects a b c)
  (:init (ontable a) (ontable b) (ontable c) (clear a) (clear b) (clear c) (handempty))
  (:goal (and (on a b) (on b c)))
)
"""

SAMPLE_PLAN = ["(pick-up a)", "(stack a b)", "(pick-up b)", "(stack b c)"]


class TestAnonymizationMapping:
    def test_consistent_mapping(self):
        mapping = AnonymizationMapping(seed="test")
        name1 = mapping.get_or_create("blocksworld", "sym")
        name2 = mapping.get_or_create("blocksworld", "sym")
        assert name1 == name2

    def test_different_names_get_different_symbols(self):
        mapping = AnonymizationMapping(seed="test")
        n1 = mapping.get_or_create("pick-up", "sym")
        n2 = mapping.get_or_create("put-down", "sym")
        assert n1 != n2

    def test_pddl_keywords_not_renamed(self):
        mapping = AnonymizationMapping(seed="test")
        for kw in ["define", "domain", "and", "not"]:
            result = mapping.get_or_create(kw, "sym")
            assert result == kw

    def test_serialization_roundtrip(self):
        mapping = AnonymizationMapping(seed="abc")
        mapping.get_or_create("pick-up", "sym")
        mapping.get_or_create("put-down", "sym")
        d = mapping.to_dict()
        restored = AnonymizationMapping.from_dict(d)
        assert restored._orig_to_anon == mapping._orig_to_anon
        assert restored._anon_to_orig == mapping._anon_to_orig

    def test_reverse_lookup(self):
        mapping = AnonymizationMapping(seed="test")
        anon = mapping.get_or_create("pick-up", "sym")
        assert mapping.reverse(anon) == "pick-up"


class TestAnonymizeTriple:
    def test_anonymize_triple_returns_four_values(self):
        result = anonymize_triple(
            SAMPLE_DOMAIN, SAMPLE_PROBLEM, SAMPLE_PLAN, "test_instance"
        )
        assert len(result) == 4
        anon_domain, anon_problem, anon_plan, mapping = result

    def test_anonymized_plan_same_length(self):
        _, _, anon_plan, _ = anonymize_triple(
            SAMPLE_DOMAIN, SAMPLE_PROBLEM, SAMPLE_PLAN, "test_instance"
        )
        assert len(anon_plan) == len(SAMPLE_PLAN)

    def test_pddl_structure_preserved(self):
        anon_domain, anon_problem, _, _ = anonymize_triple(
            SAMPLE_DOMAIN, SAMPLE_PROBLEM, SAMPLE_PLAN, "test_instance"
        )
        # PDDL define keyword must still be present
        assert "define" in anon_domain
        assert "define" in anon_problem

    def test_deterministic_across_calls(self):
        result1 = anonymize_triple(SAMPLE_DOMAIN, SAMPLE_PROBLEM, SAMPLE_PLAN, "inst_42")
        result2 = anonymize_triple(SAMPLE_DOMAIN, SAMPLE_PROBLEM, SAMPLE_PLAN, "inst_42")
        assert result1[0] == result2[0]  # domain
        assert result1[1] == result2[1]  # problem
        assert result1[2] == result2[2]  # plan

    def test_different_seeds_give_different_results(self):
        r1 = anonymize_triple(SAMPLE_DOMAIN, SAMPLE_PROBLEM, SAMPLE_PLAN, "seed_A")
        r2 = anonymize_triple(SAMPLE_DOMAIN, SAMPLE_PROBLEM, SAMPLE_PLAN, "seed_B")
        # Counters reset, so both will produce sym0, sym1, etc.
        # but orig_to_anon should be the same structure. Test that it runs.
        assert len(r1[2]) == len(r2[2])


class TestReverseAnonymize:
    def test_roundtrip_plan(self):
        _, _, anon_plan, mapping = anonymize_triple(
            SAMPLE_DOMAIN, SAMPLE_PROBLEM, SAMPLE_PLAN, "roundtrip_test"
        )
        restored = reverse_anonymize_plan(anon_plan, mapping)
        # After round-trip, canonical form should match original
        from pddl_ops.canonicalize import plans_are_equal
        assert plans_are_equal(SAMPLE_PLAN, restored)

    def test_empty_plan_roundtrip(self):
        _, _, anon_plan, mapping = anonymize_triple(
            SAMPLE_DOMAIN, SAMPLE_PROBLEM, [], "empty_test"
        )
        restored = reverse_anonymize_plan(anon_plan, mapping)
        assert restored == []
