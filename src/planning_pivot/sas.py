from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SasVariable:
    name: str
    axiom_layer: int
    values: tuple[str, ...]


@dataclass(frozen=True)
class SasEffect:
    conditions: tuple[tuple[int, int], ...]
    var: int
    old_value: int
    new_value: int


@dataclass(frozen=True)
class SasOperator:
    name: str
    prevail: tuple[tuple[int, int], ...]
    effects: tuple[SasEffect, ...]
    cost: int


@dataclass(frozen=True)
class SasTask:
    variables: tuple[SasVariable, ...]
    init: tuple[int, ...]
    goals: tuple[tuple[int, int], ...]
    operators: tuple[SasOperator, ...]
    axioms_count: int = 0


@dataclass(frozen=True)
class SasExecutionTrace:
    valid_prefix_len: int
    first_invalid_step: int | None
    reached_goal: bool
    final_state: tuple[int, ...]
    invalid_reason: str | None


def _read_section(lines: list[str], idx: int, begin_tag: str, end_tag: str) -> tuple[list[str], int]:
    while idx < len(lines) and lines[idx].strip() != begin_tag:
        idx += 1
    idx += 1
    section: list[str] = []
    while idx < len(lines) and lines[idx].strip() != end_tag:
        section.append(lines[idx].strip())
        idx += 1
    idx += 1
    return section, idx


def parse_output_sas(path: Path) -> SasTask:
    text = path.read_text()
    lines = text.splitlines()
    idx = 0

    # version
    _, idx = _read_section(lines, idx, "begin_version", "end_version")
    # metric
    _, idx = _read_section(lines, idx, "begin_metric", "end_metric")

    # variables
    while idx < len(lines) and not lines[idx].strip().isdigit():
        idx += 1
    num_variables = int(lines[idx].strip())
    idx += 1

    variables: list[SasVariable] = []
    for _ in range(num_variables):
        section, idx = _read_section(lines, idx, "begin_variable", "end_variable")
        var_name = section[0]
        axiom_layer = int(section[1])
        num_values = int(section[2])
        values = tuple(section[3: 3 + num_values])
        variables.append(SasVariable(name=var_name, axiom_layer=axiom_layer, values=values))

    # mutex groups
    while idx < len(lines) and not lines[idx].strip().isdigit():
        idx += 1
    num_mutex = int(lines[idx].strip())
    idx += 1
    for _ in range(num_mutex):
        _, idx = _read_section(lines, idx, "begin_mutex_group", "end_mutex_group")

    # initial state
    section, idx = _read_section(lines, idx, "begin_state", "end_state")
    init = tuple(int(v) for v in section)

    # goal
    section, idx = _read_section(lines, idx, "begin_goal", "end_goal")
    num_goals = int(section[0])
    goals: list[tuple[int, int]] = []
    for i in range(1, num_goals + 1):
        parts = section[i].split()
        goals.append((int(parts[0]), int(parts[1])))

    # operators
    while idx < len(lines) and not lines[idx].strip().isdigit():
        idx += 1
    num_operators = int(lines[idx].strip())
    idx += 1

    operators: list[SasOperator] = []
    for _ in range(num_operators):
        section, idx = _read_section(lines, idx, "begin_operator", "end_operator")
        op_name = section[0]
        si = 1
        num_prevail = int(section[si])
        si += 1
        prevail: list[tuple[int, int]] = []
        for _ in range(num_prevail):
            parts = section[si].split()
            prevail.append((int(parts[0]), int(parts[1])))
            si += 1

        num_effects = int(section[si])
        si += 1
        effects: list[SasEffect] = []
        for _ in range(num_effects):
            parts = section[si].split()
            si += 1
            num_conds = int(parts[0])
            conds: list[tuple[int, int]] = []
            pi = 1
            for _ in range(num_conds):
                conds.append((int(parts[pi]), int(parts[pi + 1])))
                pi += 2
            var = int(parts[pi])
            old_val = int(parts[pi + 1])
            new_val = int(parts[pi + 2])
            effects.append(SasEffect(
                conditions=tuple(conds),
                var=var,
                old_value=old_val,
                new_value=new_val,
            ))

        cost = int(section[si])
        operators.append(SasOperator(
            name=op_name,
            prevail=tuple(prevail),
            effects=tuple(effects),
            cost=cost,
        ))

    # axiom count
    axioms_count = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.isdigit():
            axioms_count = int(line)
            break
        idx += 1

    return SasTask(
        variables=tuple(variables),
        init=init,
        goals=tuple(goals),
        operators=tuple(operators),
        axioms_count=axioms_count,
    )


def canonical_operator_plan_line(op_name: str) -> str:
    parts = op_name.strip().split()
    return "(" + " ".join(parts) + ")"


def operator_lookup(task: SasTask) -> dict[str, SasOperator]:
    lookup: dict[str, SasOperator] = {}
    for op in task.operators:
        lookup[op.name] = op
        lookup[op.name.lower()] = op
        canon = canonical_operator_plan_line(op.name)
        lookup[canon] = op
        lookup[canon.lower()] = op
    return lookup


def is_goal(task: SasTask, state: tuple[int, ...]) -> bool:
    return all(state[var] == val for var, val in task.goals)


def is_applicable(task: SasTask, state: tuple[int, ...], op: SasOperator) -> bool:
    for var, val in op.prevail:
        if state[var] != val:
            return False
    for eff in op.effects:
        if eff.old_value != -1 and state[eff.var] != eff.old_value:
            return False
        for cvar, cval in eff.conditions:
            if state[cvar] != cval:
                return False
    return True


def apply_operator(task: SasTask, state: tuple[int, ...], op: SasOperator) -> tuple[int, ...]:
    new_state = list(state)
    for eff in op.effects:
        conds_hold = all(state[cv] == cv_val for cv, cv_val in eff.conditions)
        if conds_hold and (eff.old_value == -1 or state[eff.var] == eff.old_value):
            new_state[eff.var] = eff.new_value
    return tuple(new_state)


def applicable_operators(task: SasTask, state: tuple[int, ...]) -> list[SasOperator]:
    return [op for op in task.operators if is_applicable(task, state, op)]


def simulate_plan(task: SasTask, plan_lines: list[str]) -> SasExecutionTrace:
    lookup = operator_lookup(task)
    state = task.init
    executed: int = 0

    for i, line in enumerate(plan_lines):
        line = line.strip()
        if not line or line.startswith(";"):
            continue

        op = lookup.get(line) or lookup.get(line.lower())
        if op is None:
            inner = line.strip("() ").lower()
            op = lookup.get(inner) or lookup.get("(" + inner + ")")
            if op is None:
                parts = inner.split()
                for candidate_op in task.operators:
                    c_parts = candidate_op.name.lower().split()
                    if c_parts == parts:
                        op = candidate_op
                        break

        if op is None:
            return SasExecutionTrace(
                valid_prefix_len=executed,
                first_invalid_step=i,
                reached_goal=is_goal(task, state),
                final_state=state,
                invalid_reason=f"unknown_action: {line}",
            )

        if not is_applicable(task, state, op):
            return SasExecutionTrace(
                valid_prefix_len=executed,
                first_invalid_step=i,
                reached_goal=is_goal(task, state),
                final_state=state,
                invalid_reason=f"not_applicable: {line}",
            )

        state = apply_operator(task, state, op)
        executed += 1

    return SasExecutionTrace(
        valid_prefix_len=executed,
        first_invalid_step=None,
        reached_goal=is_goal(task, state),
        final_state=state,
        invalid_reason=None,
    )


def plan_lines_from_ops(ops: list[SasOperator]) -> list[str]:
    return [canonical_operator_plan_line(op.name) for op in ops]
