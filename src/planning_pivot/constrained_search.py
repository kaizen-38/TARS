from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Any

from planning_pivot.sas import (
    SasTask,
    SasOperator,
    applicable_operators,
    apply_operator,
    is_goal,
    plan_lines_from_ops,
)
from planning_pivot.rankers import ActionRanker


@dataclass(frozen=True)
class SearchNode:
    state: tuple[int, ...]
    ops: tuple[SasOperator, ...]
    score: float
    depth: int


@dataclass(frozen=True)
class SearchResult:
    solved: bool
    plan_lines: list[str]
    final_score: float
    expansions: int
    generated: int
    seconds: float
    failure_type: str | None


def greedy_rollout(
    task: SasTask,
    ranker: ActionRanker,
    max_steps: int,
    context: dict[str, Any],
) -> SearchResult:
    t0 = time.perf_counter()
    state = task.init
    ops: list[SasOperator] = []
    expansions = 0
    generated = 0

    for _ in range(max_steps):
        if is_goal(task, state):
            return SearchResult(
                solved=True,
                plan_lines=plan_lines_from_ops(ops),
                final_score=0.0,
                expansions=expansions,
                generated=generated,
                seconds=time.perf_counter() - t0,
                failure_type=None,
            )

        candidates = applicable_operators(task, state)
        if not candidates:
            return SearchResult(
                solved=False,
                plan_lines=plan_lines_from_ops(ops),
                final_score=0.0,
                expansions=expansions,
                generated=generated,
                seconds=time.perf_counter() - t0,
                failure_type="dead_end",
            )

        expansions += 1
        generated += len(candidates)
        scores = ranker.score(task, state, candidates, context)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_op = candidates[best_idx]
        state = apply_operator(task, state, best_op)
        ops.append(best_op)

    return SearchResult(
        solved=is_goal(task, state),
        plan_lines=plan_lines_from_ops(ops),
        final_score=0.0,
        expansions=expansions,
        generated=generated,
        seconds=time.perf_counter() - t0,
        failure_type=None if is_goal(task, state) else "max_steps",
    )


def beam_search(
    task: SasTask,
    ranker: ActionRanker,
    beam_width: int,
    max_steps: int,
    max_expansions: int,
    context: dict[str, Any],
) -> SearchResult:
    t0 = time.perf_counter()
    start_state = context.get("start_state", task.init)
    start_ops = tuple(context.get("prefix_ops", []))

    beam: list[SearchNode] = [
        SearchNode(state=start_state, ops=start_ops, score=0.0, depth=0)
    ]
    best_states: dict[tuple[int, ...], float] = {start_state: 0.0}
    expansions = 0
    generated = 0

    for step in range(max_steps):
        if expansions >= max_expansions:
            break

        next_beam: list[tuple[float, int, SearchNode]] = []

        for node in beam:
            if is_goal(task, node.state):
                return SearchResult(
                    solved=True,
                    plan_lines=plan_lines_from_ops(list(node.ops)),
                    final_score=node.score,
                    expansions=expansions,
                    generated=generated,
                    seconds=time.perf_counter() - t0,
                    failure_type=None,
                )

            candidates = applicable_operators(task, node.state)
            if not candidates:
                continue

            expansions += 1
            generated += len(candidates)
            scores = ranker.score(task, node.state, candidates, context)

            for op, sc in zip(candidates, scores):
                new_state = apply_operator(task, node.state, op)
                new_score = node.score + sc

                prev_best = best_states.get(new_state)
                if prev_best is not None and prev_best >= new_score:
                    continue
                best_states[new_state] = new_score

                child = SearchNode(
                    state=new_state,
                    ops=node.ops + (op,),
                    score=new_score,
                    depth=node.depth + 1,
                )
                next_beam.append((new_score, generated, child))

        if not next_beam:
            break

        next_beam.sort(key=lambda x: -x[0])
        beam = [item[2] for item in next_beam[:beam_width]]

    for node in beam:
        if is_goal(task, node.state):
            return SearchResult(
                solved=True,
                plan_lines=plan_lines_from_ops(list(node.ops)),
                final_score=node.score,
                expansions=expansions,
                generated=generated,
                seconds=time.perf_counter() - t0,
                failure_type=None,
            )

    best_node = max(beam, key=lambda n: n.score) if beam else SearchNode(
        state=task.init, ops=(), score=0.0, depth=0,
    )
    return SearchResult(
        solved=False,
        plan_lines=plan_lines_from_ops(list(best_node.ops)),
        final_score=best_node.score,
        expansions=expansions,
        generated=generated,
        seconds=time.perf_counter() - t0,
        failure_type="max_steps" if expansions < max_expansions else "max_expansions",
    )


def run_constrained_search_experiment(config) -> "pd.DataFrame":
    import pandas as pd
    from pathlib import Path
    from planning_pivot.sas import parse_output_sas
    from planning_pivot.rankers import RandomRanker, GoalCountRanker, TeacherFrequencyRanker
    from planning_pivot.plan_io import PlanAction, write_plan_file
    from planning_pivot.diagnostics import build_instance_index

    output_root = Path(config.data.output_root)
    cs_dir = output_root / "constrained_search"
    cs_dir.mkdir(parents=True, exist_ok=True)

    instances = build_instance_index(config)
    sas_cache_dir = output_root / "sas_cache"

    rankers_list = [
        RandomRanker(seed=config.experiment.random_seed),
        GoalCountRanker(),
    ]

    teacher_paths = [
        Path(p) for p in instances["teacher_plan_path"].dropna().unique()
        if p and Path(p).exists()
    ]
    if teacher_paths:
        rankers_list.append(TeacherFrequencyRanker(teacher_paths))

    beam_widths = [1, 4, 8]
    rows: list[dict] = []

    seen_instances = set()
    for _, irow in instances.iterrows():
        instance_base = irow["instance_id"].rsplit("_", 1)[0] if "_" in str(irow["instance_id"]) else str(irow["instance_id"])
        if instance_base in seen_instances:
            continue
        seen_instances.add(instance_base)

        sas_path = sas_cache_dir / instance_base / "output.sas"
        if not sas_path.exists():
            continue

        try:
            task = parse_output_sas(sas_path)
        except Exception:
            continue

        teacher_len_val = irow.get("teacher_plan_len")
        teacher_len = int(teacher_len_val) if pd.notna(teacher_len_val) else None
        max_steps = min(
            int((teacher_len or 20) * config.experiment.max_steps_factor),
            config.experiment.max_steps_cap,
        )

        for ranker in rankers_list:
            for bw in beam_widths:
                ctx: dict = {}
                result = beam_search(
                    task, ranker, beam_width=bw,
                    max_steps=max_steps,
                    max_expansions=max_steps * bw * 2,
                    context=ctx,
                )

                plan_path = cs_dir / f"{instance_base}_{ranker.name}_bw{bw}.plan"
                if result.plan_lines:
                    actions = [
                        PlanAction(name=l.strip("() ").split()[0], args=tuple(l.strip("() ").split()[1:]), raw=l)
                        for l in result.plan_lines
                    ]
                    write_plan_file(actions, plan_path)

                rows.append({
                    "instance_id": instance_base,
                    "domain_name": irow["domain_name"],
                    "split": irow["split"],
                    "mode": f"beam_{ranker.name}",
                    "ranker": ranker.name,
                    "beam_width": bw,
                    "solved_sas": result.solved,
                    "val_valid": False,
                    "plan_path": str(plan_path) if result.plan_lines else "",
                    "teacher_plan_len": teacher_len,
                    "plan_len": len(result.plan_lines),
                    "length_ratio": len(result.plan_lines) / teacher_len if teacher_len else None,
                    "expansions": result.expansions,
                    "generated": result.generated,
                    "val_calls": 0,
                    "seconds": result.seconds,
                    "failure_type": result.failure_type,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "instance_id", "domain_name", "split", "mode", "ranker", "beam_width",
            "solved_sas", "val_valid", "plan_path", "teacher_plan_len", "plan_len",
            "length_ratio", "expansions", "generated", "val_calls", "seconds", "failure_type",
        ])

    df.to_csv(cs_dir / "mode_results.csv", index=False)

    if not df.empty:
        summary = df.groupby(["ranker", "beam_width", "split"]).agg(
            count=("solved_sas", "count"),
            solved_rate=("solved_sas", "mean"),
            mean_plan_len=("plan_len", "mean"),
            mean_seconds=("seconds", "mean"),
            mean_expansions=("expansions", "mean"),
        ).reset_index()
        summary.to_csv(cs_dir / "summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(cs_dir / "summary.csv", index=False)

    return df
