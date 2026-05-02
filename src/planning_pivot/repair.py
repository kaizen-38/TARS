from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from planning_pivot.sas import (
    SasTask,
    SasOperator,
    SasExecutionTrace,
    simulate_plan,
    is_goal,
    applicable_operators,
    apply_operator,
    operator_lookup,
    plan_lines_from_ops,
)
from planning_pivot.plan_io import ExtractedPlan, PlanAction, write_plan_file
from planning_pivot.rankers import ActionRanker
from planning_pivot.constrained_search import beam_search, SearchResult


@dataclass(frozen=True)
class RepairResult:
    run_id: str
    instance_id: str
    domain_name: str
    split: str
    source_model_id: str
    source_representation: str
    original_plan_len: int
    valid_prefix_len: int
    repaired: bool
    val_valid: bool
    repaired_plan_path: Path | None
    completion_len: int | None
    seconds: float
    failure_type: str | None


def find_valid_prefix(
    task: SasTask,
    plan: ExtractedPlan,
) -> tuple[list[SasOperator], tuple[int, ...], SasExecutionTrace]:
    lines = [f"({a.name} {' '.join(a.args)})" for a in plan.actions]
    trace = simulate_plan(task, lines)

    lookup = operator_lookup(task)
    prefix_ops: list[SasOperator] = []
    state = task.init

    for i, line in enumerate(lines):
        if i >= trace.valid_prefix_len:
            break
        line_clean = line.strip()
        op = lookup.get(line_clean) or lookup.get(line_clean.lower())
        if op is None:
            inner = line_clean.strip("() ").lower()
            op = lookup.get(inner) or lookup.get("(" + inner + ")")
        if op is None:
            break
        state = apply_operator(task, state, op)
        prefix_ops.append(op)

    return prefix_ops, state, trace


def complete_from_state(
    task: SasTask,
    start_state: tuple[int, ...],
    ranker: ActionRanker,
    max_steps: int,
    beam_width: int,
    context: dict[str, Any],
) -> SearchResult:
    ctx = dict(context)
    ctx["start_state"] = start_state
    return beam_search(
        task, ranker,
        beam_width=beam_width,
        max_steps=max_steps,
        max_expansions=max_steps * beam_width * 2,
        context=ctx,
    )


def repair_by_prefix_completion(
    task: SasTask,
    plan: ExtractedPlan,
    ranker: ActionRanker,
    max_completion_steps: int,
    beam_width: int,
    context: dict[str, Any],
) -> SearchResult:
    import time
    t0 = time.perf_counter()

    prefix_ops, prefix_state, trace = find_valid_prefix(task, plan)

    if is_goal(task, prefix_state):
        return SearchResult(
            solved=True,
            plan_lines=plan_lines_from_ops(prefix_ops),
            final_score=0.0,
            expansions=0,
            generated=0,
            seconds=time.perf_counter() - t0,
            failure_type=None,
        )

    ctx = dict(context)
    ctx["prefix_ops"] = prefix_ops
    ctx["prefix_actions"] = plan_lines_from_ops(prefix_ops)
    ctx["start_state"] = prefix_state

    completion = beam_search(
        task, ranker,
        beam_width=beam_width,
        max_steps=max_completion_steps,
        max_expansions=max_completion_steps * beam_width * 2,
        context=ctx,
    )

    return SearchResult(
        solved=completion.solved,
        plan_lines=completion.plan_lines,
        final_score=completion.final_score,
        expansions=completion.expansions,
        generated=completion.generated,
        seconds=time.perf_counter() - t0,
        failure_type=completion.failure_type,
    )


def run_repair_experiment(config) -> pd.DataFrame:
    from planning_pivot.config import PivotConfig
    from planning_pivot.sas import parse_output_sas
    from planning_pivot.rankers import RandomRanker, GoalCountRanker, TeacherFrequencyRanker
    from planning_pivot.diagnostics import build_instance_index

    output_root = Path(config.data.output_root)
    repair_dir = output_root / "repair"
    repair_dir.mkdir(parents=True, exist_ok=True)

    instances = build_instance_index(config)
    sas_cache_dir = output_root / "sas_cache"

    diag_path = output_root / "diagnostics" / "diagnostics.csv"
    if diag_path.exists():
        diag_df = pd.read_csv(diag_path)
    else:
        diag_df = pd.DataFrame()

    teacher_paths = [
        Path(p) for p in instances["teacher_plan_path"].dropna().unique()
        if p and Path(p).exists()
    ]

    rankers = {
        "random": RandomRanker(seed=config.experiment.random_seed),
        "goal_count": GoalCountRanker(),
    }
    if teacher_paths:
        rankers["teacher_frequency"] = TeacherFrequencyRanker(teacher_paths)

    rows: list[dict] = []

    result_cols = [
        "run_id", "instance_id", "domain_name", "split",
        "source_model_id", "source_representation",
        "original_plan_len", "valid_prefix_len",
        "repaired", "val_valid", "repaired_plan_path",
        "completion_len", "seconds", "failure_type", "ranker",
    ]

    if diag_df.empty:
        df = pd.DataFrame(columns=result_cols)
        df.to_csv(repair_dir / "repair_results.csv", index=False)
        pd.DataFrame().to_csv(repair_dir / "summary.csv", index=False)
        return df

    gen_path = output_root / "diagnostics" / "generation_records.csv"
    if gen_path.exists():
        gen_df = pd.read_csv(gen_path)
    else:
        gen_df = pd.DataFrame()

    invalid_rows = diag_df[diag_df["val_valid"] == False].copy()
    if not gen_df.empty and "instance_id" in gen_df.columns:
        invalid_rows = invalid_rows.merge(
            gen_df[["instance_id", "raw_output_path"]],
            on="instance_id", how="left",
        )

    from planning_pivot.plan_io import parse_plan_actions, extract_plan_text

    sas_cache_local: dict[str, SasTask] = {}
    for _, drow in invalid_rows.iterrows():
        instance_id = str(drow.get("instance_id", ""))
        problem_id = str(drow.get("problem_id", instance_id))
        problem_base = problem_id.rsplit("_", 1)[0] if "_" in problem_id else problem_id

        if problem_id not in sas_cache_local:
            sas_path = sas_cache_dir / problem_id / "output.sas"
            if not sas_path.exists():
                sas_path = sas_cache_dir / problem_base / "output.sas"
            if sas_path.exists():
                try:
                    sas_cache_local[problem_id] = parse_output_sas(sas_path)
                except Exception:
                    continue
            else:
                continue

        task = sas_cache_local.get(problem_id)
        if task is None:
            continue

        raw_path_str = drow.get("raw_output_path", "")
        if pd.isna(raw_path_str) or not raw_path_str:
            continue
        raw_path = Path(str(raw_path_str))
        if not raw_path.exists():
            continue

        raw_text = raw_path.read_text()
        plan = parse_plan_actions(extract_plan_text(raw_text))

        if not plan.actions:
            continue

        teacher_len_val = drow.get("teacher_plan_len")
        teacher_len = int(teacher_len_val) if pd.notna(teacher_len_val) else 20
        max_steps = min(
            int(teacher_len * config.experiment.max_steps_factor),
            config.experiment.max_steps_cap,
        )

        for ranker_name, ranker in rankers.items():
            import time
            t0 = time.perf_counter()

            result = repair_by_prefix_completion(
                task, plan, ranker,
                max_completion_steps=max_steps,
                beam_width=4,
                context={},
            )

            plan_path: Path | None = None
            if result.plan_lines:
                plan_path = repair_dir / f"{problem_base}_{ranker_name}_repair.plan"
                actions = [
                    PlanAction(
                        name=l.strip("() ").split()[0],
                        args=tuple(l.strip("() ").split()[1:]),
                        raw=l,
                    )
                    for l in result.plan_lines
                ]
                write_plan_file(actions, plan_path)

            prefix_ops, _, trace = find_valid_prefix(task, plan)

            rows.append({
                "run_id": str(drow.get("run_id", "")),
                "instance_id": instance_id,
                "domain_name": str(drow.get("domain_name", "")),
                "split": str(drow.get("split", "")),
                "source_model_id": str(drow.get("model_id", "")),
                "source_representation": str(drow.get("representation", "")),
                "original_plan_len": len(plan.actions),
                "valid_prefix_len": len(prefix_ops),
                "repaired": result.solved,
                "val_valid": False,
                "repaired_plan_path": str(plan_path) if plan_path else "",
                "completion_len": len(result.plan_lines) - len(prefix_ops) if result.solved else None,
                "seconds": result.seconds,
                "failure_type": result.failure_type,
                "ranker": ranker_name,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "run_id", "instance_id", "domain_name", "split",
            "source_model_id", "source_representation",
            "original_plan_len", "valid_prefix_len",
            "repaired", "val_valid", "repaired_plan_path",
            "completion_len", "seconds", "failure_type", "ranker",
        ])

    df.to_csv(repair_dir / "repair_results.csv", index=False)

    if not df.empty:
        summary = df.groupby(["ranker", "split"]).agg(
            count=("repaired", "count"),
            repair_rate=("repaired", "mean"),
            mean_prefix_len=("valid_prefix_len", "mean"),
            mean_seconds=("seconds", "mean"),
        ).reset_index()
        summary.to_csv(repair_dir / "summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(repair_dir / "summary.csv", index=False)

    return df
