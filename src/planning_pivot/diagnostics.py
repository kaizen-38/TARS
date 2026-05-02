from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from planning_pivot.config import PivotConfig
from planning_pivot.plan_io import ExtractedPlan, parse_plan_actions, extract_plan_text, write_plan_file
from planning_pivot.val import validate_plan, ValidationResult
from planning_pivot.sas import SasTask, simulate_plan, operator_lookup


@dataclass(frozen=True)
class GenerationRecord:
    run_id: str
    instance_id: str
    domain_name: str
    split: str
    model_id: str
    checkpoint: str
    representation: str
    seed: int | None
    raw_output_path: Path
    extracted_plan_path: Path | None = None


@dataclass(frozen=True)
class DiagnosticRecord:
    run_id: str
    instance_id: str
    domain_name: str
    split: str
    model_id: str
    representation: str
    surface_status: str
    val_valid: bool
    failure_type: str
    plan_len: int
    teacher_plan_len: int | None
    length_ratio: float | None
    executable_prefix_len: int | None
    first_invalid_step: int | None
    schema_valid_action_frac: float | None
    val_seconds: float


def build_instance_index(config: PivotConfig) -> pd.DataFrame:
    repo_root = Path(config.data.repo_root)
    rows: list[dict[str, Any]] = []

    instance_dirs = [
        repo_root / "data" / "generated" / "instances",
        repo_root / "data" / "static",
    ]
    if config.data.domains_root:
        instance_dirs.append(Path(config.data.domains_root))

    all_domains = config.experiment.train_domains + config.experiment.heldout_domains
    train_set = set(config.experiment.train_domains)
    heldout_set = set(config.experiment.heldout_domains)

    for base_dir in instance_dirs:
        if not base_dir.exists():
            continue
        for domain_dir in sorted(base_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain_name = domain_dir.name
            if all_domains and domain_name not in all_domains:
                continue

            split = "train" if domain_name in train_set else "heldout" if domain_name in heldout_set else "unknown"

            for split_dir in sorted(domain_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                if split_dir.name in ("train", "heldout", "test", "val"):
                    actual_split = split_dir.name if split_dir.name in ("train", "heldout") else split
                    _scan_instance_dir(split_dir, domain_name, actual_split, repo_root, rows, config)
                else:
                    _scan_instance_dir(domain_dir, domain_name, split, repo_root, rows, config)
                    break

    if not rows:
        manifest_path = repo_root / "manifests" / "solve_manifest.tsv"
        if manifest_path.exists():
            _scan_from_manifest(manifest_path, repo_root, config, rows)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "instance_id", "domain_name", "split", "domain_path",
            "problem_path", "teacher_plan_path", "teacher_plan_len",
            "representation", "notes",
        ])
    return df


def _scan_instance_dir(
    directory: Path, domain_name: str, split: str,
    repo_root: Path, rows: list[dict], config: PivotConfig,
) -> None:
    problem_files = sorted(directory.glob("*problem*.pddl")) + sorted(directory.glob("instance_*.pddl"))
    domain_files = sorted(directory.glob("*domain*.pddl")) + sorted(directory.glob("domain.pddl"))

    if not problem_files:
        problem_files = [f for f in sorted(directory.glob("*.pddl")) if "domain" not in f.name.lower()]

    domain_file = domain_files[0] if domain_files else None

    seen_problem_ids: set[str] = set()
    for pf in problem_files:
        problem_id = pf.stem.replace("_problem", "")
        if problem_id in seen_problem_ids:
            continue
        seen_problem_ids.add(problem_id)

        df_path = domain_file
        if not df_path:
            candidate = pf.parent / pf.name.replace("problem", "domain")
            if candidate.exists():
                df_path = candidate

        plan_path = _find_teacher_plan(problem_id, repo_root)
        plan_len = _count_plan_actions(plan_path) if plan_path else None

        rows.append({
            "instance_id": problem_id,
            "domain_name": domain_name,
            "split": split,
            "domain_path": str(df_path) if df_path else "",
            "problem_path": str(pf),
            "teacher_plan_path": str(plan_path) if plan_path else "",
            "teacher_plan_len": plan_len,
            "representation": "all",
            "notes": "",
        })


def _scan_from_manifest(
    manifest_path: Path, repo_root: Path, config: PivotConfig, rows: list[dict],
) -> None:
    df = pd.read_csv(manifest_path, sep="\t")
    train_set = set(config.experiment.train_domains)
    heldout_set = set(config.experiment.heldout_domains)

    for _, row in df.iterrows():
        instance_id = str(row.get("instance_id", ""))
        domain_path = repo_root / row.get("domain_file", "")
        problem_path = repo_root / row.get("problem_file", "")

        parts = instance_id.split("_")
        domain_name = parts[0] if parts else "unknown"
        split = "train" if domain_name in train_set else "heldout" if domain_name in heldout_set else "unknown"

        plan_path = _find_teacher_plan(instance_id, repo_root)
        plan_len = _count_plan_actions(plan_path) if plan_path else None

        rows.append({
            "instance_id": instance_id,
            "domain_name": domain_name,
            "split": split,
            "domain_path": str(domain_path),
            "problem_path": str(problem_path),
            "teacher_plan_path": str(plan_path) if plan_path else "",
            "teacher_plan_len": plan_len,
            "representation": "all",
            "notes": "from_manifest",
        })


def _find_teacher_plan(problem_id: str, repo_root: Path) -> Path | None:
    plans_root = repo_root / "data" / "generated" / "plans"
    candidates = [
        plans_root / f"{problem_id}.plan.pddl",
        plans_root / f"{problem_id}.plan",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _count_plan_actions(plan_path: Path) -> int | None:
    if not plan_path or not plan_path.exists():
        return None
    text = plan_path.read_text()
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith(";"):
            count += 1
    return count


def classify_surface(plan: ExtractedPlan) -> str:
    return plan.surface_status


def compute_schema_valid_action_frac(plan: ExtractedPlan, task: SasTask | None) -> float | None:
    if task is None or not plan.actions:
        return None
    op_schemas: set[str] = set()
    for op in task.operators:
        op_schemas.add(op.name.split()[0].lower())

    valid = 0
    for action in plan.actions:
        if action.name.lower() in op_schemas:
            valid += 1
    return valid / len(plan.actions)


def diagnose_one_generation(
    row: pd.Series,
    instances: pd.DataFrame,
    sas_task: SasTask | None,
    val_bin: Path | None,
    output_dir: Path,
) -> DiagnosticRecord:
    raw_path = Path(row["raw_output_path"])
    run_id = str(row.get("run_id", ""))
    instance_id = str(row.get("instance_id", ""))
    problem_id = str(row.get("problem_id", instance_id))
    domain_name = str(row.get("domain_name", ""))
    split = str(row.get("split", ""))
    model_id = str(row.get("model_id", ""))
    representation = str(row.get("representation", ""))

    raw_text = raw_path.read_text() if raw_path.exists() else ""
    plan_text = extract_plan_text(raw_text)
    plan = parse_plan_actions(plan_text)
    surface = classify_surface(plan)

    plan_len = len(plan.actions)

    teacher_plan_len: int | None = None
    inst_match = instances[instances["instance_id"] == problem_id]
    if not inst_match.empty:
        tpl = inst_match.iloc[0].get("teacher_plan_len")
        if pd.notna(tpl):
            teacher_plan_len = int(tpl)

    length_ratio = plan_len / teacher_plan_len if teacher_plan_len and teacher_plan_len > 0 else None

    exec_prefix_len: int | None = None
    first_invalid: int | None = None
    if sas_task and plan.actions:
        lines = [f"({a.name} {' '.join(a.args)})" for a in plan.actions]
        trace = simulate_plan(sas_task, lines)
        exec_prefix_len = trace.valid_prefix_len
        first_invalid = trace.first_invalid_step

    schema_frac = compute_schema_valid_action_frac(plan, sas_task)

    val_valid = False
    failure_type = "no_validation"
    val_seconds = 0.0

    if val_bin and plan.actions and not inst_match.empty:
        irow = inst_match.iloc[0]
        domain_path = Path(str(irow["domain_path"]))
        problem_path = Path(str(irow["problem_path"]))

        if domain_path.exists() and problem_path.exists():
            safe_name = f"{problem_id}_{representation}".replace("/", "_")
            plan_file = output_dir / f"{safe_name}_plan.pddl"
            write_plan_file(plan.actions, plan_file)
            vr = validate_plan(domain_path, problem_path, plan_file, val_bin)
            val_valid = vr.valid
            failure_type = vr.failure_type
            val_seconds = vr.seconds

    if failure_type == "no_validation":
        if not plan.actions:
            failure_type = surface
        elif val_bin is None:
            failure_type = "not_validated"

    return DiagnosticRecord(
        run_id=run_id,
        instance_id=instance_id,
        domain_name=domain_name,
        split=split,
        model_id=model_id,
        representation=representation,
        surface_status=surface,
        val_valid=val_valid,
        failure_type=failure_type,
        plan_len=plan_len,
        teacher_plan_len=teacher_plan_len,
        length_ratio=length_ratio,
        executable_prefix_len=exec_prefix_len,
        first_invalid_step=first_invalid,
        schema_valid_action_frac=schema_frac,
        val_seconds=val_seconds,
    )


def load_generation_records(config: PivotConfig, instances: pd.DataFrame) -> pd.DataFrame:
    repo_root = Path(config.data.repo_root)
    rows: list[dict[str, Any]] = []

    eval_dirs = sorted((repo_root / "runs").iterdir()) if (repo_root / "runs").exists() else []
    for eval_dir in eval_dirs:
        if not eval_dir.is_dir():
            continue
        run_log = eval_dir / "run_log.jsonl"
        raw_dir = eval_dir / "raw"
        if not run_log.exists():
            continue

        for line in run_log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            problem_id = rec.get("problem_id", "")
            representation = rec.get("representation", "unknown")
            raw_file = raw_dir / f"{problem_id}_{representation}.txt"

            rows.append({
                "run_id": rec.get("run_id", eval_dir.name),
                "instance_id": f"{problem_id}_{representation}",
                "problem_id": problem_id,
                "domain_name": rec.get("domain", ""),
                "split": rec.get("split", "unknown"),
                "model_id": rec.get("model_name", "unknown"),
                "checkpoint": rec.get("checkpoint_path", ""),
                "representation": representation,
                "seed": rec.get("seed"),
                "raw_output_path": str(raw_file),
                "num_actions_logged": rec.get("num_actions", 0),
                "valid_plan_logged": rec.get("valid_plan", False),
                "generated_tokens": rec.get("generated_tokens", 0),
            })

    if not rows:
        gen_roots = []
        if config.data.generations_root:
            gen_roots.append(Path(config.data.generations_root))
        for gen_root in gen_roots:
            if not gen_root.exists():
                continue
            for output_file in sorted(gen_root.rglob("*.txt")):
                if output_file.stat().st_size == 0:
                    continue
                rows.append({
                    "run_id": hashlib.md5(str(output_file).encode()).hexdigest()[:12],
                    "instance_id": output_file.stem,
                    "problem_id": output_file.stem,
                    "domain_name": output_file.stem.split("_")[0],
                    "split": "unknown",
                    "model_id": "unknown",
                    "checkpoint": "",
                    "representation": "unknown",
                    "seed": None,
                    "raw_output_path": str(output_file),
                    "num_actions_logged": None,
                    "valid_plan_logged": None,
                    "generated_tokens": None,
                })

    cols = [
        "run_id", "instance_id", "problem_id", "domain_name", "split",
        "model_id", "checkpoint", "representation", "seed",
        "raw_output_path", "num_actions_logged", "valid_plan_logged",
        "generated_tokens",
    ]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def run_diagnostics(config: PivotConfig) -> pd.DataFrame:
    from planning_pivot.paths import discover_val
    from planning_pivot.sas import parse_output_sas

    output_root = Path(config.data.output_root)
    diag_dir = output_root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    instances = build_instance_index(config)
    instances.to_csv(output_root / "instances.csv", index=False)

    gen_records = load_generation_records(config, instances)
    gen_records.to_csv(diag_dir / "generation_records.csv", index=False)

    try:
        val_bin = discover_val(config.tools.val)
    except FileNotFoundError:
        val_bin = None

    sas_cache_dir = output_root / "sas_cache"
    sas_cache: dict[str, SasTask] = {}

    records: list[dict] = []
    total = len(gen_records)
    for idx, (_, row) in enumerate(gen_records.iterrows()):
        if idx % 100 == 0 and idx > 0:
            print(f"  Diagnosing {idx}/{total}...")

        try:
            problem_id = str(row.get("problem_id", ""))

            sas_task = sas_cache.get(problem_id)
            if sas_task is None and problem_id:
                sas_path = sas_cache_dir / problem_id / "output.sas"
                if not sas_path.exists():
                    stripped = problem_id.rsplit("_", 1)[0] if "_" in problem_id else problem_id
                    sas_path = sas_cache_dir / stripped / "output.sas"
                if sas_path.exists():
                    try:
                        sas_task = parse_output_sas(sas_path)
                        sas_cache[problem_id] = sas_task
                    except Exception:
                        pass

            dr = diagnose_one_generation(row, instances, sas_task, val_bin, diag_dir)
            records.append({
                "run_id": dr.run_id,
                "instance_id": dr.instance_id,
                "problem_id": problem_id,
                "domain_name": dr.domain_name,
                "split": dr.split,
                "model_id": dr.model_id,
                "representation": dr.representation,
                "surface_status": dr.surface_status,
                "val_valid": dr.val_valid,
                "failure_type": dr.failure_type,
                "plan_len": dr.plan_len,
                "teacher_plan_len": dr.teacher_plan_len,
                "length_ratio": dr.length_ratio,
                "executable_prefix_len": dr.executable_prefix_len,
                "first_invalid_step": dr.first_invalid_step,
                "schema_valid_action_frac": dr.schema_valid_action_frac,
                "val_seconds": dr.val_seconds,
            })
        except Exception as e:
            records.append({
                "run_id": str(row.get("run_id", "")),
                "instance_id": str(row.get("instance_id", "")),
                "problem_id": str(row.get("problem_id", "")),
                "domain_name": str(row.get("domain_name", "")),
                "split": str(row.get("split", "")),
                "model_id": str(row.get("model_id", "")),
                "representation": str(row.get("representation", "")),
                "surface_status": "error",
                "val_valid": False,
                "failure_type": f"diagnostic_error: {e}",
                "plan_len": 0,
                "teacher_plan_len": None,
                "length_ratio": None,
                "executable_prefix_len": None,
                "first_invalid_step": None,
                "schema_valid_action_frac": None,
                "val_seconds": 0.0,
            })

    diag_df = pd.DataFrame(records)
    diag_df.to_csv(diag_dir / "diagnostics.csv", index=False)

    if not diag_df.empty and "model_id" in diag_df.columns:
        summary_mr = diag_df.groupby(["model_id", "representation"]).agg(
            count=("val_valid", "count"),
            val_valid_rate=("val_valid", "mean"),
            empty_plan_rate=("surface_status", lambda x: (x == "empty_plan").mean()),
            parsed_action_rate=("surface_status", lambda x: (x == "parsed_actions").mean()),
            mean_plan_len=("plan_len", "mean"),
            mean_schema_valid_frac=("schema_valid_action_frac", "mean"),
            mean_exec_prefix=("executable_prefix_len", "mean"),
        ).reset_index()
        summary_mr.to_csv(diag_dir / "summary_by_model_representation.csv", index=False)

        summary_dom = diag_df.groupby(["domain_name", "representation"]).agg(
            count=("val_valid", "count"),
            val_valid_rate=("val_valid", "mean"),
            mean_plan_len=("plan_len", "mean"),
            parsed_action_rate=("surface_status", lambda x: (x == "parsed_actions").mean()),
            mean_exec_prefix=("executable_prefix_len", "mean"),
        ).reset_index()
        summary_dom.to_csv(diag_dir / "summary_by_domain.csv", index=False)
    else:
        pd.DataFrame().to_csv(diag_dir / "summary_by_model_representation.csv", index=False)
        pd.DataFrame().to_csv(diag_dir / "summary_by_domain.csv", index=False)

    return diag_df
