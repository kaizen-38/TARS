"""thicket_phase1 CLI entry point.

Usage:
    thicket --help
    thicket generate-smoke
    thicket solve-smoke
    thicket build-dataset
    thicket train --mode lora_debug
    thicket eval --split heldout
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="thicket",
    help="Phase 1 PDDL planning baseline — data + training + evaluation pipeline.",
    add_completion=False,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPLITS_CONFIG = _REPO_ROOT / "configs" / "splits" / "phase1_v1.yaml"


@app.command()
def generate_smoke(
    seed: int = typer.Option(42, help="Random seed"),
    splits_config: Path = typer.Option(_SPLITS_CONFIG),
    output_dir: Path = typer.Option(_REPO_ROOT / "data" / "generated" / "instances"),
) -> None:
    """Generate smoke-test instances (5 per domain)."""
    from utils.io import load_yaml
    from utils.seeds import set_global_seed
    from generation.generate_instances import generate_domain_split

    set_global_seed(seed)
    cfg = load_yaml(splits_config)
    n = cfg["smoke_test"]["instances_per_domain"]

    all_domains = cfg["train_domains"] + cfg["heldout_domains"]
    for domain in all_domains:
        split = "train" if domain in cfg["train_domains"] else "heldout"
        typer.echo(f"Generating {n} instances for {domain} ({split})")
        try:
            generate_domain_split(
                domain=domain,
                split=split,
                n_instances=n,
                seed=seed,
                output_dir=output_dir,
            )
        except Exception as exc:
            typer.echo(f"  WARNING: {exc}", err=True)


@app.command()
def solve_smoke(
    seed: int = typer.Option(42),
    backend: str = typer.Option("fd", help="Planner backend"),
    splits_config: Path = typer.Option(_SPLITS_CONFIG),
    instances_dir: Path = typer.Option(_REPO_ROOT / "data" / "generated" / "instances"),
    plans_dir: Path = typer.Option(_REPO_ROOT / "data" / "generated" / "plans"),
    tuples_dir: Path = typer.Option(_REPO_ROOT / "data" / "generated" / "tuples_standard"),
) -> None:
    """Solve smoke instances with the teacher planner and run VAL."""
    from generation.solve_with_fd import solve_instance
    from generation.validate_with_val import validate_plan
    from pddl_ops.anonymize import anonymize_triple, save_anonymized_triple
    from pddl_ops.compact_serialize import actions_to_compact, save_compact_plan
    from dataset.build_sft_dataset import build_tuple_json
    from utils.io import load_json
    from utils.logging import get_logger

    log = get_logger("solve_smoke")

    meta_files = sorted(instances_dir.rglob("*_meta.json"))
    log.info("Found %d instances to solve", len(meta_files))

    for meta_path in meta_files:
        meta = load_json(meta_path)
        instance_id = meta["instance_id"]
        domain_file = _REPO_ROOT / meta["domain_file"]
        problem_file = _REPO_ROOT / meta["problem_file"]

        if not domain_file.exists() or not problem_file.exists():
            log.warning("Missing PDDL for %s", instance_id)
            continue

        # Solve
        solve_result = solve_instance(
            problem_id=instance_id,
            domain_file=domain_file,
            problem_file=problem_file,
            output_dir=plans_dir / meta["domain"],
            backend=backend,
        )

        if not solve_result.success or not solve_result.action_sequence:
            log.warning("No plan for %s", instance_id)
            continue

        # Validate
        plan_file = Path(solve_result.normalized_plan_file)
        validate_plan(
            problem_id=instance_id,
            domain_file=domain_file,
            problem_file=problem_file,
            plan_file=plan_file,
            output_dir=plans_dir / meta["domain"],
        )

        # Build anonymized + compact versions
        domain_text = domain_file.read_text()
        problem_text = problem_file.read_text()
        plan_actions = solve_result.action_sequence

        anon_d, anon_p, anon_plan, mapping = anonymize_triple(
            domain_text, problem_text, plan_actions, instance_id
        )
        compact_text = actions_to_compact(plan_actions)

        # Write tuple JSON for dataset builder
        build_tuple_json(
            instance_id=instance_id,
            domain_text=domain_text,
            problem_text=problem_text,
            plan_actions=plan_actions,
            anon_domain_text=anon_d,
            anon_problem_text=anon_p,
            anon_plan_actions=anon_plan,
            compact_plan=compact_text,
            output_dir=tuples_dir,
        )
        log.info("Done: %s", instance_id)


@app.command()
def build_dataset(
    split: str = typer.Option("train"),
) -> None:
    """Build Alpaca JSONL datasets from solved tuples."""
    from dataset.build_sft_dataset import SFTDatasetBuilder
    from dataset.dedupe import deduplicate_all
    from dataset.stats import print_dataset_stats
    from utils.io import ensure_dir

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    builder = SFTDatasetBuilder()
    counts = builder.build_all(split=split)

    typer.echo(f"Dataset counts: {counts}")
    deduplicate_all(_REPO_ROOT / "data" / "datasets" / "alpaca")
    print_dataset_stats(_REPO_ROOT / "data" / "datasets" / "alpaca")


@app.command()
def train(
    mode: str = typer.Option("lora_debug", help="'full' or 'lora_debug'"),
    model: str = typer.Option("Qwen/Qwen3-1.7B"),
    dry_run: bool = typer.Option(False),
) -> None:
    """Launch SFT training via LLaMA-Factory."""
    import subprocess
    import sys

    cmd = [
        sys.executable, str(_REPO_ROOT / "src" / "training" / "launch_sft.py"),
        "--mode", mode,
        "--model", model,
    ]
    if dry_run:
        cmd.append("--dry-run")
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)


@app.command()
def eval(
    checkpoint_path: str = typer.Option(...),
    split: str = typer.Option("heldout"),
    seed: int = typer.Option(42),
) -> None:
    """Run greedy evaluation."""
    import subprocess
    import sys

    cmd = [
        sys.executable, str(_REPO_ROOT / "src" / "inference" / "run_greedy_eval.py"),
        "--checkpoint-path", checkpoint_path,
        "--split", split,
        "--seed", str(seed),
    ]
    subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)


if __name__ == "__main__":
    app()
