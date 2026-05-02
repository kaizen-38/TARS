from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Verifier-grounded LLM planning evaluation framework")
console = Console()


@app.command()
def audit(config: Path = typer.Option(Path("configs/pivot_local.yaml"), "--config")):
    """Audit repo structure and tool availability."""
    from planning_pivot.config import load_config
    from planning_pivot.paths import discover_fast_downward, discover_val

    cfg = load_config(config)
    console.print("[bold]Repo audit[/bold]")
    console.print(f"  Repo root: {cfg.data.repo_root}")
    console.print(f"  Output root: {cfg.data.output_root}")

    try:
        fd = discover_fast_downward(cfg.tools.fast_downward)
        console.print(f"  Fast Downward: [green]{fd}[/green]")
    except FileNotFoundError as e:
        console.print(f"  Fast Downward: [red]NOT FOUND[/red] ({e})")

    try:
        val = discover_val(cfg.tools.val)
        console.print(f"  VAL: [green]{val}[/green]")
    except FileNotFoundError as e:
        console.print(f"  VAL: [red]NOT FOUND[/red] ({e})")

    console.print(f"  Train domains: {cfg.experiment.train_domains}")
    console.print(f"  Heldout domains: {cfg.experiment.heldout_domains}")
    console.print(f"  Representations: {cfg.experiment.representations}")


@app.command()
def build_index(config: Path = typer.Option(Path("configs/pivot_local.yaml"), "--config")):
    """Build instance index CSV."""
    from planning_pivot.config import load_config
    from planning_pivot.diagnostics import build_instance_index

    cfg = load_config(config)
    output_root = Path(cfg.data.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df = build_instance_index(cfg)
    out_path = output_root / "instances.csv"
    df.to_csv(out_path, index=False)
    console.print(f"Built instance index: {len(df)} rows -> {out_path}")


@app.command()
def cache_sas(config: Path = typer.Option(Path("configs/pivot_local.yaml"), "--config")):
    """Cache SAS files via Fast Downward translate."""
    from planning_pivot.config import load_config
    from planning_pivot.diagnostics import build_instance_index
    from planning_pivot.paths import discover_fast_downward
    from planning_pivot.fd import run_fd_translate

    cfg = load_config(config)
    output_root = Path(cfg.data.output_root)
    sas_dir = output_root / "sas_cache"

    instances = build_instance_index(cfg)
    try:
        fd_bin = discover_fast_downward(cfg.tools.fast_downward)
    except FileNotFoundError as e:
        console.print(f"[red]Cannot cache SAS: {e}[/red]")
        return

    seen = set()
    success = 0
    fail = 0
    for _, row in instances.iterrows():
        iid = str(row["instance_id"])
        base = iid.rsplit("_", 1)[0] if "_" in iid else iid
        if base in seen:
            continue
        seen.add(base)

        out_dir = sas_dir / base
        if (out_dir / "output.sas").exists():
            success += 1
            continue

        domain_path = Path(row["domain_path"])
        problem_path = Path(row["problem_path"])
        if not domain_path.exists() or not problem_path.exists():
            fail += 1
            continue

        try:
            run_fd_translate(domain_path, problem_path, fd_bin, out_dir)
            success += 1
        except Exception as e:
            console.print(f"  [yellow]Failed {base}: {e}[/yellow]")
            fail += 1

    console.print(f"SAS cache: {success} success, {fail} failed")


@app.command()
def diagnose(config: Path = typer.Option(Path("configs/pivot_local.yaml"), "--config")):
    """Run VAL-grounded failure diagnostics."""
    from planning_pivot.config import load_config
    from planning_pivot.diagnostics import run_diagnostics

    cfg = load_config(config)
    df = run_diagnostics(cfg)
    console.print(f"Diagnostics: {len(df)} records")
    if not df.empty and "val_valid" in df.columns:
        console.print(f"  VAL-valid rate: {df['val_valid'].mean():.3f}")


@app.command()
def constrained_search(config: Path = typer.Option(Path("configs/pivot_local.yaml"), "--config")):
    """Run legality-constrained action search experiment."""
    from planning_pivot.config import load_config
    from planning_pivot.constrained_search import run_constrained_search_experiment

    cfg = load_config(config)
    df = run_constrained_search_experiment(cfg)
    console.print(f"Constrained search: {len(df)} results")
    if not df.empty and "solved_sas" in df.columns:
        console.print(f"  SAS-solved rate: {df['solved_sas'].mean():.3f}")


@app.command()
def repair(config: Path = typer.Option(Path("configs/pivot_local.yaml"), "--config")):
    """Run prefix-salvage repair experiment."""
    from planning_pivot.config import load_config
    from planning_pivot.repair import run_repair_experiment

    cfg = load_config(config)
    df = run_repair_experiment(cfg)
    console.print(f"Repair: {len(df)} results")
    if not df.empty and "repaired" in df.columns:
        console.print(f"  Repair rate: {df['repaired'].mean():.3f}")


@app.command()
def portfolio(config: Path = typer.Option(Path("configs/pivot_local.yaml"), "--config")):
    """Run budgeted portfolio selector experiment."""
    from planning_pivot.config import load_config
    from planning_pivot.portfolio import run_portfolio_experiment

    cfg = load_config(config)
    df = run_portfolio_experiment(cfg)
    console.print(f"Portfolio: {len(df)} results")


@app.command()
def report(config: Path = typer.Option(Path("configs/pivot_local.yaml"), "--config")):
    """Generate figures, tables, and report stub."""
    from planning_pivot.config import load_config
    from planning_pivot.reporting import generate_all_figures, generate_report_tables, generate_report_stub

    cfg = load_config(config)
    output_root = Path(cfg.data.output_root)

    generate_all_figures(output_root)
    generate_report_tables(output_root)
    generate_report_stub(output_root)
    console.print(f"Report generated in {output_root}")


@app.command()
def run_all(config: Path = typer.Option(Path("configs/pivot_local.yaml"), "--config")):
    """Run all pipeline steps in order."""
    audit(config)
    build_index(config)
    diagnose(config)
    constrained_search(config)
    repair(config)
    portfolio(config)
    report(config)


def main():
    app()


if __name__ == "__main__":
    main()
