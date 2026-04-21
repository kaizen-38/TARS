"""Generate TSV manifests for Slurm job arrays.

Usage:
    # Manifest for generate step (one row per domain):
    python scripts/gen_manifests.py generate --mode smoke
    python scripts/gen_manifests.py generate --mode pilot

    # Manifests for solve/validate steps (one row per instance):
    python scripts/gen_manifests.py solve
    python scripts/gen_manifests.py validate

    # Manifest for eval step (one row per domain):
    python scripts/gen_manifests.py eval --split heldout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import yaml

app = typer.Typer(add_completion=False)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SPLITS_CONFIG = _REPO_ROOT / "configs" / "splits" / "phase1_v1.yaml"
_INSTANCES_DIR = _REPO_ROOT / "data" / "generated" / "instances"
_PLANS_DIR = _REPO_ROOT / "data" / "generated" / "plans"
_MANIFESTS_DIR = _REPO_ROOT / "manifests"


def _load_config() -> dict:
    with open(_SPLITS_CONFIG) as f:
        return yaml.safe_load(f)


@app.command()
def generate(mode: str = typer.Option("smoke", help="'smoke' or 'pilot'")) -> None:
    """Write manifests/generate_manifest.tsv (one row per domain)."""
    cfg = _load_config()
    n = cfg[mode + "_test"]["instances_per_domain"] if mode == "smoke" else cfg["pilot"]["instances_per_domain"]
    all_domains = cfg["train_domains"] + cfg["heldout_domains"]

    _MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _MANIFESTS_DIR / f"generate_{mode}.tsv"
    rows = []
    for domain in all_domains:
        split = "train" if domain in cfg["train_domains"] else "heldout"
        rows.append(f"{domain}\t{split}\t{n}\t42")
    out.write_text("domain\tsplit\tn_instances\tseed\n" + "\n".join(rows) + "\n")
    print(f"Wrote {len(rows)} rows → {out}")


@app.command()
def solve() -> None:
    """Write manifests/solve_manifest.tsv from generated instance metadata."""
    meta_files = sorted(_INSTANCES_DIR.rglob("*_meta.json"))
    if not meta_files:
        print("ERROR: No instance metadata found. Run generate step first.", file=sys.stderr)
        raise SystemExit(1)

    _MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _MANIFESTS_DIR / "solve_manifest.tsv"
    rows = []
    for mf in meta_files:
        meta = json.loads(mf.read_text())
        rows.append(
            f"{meta['instance_id']}\t{meta['domain_file']}\t{meta['problem_file']}"
        )
    out.write_text("instance_id\tdomain_file\tproblem_file\n" + "\n".join(rows) + "\n")
    print(f"Wrote {len(rows)} rows → {out}")


@app.command()
def validate() -> None:
    """Write manifests/validate_manifest.tsv from solved plan metadata."""
    solve_meta_files = sorted(_PLANS_DIR.rglob("*.solve_meta.json"))
    if not solve_meta_files:
        print("ERROR: No solve metadata found. Run solve step first.", file=sys.stderr)
        raise SystemExit(1)

    _MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _MANIFESTS_DIR / "validate_manifest.tsv"
    rows = []
    for smf in solve_meta_files:
        meta = json.loads(smf.read_text())
        if not meta.get("success") or not meta.get("normalized_plan_file"):
            continue
        rows.append(
            f"{meta['problem_id']}\t{meta['domain_file']}\t"
            f"{meta['problem_file']}\t{meta['normalized_plan_file']}"
        )
    out.write_text(
        "instance_id\tdomain_file\tproblem_file\tplan_file\n" + "\n".join(rows) + "\n"
    )
    print(f"Wrote {len(rows)} rows → {out}")


@app.command()
def eval(split: str = typer.Option("heldout")) -> None:
    """Write manifests/eval_{split}.tsv (one row per domain for array jobs)."""
    cfg = _load_config()
    if split == "train":
        domains = cfg["train_domains"]
    elif split == "heldout":
        domains = cfg["heldout_domains"]
    else:
        domains = cfg["train_domains"] + cfg["heldout_domains"]

    _MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _MANIFESTS_DIR / f"eval_{split}.tsv"
    rows = [f"{d}\t{split}" for d in domains]
    out.write_text("domain\tsplit\n" + "\n".join(rows) + "\n")
    print(f"Wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    app()
